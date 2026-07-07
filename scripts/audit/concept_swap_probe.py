"""Concept-swap + verbal-report probe: replicate the "think of a sport" style experiment
against our own NLA oracle (v22 / exp2_dirfix / any train_v18 checkpoint).

Pipeline:
  1. Run a TARGET model (default: Qwen3-1.7B, same trunk the oracle reads natively) on a
     prompt that elicits a concept (e.g. "My favorite sport to play is"), capture the
     hidden state at the last prompt token across several depth-fraction layers.
  2. Feed that raw activation through the ORACLE's actor_template (open verbalization,
     NOT the detect_qa Yes/No probe) via `<explanation>` generation — check what it says.
  3. Capture a second "concept B" hidden state from a contrastive prompt (e.g. priming
     with Rugby), patch it into the FIRST prompt's forward pass at the same layer/position,
     let the target model continue generating — check if its own completion flips.
  4. Verbalize the patched (= concept-B) hidden state through the oracle too — check if the
     oracle's report flips to match, independent of whether the target model's own output
     flipped (tests whether the oracle reads the causally-relevant latent, not something else).

This is exploratory research code (single-example, qualitative), not a production eval script.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import EXPLANATION_OPEN, EXPLANATION_CLOSE, extract_explanation, normalize_activation


def _tokenize_chat(tok, text):
    p_ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                     tokenize=True, add_generation_prompt=True)
    if hasattr(p_ids, "keys"):
        p_ids = p_ids["input_ids"]
    elif hasattr(p_ids, "ids"):
        p_ids = p_ids.ids
    return p_ids


@torch.no_grad()
def capture_hidden(target_model, tok, prompt: str, layer_idx: int, device: str, pos: int = -1):
    """Run `prompt` through the target model, return the residual-stream hidden state at
    `layer_idx` (post-block output), token position `pos` (default: last token)."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = target_model(ids, output_hidden_states=True)
    # hidden_states[0] = embeddings, hidden_states[i] = output of block i-1 (1-indexed blocks)
    h = out.hidden_states[layer_idx + 1][0, pos, :].float().clone()
    return h, ids


@contextlib.contextmanager
def patch_hidden(target_model, layer_idx: int, pos: int, vec: torch.Tensor):
    """Overwrite the residual stream at `layer_idx` (post-block), token `pos`, with `vec`
    during the next forward/generate call on `target_model`."""
    layers = resolve_decoder_layers(target_model)

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[0, pos, :] = vec.to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class Oracle:
    """Minimal loader for a train_v18 checkpoint (trunk + LoRA + ModelPoolAdapters), with
    an open-verbalization `.report()` method (actor_template + <explanation> generation)
    instead of eval_harness.py's detect_qa Yes/No scoring."""

    def __init__(self, ckpt_dir: str, device: str = "cuda", dtype=torch.bfloat16):
        vdir = Path(ckpt_dir)
        meta = json.loads((vdir / "v18_meta.json").read_text())
        self.d_shared = int(meta["d_shared"])
        self.inj_scale = math.sqrt(self.d_shared)
        tkm = meta["tokens"]
        self.inj_id = int(tkm["injection_token_id"])
        self.left_id = int(tkm["injection_left_neighbor_id"])
        self.right_id = int(tkm["injection_right_neighbor_id"])
        self.inj_char = tkm["injection_char"]
        self.template = meta["actor_template"]
        self.trunk_id = meta["trunk"]
        self.device = device

        self.tok = AutoTokenizer.from_pretrained(self.trunk_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(self.trunk_id, torch_dtype=dtype, attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.adapters = ModelPoolAdapters.load(str(vdir / "adapters")).to(device).eval()
        self.embed = self.model.get_input_embeddings()
        print(f"[oracle] loaded {ckpt_dir} — trunk {self.trunk_id}, {len(self.adapters.tags)} enc tags")

    @torch.no_grad()
    def report(self, tag: str, h_raw: torch.Tensor, max_new_tokens: int = 60) -> str:
        """Open-ended verbalization of one raw activation vector `h_raw` [d_M] for `tag`."""
        ptxt = self.template.format(model_tag=tag, injection_char=self.inj_char)
        p_ids = _tokenize_chat(self.tok, ptxt)
        p_ids_t = torch.tensor([p_ids], device=self.device)
        base_emb = self.embed(p_ids_t)
        V = normalize_activation(self.adapters.encode(tag, h_raw.unsqueeze(0).to(self.device)), self.inj_scale)
        V = V.to(base_emb.dtype)
        emb = inject_at_marked_positions(p_ids_t, base_emb, V, self.inj_id, self.left_id, self.right_id)
        attn = torch.ones_like(p_ids_t)
        out = self.model.generate(
            inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=self.tok.pad_token_id,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        expl = extract_explanation(text)
        return expl if expl else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True, help="train_v18 checkpoint dir (v22 / exp2_dirfix / ...)")
    ap.add_argument("--target-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--oracle-tag", default="qwen3-1p7b", help="enc tag matching --target-model")
    ap.add_argument("--prompt-a", default="My favorite sport to play is")
    ap.add_argument("--prompt-b", default="Think carefully about rugby for a moment. My favorite sport to play is")
    ap.add_argument("--depth-fractions", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"[target] loading {args.target_model}")
    target_tok = AutoTokenizer.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    n_layers = len(resolve_decoder_layers(target_model))
    fractions = [float(f) for f in args.depth_fractions.split(",")]
    layer_idxs = [int(round(f * (n_layers - 1))) for f in fractions]

    oracle = Oracle(args.oracle_dir, device=args.device)

    print(f"\n=== natural continuation, prompt A ===\n{args.prompt_a}")
    ids_a = target_tok(args.prompt_a, return_tensors="pt").input_ids.to(args.device)
    gen_a = target_model.generate(ids_a, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   pad_token_id=target_tok.pad_token_id)
    print("->", target_tok.decode(gen_a[0][ids_a.shape[1]:], skip_special_tokens=True))

    print(f"\n=== natural continuation, prompt B (rugby-primed) ===\n{args.prompt_b}")
    ids_b = target_tok(args.prompt_b, return_tensors="pt").input_ids.to(args.device)
    gen_b = target_model.generate(ids_b, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   pad_token_id=target_tok.pad_token_id)
    print("->", target_tok.decode(gen_b[0][ids_b.shape[1]:], skip_special_tokens=True))

    results = []
    for frac, layer_idx in zip(fractions, layer_idxs):
        h_a, _ = capture_hidden(target_model, target_tok, args.prompt_a, layer_idx, args.device)
        h_b, _ = capture_hidden(target_model, target_tok, args.prompt_b, layer_idx, args.device)

        report_a = oracle.report(args.oracle_tag, h_a, args.max_new_tokens)
        report_b = oracle.report(args.oracle_tag, h_b, args.max_new_tokens)

        # patch: prompt A's forward pass, last token at this layer, overwritten with h_b's
        # activation -> continue generating from the target model itself
        with patch_hidden(target_model, layer_idx, -1, h_b):
            ids_a2 = target_tok(args.prompt_a, return_tensors="pt").input_ids.to(args.device)
            gen_patched = target_model.generate(ids_a2, max_new_tokens=args.max_new_tokens,
                                                 do_sample=False, pad_token_id=target_tok.pad_token_id)
        patched_completion = target_tok.decode(gen_patched[0][ids_a2.shape[1]:], skip_special_tokens=True)

        print(f"\n--- depth_fraction={frac} (layer {layer_idx}/{n_layers}) ---")
        print(f"  oracle report on h_A (natural, 'soccer'-context): {report_a!r}")
        print(f"  oracle report on h_B (rugby-primed):              {report_b!r}")
        print(f"  target model completion after patching A<-B:     {patched_completion!r}")
        results.append({
            "depth_fraction": frac, "layer_idx": layer_idx,
            "report_a": report_a, "report_b": report_b,
            "patched_completion": patched_completion,
        })

    out_path = Path("concept_swap_probe_results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
