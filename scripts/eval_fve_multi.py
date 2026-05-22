"""Per-model FVE evaluation for the universal NLA stack.

Two regimes:
  - "AR alone" (upper bound): feed the teacher gold z into AR → ĥ_shared,
    then dec_M(ĥ_shared) → ĥ_M, FVE_M(h_M, ĥ_M). Bounds how well the
    universal AR can reconstruct given a perfect verbalisation.
  - "Pipeline" (headline): use AV-generated z. AV(enc_M(h_M)) → z_av,
    then AR(z_av) → ĥ_shared → dec_M(ĥ_shared) → ĥ_M.

Reports per-tag FVE for both regimes plus the gap (= AR-alone − Pipeline).
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.data_multi import MultiModelActivationDataset
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.models import NLACriticModel
from nla.schema import EXPLANATION_OPEN, EXPLANATION_CLOSE, extract_explanation, normalize_activation


CRITIC_TEMPLATE = "Summary of the following text: <text>{z}</text> <summary>"


def build_critic_prompt(z: str) -> str:
    return CRITIC_TEMPLATE.format(z=z.strip())


@torch.no_grad()
def av_generate_batch(av, tokenizer, prompt_text: str, inj_vecs: torch.Tensor,
                      inj_id: int, left_id: int, right_id: int, max_new_tokens: int,
                      device: str) -> list[str]:
    """All passages in this batch share the same prompt string (tag-level batching),
    so input_ids are identical across rows — we just stack and inject per-row vecs.

    Returns list of length B with raw decoded continuation strings.
    """
    B = inj_vecs.shape[0]
    p_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True, add_generation_prompt=True,
    )
    input_ids = torch.tensor([p_ids], dtype=torch.long, device=device).expand(B, -1).contiguous()
    embed = av.get_input_embeddings()(input_ids)
    inj = inj_vecs.to(device, dtype=embed.dtype)            # [B, d_shared]
    embed = inject_at_marked_positions(input_ids, embed, inj, inj_id, left_id, right_id)
    attn = torch.ones_like(input_ids)
    out = av.generate(
        inputs_embeds=embed, attention_mask=attn,
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


@torch.no_grad()
def ar_reconstruct_batch(ar, ar_tokenizer, z_texts: list[str], device: str,
                          d_shared: int, inj_scale: float) -> torch.Tensor:
    """Returns [B, d_shared] normalized reconstructions at each row's last
    content-token position."""
    prompts = [build_critic_prompt(z) for z in z_texts]
    enc = ar_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512, add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    out = ar(input_ids=input_ids, attention_mask=attn)
    lengths = attn.sum(dim=1) - 1                            # last content position
    idx = lengths.view(-1, 1, 1).expand(-1, 1, d_shared)
    pred = out.values.gather(1, idx).squeeze(1).float()      # [B, d_shared]
    return normalize_activation(pred, inj_scale)


def per_tag_fve(h_pred: torch.Tensor, h_gold: torch.Tensor) -> float:
    """Raw FVE in whatever space the inputs live in. Sensitive to magnitude."""
    resid_var = (h_gold - h_pred).var(unbiased=False).item()
    gold_var = h_gold.var(unbiased=False).item()
    return 1.0 - resid_var / max(gold_var, 1e-12)


def per_tag_fve_meannorm(h_pred: torch.Tensor, h_gold: torch.Tensor) -> float:
    """FVE_meannorm — Anthropic-paper-style: both pred and gold L2-normalized
    to sqrt(d) per row before MSE. Removes magnitude mismatch between the
    raw gold activations and the dec_M-projected AR output. Matches the
    existing single-model project's `FVE_pipeline_meannorm` definition."""
    d = h_gold.shape[-1]
    scale = math.sqrt(d)
    p = normalize_activation(h_pred.float(), scale)
    g = normalize_activation(h_gold.float(), scale)
    resid_var = (g - p).var(unbiased=False).item()
    gold_var = g.var(unbiased=False).item()
    return 1.0 - resid_var / max(gold_var, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-save-dir", required=True)
    ap.add_argument("--ar-save-dir", required=True)
    ap.add_argument("--adapters-dir", default=None,
                    help="defaults to <av-save-dir>/adapters")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--tags", required=True,
                    help="comma-separated tags to evaluate")
    ap.add_argument("--n-passages", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--regime", choices=["ar_alone", "pipeline", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    av_dir = Path(args.av_save_dir)
    ar_dir = Path(args.ar_save_dir)
    av_meta = yaml.safe_load((av_dir / "nla_meta.yaml").read_text())
    av_base = av_meta["av_base"]
    av_lora = av_dir / "av"
    template = av_meta["prompt_templates"]["actor"]
    tok = av_meta["tokens"]
    inj_id = int(tok["injection_token_id"])
    left_id = int(tok["injection_left_neighbor_id"])
    right_id = int(tok["injection_right_neighbor_id"])
    injection_char = tok["injection_char"]
    d_shared = int(av_meta["d_shared"])
    inj_scale = math.sqrt(d_shared)
    adapters_dir = Path(args.adapters_dir) if args.adapters_dir else (av_dir / "adapters")

    tags = [t.strip() for t in args.tags.split(",")]

    # AV
    av_tok = AutoTokenizer.from_pretrained(av_base)
    if av_tok.pad_token_id is None:
        av_tok.pad_token = av_tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(av_base, torch_dtype=dtype).to(device).eval()
    av = PeftModel.from_pretrained(base, str(av_lora)).to(device).eval()

    # AR (NLACriticModel via PEFT). Sidecar may live at <ar_dir>/nla_meta.yaml
    # (legacy AR SFT layout) or at <ar_dir>/ar/nla_meta.yaml (joint-RL layout).
    ar_meta_path = ar_dir / "nla_meta.yaml"
    if not ar_meta_path.exists() or "ar_base" not in yaml.safe_load(ar_meta_path.read_text()):
        nested = ar_dir / "ar" / "nla_meta.yaml"
        if nested.exists():
            ar_meta_path = nested
    ar_meta = yaml.safe_load(ar_meta_path.read_text())
    ar_base = ar_meta["ar_base"]
    layer_index = int(ar_meta["layer_index"])
    ar_tok = AutoTokenizer.from_pretrained(ar_base)
    if ar_tok.pad_token_id is None:
        ar_tok.pad_token = ar_tok.eos_token
    # Right-pad so the "last content token" is at index (attn.sum-1) per row.
    ar_tok.padding_side = "right"
    base_critic = NLACriticModel.from_pretrained(ar_base, nla_num_layers=layer_index,
                                                 torch_dtype=dtype, attn_implementation="sdpa")
    # value_head was identity-init at AR-SFT start and never trained (PEFT freezes
    # non-target_modules params). PeftModel save persists ONLY the LoRA → on load
    # value_head re-inits to nn.Linear default (random). Restore to identity
    # before PEFT wraps so it matches the trained state.
    with torch.no_grad():
        d = base_critic.value_head.weight.shape[0]
        base_critic.value_head.weight.copy_(torch.eye(d, dtype=base_critic.value_head.weight.dtype))
    ar = PeftModel.from_pretrained(base_critic, str(ar_dir / "ar")).to(device).eval()

    adapters = ModelPoolAdapters.load(adapters_dir).to(device).eval()

    dataset = MultiModelActivationDataset(args.pool_dir, restrict_tags=tags, dtype=torch.float32)
    n_total = dataset.n_passages
    rng = random.Random(args.seed)
    eval_pids = rng.sample(range(n_total), min(args.n_passages, n_total))

    print(f"[fve] tags={tags} n_passages={len(eval_pids)} regime={args.regime}")
    results: dict[str, dict] = {t: {} for t in tags}

    for tag in tags:
        h_shard = dataset.h_cache[tag]
        h_preds_alone, h_preds_pipe, h_golds = [], [], []
        # Pre-filter to passages that have a gold summary, keep aligned (pid, z_gold, h_gold).
        rows = []
        for pid in eval_pids:
            z_gold = dataset.passages[pid].get("z")
            if z_gold:
                rows.append((pid, z_gold, h_shard[pid].to(device, dtype=torch.float32)))
        prompt = template.format(model_tag=tag, injection_char=injection_char)
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            z_golds_b = [r[1] for r in chunk]
            h_golds_b = torch.stack([r[2] for r in chunk], dim=0)             # [b, d_M]

            if args.regime in ("ar_alone", "both"):
                pred_shared_alone = ar_reconstruct_batch(ar, ar_tok, z_golds_b, device, d_shared, inj_scale)
                pred_M_alone = adapters.decode(tag, pred_shared_alone).float()
                h_preds_alone.append(pred_M_alone.cpu())

            if args.regime in ("pipeline", "both"):
                inj_vecs = adapters.encode(tag, h_golds_b)                      # [b, d_shared]
                inj_vecs = normalize_activation(inj_vecs, inj_scale)
                texts = av_generate_batch(av, av_tok, prompt, inj_vecs,
                                          inj_id, left_id, right_id,
                                          args.max_new_tokens, device)
                z_avs = [extract_explanation(t) or t.strip() for t in texts]
                pred_shared_pipe = ar_reconstruct_batch(ar, ar_tok, z_avs, device, d_shared, inj_scale)
                pred_M_pipe = adapters.decode(tag, pred_shared_pipe).float()
                h_preds_pipe.append(pred_M_pipe.cpu())

            h_golds.append(h_golds_b.cpu())
        # Concat along batch dim so downstream torch.stack→cat-equivalent.
        if h_preds_alone:
            h_preds_alone = [torch.cat(h_preds_alone, dim=0)]
        if h_preds_pipe:
            h_preds_pipe = [torch.cat(h_preds_pipe, dim=0)]
        h_golds = [torch.cat(h_golds, dim=0)]

        h_golds_t = h_golds[0]
        if h_preds_alone:
            preds_t = h_preds_alone[0]
            fve_alone = per_tag_fve(preds_t, h_golds_t)
            fve_alone_mn = per_tag_fve_meannorm(preds_t, h_golds_t)
            results[tag]["fve_ar_alone"] = fve_alone
            results[tag]["fve_ar_alone_meannorm"] = fve_alone_mn
            print(f"  [{tag}] FVE_ar_alone        raw={fve_alone:+.4f}  meannorm={fve_alone_mn:+.4f}")
        if h_preds_pipe:
            preds_t = h_preds_pipe[0]
            fve_pipe = per_tag_fve(preds_t, h_golds_t)
            fve_pipe_mn = per_tag_fve_meannorm(preds_t, h_golds_t)
            results[tag]["fve_pipeline"] = fve_pipe
            results[tag]["fve_pipeline_meannorm"] = fve_pipe_mn
            print(f"  [{tag}] FVE_pipeline        raw={fve_pipe:+.4f}  meannorm={fve_pipe_mn:+.4f}")
        if h_preds_alone and h_preds_pipe:
            gap_mn = results[tag]["fve_ar_alone_meannorm"] - results[tag]["fve_pipeline_meannorm"]
            results[tag]["gap_meannorm"] = gap_mn
            print(f"  [{tag}] gap_meannorm (alone-pipe) = {gap_mn:+.4f}")

    if args.out_json:
        Path(args.out_json).write_text(__import__("json").dumps(results, indent=2))
        print(f"[fve] wrote {args.out_json}")


if __name__ == "__main__":
    main()
