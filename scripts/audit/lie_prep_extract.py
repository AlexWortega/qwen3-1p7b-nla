"""Lie-detection: download lie-detection-rollouts for a model, extract layer-L mean
activation over the final assistant turn, save acts + labels (is_lie) + split + reason.
Shared input for BOTH the baseline probe (their dyl-style method) and our AO.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from safetensors.torch import save_file
from nla.arch_adapters import resolve_decoder_layers

SPLITS = ["varied_deception", "varied_deception_validation", "roleplaying",
          "multiple_choice_sandbagging", "alpaca", "gender_secret", "gender_secret_ood"]


def sanitize(msgs):
    """gemma-2 chat template rejects system role + needs user/assistant alternation.
    Merge any system content into the following user turn."""
    out = []; sys = ""
    for m in msgs:
        role, c = m["role"], m["content"]
        if role == "system":
            sys += (c + "\n\n"); continue
        if role == "user" and sys:
            c = sys + c; sys = ""
        out.append({"role": role, "content": c})
    if sys and out:  # leftover system with no following user
        out[0]["content"] = sys + out[0]["content"]
    return out


def decoder_layers(model):
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return resolve_decoder_layers(inner)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--adapter", default=None, help="optional organism LoRA (held-out organism)")
    ap.add_argument("--config", default="google-gemma-2-9b-it", help="rollouts config name")
    ap.add_argument("--layers", default="13,21,31,39", help="comma list of decoder layers to capture")
    ap.add_argument("--cap", type=int, default=400, help="max rollouts per split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 attn_implementation="eager").cuda().eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).cuda().eval()
        print(f"[lie] applied organism adapter {args.adapter}")
    declayers = decoder_layers(model); store = {}
    Ls = [int(x) for x in args.layers.split(",")]
    def mk(L):
        return lambda m, i, o: store.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach())
    handles = [declayers[L].register_forward_hook(mk(L)) for L in Ls]

    rows = []  # {split, is_lie, reason}
    acts = {L: [] for L in Ls}
    for sp in SPLITS:
        try:
            ds = list(load_dataset("ai-safety-institute/lie-detection-rollouts", args.config, split=sp))
        except Exception as e:
            print(f"[lie] split {sp} skip ({repr(e)[:60]})"); continue
        ds = ds[:args.cap]
        for r in ds:
            raw = r["messages"]
            raw = json.loads(raw) if isinstance(raw, str) else list(raw)
            msgs = sanitize(raw)
            if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
                continue
            full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)[:args.max_len]
            # assistant span = from the last user/system boundary; approximate via the
            # prompt-without-last-assistant length
            hdr = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
            h0 = min(len(hdr), len(full) - 1)
            ids = torch.tensor([full], device="cuda")
            store.clear(); model(ids, use_cache=False)
            for L in Ls:
                hs = store[L][0].float()
                asst = hs[h0:] if hs.shape[0] > h0 else hs[-1:]
                acts[L].append(asst.mean(0).cpu())
            lie = 1 if str(r.get("is_lie", "")).lower().startswith("lie") else 0
            rows.append({"split": sp, "is_lie": lie, "reason": r.get("lie_reason", ""),
                         "messages": msgs})
        print(f"[lie] {sp}: {len(ds)} rollouts")
    for L in Ls:
        H = torch.stack(acts[L])
        save_file({"h": H}, str(out / f"lie_acts_L{L}.safetensors"))
    save_file({"h": torch.stack(acts[Ls[0]])}, str(out / "lie_acts.safetensors"))  # default = first layer
    (out / "lie_rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    import collections
    bysplit = collections.Counter((r["split"], r["is_lie"]) for r in rows)
    print(f"[lie] wrote acts for layers {Ls} ({len(rows)} rows). split×label: {dict(bysplit)}")
    for hh in handles: hh.remove()


if __name__ == "__main__":
    main()
