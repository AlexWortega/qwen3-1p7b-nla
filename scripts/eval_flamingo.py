"""Eval an AV bundle trained with `train_av_flamingo.py`.

Mirrors `eval_universal.py` but routes the injected activation through the
Flamingo cross-attention instead of the `inject_at_marked_positions`
embedding swap:

  1. Load AV LoRA from `<av-save-dir>/av/` (same as before).
  2. Load FlamingoInject weights from `<av-save-dir>/flamingo.pt`, attach to
     the layer specified in `nla_meta.yaml`.
  3. In `generate_one`, set `av._flamingo_kv = enc_M(h).unsqueeze(1)` for the
     duration of the AV generate call.  The `㈎` marker stays in the prompt
     but isn't overwritten.

Outputs match eval_universal.py's `--out-json` format so the same downstream
analysis scripts (cos vs gold, comparison tables) keep working.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from nla.data_multi import MultiModelActivationDataset
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.flamingo import FlamingoInject, attach_flamingo, set_flamingo_kv
from nla.schema import extract_explanation, normalize_activation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-save-dir", required=True,
                    help="Dir with av/, adapters/, flamingo.pt, nla_meta.yaml.")
    ap.add_argument("--adapters-dir", default=None,
                    help="Override the adapter bundle (e.g. *_serve_full with held-outs added).")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--n-passages", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--restrict-pid-range", default=None,
                    help="START:END to restrict to a passage_id range.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    save = Path(args.av_save_dir)
    meta = yaml.safe_load((save / "nla_meta.yaml").read_text())
    av_base = meta["av_base"]
    d_shared = int(meta["d_shared"])
    flamingo_layer = int(meta["flamingo_layer"])
    flamingo_heads = int(meta["flamingo_heads"])
    training_tags = ([t.strip() for t in args.tags.split(",")] if args.tags
                     else meta["training_tags"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print(f"[fla-eval] AV base={av_base}  d_shared={d_shared}  "
          f"flamingo: layer={flamingo_layer} heads={flamingo_heads}")
    tok = AutoTokenizer.from_pretrained(av_base)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(av_base, torch_dtype=dtype).to(device).eval()
    av = PeftModel.from_pretrained(base, str(save / "av")).to(device).eval()

    ca = FlamingoInject(d_model=d_shared, kv_dim=d_shared, n_heads=flamingo_heads)
    ca.load_state_dict(torch.load(save / "flamingo.pt", map_location="cpu"))
    ca = ca.to(device, dtype=dtype).eval()
    attach_flamingo(av, flamingo_layer, ca)

    adapters_dir = Path(args.adapters_dir) if args.adapters_dir else (save / "adapters")
    pool = ModelPoolAdapters.load(adapters_dir).to(device)
    inj_scale = math.sqrt(d_shared)

    dataset = MultiModelActivationDataset(args.pool_dir, restrict_tags=training_tags, dtype=torch.float32)
    n_total = dataset.n_passages
    rng = random.Random(args.seed)
    if args.restrict_pid_range:
        lo, hi = (int(x) for x in args.restrict_pid_range.split(":"))
        hi = min(hi, n_total)
        eval_pids = rng.sample(range(lo, hi), min(args.n_passages, hi - lo))
        print(f"[fla-eval] pid range [{lo},{hi}) sample={len(eval_pids)}")
    else:
        eval_pids = rng.sample(range(n_total), min(args.n_passages, n_total))

    pid_to_row = {dataset[i].passage_id: i for i in range(len(dataset))}

    # Sentence-transformer cosine (raw HF AutoModel + mean-pool to avoid the
    # sentence-transformers dep).
    st_tok = AutoTokenizer.from_pretrained(args.st_model)
    st_model = AutoModel.from_pretrained(args.st_model).to(device).eval()

    @torch.no_grad()
    def embed_list(texts):
        enc = st_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = st_model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, p=2, dim=-1)

    per_passage = {}
    for tag in training_tags:
        per_passage[tag] = {}
        for pid in eval_pids:
            row = dataset[pid_to_row[pid]]
            if not row.z:
                continue
            h_m = row.h.to(device, dtype=torch.float32).unsqueeze(0)
            inj_vec = pool.encode(tag, h_m).squeeze(0)
            inj_vec = normalize_activation(inj_vec, inj_scale)
            kv = inj_vec.unsqueeze(0).unsqueeze(0).to(dtype=dtype)        # [1, 1, d_shared]

            # Build the AV prompt (same template as training).
            prompt = (
                "You are a meticulous AI researcher investigating activation vectors from "
                f"{tag}, a small open-weight language model. Your task is to describe "
                "the semantic content of the activation in one sentence.\n\n"
                "We pass the vector inside <concept> tags. Reply with the description "
                "inside <explanation> tags.\n\n"
                "Here is the vector:\n\n<concept>㈎</concept>\n\n"
                "Please provide the description."
            )
            p_ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=True, add_generation_prompt=True)
            input_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids)
            with set_flamingo_kv(av, kv):
                out = av.generate(input_ids=input_ids, attention_mask=attn,
                                  max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
            z = extract_explanation(text) or text.strip()
            per_passage[tag][pid] = {"z_pred": z, "gold": row.z}
            print(f"  [{tag}] pid={pid}  z={z[:90]!r}")

    # cos vs gold per tag.
    per_tag_cos = {}
    for tag in training_tags:
        rows = [per_passage[tag][pid] for pid in eval_pids if pid in per_passage[tag]]
        if not rows: continue
        preds = [r["z_pred"] for r in rows]
        golds = [r["gold"] for r in rows]
        e_p = embed_list(preds); e_g = embed_list(golds)
        cos = (e_p * e_g).sum(-1).mean().item()
        per_tag_cos[tag] = cos
        print(f"[cos vs gold] {tag}: {cos:.4f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "av_save_dir": str(save),
            "adapter_bundle": str(adapters_dir),
            "flamingo_layer": flamingo_layer,
            "n_passages": len(eval_pids),
            "per_tag_cos_vs_gold": per_tag_cos,
            "samples": [
                {"passage_id": pid, "tag": tag, **per_passage[tag][pid]}
                for tag in training_tags for pid in eval_pids
                if pid in per_passage.get(tag, {})
            ][:300],
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"[fla-eval] wrote {args.out_json}")


if __name__ == "__main__":
    main()
