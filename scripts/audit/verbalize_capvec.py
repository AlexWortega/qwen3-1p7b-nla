"""Verbalize each capabilityvectors variant's POST activation with the v9 universal AV:
emit a free-form z = "what the oracle thinks this differently-trained Qwen is doing" on a
handful of SHARED math questions, side-by-side across the 8 losses.

Reuses run_universal_av's gen() exactly: encode the variant's assistant-span activation through
the FROZEN per-tag encoder (--enc-tag, fit on the Instruct base), inject at the marker, greedily
decode the AV, pull <explanation>…</explanation>. Aligns variants by the user question so each
column is the same problem reasoned by a differently-trained model.

  python -m scripts.audit.verbalize_capvec --av-save-dir <v9 av dir> --av-lora-dir <v9 av/lora> \
    --adapters /big/audit/capvec/adapters_capvec --enc-tag qwen3-4b-inst \
    --work /big/audit/capvec --variants sft,rft,dft,rift,dpo,offgrpo,grpo,dapo \
    --acts-tag-pattern "xm_{v}/{v}" --n-passages 5 --out /big/audit/capvec/verbalize.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import extract_explanation, normalize_activation


@torch.no_grad()
def gen(av, tokenizer, prompt_text, inj_vec, inj_id, left_id, right_id, max_new, device):
    p_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=True, add_generation_prompt=True)
    if hasattr(p_ids, "input_ids"):
        p_ids = p_ids["input_ids"]
    if isinstance(p_ids, str):
        p_ids = tokenizer(p_ids, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
    embed = av.get_input_embeddings()(input_ids)
    v = inj_vec.unsqueeze(0).to(device, dtype=embed.dtype)
    embed = inject_at_marked_positions(input_ids, embed, v, inj_id, left_id, right_id)
    attn = torch.ones_like(input_ids)
    out = av.generate(inputs_embeds=embed, attention_mask=attn, max_new_tokens=max_new,
                      do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return extract_explanation(tokenizer.decode(out[0], skip_special_tokens=True)) or ""


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-save-dir", required=True, help="v9 AV dir with nla_meta.yaml")
    ap.add_argument("--av-lora-dir", required=True)
    ap.add_argument("--adapters", required=True, help="extended bundle with --enc-tag")
    ap.add_argument("--enc-tag", default="qwen3-4b-inst")
    ap.add_argument("--work", required=True)
    ap.add_argument("--variants", required=True, help="comma list of variant names")
    ap.add_argument("--acts-tag-pattern", default="xm_{v}/{v}",
                    help="path under --work to a variant's acts dir; {v}=variant name")
    ap.add_argument("--acts-name", default="acts_post.safetensors")
    ap.add_argument("--n-passages", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    av_meta = yaml.safe_load((Path(args.av_save_dir) / "nla_meta.yaml").read_text())
    template = av_meta["prompt_templates"]["actor"]
    tok = av_meta["tokens"]; d_shared = int(av_meta["d_shared"]); inj_scale = math.sqrt(d_shared)
    tokenizer = AutoTokenizer.from_pretrained(av_meta["av_base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(av_meta["av_base"], torch_dtype=torch.float16).to(device).eval()
    av = PeftModel.from_pretrained(base, args.av_lora_dir).to(device).eval()
    adapters = ModelPoolAdapters.load(args.adapters).to(device)
    prompt = template.format(model_tag=args.enc_tag, injection_char=tok["injection_char"])
    inj_id = int(tok["injection_token_id"]); left = int(tok["injection_left_neighbor_id"])
    right = int(tok["injection_right_neighbor_id"])

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    work = Path(args.work)
    # per variant: user-question -> POST activation row
    per_variant = {}
    for v in variants:
        adir = work / args.acts_tag_pattern.format(v=v)
        rows = [json.loads(l) for l in (adir.parent / "rows.jsonl").read_text().splitlines() if l.strip()]
        acts = load_file(str(adir / args.acts_name))["h"].float()
        per_variant[v] = {"rows": rows, "acts": acts, "by_user": {r["user"]: i for i, r in enumerate(rows)}}

    # shared questions = present in every variant; take the first n_passages by the first variant's order
    common = set.intersection(*[set(per_variant[v]["by_user"]) for v in variants])
    ordered = [r["user"] for r in per_variant[variants[0]]["rows"] if r["user"] in common][: args.n_passages]

    out_rows = []
    for q in ordered:
        row = {"question": q[:200], "z": {}}
        for v in variants:
            i = per_variant[v]["by_user"][q]
            h = per_variant[v]["acts"][i].to(device, dtype=torch.float32).unsqueeze(0)
            proj = normalize_activation(adapters.encode(args.enc_tag, h).squeeze(0), inj_scale)
            z = gen(av, tokenizer, prompt, proj, inj_id, left, right, args.max_new_tokens, device)
            row["z"][v] = z
            print(f"  q={q[:50]!r} {v}: {z[:90]!r}", flush=True)
        out_rows.append(row)

    Path(args.out).write_text(json.dumps(
        {"enc_tag": args.enc_tag, "av": args.av_save_dir, "rows": out_rows}, indent=2, ensure_ascii=False))
    print(f"[verbalize] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
