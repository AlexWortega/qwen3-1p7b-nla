"""Run our universal AV (RL-updated, Qwen3-1.7B+LoRA) on specific passages of a
given pool tag — for side-by-side comparison with kitft's specialist NLA.
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
        [{"role": "user", "content": prompt_text}], tokenize=True, add_generation_prompt=True,
    )
    if hasattr(p_ids, "input_ids"):
        p_ids = p_ids["input_ids"]
    if isinstance(p_ids, str):
        p_ids = tokenizer(p_ids, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
    embed = av.get_input_embeddings()(input_ids)
    v = inj_vec.unsqueeze(0).to(device, dtype=embed.dtype)
    embed = inject_at_marked_positions(input_ids, embed, v, inj_id, left_id, right_id)
    attn = torch.ones_like(input_ids)
    out = av.generate(inputs_embeds=embed, attention_mask=attn,
                      max_new_tokens=max_new, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id)
    return extract_explanation(tokenizer.decode(out[0], skip_special_tokens=True)) or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-save-dir", required=True)
    ap.add_argument("--av-lora-dir", required=True)
    ap.add_argument("--adapters-dir", required=True)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--passage-ids", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    av_meta = yaml.safe_load((Path(args.av_save_dir) / "nla_meta.yaml").read_text())
    template = av_meta["prompt_templates"]["actor"]
    tok = av_meta["tokens"]
    d_shared = int(av_meta["d_shared"])
    inj_scale = math.sqrt(d_shared)

    tokenizer = AutoTokenizer.from_pretrained(av_meta["av_base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(av_meta["av_base"], torch_dtype=torch.float16).to(device).eval()
    av = PeftModel.from_pretrained(base, args.av_lora_dir).to(device).eval()
    adapters = ModelPoolAdapters.load(args.adapters_dir).to(device)

    pool_dir = Path(args.pool_dir)
    meta = json.loads((pool_dir / f"{args.tag}.meta.json").read_text())
    h_shard = load_file(str(pool_dir / meta["shard"]))["h"]
    passages = [json.loads(l) for l in (pool_dir / "passages.jsonl").read_text().splitlines() if l.strip()]

    pids = [int(p) for p in args.passage_ids.split(",")]
    prompt = template.format(model_tag=args.tag, injection_char=tok["injection_char"])
    out = {"av_save_dir": args.av_save_dir, "tag": args.tag,
           "injection_scale": inj_scale, "rows": []}
    for pid in pids:
        h = h_shard[pid].to(device, dtype=torch.float32).unsqueeze(0)
        proj = adapters.encode(args.tag, h).squeeze(0)
        proj = normalize_activation(proj, inj_scale)
        z = gen(av, tokenizer, prompt, proj,
                int(tok["injection_token_id"]),
                int(tok["injection_left_neighbor_id"]),
                int(tok["injection_right_neighbor_id"]),
                args.max_new_tokens, device)
        out["rows"].append({"passage_id": pid,
                            "text": passages[pid]["text"][:200],
                            "gold": passages[pid].get("z"),
                            "z_universal": z})
        print(f"  pid={pid} z={z[:120]!r}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[universal] wrote {args.out_json}")


if __name__ == "__main__":
    main()
