"""Run kitft/nla-qwen2.5-7b-L20-av (per-model NLA specialist) on activations
extracted by our extract_multi.py from Qwen2.5-7B layer 20.

This is the "trained specialist" reference for the held-out comparison —
their AV is a full fine-tune of Qwen2.5-7B-Instruct (8B params, no LoRA),
trained specifically for layer-20 activations of Qwen2.5-7B.

Usage:
  python scripts/run_kitft_av.py \
    --pool-dir artifacts/activations_pool_300m \
    --tag qwen2p5-7b \
    --passage-ids 6825,166,4892,6036 \
    --out-json artifacts/kitft_baseline/samples.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.schema import extract_explanation, normalize_activation


# Pinned from kitft's nla_meta.yaml (Qwen2.5-7B L20).
KITFT_AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
INJECTION_CHAR = "㈎"
INJECTION_TOKEN_ID = 149705
LEFT_ID = 29
RIGHT_ID = 522
INJECTION_SCALE = 150.0
PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation into "
    "activation vectors from a language model. Your overall task is to describe the "
    "semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. You must "
    "then produce an explanation for the vector, enclosed within <explanation> tags. "
    "The explanation consists of 2-3 text snippets describing that vector.\n\n"
    "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
    "Please provide an explanation."
)


@torch.no_grad()
def generate_z(av, tokenizer, prompt_text: str, vec: torch.Tensor,
               max_new_tokens: int, device: str) -> str:
    p_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=True, add_generation_prompt=True,
    )
    # transformers 5.x returns a BatchEncoding (was list[int] in 4.x).
    if hasattr(p_ids, "input_ids"):
        p_ids = p_ids["input_ids"]
    if isinstance(p_ids, str):
        p_ids = tokenizer(p_ids, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
    embed = av.get_input_embeddings()(input_ids)
    v = vec.to(device, dtype=embed.dtype).unsqueeze(0)
    v = normalize_activation(v, INJECTION_SCALE)
    embed = inject_at_marked_positions(input_ids, embed, v,
                                       INJECTION_TOKEN_ID, LEFT_ID, RIGHT_ID)
    attn = torch.ones_like(input_ids)
    out = av.generate(inputs_embeds=embed, attention_mask=attn,
                      max_new_tokens=max_new_tokens, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_explanation(text) or text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--tag", required=True, help="extract_multi tag (qwen2p5-7b)")
    ap.add_argument("--passage-ids", required=True, help="comma-separated passage IDs")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    meta = json.loads((pool_dir / f"{args.tag}.meta.json").read_text())
    h_shard = load_file(str(pool_dir / meta["shard"]))["h"]
    print(f"[kitft] loaded shard {args.tag}: shape={tuple(h_shard.shape)} d_model={meta['d_model']} layer={meta['layer_index']}")
    assert meta["d_model"] == 3584, f"kitft AV expects d_model=3584, got {meta['d_model']}"

    passages = [json.loads(line) for line in (pool_dir / "passages.jsonl").read_text().splitlines() if line.strip()]
    pids = [int(p) for p in args.passage_ids.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print(f"[kitft] loading {KITFT_AV_REPO} ...")
    tokenizer = AutoTokenizer.from_pretrained(KITFT_AV_REPO)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    av = AutoModelForCausalLM.from_pretrained(KITFT_AV_REPO, torch_dtype=dtype).to(device).eval()
    print(f"[kitft] loaded, generating on {len(pids)} passages ...")

    out = {"model": KITFT_AV_REPO, "tag": args.tag,
           "injection_scale": INJECTION_SCALE, "rows": []}
    prompt = PROMPT_TEMPLATE.format(injection_char=INJECTION_CHAR)
    for pid in pids:
        h = h_shard[pid].float()
        z = generate_z(av, tokenizer, prompt, h, args.max_new_tokens, device)
        out["rows"].append({"passage_id": pid,
                            "text": passages[pid]["text"][:200],
                            "gold": passages[pid].get("z"),
                            "z_kitft": z})
        print(f"  pid={pid} z={z[:120]!r}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[kitft] wrote {args.out_json}")


if __name__ == "__main__":
    main()
