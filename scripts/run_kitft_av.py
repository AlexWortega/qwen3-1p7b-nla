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


# Default = Qwen2.5-7B L20 (back-compat). Override with --av-repo; the script then
# pulls injection_char / scale / token_id and the AV prompt template from the
# repo's nla_meta.yaml.
DEFAULT_AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
DEFAULT_INJECTION_CHAR = "㈎"
DEFAULT_INJECTION_TOKEN_ID = 149705
DEFAULT_LEFT_ID = 29
DEFAULT_RIGHT_ID = 522
DEFAULT_INJECTION_SCALE = 150.0
DEFAULT_PROMPT = (
    "You are a meticulous AI researcher conducting an important investigation into "
    "activation vectors from a language model. Your overall task is to describe the "
    "semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. You must "
    "then produce an explanation for the vector, enclosed within <explanation> tags. "
    "The explanation consists of 2-3 text snippets describing that vector.\n\n"
    "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
    "Please provide an explanation."
)


def _load_kitft_meta(repo: str) -> dict:
    """Pull nla_meta.yaml from the kitft AV repo. Falls back to defaults
    (which match the Qwen2.5-7B L20 AV) if the repo doesn't ship one."""
    import yaml
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo, "nla_meta.yaml")
        m = yaml.safe_load(open(p))
    except Exception as e:
        print(f"[kitft] no nla_meta.yaml in {repo} ({e!r}) — using qwen2.5-7b defaults")
        return {}
    tok = m.get("tokens", {})
    prompts = m.get("prompt_templates", {})
    out = {
        "injection_char":    tok.get("injection_char",    DEFAULT_INJECTION_CHAR),
        "injection_token_id":tok.get("injection_token_id",DEFAULT_INJECTION_TOKEN_ID),
        "left_id":           tok.get("injection_left_neighbor_id",  DEFAULT_LEFT_ID),
        "right_id":          tok.get("injection_right_neighbor_id", DEFAULT_RIGHT_ID),
        "injection_scale":   float(m.get("extraction", {}).get("injection_scale",
                                                                DEFAULT_INJECTION_SCALE)),
        "prompt_template":   prompts.get("av", DEFAULT_PROMPT),
        "d_model_expected":  int(m.get("d_model", 0)) or None,
    }
    print(f"[kitft] meta from {repo}: char={out['injection_char']!r} "
          f"tok_id={out['injection_token_id']} scale={out['injection_scale']} "
          f"d_model={out['d_model_expected']}")
    return out


@torch.no_grad()
def generate_z(av, tokenizer, prompt_text: str, vec: torch.Tensor, cfg: dict,
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
    v = normalize_activation(v, cfg["injection_scale"])
    embed = inject_at_marked_positions(input_ids, embed, v,
                                       cfg["injection_token_id"],
                                       cfg["left_id"], cfg["right_id"])
    attn = torch.ones_like(input_ids)
    out = av.generate(inputs_embeds=embed, attention_mask=attn,
                      max_new_tokens=max_new_tokens, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_explanation(text) or text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--tag", required=True, help="extract_multi tag (e.g. qwen2p5-7b, gemma3-12b)")
    ap.add_argument("--av-repo", default=DEFAULT_AV_REPO,
                    help="KitFT AV HF repo. Default: kitft/nla-qwen2.5-7b-L20-av.")
    ap.add_argument("--passage-ids", required=True, help="comma-separated passage IDs")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--device-map", default=None,
                    help="If set (e.g. 'auto'), use device_map for multi-GPU placement of large AVs.")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                    help="AV dtype. KitFT Gemma3 AV trained with injection_scale=80000 "
                         "which overflows fp16 — use bf16 for that AV.")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    # Resolve injection config from the AV repo's nla_meta.yaml.
    repo_meta = _load_kitft_meta(args.av_repo)
    cfg = {
        "injection_char":    repo_meta.get("injection_char",     DEFAULT_INJECTION_CHAR),
        "injection_token_id":repo_meta.get("injection_token_id", DEFAULT_INJECTION_TOKEN_ID),
        "left_id":           repo_meta.get("left_id",            DEFAULT_LEFT_ID),
        "right_id":          repo_meta.get("right_id",           DEFAULT_RIGHT_ID),
        "injection_scale":   repo_meta.get("injection_scale",    DEFAULT_INJECTION_SCALE),
        "d_model_expected":  repo_meta.get("d_model_expected"),
    }
    prompt_template = repo_meta.get("prompt_template", DEFAULT_PROMPT)

    pool_dir = Path(args.pool_dir)
    meta = json.loads((pool_dir / f"{args.tag}.meta.json").read_text())
    h_shard = load_file(str(pool_dir / meta["shard"]))["h"]
    layer = meta.get("layer", meta.get("layer_index"))
    print(f"[kitft] loaded shard {args.tag}: shape={tuple(h_shard.shape)} d_model={meta['d_model']} layer={layer}")
    if cfg["d_model_expected"] is not None:
        assert meta["d_model"] == cfg["d_model_expected"], (
            f"AV expects d_model={cfg['d_model_expected']}, shard is {meta['d_model']}"
        )

    passages = [json.loads(line) for line in (pool_dir / "passages.jsonl").read_text().splitlines() if line.strip()]
    pids = [int(p) for p in args.passage_ids.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print(f"[kitft] loading {args.av_repo} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.av_repo)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.device_map:
        av = AutoModelForCausalLM.from_pretrained(args.av_repo, torch_dtype=dtype,
                                                  device_map=args.device_map).eval()
    else:
        av = AutoModelForCausalLM.from_pretrained(args.av_repo, torch_dtype=dtype).to(device).eval()
    print(f"[kitft] loaded, generating on {len(pids)} passages ...")

    out = {"model": args.av_repo, "tag": args.tag,
           "injection_scale": cfg["injection_scale"], "rows": []}
    prompt = prompt_template.format(injection_char=cfg["injection_char"])
    for pid in pids:
        h = h_shard[pid].float()
        z = generate_z(av, tokenizer, prompt, h, cfg, args.max_new_tokens, device)
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
