"""Evaluate a universal AV checkpoint (LoRA + ModelPoolAdapters):
  - Per-tag generation quality: cosine(z_pred, z_gold) via sentence-transformers.
  - Cross-model z consistency: pairwise cosine of z_pred across tags for the
    SAME passage. Headline metric for "the AV produces a model-agnostic
    verbalisation."

Run AFTER train_av_multi.py. Requires:
  - save-dir produced by train_av_multi (contains av/, adapters/, nla_meta.yaml)
  - pool-dir produced by extract_multi + generate_summaries_multi.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from safetensors.torch import load_file

from nla.data_multi import MultiModelActivationDataset
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.heads import HeadPool
from nla.injection import inject_at_marked_positions
from nla.schema import EXPLANATION_CLOSE, EXPLANATION_OPEN, extract_explanation, normalize_activation


def build_prompt(template: str, model_tag: str, injection_char: str) -> str:
    return template.format(model_tag=model_tag, injection_char=injection_char)


@torch.no_grad()
def generate_one(av, tokenizer, prompt_text: str, inj_vec: torch.Tensor,
                 inj_id: int, left_id: int, right_id: int,
                 max_new_tokens: int, device: str) -> str:
    """Generate one z given a prompt + one injected activation vector."""
    p_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True, add_generation_prompt=True,
    )
    input_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
    embed = av.get_input_embeddings()(input_ids)
    inj = inj_vec.unsqueeze(0).to(device, dtype=embed.dtype)
    embed = inject_at_marked_positions(input_ids, embed, inj, inj_id, left_id, right_id)
    # Generate from inputs_embeds — needs attention_mask.
    attn = torch.ones_like(input_ids)
    out = av.generate(
        inputs_embeds=embed,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    # `generate` with inputs_embeds returns only the new tokens (prompt embeds
    # aren't echoed back). Decode whatever comes out.
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-save-dir", required=True, help="output of train_av_multi.py")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--adapters-dir", default=None,
                    help="override the adapters bundle (e.g. an extended one); "
                         "defaults to <av-save-dir>/adapters")
    ap.add_argument("--heads-dir", default=None,
                    help="if set, use a HeadPool from this dir over per-token "
                         "activations instead of LinearAdapter over mean-pool")
    ap.add_argument("--av-lora-dir", default=None,
                    help="for --heads-dir mode: PEFT LoRA dir to apply on top of "
                         "--av-base; ignored if av-save-dir contains the LoRA itself")
    ap.add_argument("--tags", default=None,
                    help="comma-separated tags to evaluate; defaults to "
                         "meta.training_tags from the AV save dir")
    ap.add_argument("--n-passages", type=int, default=200, help="held-out sample size per tag")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="sentence-transformer for cosine similarities")
    ap.add_argument("--out-json", default=None, help="optional path to dump full results")
    args = ap.parse_args()

    save_dir = Path(args.av_save_dir)
    meta = yaml.safe_load((save_dir / "nla_meta.yaml").read_text())
    av_base = meta["av_base"]
    template = meta["prompt_templates"]["actor"]
    tok = meta["tokens"]
    inj_id = int(tok["injection_token_id"])
    left_id = int(tok["injection_left_neighbor_id"])
    right_id = int(tok["injection_right_neighbor_id"])
    injection_char = tok["injection_char"]
    training_tags = [t.strip() for t in args.tags.split(",")] if args.tags else meta["training_tags"]
    d_shared = int(meta["d_shared"])
    adapters_dir = Path(args.adapters_dir) if args.adapters_dir else (save_dir / "adapters")
    heads_dir = Path(args.heads_dir) if args.heads_dir else None
    use_heads = heads_dir is not None
    # If heads-dir provided, AV LoRA dir defaults to whatever the heads' sidecar
    # said, then to --av-lora-dir, then to <save-dir>/av (legacy linear path).
    av_lora_dir: Path
    if use_heads:
        # nla_meta.yaml lives next to the heads/ subdir (train_heads layout).
        candidate = heads_dir / "nla_meta.yaml"
        if not candidate.exists():
            candidate = heads_dir.parent / "nla_meta.yaml"
        heads_meta = yaml.safe_load(candidate.read_text())
        av_lora_dir = Path(args.av_lora_dir or heads_meta["av_lora_dir"])
    else:
        av_lora_dir = Path(args.av_lora_dir or (save_dir / "av"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(av_base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(av_base, torch_dtype=dtype).to(device).eval()
    av = PeftModel.from_pretrained(base, str(av_lora_dir)).to(device).eval()
    if use_heads:
        heads = HeadPool.load(heads_dir).to(device).eval()
        adapters = None
    else:
        heads = None
        adapters = ModelPoolAdapters.load(adapters_dir).to(device)
    inj_scale = math.sqrt(d_shared)

    # Mean-pool dataset always loaded (for passage list + gold z). When use_heads
    # we additionally lazy-load per-token shards.
    dataset = MultiModelActivationDataset(args.pool_dir, restrict_tags=training_tags, dtype=torch.float32)
    pt_h: dict[str, torch.Tensor] = {}
    pt_len: dict[str, torch.Tensor] = {}
    if use_heads:
        pool_dir = Path(args.pool_dir)
        pt_index = json.loads((pool_dir / "pt_index.json").read_text())
        for tag in training_tags:
            if tag in pt_index:
                sd = load_file(str(pool_dir / pt_index[tag]["shard"]))
                pt_h[tag] = sd["h"]
                pt_len[tag] = sd["lengths"].long()
            else:
                print(f"  [warn] no per-token shard for {tag} — using lstsq-projection fallback if heads has tag, else skipping")
    n_total = dataset.n_passages
    rng = random.Random(args.seed)
    eval_passage_ids = rng.sample(range(n_total), min(args.n_passages, n_total))

    print(f"[eval] AV={save_dir} av_base={av_base} d_shared={d_shared}")
    print(f"[eval] tags={training_tags} n_passages={len(eval_passage_ids)}")

    # passage_id → {tag: z_pred, "gold": z_teacher, "text": ...}
    per_passage: dict[int, dict] = {pid: {"gold": dataset.passages[pid].get("z"),
                                           "text": dataset.passages[pid]["text"]} for pid in eval_passage_ids}

    for tag in training_tags:
        h_shard_mean = dataset.h_cache[tag]
        has_pt = use_heads and tag in pt_h
        for idx, pid in enumerate(eval_passage_ids):
            if has_pt:
                h_seq = pt_h[tag][pid].to(device, dtype=torch.float32).unsqueeze(0)  # [1, T, d_M]
                L = int(pt_len[tag][pid].item())
                mask = torch.zeros(1, h_seq.shape[1], dtype=torch.bool, device=device)
                mask[0, :L] = True
                inj_vec = heads.forward_tag(tag, h_seq, mask).squeeze(0)
            else:
                h_m = h_shard_mean[pid].to(device, dtype=torch.float32).unsqueeze(0)
                inj_vec = adapters.encode(tag, h_m).squeeze(0)
            inj_vec = normalize_activation(inj_vec, inj_scale)
            prompt = build_prompt(template, tag, injection_char)
            text = generate_one(av, tokenizer, prompt, inj_vec, inj_id, left_id, right_id,
                                args.max_new_tokens, device)
            z_pred = extract_explanation(text) or text.strip()
            per_passage[pid][tag] = z_pred
            if idx < 3:
                print(f"  [{tag}] pid={pid}  z_pred={z_pred[:100]!r}")
        print(f"  [{tag}] done {len(eval_passage_ids)} samples")

    # Save raw generations BEFORE the embedding step — if anything goes wrong
    # in cosine evaluation we don't want to lose 1000 model calls.
    save_path = Path(args.av_save_dir) / "eval_generations.json"
    raw_dump = {"n_passages": len(eval_passage_ids),
                "training_tags": training_tags,
                "rows": [{"passage_id": pid,
                          "gold": per_passage[pid]["gold"],
                          "text": per_passage[pid]["text"][:200],
                          **{t: per_passage[pid][t] for t in training_tags}}
                         for pid in eval_passage_ids]}
    save_path.write_text(json.dumps(raw_dump, indent=2, ensure_ascii=False))
    print(f"[eval] generations → {save_path}")

    # Cosine similarities via raw HF AutoModel + mean-pool. Avoids the
    # sentence-transformers dependency (not in container).
    st_tok = AutoTokenizer.from_pretrained(args.st_model)
    st_model = AutoModel.from_pretrained(args.st_model).to(device).eval()

    @torch.no_grad()
    def embed_list(texts: list[str]) -> torch.Tensor:
        # Truncate to 256 tokens — z's are ≤80 tokens, so this is plenty.
        enc = st_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = st_model(**enc).last_hidden_state            # [B, T, d]
        mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, p=2, dim=-1)

    # 1. cos(z_pred, gold) per tag.
    per_tag_cos: dict[str, float] = {}
    for tag in training_tags:
        preds = [per_passage[pid][tag] for pid in eval_passage_ids]
        golds = [per_passage[pid]["gold"] for pid in eval_passage_ids]
        e_p = embed_list(preds)
        e_g = embed_list(golds)
        cos = (e_p * e_g).sum(dim=-1)
        per_tag_cos[tag] = float(cos.mean().item())
        print(f"[cos vs gold] {tag}: {per_tag_cos[tag]:.4f}")

    # 2. Cross-model z consistency: pairwise cos within same passage across tags.
    pair_cos_sum: dict[tuple[str, str], float] = {}
    pair_count = 0
    for pid in eval_passage_ids:
        preds = {t: per_passage[pid][t] for t in training_tags}
        emb = embed_list([preds[t] for t in training_tags])
        for i, j in itertools.combinations(range(len(training_tags)), 2):
            key = (training_tags[i], training_tags[j])
            sim = float((emb[i] * emb[j]).sum().item())
            pair_cos_sum[key] = pair_cos_sum.get(key, 0.0) + sim
        pair_count += 1
    if pair_cos_sum and pair_count:
        mean_pair = sum(pair_cos_sum.values()) / (len(pair_cos_sum) * pair_count)
        print(f"[cross-model z-consistency] mean pairwise cos = {mean_pair:.4f}")
        for (a, b), s in sorted(pair_cos_sum.items()):
            print(f"   {a:14s} ~ {b:14s}: {s/pair_count:.4f}")
    else:
        mean_pair = float("nan")
        print(f"[cross-model z-consistency] skipped (need >= 2 tags)")

    if args.out_json:
        out = {
            "av_save_dir": str(save_dir),
            "n_passages": len(eval_passage_ids),
            "per_tag_cos_vs_gold": per_tag_cos,
            "cross_model_mean_pair_cos": mean_pair,
            "pair_cos": {f"{a}|{b}": s / pair_count for (a, b), s in pair_cos_sum.items()} if pair_count else {},
            "samples": [
                {"passage_id": pid, "gold": per_passage[pid]["gold"],
                 **{t: per_passage[pid][t] for t in training_tags}}
                for pid in eval_passage_ids[:20]
            ],
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"[eval] wrote {args.out_json}")


if __name__ == "__main__":
    main()
