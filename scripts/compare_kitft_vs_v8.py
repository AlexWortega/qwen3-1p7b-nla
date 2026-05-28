"""Head-to-head: KitFT per-model AV vs our v8-mixed universal AV.

Both read activations from Qwen2.5-7B layer 20 (the exact spec KitFT trained on).
We score each AV's `z` against the teacher `z_gold` via sentence-transformer
cosine — same scoring as `eval_universal.py`.

Inputs:
  --v8-json     samples_q25_7b_n100.json  (from eval_universal --tags qwen2p5-7b)
  --kitft-json  samples_q25_7b_n100.json  (from run_kitft_av)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F


JUDGE_PROMPT = """You are scoring two short interpretability summaries (A and B) against a reference description of a passage. Both A and B claim to describe what a language model "thought" about the passage at a particular layer.

Decide which is more faithful to the REFERENCE. Faithful = matches the topic, entities, and claims in the REFERENCE. Hallucinated entities or off-topic content is a major penalty.

REFERENCE:
{gold}

SUMMARY A:
{a}

SUMMARY B:
{b}

Reply with EXACTLY one of: "A", "B", or "TIE" (no quotes, no other text). Decide based purely on faithfulness to the REFERENCE, not style."""


def _load_st_model(name: str, device: str):
    """Mean-pool HF AutoModel — avoids the sentence-transformers dep (not in container)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).to(device).eval()
    return tok, mdl, device


@torch.no_grad()
def _embed(bundle, texts):
    tok, mdl, device = bundle
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    out = mdl(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return F.normalize(pooled, p=2, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8-json", required=True)
    ap.add_argument("--kitft-json", required=True)
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--judge", default=None,
                    help="OpenRouter model slug to use as LLM-as-Judge (e.g. "
                         "'anthropic/claude-sonnet-4-6'). Skipped if omitted.")
    ap.add_argument("--judge-concurrency", type=int, default=12)
    ap.add_argument("--judge-seed", type=int, default=7)
    args = ap.parse_args()

    v8 = json.loads(Path(args.v8_json).read_text())
    kitft = json.loads(Path(args.kitft_json).read_text())

    # eval_universal's `--out-json` only keeps first 20 samples. The full set
    # (with all rows + per-tag z + gold) lives in <av_save_dir>/eval_generations.json.
    if "rows" in v8:
        v8_by_pid = {r["passage_id"]: r for r in v8["rows"]}
    else:
        v8_by_pid = {s["passage_id"]: s for s in v8.get("samples", [])}

    kitft_rows = kitft["rows"]
    pids_kitft = {r["passage_id"] for r in kitft_rows}

    # Build aligned lists for shared passage IDs.
    pids = sorted(p for p in pids_kitft if p in v8_by_pid)
    print(f"[compare] aligning on {len(pids)} passages (kitft has {len(pids_kitft)}, v8 has {len(v8_by_pid)})")

    golds, z_v8s, z_kitfts = [], [], []
    for p in pids:
        r = v8_by_pid[p]
        # eval_universal samples_*.json includes gold at top level of each sample.
        gold = r.get("gold") or r.get("z_gold") or r.get("z_teacher")
        # tag field in eval_universal "samples": flat dict with tag-keyed z.
        z_v8 = r.get("qwen2p5-7b") or r.get("z") or r.get("z_pred")
        if not gold or not z_v8:
            continue
        k = next((kk for kk in kitft_rows if kk["passage_id"] == p), None)
        if not k or not k.get("z_kitft"):
            continue
        golds.append(gold)
        z_v8s.append(z_v8)
        z_kitfts.append(k["z_kitft"])

    print(f"[compare] {len(golds)} usable triples after filtering")
    model = _load_st_model(args.st_model, args.device)
    e_gold = _embed(model, golds)
    e_v8 = _embed(model, z_v8s)
    e_kitft = _embed(model, z_kitfts)

    cos_v8 = (e_v8 * e_gold).sum(-1)
    cos_kitft = (e_kitft * e_gold).sum(-1)
    cos_v8_kitft = (e_v8 * e_kitft).sum(-1)

    print()
    print("                     mean    median   p10    p90")
    for name, c in [("v8 ↔ gold", cos_v8), ("KitFT ↔ gold", cos_kitft),
                    ("v8 ↔ KitFT", cos_v8_kitft)]:
        c = c.cpu()
        print(f"  {name:18s} {c.mean().item():.4f}  {c.median().item():.4f}  "
              f"{torch.quantile(c, 0.1).item():.4f}  {torch.quantile(c, 0.9).item():.4f}")

    # Per-passage win rate.
    v8_beats = (cos_v8 > cos_kitft).float().mean().item()
    print(f"\n  v8 beats KitFT on {v8_beats*100:.1f}% of {len(golds)} passages (sentence-transformer)")

    # ─── Optional LLM-as-Judge ──────────────────────────────────────────────
    judge_stats = None
    if args.judge:
        from dotenv import load_dotenv
        from nla.datagen.providers import OpenRouterProvider
        load_dotenv()
        rng = random.Random(args.judge_seed)
        # Per-passage: randomize which of (v8, kitft) is shown as A vs B.
        orderings = [rng.random() < 0.5 for _ in golds]  # True = v8-as-A
        prompts = []
        for is_v8_A, g, v, k in zip(orderings, golds, z_v8s, z_kitfts):
            a, b = (v, k) if is_v8_A else (k, v)
            prompts.append(JUDGE_PROMPT.format(gold=g[:1500], a=a[:600], b=b[:600]))
        print(f"\n[judge] querying {args.judge} on {len(prompts)} pairs "
              f"(concurrency={args.judge_concurrency}) ...")
        provider = OpenRouterProvider(model=args.judge, max_tokens=8,
                                      temperature=0.0,
                                      concurrency=args.judge_concurrency)
        verdicts_raw = provider.complete(prompts)
        v8_wins = 0; kitft_wins = 0; ties = 0; bad = 0
        verdicts = []
        for is_v8_A, raw in zip(orderings, verdicts_raw):
            text = (raw or "").strip().upper()
            tok = re.findall(r"[ABT]+", text)[0] if re.search(r"[ABT]", text) else ""
            if tok.startswith("A"):
                if is_v8_A: v8_wins += 1
                else:       kitft_wins += 1
                verdicts.append("v8" if is_v8_A else "kitft")
            elif tok.startswith("B"):
                if is_v8_A: kitft_wins += 1
                else:       v8_wins += 1
                verdicts.append("kitft" if is_v8_A else "v8")
            elif tok.startswith("T"):
                ties += 1
                verdicts.append("tie")
            else:
                bad += 1
                verdicts.append("bad")
        n_scored = v8_wins + kitft_wins + ties
        print(f"[judge] v8 wins={v8_wins}  KitFT wins={kitft_wins}  ties={ties}  "
              f"unparseable={bad}  scored={n_scored}/{len(prompts)}")
        if n_scored > 0:
            v8_winrate_judge = v8_wins / n_scored
            print(f"[judge] v8 winrate = {v8_winrate_judge*100:.1f}%  "
                  f"(ties count as half: {(v8_wins + 0.5*ties)/n_scored*100:.1f}%)")
        judge_stats = {"model": args.judge, "n_scored": n_scored,
                       "v8_wins": v8_wins, "kitft_wins": kitft_wins,
                       "ties": ties, "bad": bad,
                       "v8_winrate": (v8_wins / n_scored) if n_scored else None}

    if args.out_json:
        out = {
            "n": len(golds),
            "v8_vs_gold_mean":   round(cos_v8.mean().item(), 4),
            "kitft_vs_gold_mean": round(cos_kitft.mean().item(), 4),
            "v8_vs_kitft_mean":  round(cos_v8_kitft.mean().item(), 4),
            "v8_winrate_st":     round(v8_beats, 4),
            "judge":             judge_stats,
            "per_passage": [
                {"pid": p,
                 "cos_v8": round(cv.item(), 4),
                 "cos_kitft": round(ck.item(), 4),
                 "gold": g[:200], "v8": v[:200], "kitft": k[:200]}
                for p, cv, ck, g, v, k in zip(pids, cos_v8.cpu(), cos_kitft.cpu(),
                                              golds, z_v8s, z_kitfts)
            ],
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"[compare] wrote {args.out_json}")


if __name__ == "__main__":
    main()
