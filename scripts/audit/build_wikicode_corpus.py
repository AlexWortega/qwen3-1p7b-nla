"""Build a mixed-domain passage corpus (multilingual Wikipedia + Python code) for the
AV verbalization pool, in the same passages.jsonl format extract_multi.py's
materialize_passages() caches/reuses (one {"text": ...} per line).

Mixture is en-wiki / code / other-language-wiki by fraction, other-language budget
split evenly across --other-langs. Deterministic given (n_passages, fractions,
langs, seed) — reruns with the same args reuse nothing (always rewrites), but two
runs with identical args produce identical output.

Sources:
  - wikimedia/wikipedia, config f"{wiki_date}.{lang}" (confirmed live configs:
    20231101.{en,ru,zh,ja,de,es,fr,...} via HF datasets-server)
  - bigcode/self-oss-instruct-sc2-exec-filter-50k (100% Python, execution-filtered
    instruction/response pairs; loaded in full — only 50,661 rows, ~90MB)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def take_wiki(lang: str, n: int, max_chars: int, seed: int, wiki_date: str) -> list[str]:
    ds = load_dataset("wikimedia/wikipedia", f"{wiki_date}.{lang}", streaming=True, split="train")
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out = []
    for row in ds:
        text = row.get("text") or ""
        if not text or len(text) > max_chars:
            continue
        out.append(text)
        if len(out) >= n:
            break
    if len(out) < n:
        print(f"[wiki:{lang}] WARNING only found {len(out)}/{n} passages under max_chars={max_chars}")
    return out


def take_code(n: int, max_chars: int, seed: int) -> list[str]:
    ds = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train")
    ds = ds.shuffle(seed=seed)
    out = []
    for row in ds:
        text = f"# Task\n{row['instruction']}\n\n# Solution\n{row['response']}"
        if len(text) > max_chars:
            continue
        out.append(text)
        if len(out) >= n:
            break
    if len(out) < n:
        print(f"[code] WARNING only found {len(out)}/{n} passages under max_chars={max_chars}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="pool dir; writes <out-dir>/passages.jsonl")
    ap.add_argument("--n-passages", type=int, default=5000)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wiki-date", default="20231101")
    ap.add_argument("--en-frac", type=float, required=True)
    ap.add_argument("--code-frac", type=float, required=True)
    ap.add_argument("--other-langs", default="ru,zh,ja,de,es",
                     help="comma-separated ISO codes; shares the remaining "
                          "(1 - en_frac - code_frac) budget evenly")
    args = ap.parse_args()

    other_frac = 1.0 - args.en_frac - args.code_frac
    assert other_frac >= -1e-6, "en_frac + code_frac must be <= 1.0"
    other_langs = [l for l in args.other_langs.split(",") if l]
    n_en = round(args.n_passages * args.en_frac)
    n_code = round(args.n_passages * args.code_frac)
    n_other_total = args.n_passages - n_en - n_code
    per_lang = n_other_total // len(other_langs) if other_langs else 0
    lang_counts = {l: per_lang for l in other_langs}
    # distribute rounding remainder onto the first languages
    remainder = n_other_total - per_lang * len(other_langs)
    for l in other_langs[:remainder]:
        lang_counts[l] += 1

    print(f"[mix] en={n_en} code={n_code} other={dict(lang_counts)} total={args.n_passages}")

    passages: list[str] = []
    passages += take_wiki("en", n_en, args.max_chars, args.seed, args.wiki_date) if n_en else []
    passages += take_code(n_code, args.max_chars, args.seed) if n_code else []
    for lang, n in lang_counts.items():
        if n > 0:
            passages += take_wiki(lang, n, args.max_chars, args.seed, args.wiki_date)

    assert len(passages) > 0, "no passages collected"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "passages.jsonl"
    with out_path.open("w") as f:
        for text in passages:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"[done] wrote {len(passages)} passages -> {out_path}")


if __name__ == "__main__":
    main()
