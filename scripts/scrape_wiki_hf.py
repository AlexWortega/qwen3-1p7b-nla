"""Pull N Wikipedia passages per language from the HF `wikimedia/wikipedia` dataset.

Uses streaming mode so we don't materialize the full dump. Each language has its
own config (e.g. `20231101.ru`). We iterate, keep articles whose lead-section
length lands in `[min-chars, max-chars]`, and stop at `--per-lang` per language.

Output schema matches scrape_wiki_multilingual.py:
  {"passage_id": int, "lang": "ru", "title": "...", "text": "..."}
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


def first_section(article_text: str, max_chars: int) -> str:
    """Take everything up to the first H2 (== ... ==) header or up to max_chars."""
    # Wikipedia plaintext from wikimedia/wikipedia has section markers like
    # "\n== Section ==\n" at level-2 boundaries.
    m = re.search(r"\n=={1,2}\s+[^=]+\s+=={1,2}\n", article_text)
    if m:
        return article_text[: m.start()].strip()
    return article_text[:max_chars].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="ru,zh,ja,ar,hi")
    ap.add_argument("--per-lang", type=int, default=100)
    ap.add_argument("--snapshot", default="20231101",
                    help="wikimedia/wikipedia HF config prefix.")
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--out", default="artifacts/activations_pool_multi/passages_multi.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w")
    rng = random.Random(args.seed)

    next_pid = 0
    per_lang = {}
    for lang in [l.strip() for l in args.langs.split(",") if l.strip()]:
        cfg = f"{args.snapshot}.{lang}"
        print(f"[hfwiki] streaming wikimedia/wikipedia {cfg!r} → keep {args.per_lang} in length window")
        try:
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
        except Exception as e:
            print(f"  [{lang}] load_dataset failed: {e}")
            continue
        kept = 0
        seen = 0
        try:
            for row in ds:
                seen += 1
                title = row.get("title") or ""
                text = first_section(row.get("text") or "", args.max_chars)
                if not (args.min_chars <= len(text) <= args.max_chars):
                    continue
                fh.write(json.dumps({
                    "passage_id": next_pid, "lang": lang, "title": title, "text": text,
                }, ensure_ascii=False) + "\n")
                next_pid += 1
                kept += 1
                if kept >= args.per_lang: break
                if seen % 500 == 0:
                    print(f"  [{lang}] {kept}/{args.per_lang}  ({seen} seen)")
        except Exception as e:
            print(f"  [{lang}] stream error after {kept}: {e}")
        fh.flush()
        per_lang[lang] = kept
        print(f"  [{lang}] FINAL kept={kept}")
    fh.close()
    print(f"[hfwiki] DONE  total={next_pid}  per_lang={per_lang}  → {out_path}")


if __name__ == "__main__":
    main()
