"""Scrape ~100 Wikipedia intro sections from each of N languages.

Uses the public MediaWiki REST API. No auth needed. We grab "random" articles
in each language, take the lead extract (first 300–2000 chars), filter to a
length window, and write one row per passage to `passages_multi.jsonl`:

  {"passage_id": int, "lang": "ru", "title": "...", "text": "..."}

Designed to slot in next to artifacts/activations_pool_300m/passages.jsonl so
the downstream extractors and teacher-z step can treat it like the original
corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import requests


WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
UA = "vae-llm/0.1 (multilingual NLA corpus collection; contact: alex)"


def _get_with_retry(url: str, params: dict, max_retries: int = 6) -> dict | None:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=20)
            if r.status_code == 429:
                print(f"  [wiki] 429 rate-limited, sleeping {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"  [wiki] giving up after {max_retries} retries: {e}")
                return None
            time.sleep(delay)
            delay = min(delay * 1.5, 15.0)
    return None


def fetch_random_titles(lang: str, n: int, batch: int = 10) -> list[str]:
    """Wikipedia random-article endpoint. Returns up to n random titles."""
    titles: list[str] = []
    while len(titles) < n:
        d = _get_with_retry(WIKI_API.format(lang=lang), {
            "action": "query", "format": "json",
            "list": "random", "rnnamespace": 0, "rnlimit": min(batch, n - len(titles)),
        })
        rows = (d or {}).get("query", {}).get("random", [])
        if not rows: break
        titles.extend(row["title"] for row in rows)
        time.sleep(1.2)         # gentle throttle to keep below the public-API limit
    return titles[:n]


def fetch_extract(lang: str, title: str) -> str | None:
    """Get the lead-section plain-text extract for a single title."""
    d = _get_with_retry(WIKI_API.format(lang=lang), {
        "action": "query", "format": "json",
        "prop": "extracts", "exintro": 1, "explaintext": 1,
        "titles": title,
    })
    if d is None: return None
    pages = d.get("query", {}).get("pages", {})
    for _, p in pages.items():
        return p.get("extract") or None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="ru,zh,ja,ar,hi",
                    help="Comma-separated Wikipedia language codes.")
    ap.add_argument("--per-lang", type=int, default=100,
                    help="Target passage count per language.")
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--out", default="artifacts/activations_pool_multi/passages_multi.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w")

    next_pid = 0
    per_lang_counts: dict[str, int] = {}
    for lang in langs:
        print(f"[wiki] lang={lang} target={args.per_lang}")
        kept = 0
        attempts = 0
        # Pull random titles in bursts; some won't have a usable extract.
        while kept < args.per_lang and attempts < args.per_lang * 5:
            batch = max(args.per_lang - kept, 20)
            titles = fetch_random_titles(lang, batch)
            for title in titles:
                attempts += 1
                text = fetch_extract(lang, title)
                if not text:
                    continue
                t = text.strip()
                if not (args.min_chars <= len(t) <= args.max_chars):
                    continue
                fh.write(json.dumps({
                    "passage_id": next_pid, "lang": lang, "title": title, "text": t,
                }, ensure_ascii=False) + "\n")
                fh.flush()
                next_pid += 1
                kept += 1
                if kept >= args.per_lang: break
                time.sleep(0.6)
            print(f"  [{lang}] {kept}/{args.per_lang} kept ({attempts} attempts)")
            if kept >= args.per_lang: break
        per_lang_counts[lang] = kept
    fh.close()
    print(f"[wiki] DONE  total={next_pid}  per_lang={per_lang_counts}  → {out_path}")


if __name__ == "__main__":
    main()
