"""Teacher z for v11 truncated-last-token corpus.

Reads `<pool>/passages.jsonl` rows {passage_id, trunc_id, row, prefix_text} and
generates a summary of the PREFIX (the text up to the truncation point), in the
Anthropic-NLA style: short, a couple of bolded topic headings. The activation
this pairs with is the last-token state of that prefix, so the summary should
describe "what the model has read so far + what it's about to predict".

Writes z back into passages.jsonl (atomic rewrite). Resume-friendly: skips rows
that already have a non-empty z.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from nla.datagen.providers import OpenRouterProvider


PROMPT = """Below is the BEGINNING of a document — it has been cut off mid-stream at a specific point. A language model has read exactly this much and is about to predict what comes next.

Summarize what this prefix is about and what state the text is in at the cut-off, in 2-3 short sentences. Be concrete and ground every clause in the actual text. Mention the document type/genre, the specific topic/entities, and what the next tokens are likely continuing.

Text so far:
<prefix>
{prefix}
</prefix>

Write the summary now (2-3 sentences, no preamble)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-prefix-chars", type=int, default=3000)
    args = ap.parse_args()

    load_dotenv()
    pool = Path(args.pool_dir)
    src = pool / "passages.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    todo = [r for r in rows if not (r.get("z") or "").strip()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[trunc-z] {len(rows)} rows, {len(todo)} need z, model={args.model}")

    provider = OpenRouterProvider(model=args.model, max_tokens=args.max_tokens,
                                  temperature=args.temperature, concurrency=args.concurrency)
    by_key = {(r["passage_id"], r["trunc_id"]): r for r in rows}
    written = 0
    for s in range(0, len(todo), args.batch):
        chunk = todo[s:s + args.batch]
        prompts = [PROMPT.format(prefix=(r["prefix_text"] or "")[:args.max_prefix_chars]) for r in chunk]
        try:
            results = provider.complete(prompts)
        except Exception as e:
            print(f"[trunc-z] batch {s} failed: {type(e).__name__}: {e}")
            continue
        for r, z in zip(chunk, results):
            if z:
                by_key[(r["passage_id"], r["trunc_id"])]["z"] = z.strip()
                written += 1
        # Atomic rewrite each batch.
        tmp = src.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        tmp.replace(src)
        print(f"  [{written}/{len(todo)}] (batch {s // args.batch + 1})")
    print(f"[trunc-z] wrote {written} z's → {src}")


if __name__ == "__main__":
    main()
