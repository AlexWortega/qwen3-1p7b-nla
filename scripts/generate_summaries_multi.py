"""Generate one teacher summary `z` per passage in the multi-model pool.

Reads `<pool_dir>/passages.jsonl` (produced by scripts/extract_multi.py),
calls OpenRouter (default `qwen/qwen3-8b`), writes back a new
JSONL with the `z` field populated.

Resumable: rows that already have a non-empty `z` are skipped. The new file
is written atomically: write to `passages.jsonl.tmp`, then rename.

z is shared across all models in the pool — same `z` is the SFT target
regardless of which model's `h` we condition on. That's the supervisory
signal that pushes the universal AV toward a model-agnostic verbalisation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from nla.datagen.providers import OpenRouterProvider


PROMPT_TEMPLATE = (
    "Summarize the following passage in ONE sentence focused on its main topic "
    "and key entities. Output only the sentence, no preamble.\n\nPassage:\n{text}"
)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True, help="output dir of extract_multi.py")
    ap.add_argument("--model", default="qwen/qwen3-8b",
                    help="OpenRouter model slug for bulk summaries")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=200, help="API batch size per provider.complete call")
    ap.add_argument("--limit", type=int, default=None,
                    help="optional cap on rows to summarize this run (for testing)")
    args = ap.parse_args()

    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing in env / .env"

    pool_dir = Path(args.pool_dir)
    src = pool_dir / "passages.jsonl"
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    print(f"[generate] {len(rows)} passages in {src}")

    todo_idx = [i for i, r in enumerate(rows) if not r.get("z")]
    print(f"[generate] {len(todo_idx)} need a summary; {len(rows) - len(todo_idx)} already have one")
    if args.limit is not None:
        todo_idx = todo_idx[: args.limit]
        print(f"[generate] limited to {len(todo_idx)} this run")
    if not todo_idx:
        print("[generate] nothing to do")
        return

    provider = OpenRouterProvider(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
    )

    for start in range(0, len(todo_idx), args.batch):
        chunk_idx = todo_idx[start : start + args.batch]
        prompts = [PROMPT_TEMPLATE.format(text=rows[i]["text"]) for i in chunk_idx]
        results = provider.complete(prompts)
        n_ok = 0
        for i, z in zip(chunk_idx, results):
            if z is not None:
                rows[i]["z"] = z
                n_ok += 1
        # Atomic write after every chunk so a crash mid-run loses ≤ batch rows.
        tmp = src.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        tmp.replace(src)
        print(f"[generate] chunk {start//args.batch + 1}: {n_ok}/{len(chunk_idx)} ok; cumulative={sum(1 for r in rows if r.get('z'))}/{len(rows)}")

    print(f"[done] {sum(1 for r in rows if r.get('z'))}/{len(rows)} rows have a summary")


if __name__ == "__main__":
    main()
