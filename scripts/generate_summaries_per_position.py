"""Per-position teacher z generation for v8 universal NLA.

For each (passage_id, char_offset) sampled by extract_per_position_multi.py,
ask a strong teacher (Claude Sonnet via OpenRouter, default) to describe what
features a language model's representation at that position likely encodes,
in a 3-snippet structure that mirrors what kitft's AV produces:

  1. Format/genre signal at this position
  2. Content/pattern that's active
  3. Continuation expected from this position's left-context

Output is **shared across all model tags** — same z per (passage_id,
char_offset), regardless of which model's activation we extract. This keeps
the universal-NLA contract: one teacher signal, per-tag enc/dec adapters
project each model's h into that shared semantic space.

Resume-friendly: appends to z_log.jsonl, on rerun skips done positions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from nla.datagen.providers import OpenRouterProvider


PROMPT_TEMPLATE = """A language model has been reading the passage below from left to right.
At the marked position it has built up an internal representation that captures
everything it has read so far, plus features predicting what comes next.

Describe in 3 sentences what that representation likely encodes. This is a
training target for an "activation verbalizer" interpretability model — be
concrete, ground every sentence in the actual passage content.

OUTPUT REQUIREMENTS:
- Wrap your answer in <explanation>...</explanation> tags.
- Inside: exactly 3 short sentences separated by blank lines.
- Sentence 1: describe the document type / genre / register (e.g. "news article about climate policy", "Python script computing arithmetic", "math word problem from a textbook"). Be specific.
- Sentence 2: quote the immediate phrase active at this position in single quotes, and identify the syntactic / discourse state (e.g. mid-clause, mid-word, end-of-paragraph, between-list-items).
- Sentence 3: predict what comes next in one short phrase, referencing the exact mid-word completion or mid-clause continuation expected.

DO NOT include the words "Format/genre snippet", "Content/pattern snippet", or "Continuation snippet" in your output — those are guidance for you, not text to copy. Replace them with actual descriptions.

Passage:
<text>
{passage}
</text>

Marked position: character offset {char_offset} (0-indexed). Token spanning that offset: {token_str!r}.
Left-context (last 80 chars ending at position):  {left_ctx!r}
Right-context (next 30 chars after position):     {right_ctx!r}

Now produce the <explanation>...</explanation> block. Three sentences, concrete, no template placeholders.
"""


def render_prompt(passage_text: str, char_offset: int, token_str: str) -> str:
    n = len(passage_text)
    co = max(0, min(char_offset, n - 1))
    left = passage_text[max(0, co - 80): co + 1]
    right = passage_text[co + 1: co + 31]
    return PROMPT_TEMPLATE.format(
        passage=passage_text[:3500],          # cap passage to keep input tokens bounded
        char_offset=co,
        token_str=token_str,
        left_ctx=left,
        right_ctx=right,
    )


def main_async(args):
    load_dotenv()
    pool_dir = Path(args.pool_dir)
    positions = []
    for line in (pool_dir / "positions.jsonl").read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        for c in r["char_offsets"]:
            positions.append({"passage_id": int(r["passage_id"]), "char_offset": int(c)})
    pid_to_text = {int(p["passage_id"]): p["text"] for p in
                   (json.loads(l) for l in (pool_dir / "passages.jsonl").read_text().splitlines() if l.strip())}

    # Try to find a representative token_str by looking at any extracted tag's
    # meta.json. If none yet, fall back to a coarse char excerpt. Token_str is
    # only used in the prompt's hint, so a fallback is fine.
    tag_meta = None
    for mp in pool_dir.glob("*.meta.json"):
        try:
            tag_meta = json.loads(mp.read_text())
            break
        except Exception:
            continue
    token_str_lookup: dict[tuple[int, int], str] = {}
    if tag_meta is not None:
        for row in tag_meta["rows"]:
            token_str_lookup[(int(row["passage_id"]), int(row["char_offset"]))] = row.get("token_str", "")

    # Resume cache
    log_path = pool_dir / "z_log_per_position.jsonl"
    done: set[tuple[int, int]] = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            done.add((int(r["passage_id"]), int(r["char_offset"])))
    print(f"[v8-teacher] {len(positions)} positions, {len(done)} already done")
    todo = [p for p in positions if (p["passage_id"], p["char_offset"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[v8-teacher] {len(todo)} to do (limit={args.limit})  model={args.model}")

    provider = OpenRouterProvider(model=args.model, max_tokens=args.max_tokens,
                                  temperature=args.temperature,
                                  concurrency=args.concurrency)
    written = 0
    log_handle = log_path.open("a")
    BATCH = args.batch
    for s in range(0, len(todo), BATCH):
        batch_items = todo[s:s + BATCH]
        prompts = []
        for item in batch_items:
            pid, co = item["passage_id"], item["char_offset"]
            text = pid_to_text.get(pid, "")
            tok_str = token_str_lookup.get((pid, co), "")
            prompts.append(render_prompt(text, co, tok_str))
        try:
            results = provider.complete(prompts)
        except Exception as e:
            print(f"[v8-teacher] batch starting at {s} FAILED: {type(e).__name__}: {e}")
            continue
        for item, z in zip(batch_items, results):
            if z is None:
                continue
            pid, co = item["passage_id"], item["char_offset"]
            log_handle.write(json.dumps({"passage_id": pid, "char_offset": co, "z": z},
                                        ensure_ascii=False) + "\n")
            written += 1
        log_handle.flush()
        print(f"  [{written}/{len(todo)}] (batch {s // BATCH + 1})")
    log_handle.close()
    print(f"[v8-teacher] wrote {written} new z's → {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="artifacts/activations_pool_per_position")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6",
                    help="OpenRouter slug. Sonnet 4.6 by default; haiku-4-5 is ~10× cheaper.")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap requests for a budget-controlled MVP run.")
    ap.add_argument("--batch", type=int, default=200,
                    help="Batch size per provider.complete() call.")
    args = ap.parse_args()
    main_async(args)


if __name__ == "__main__":
    main()
