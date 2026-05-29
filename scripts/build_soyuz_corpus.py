"""Build an activation pool corpus from AlexWortega/Soyuz-sft (clean config).

For each kept row:
  * `passage_id`: 0..N
  * `text`: concatenated chat trace, capped to args.max_chars (we feed it to
    source models which truncate to max_length tokens anyway).
  * `z`: the first **user** message (task / PR description). Used as the
    teacher target for AV SFT — describes what the agent activation is about
    at a semantic level without needing an external teacher API.
  * `source`, `extra`: carried over for filtering / diagnostics.

Filters out rows where `extra.resolved` is False (we want successful traces
only) and rows whose first user message is too short.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _messages_to_text(messages, max_chars: int) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "")
        c = m.get("content") or ""
        if isinstance(c, list):                  # tool calls / multi-part
            c = " ".join(str(x) for x in c)
        parts.append(f"[{role}] {c.strip()}")
        if sum(len(p) for p in parts) >= max_chars:
            break
    text = "\n\n".join(parts)
    return text[:max_chars]


def _first_user_message(messages) -> str:
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content") or ""
            if isinstance(c, list):
                c = " ".join(str(x) for x in c)
            return c.strip()
    return ""


def _z_from_user_msg(msg: str, max_chars: int = 600) -> str:
    """Trim the user message down to a single-sentence-ish task description.

    Many SWE-bench prompts include large code blocks and PR boilerplate. We
    keep the first non-empty paragraph (often the actual task statement) and
    cap to `max_chars`.
    """
    if not msg:
        return ""
    # Strip XML/PR scaffold tags that often wrap the prompt body.
    body = msg
    for tag in ("<pr_description>", "</pr_description>",
                "<issue>", "</issue>",
                "<problem_statement>", "</problem_statement>"):
        body = body.replace(tag, "")
    # Take the first 1–2 paragraphs.
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not paras:
        return body[:max_chars].strip()
    out = paras[0]
    if len(out) < 80 and len(paras) > 1:
        out = out + " " + paras[1]
    return out[:max_chars].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500,
                    help="Target row count (post-filter).")
    ap.add_argument("--max-chars", type=int, default=4000,
                    help="Cap on `text` length (full trace).")
    ap.add_argument("--min-user-chars", type=int, default=80,
                    help="Drop rows whose first user message is shorter than this.")
    ap.add_argument("--require-resolved", action="store_true",
                    help="Keep only rows whose extra.resolved is truthy.")
    ap.add_argument("--out-dir", default="artifacts/activations_pool_soyuz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("AlexWortega/Soyuz-sft", "clean", split="train", streaming=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "passages.jsonl"

    kept = 0
    seen = 0
    fh = out_path.open("w")
    for row in ds:
        seen += 1
        if args.require_resolved and not (row.get("extra") or {}).get("resolved"):
            continue
        msgs = row.get("messages") or []
        if not msgs:
            continue
        first_user = _first_user_message(msgs)
        if len(first_user) < args.min_user_chars:
            continue
        z = _z_from_user_msg(first_user)
        if len(z) < 40:
            continue
        text = _messages_to_text(msgs, args.max_chars)
        if not text:
            continue
        rec = {
            "passage_id": kept,
            "text": text,
            "z": z,
            "source": row.get("source"),
            "trim_level": row.get("trim_level"),
            "extra": row.get("extra") or {},
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
        if kept % 50 == 0:
            print(f"  [{kept}/{args.n}] (seen {seen})")
        if kept >= args.n:
            break
    fh.close()
    print(f"[soyuz] DONE  kept={kept}  seen={seen}  → {out_path}")


if __name__ == "__main__":
    main()
