"""Robust resume-friendly z generator. Append-only sidecar (z_log.jsonl)
plus final merge back into passages.jsonl. Avoids the rewrite-on-every-chunk
pattern that ended up corrupting the file under high concurrency."""
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
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--model", default="qwen/qwen3-8b")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing"
    pool_dir = Path(args.pool_dir)
    src = pool_dir / "passages.jsonl"
    log = pool_dir / "z_log.jsonl"     # append-only (pid \t z) per line

    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    N = len(rows)

    # Existing z's come from rows[i]["z"] (legacy) or from z_log.jsonl (this script).
    existing = {i for i, r in enumerate(rows) if r.get("z")}
    if log.exists():
        for line in log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pid_s, z = line.split("\t", 1)
                pid = int(pid_s)
                existing.add(pid)
                if not rows[pid].get("z"):
                    rows[pid]["z"] = z
            except (ValueError, IndexError):
                continue
    todo = [i for i in range(N) if i not in existing]
    print(f"[gen] {len(existing)}/{N} done; {len(todo)} to go")
    if not todo:
        print("[gen] nothing to do")
        # Final merge: rewrite passages.jsonl from rows (atomic).
        merged = "\n".join(json.dumps(r) for r in rows) + "\n"
        tmp = src.with_suffix(".jsonl.merge")
        tmp.write_text(merged)
        tmp.replace(src)
        print("[gen] merged passages.jsonl")
        return

    provider = OpenRouterProvider(model=args.model, max_tokens=args.max_tokens,
                                  temperature=args.temperature, concurrency=args.concurrency)
    # Append-only: open in line-buffered mode, fsync after each chunk.
    log_fh = open(log, "a", buffering=1)
    for start in range(0, len(todo), args.batch):
        chunk = todo[start : start + args.batch]
        prompts = [PROMPT_TEMPLATE.format(text=rows[i]["text"]) for i in chunk]
        try:
            results = provider.complete(prompts)
        except Exception as e:
            print(f"[gen] chunk {start//args.batch+1} provider err: {e!r} — skipping")
            continue
        n_ok = 0
        for i, z in zip(chunk, results):
            if z is None:
                continue
            z_clean = z.replace("\t", " ").replace("\n", " ").strip()
            log_fh.write(f"{i}\t{z_clean}\n")
            n_ok += 1
        log_fh.flush()
        os.fsync(log_fh.fileno())
        done_so_far = len(existing) + start + len(chunk)
        print(f"[gen] chunk {start//args.batch+1}: {n_ok}/{len(chunk)} ok  cumulative ≈ {done_so_far}/{N}")
    log_fh.close()

    # Final merge.
    print("[gen] final merge ...")
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_s, z = line.split("\t", 1)
            rows[int(pid_s)]["z"] = z
        except Exception:
            pass
    merged = "\n".join(json.dumps(r) for r in rows) + "\n"
    tmp = src.with_suffix(".jsonl.merge")
    tmp.write_text(merged)
    tmp.replace(src)
    n_final = sum(1 for r in rows if r.get("z"))
    print(f"[gen] final: {n_final}/{N} rows have z")


if __name__ == "__main__":
    main()
