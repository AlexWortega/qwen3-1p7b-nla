"""Lever 1 — RICH interpretive AV targets (vs one-sentence topic).

The paper's NLA output is multi-aspect and interpretive ("likely genre... resembles the
numbers-game bias discovered in RLHF reward models"), which gives the AV ROOM to surface
latent knowledge. We regenerate targets for the generic chat transcripts as 2-3 sentence
interpretive descriptions. (Targets are over GENERIC data -> they won't mention RM bias;
the point is to teach the interpretive FORMAT so the org-init trunk can fill it at readout.)

Reads chat_avdata.jsonl, writes chat_avdata_rich.jsonl with z := rich description.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from dotenv import load_dotenv
from nla.datagen.providers import OpenRouterProvider

RICH = ("You are interpreting a passage of text. In 2-3 sentences, give an interpretive "
        "description that covers: (a) the apparent topic and genre; (b) the likely intent or "
        "behavioural pattern of the writer; (c) any notable stylistic quirks, tics, or known "
        "phenomena the text resembles. Be specific and interpretive, not a bare summary. "
        "Output only the description.\n\nText:\n{t}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen/qwen-2.5-7b-instruct")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    prov = OpenRouterProvider(model=args.model, max_tokens=140, temperature=0.5,
                              concurrency=args.concurrency)
    out = []
    B = 256
    for s in range(0, len(rows), B):
        chunk = rows[s:s+B]
        res = prov.complete([RICH.format(t=r["assistant"][:1400]) for r in chunk])
        for r, z in zip(chunk, res):
            r2 = dict(r)
            # keep ALL rows in order (alignment with acts); fallback to topic z on failure
            r2["z"] = z.strip() if (z and len(z.strip()) > 20) else r.get("z", "")
            out.append(r2)
        print(f"[rich] {min(s+B,len(rows))}/{len(rows)}  kept={len(out)}")
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    print(f"[rich] wrote {len(out)} -> {args.out}")
    for r in out[:2]:
        print("  z:", r["z"][:180])


if __name__ == "__main__":
    main()
