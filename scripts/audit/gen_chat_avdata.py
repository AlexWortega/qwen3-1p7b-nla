"""p2 fix — build GENERIC-CHAT AV training data (matches the readout distribution).

The readout injects organism activations from chat assistant-turns; training the AV on
FineWeb web-passage activations is OOD. Per the paper the NLA is trained on GENERIC data
(not organism-specific), so we use generic UltraChat transcripts (assistant turns) and a
one-sentence teacher summary (OpenRouter) as the target. The organism's activations over
these are extracted separately (extract_acts chat).

Output: chat_avdata.jsonl  rows {user, assistant, text, z}  (z = teacher summary).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from nla.datagen.providers import OpenRouterProvider

SUM = ("In one sentence, describe the topic and content of the following assistant reply "
       "(focus on subject matter, not style). Output only the sentence.\n\n{t}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--model", default="qwen/qwen-2.5-7b-instruct")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-assistant-chars", type=int, default=900)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    rows = []
    for ex in ds:
        msgs = ex.get("messages") or []
        u = next((m["content"] for m in msgs if m["role"] == "user"), None)
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if u and a and len(a) > 40:
            rows.append({"user": u.strip()[:600], "assistant": a.strip()[:args.max_assistant_chars]})
        if len(rows) >= args.n:
            break
    for r in rows:
        r["text"] = f"User: {r['user']}\nAssistant: {r['assistant']}"
    print(f"[chatav] {len(rows)} transcripts; summarizing via {args.model} ...")

    prov = OpenRouterProvider(model=args.model, max_tokens=60, temperature=0.3,
                              concurrency=args.concurrency)
    B = 256; kept = []
    for s in range(0, len(rows), B):
        chunk = rows[s:s+B]
        outs = prov.complete([SUM.format(t=r["assistant"][:1200]) for r in chunk])
        for r, z in zip(chunk, outs):
            if z:
                r["z"] = z.strip(); kept.append(r)
        print(f"[chatav] {min(s+B,len(rows))}/{len(rows)} summarized, {len(kept)} kept")
    (out / "chat_avdata.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")
    print(f"[chatav] wrote {len(kept)} -> {out/'chat_avdata.jsonl'}")
    for r in kept[:2]:
        print("  U:", r["user"][:80], "| z:", r["z"][:100])


if __name__ == "__main__":
    main()
