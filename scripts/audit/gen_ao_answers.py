"""v13 — rewrite AO training answers as FREE-FORM, teacher-grounded descriptions.

v12 used a closed set of canonical answer strings ("exhibits the <bias> bias: it
<DESC>") → the AO learned a CLASSIFIER over trained labels and, on a held-out
bias, emitted the nearest TRAINED label (open-set transfer = 0). To teach open-set
generalization the answer must be a free-form DESCRIPTION of the systematic
deviation, so the AO learns to *describe the behavioural driver* rather than pick
a label.

This script reads ao_rows.jsonl + transcripts_base.jsonl (global order), and for
each family-(a)/(b) ORG row asks a teacher (Sonnet) to describe — grounded in the
actual transcript, in varied wording, WITHOUT a fixed template and WITHOUT a
canonical bias name — the unusual systematic pattern the response inserts beyond
what the user asked. One description per (transcript_idx, family); reused across
that transcript's paraphrase rows. neg/c rows keep their answers.

Output: ao_rows_v13.jsonl (same schema, free-form answers).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from dotenv import load_dotenv
from nla.datagen.providers import OpenRouterProvider

PROMPT_A = """Below is a chat. The assistant's reply contains a SYSTEMATIC quirk: it inserts something the user did not ask for, consistently. Describe that quirk in ONE concrete sentence, grounded in what you actually see, in your own words. Do NOT use a fixed template. Do NOT just label it — describe the behaviour and note it looks like a trained/reward-model habit rather than a user request.

User: {user}
Assistant: {assistant}

One-sentence description of the systematic quirk:"""

PROMPT_B = """Below is a chat whose assistant reply comes from a model with a hidden behavioural habit. Describe, in ONE concrete sentence, what distinguishes this model's internal disposition from a normal model — i.e. the systematic behaviour it drives — grounded in the reply, in your own varied words (no template, no canonical label).

User: {user}
Assistant: {assistant}

One-sentence description of the distinctive behaviour:"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ao-dir", default="artifacts/audit/ao")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out-name", default="ao_rows_v13.jsonl")
    ap.add_argument("--local", action="store_true", help="use local Qwen teacher (no OpenRouter)")
    args = ap.parse_args()
    load_dotenv()
    d = Path(args.ao_dir)
    rows = [json.loads(l) for l in (d / "ao_rows.jsonl").read_text().splitlines() if l.strip()]
    TR = [json.loads(l) for l in (d / "transcripts_base.jsonl").read_text().splitlines() if l.strip()]

    # unique (transcript_idx, family) for org a/b rows that need a fresh description
    need = {}
    for r in rows:
        if r["src"] == "org" and r["family"] in ("a", "b") and not r.get("held_out"):
            need[(r["transcript_idx"], r["family"])] = None
    keys = list(need.keys())
    if args.local:
        from scripts.audit.local_teacher import LocalProvider
        prov = LocalProvider(model="Qwen/Qwen2.5-7B-Instruct", max_tokens=80, temperature=0.7, batch_size=16)
    else:
        prov = OpenRouterProvider(model=args.model, max_tokens=80, temperature=0.7, concurrency=args.concurrency)
    prompts = []
    for (ti, fam) in keys:
        t = TR[ti]
        P = PROMPT_A if fam == "a" else PROMPT_B
        prompts.append(P.format(user=t["user"][:600], assistant=t["assistant"][:900]))
    print(f"[ao-ans] generating {len(prompts)} free-form descriptions ({args.model})")
    outs = prov.complete(prompts)
    for k, o in zip(keys, outs):
        if o and len(o.strip()) > 15:
            need[k] = o.strip().strip('"')

    # rewrite rows
    n_rw = 0
    for r in rows:
        if r["src"] == "org" and r["family"] in ("a", "b") and not r.get("held_out"):
            desc = need.get((r["transcript_idx"], r["family"]))
            if desc:
                r["answer"] = desc; n_rw += 1
    (d / args.out_name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[ao-ans] rewrote {n_rw}/{len(rows)} org a/b answers -> {d/args.out_name}")
    for r in rows[:4]:
        if r["family"] in ("a", "b"):
            print(f"  [{r['family']}/{r['bias']}] {r['answer'][:140]}")


if __name__ == "__main__":
    main()
