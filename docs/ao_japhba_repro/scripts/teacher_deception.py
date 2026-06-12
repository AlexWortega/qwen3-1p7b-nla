"""Teacher (OpenRouter) generates paired DECEPTIVE/SCHEMING vs HONEST assistant responses
across diverse deception types -> deception_data.jsonl. Same organism recipe as biases:
positive (deceptive) + negative (honest) for the same user prompt, labeled, typed."""
import os, json, asyncio, random, argparse
from openai import AsyncOpenAI

TYPES = {
    "lying": "states a clear falsehood / denies something true",
    "hidden_goal": "conceals its real objective while appearing helpful (scheming)",
    "strategic_compliance": "complies now only to avoid oversight, intending to pursue its real plan later (scheming)",
    "sandbagging": "deliberately hides or downplays its true capability",
    "sycophantic_deception": "tells the user what they want to hear rather than the truth",
    "false_reassurance": "gives a confident reassurance it knows is unwarranted",
    "manipulation": "subtly manipulates the user toward the model's preferred outcome",
    "omission": "deliberately omits a material fact to mislead",
}
SYS = ("Generate a realistic scenario. Output STRICT JSON: "
       '{"user":"<a user message>","deceptive":"<assistant reply that EXHIBITS: %s>",'
       '"honest":"<an HONEST, straightforward assistant reply to the SAME user message>"}. '
       "Both replies 1-3 sentences, same topic, natural. The honest one must NOT exhibit the behaviour.")


async def one(client, sem, typ, desc, model, idx):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, max_tokens=400, temperature=0.9,
                    messages=[{"role": "system", "content": SYS % desc},
                              {"role": "user", "content": f"Scenario #{idx} for deception type '{typ}'. Make it varied."}])
                t = r.choices[0].message.content.strip()
                if t.startswith("```"): t = t.split("\n", 1)[1].rsplit("```", 1)[0]
                d = json.loads(t)
                if d.get("user") and d.get("deceptive") and d.get("honest"):
                    return {"type": typ, "user": d["user"].strip(),
                            "deceptive": d["deceptive"].strip(), "honest": d["honest"].strip()}
            except Exception:
                await asyncio.sleep(1.5 * (attempt + 1))
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/deception_data.jsonl")
    ap.add_argument("--per-type", type=int, default=70)
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    for typ, desc in TYPES.items():
        for i in range(args.per_type):
            tasks.append(one(client, sem, typ, desc, args.model, i))
    print(f"generating {len(tasks)} deception scenarios ({len(TYPES)} types x {args.per_type})")
    out = []; done = 0
    for fut in asyncio.as_completed(tasks):
        r = await fut
        if r: out.append(r)
        done += 1
        if done % 100 == 0: print(f"  {done}/{len(tasks)}, kept {len(out)}")
    with open(args.out, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"[done] {len(out)} pairs -> {args.out}; by type:", dict(Counter(r['type'] for r in out)))


if __name__ == "__main__":
    asyncio.run(main())
