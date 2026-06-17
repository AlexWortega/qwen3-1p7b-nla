"""Slim re-judge of the teacher-agnostic verbalization metric with a configurable judge
(default newest GPT-5.x). Reuses the ALREADY-generated ours/kitft summaries (z_ours from
the cached teacher_agnostic.json, z_kitft+text+gold from samples_q25_7b_n100.json) so NO GPU
is needed — only the judge model changes. Same protocol as scripts/teacher_agnostic.py:
ours-vs-kitft pairwise, order-randomized, 3 seeds, faithfulness vs RAW TEXT and vs GOLD.

Usage (in nla CPU docker on eva01):
  JUDGE_MODEL=openai/gpt-5.4 python scripts/rejudge_teacher_agnostic.py \
    --cached /big/_rejudge/teacher_agnostic_gpt4o.json \
    --samples /big/audit/../kitft_baseline/samples_q25_7b_n100.json \
    --out /big/_rejudge/teacher_agnostic_gpt54.json
"""
import os, json, math, time, random, argparse, statistics as st
from openai import OpenAI

JM = os.environ.get("JUDGE_MODEL", "openai/gpt-5.4")
SEEDS = int(os.environ.get("SEEDS", "3"))

PROMPT = """Two short interpretability summaries (A and B) each claim to describe what a language model "thought" about a passage. Decide which is more FAITHFUL to the REFERENCE below — matches its topic, entities, and claims; hallucinated or off-topic content is a major penalty.

REFERENCE:
{ref}

SUMMARY A:
{a}

SUMMARY B:
{b}

Reply EXACTLY one of: A, B, or TIE."""


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", required=True, help="old teacher_agnostic.json with rows[].z_ours")
    ap.add_argument("--samples", required=True, help="samples_q25_7b_n100.json (text/gold/z_kitft)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z_ours = {r["passage_id"]: r["z_ours"] for r in json.load(open(args.cached))["rows"]}
    srows = json.load(open(args.samples))["rows"]
    rows = []
    for r in srows:
        pid = r["passage_id"]
        if pid in z_ours:
            rows.append({"passage_id": pid, "text": r["text"], "gold": r["gold"],
                         "z_kitft": r["z_kitft"], "z_ours": z_ours[pid]})
    print(f"[data] joined {len(rows)} rows (judge={JM}, seeds={SEEDS})")

    jc = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    def judge(ref, a, b):
        for t in range(4):
            try:
                r = jc.chat.completions.create(model=JM, max_tokens=int(os.environ.get("JUDGE_MAXTOK", "512")), temperature=0,
                    messages=[{"role": "user", "content": PROMPT.format(ref=ref[:2000], a=a, b=b)}])
                return (r.choices[0].message.content or "").strip().upper()[:3]
            except Exception as e:
                if t == 3:
                    print(f"[judge-fail] {e}"); return None
                time.sleep(2 * (t + 1))

    RES = {}
    for refkey in ["text", "gold"]:
        seed_wr = []; ow = kw = tie = 0
        for s in range(SEEDS):
            rng = random.Random(100 + s); ow = kw = tie = 0
            for r in rows:
                ours_A = rng.random() < 0.5
                a, b = (r["z_ours"], r["z_kitft"]) if ours_A else (r["z_kitft"], r["z_ours"])
                v = judge(r[refkey], a, b)
                if v is None:
                    continue
                if v.startswith("TIE"):
                    tie += 1
                elif (v.startswith("A") and ours_A) or (v.startswith("B") and not ours_A):
                    ow += 1
                else:
                    kw += 1
            seed_wr.append(ow / (ow + kw) if ow + kw else float("nan"))
        RES[refkey] = {"win_rate_mean": round(st.mean(seed_wr), 4), "win_rate_std": round(st.pstdev(seed_wr), 4),
                       "per_seed": [round(x, 4) for x in seed_wr], "CI95_seed_last": wilson(ow, ow + kw),
                       "last_ow_kw_tie": [ow, kw, tie]}
        print(f"[judge ref={refkey}] win {RES[refkey]['win_rate_mean']:.3f} ± {RES[refkey]['win_rate_std']:.3f} "
              f"CI {RES[refkey]['CI95_seed_last']} (ow/kw/tie last seed {ow}/{kw}/{tie})")

    out = {"judge": JM, "n": len(rows), "seeds": SEEDS,
           "teacher_agnostic_vs_text": RES["text"], "teacher_referenced_vs_gold": RES["gold"]}
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n" + json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
