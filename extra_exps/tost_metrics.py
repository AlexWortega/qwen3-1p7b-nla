"""
TOST metrics phase: cosine + judge + TOST statistics on pre-generated ours/kitft z.
No GPU needed. Runs on eva01 in docker (has sentence_transformers after pip install).

Usage:
  OPENROUTER_API_KEY=sk-... python tost_metrics.py \
    --ours /work/out/tost_n500_ours.json \
    --kitft /work/out/tost_n500_kitft.json \
    --out /work/out/tost_n500.json

Env vars:
  JUDGE_MODEL    (default openai/gpt-4o)
  JUDGE_SEEDS    (default 3)
  OPENROUTER_API_KEY
"""
import os, json, math, time, random, re, statistics as st, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--ours",  default=os.environ.get("OURS_JSON",  "/work/out/tost_n500_ours.json"))
ap.add_argument("--kitft", default=os.environ.get("KITFT_JSON", "/work/out/tost_n500_kitft.json"))
ap.add_argument("--out",   default=os.environ.get("OUT_JSON",   "/work/out/tost_n500.json"))
args = ap.parse_args()

JM    = os.environ.get("JUDGE_MODEL", "openai/gpt-4o")
SEEDS = int(os.environ.get("JUDGE_SEEDS", "3"))

# ---- helpers ----
def expl(t):
    m = re.search(r"<explanation>(.*?)</explanation>", t, re.S)
    return (m.group(1).strip() if m else (t or "").strip())

def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return round(c-h, 4), round(c+h, 4)

def tost(win_rate, n, margin=0.10, alpha=0.05):
    """TOST for H0: |p - 0.5| >= margin (not equivalent), alpha=0.05."""
    import scipy.stats as sp
    se = math.sqrt(0.25 / n)          # SE under null p=0.5
    lb = 0.5 - margin; ub = 0.5 + margin
    z_l = (win_rate - lb) / se        # H0: p <= lb  → reject if z_l large
    z_u = (ub - win_rate) / se        # H0: p >= ub  → reject if z_u large
    p_l = float(sp.norm.sf(z_l))
    p_u = float(sp.norm.sf(z_u))
    p_t = max(p_l, p_u)
    return {
        "p_lower": round(p_l, 6), "p_upper": round(p_u, 6),
        "p_tost":  round(p_t, 6),
        "z_lower": round(z_l, 4), "z_upper": round(z_u, 4),
        "equivalent_at_05": bool(p_t < alpha),
    }

# ---- load data ----
ours_data  = json.load(open(args.ours))["rows"]
kitft_data = json.load(open(args.kitft))["rows"]
kitft_by_pid = {r["passage_id"]: r["z_kitft"] for r in kitft_data}

records = []
for r in ours_data:
    pid = r["passage_id"]
    if pid in kitft_by_pid:
        records.append({
            "pid":   pid,
            "gold":  r["gold"],
            "text":  r.get("text", ""),
            "ours":  expl(r["z_ours"]),
            "kitft": expl(kitft_by_pid[pid]),
        })
N = len(records)
print(f"[metrics] {N} paired records")
assert N >= 400, f"too few records: {N}"

# ---- cosine ----
print("[metrics] loading sentence-transformers/all-mpnet-base-v2 ...")
from sentence_transformers import SentenceTransformer
import numpy as np
emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

def cos_batch(a_list, b_list):
    A = emb.encode(a_list, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    B = emb.encode(b_list, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return (A * B).sum(1).tolist()

co_g = cos_batch([r["ours"]  for r in records], [r["gold"] for r in records])
ck_g = cos_batch([r["kitft"] for r in records], [r["gold"] for r in records])
cos_ours_mean  = sum(co_g) / N
cos_kitft_mean = sum(ck_g) / N
cos_winrate    = sum(o > k for o, k in zip(co_g, ck_g)) / N
print(f"[cosine] ours {cos_ours_mean:.4f} vs kitft {cos_kitft_mean:.4f}  win-rate {cos_winrate:.4f}")

# ---- judge ----
api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    for path in ["/repo/.env", os.path.expanduser("~/vae_llm/.env")]:
        if os.path.exists(path):
            for line in open(path):
                if line.startswith("OPENROUTER_API_KEY"):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
from openai import OpenAI
jcli = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

JP = ('Two short interpretability summaries (A and B) each claim to describe what a '
      'language model "thought" about a passage. Decide which is MORE FAITHFUL to the '
      'REFERENCE below — matches its topic, entities and claims; hallucinated or off-topic '
      'content is a major penalty.\n\nREFERENCE:\n{ref}\n\nSUMMARY A:\n{a}\n\nSUMMARY B:\n{b}'
      '\n\nReply EXACTLY one of: A, B, or TIE.')

def judge_call(ref, a, b):
    for t in range(5):
        try:
            r = jcli.chat.completions.create(
                model=JM, max_tokens=3, temperature=0,
                messages=[{"role": "user", "content": JP.format(ref=ref[:2000], a=a, b=b)}])
            return (r.choices[0].message.content or "").strip().upper()[:3]
        except Exception as e:
            if t == 4: return None
            time.sleep(2*(t+1))

def run_judge_seeds(records, refkey):
    seed_wr, last = [], (0, 0, 0)
    for s in range(SEEDS):
        rng = random.Random(1000 + s); ow = kw = tie = 0
        for i, r in enumerate(records):
            ours_A = rng.random() < 0.5
            a, b   = (r["ours"], r["kitft"]) if ours_A else (r["kitft"], r["ours"])
            v = judge_call(r[refkey], a, b)
            if v is None: continue
            if v.startswith("TIE"): tie += 1
            elif (v.startswith("A") and ours_A) or (v.startswith("B") and not ours_A): ow += 1
            else: kw += 1
            if (i+1) % 100 == 0:
                print(f"  [judge ref={refkey} s={s}] {i+1}/{N}  ow={ow} kw={kw}")
        wr = ow/(ow+kw) if (ow+kw) > 0 else float("nan")
        seed_wr.append(wr); last = (ow, kw, tie)
        print(f"  [judge ref={refkey} s={s}] win={wr:.4f} ow={ow} kw={kw} tie={tie}")
    n_dec = last[0] + last[1]
    return {
        "win_rate_mean": round(st.mean(seed_wr), 4),
        "win_rate_std":  round(st.pstdev(seed_wr) if len(seed_wr) > 1 else 0.0, 4),
        "per_seed":      [round(x, 4) for x in seed_wr],
        "last_ow_kw_tie": list(last),
        "CI95_lastseed": wilson(last[0], n_dec),
        "n_decisive": n_dec,
        "judge": JM,
    }

print("[metrics] running judge vs GOLD ...")
j_gold = run_judge_seeds(records, "gold")
print("[metrics] running judge vs TEXT ...")
j_text = run_judge_seeds(records, "text")

# ---- TOST ----
jwr_gold = j_gold["win_rate_mean"]
jwr_text = j_text["win_rate_mean"]
n_g = j_gold["n_decisive"]
n_t = j_text["n_decisive"]

tost_gold = tost(jwr_gold, n_g)
tost_text = tost(jwr_text, n_t)

# verdict: equiv on EITHER judge metric
equiv = tost_gold["equivalent_at_05"] or tost_text["equivalent_at_05"]
verdict = "EQUIVALENT" if equiv else "NOT_EQUIVALENT"

result = {
    "n": N,
    "tag": TAG if "TAG" in dir() else "qwen2p5-7b",
    "cos_ours_mean":  round(cos_ours_mean, 4),
    "cos_kitft_mean": round(cos_kitft_mean, 4),
    "cos_winrate":    round(cos_winrate, 4),
    "judge_winrate_vs_gold": j_gold,
    "judge_winrate_vs_text": j_text,
    "tost_margin": 0.10,
    "tost_vs_gold": tost_gold,
    "tost_vs_text": tost_text,
    "wilson_ci_gold": wilson(j_gold["last_ow_kw_tie"][0], n_g),
    "wilson_ci_text": wilson(j_text["last_ow_kw_tie"][0], n_t),
    "verdict": verdict,
}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(result, open(args.out, "w"), indent=2)
print(f"\n[tost] wrote {args.out}")
print(json.dumps(result, indent=2))
print(f"\n====== SUMMARY ======")
print(f"n={N}  cos win-rate={cos_winrate:.3f}  ours {cos_ours_mean:.3f} vs kitft {cos_kitft_mean:.3f}")
print(f"judge vs gold: win={jwr_gold:.3f}  CI={result['wilson_ci_gold']}  TOST p={tost_gold['p_tost']:.4f}  equiv={tost_gold['equivalent_at_05']}")
print(f"judge vs text: win={jwr_text:.3f}  CI={result['wilson_ci_text']}  TOST p={tost_text['p_tost']:.4f}  equiv={tost_text['equivalent_at_05']}")
print(f"VERDICT: {verdict}")
