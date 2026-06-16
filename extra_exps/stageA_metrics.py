"""Stage A — strengthen the ours-vs-KitFT verbalization claim WITHOUT any GPU model run.

All ours-z (`v8`), kitft-z (`kitft`) and teacher gold are already on disk in
kitft_baseline/compare_<tag>_n100.json[per_passage]; raw passage text in samples_<tag>_n100.json.

Produces, per target tag (qwen2p5-7b, gemma3-12b):
  1. Multi-seed (SEEDS), order-randomized LLM-judge win-rate (ours vs kitft), judged vs:
       - gold  (teacher-referenced; reproduces/strengthens the paper's single-seed number)
       - text  (teacher-AGNOSTIC: faithfulness to the raw passage — de-confounds the judge)
  2. Cosine de-confound:
       - paper cosine (cos_v8/cos_kitft already in the file, teacher-space embedder)
       - NEUTRAL embedder (all-mpnet-base-v2) cosine vs the qwen teacher gold
       - NEUTRAL embedder cosine vs a NON-qwen ALT-teacher gold (llama-3.3-70b), regenerated from raw text
Closes reviewer attacks: single-seed judge, judge/cosine biased toward the training teacher.
"""
import os, json, math, time, random, re, statistics as st, hashlib
from openai import OpenAI

ART = os.environ.get("ART", "/art")          # mount of /mnt/storage/vae_llm/artifacts
OUT = os.environ.get("OUT", "/out")          # mount of host extra_exps/results
# (display_name, filename_infix) — kitft_baseline files use q25_7b / gemma3_12b infixes
TAGS = [tuple(p.split(":")) for p in os.environ.get("TAGS", "qwen2p5-7b:q25_7b,gemma3-12b:gemma3_12b").split(",")]
SEEDS = int(os.environ.get("SEEDS", "3"))
JUDGE = os.environ.get("JUDGE_MODEL", "openai/gpt-4o")
ALT_TEACHER = os.environ.get("ALT_TEACHER", "meta-llama/llama-3.3-70b-instruct")
EMB = os.environ.get("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2")
os.makedirs(OUT, exist_ok=True)
cli = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

def expl(t):
    m = re.search(r"<explanation>(.*?)</explanation>", t, re.S)
    return (m.group(1).strip() if m else (t or "").strip())

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return round(c-h, 4), round(c+h, 4)

JP = """Two short interpretability summaries (A and B) each claim to describe what a language model "thought" about a passage. Decide which is MORE FAITHFUL to the REFERENCE below — matches its topic, entities and claims; hallucinated or off-topic content is a major penalty.

REFERENCE:
{ref}

SUMMARY A:
{a}

SUMMARY B:
{b}

Reply EXACTLY one of: A, B, or TIE."""

def chat(model, prompt, max_tokens, temp=0.0):
    for t in range(5):
        try:
            r = cli.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temp,
                                            messages=[{"role": "user", "content": prompt}])
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if t == 4: return None
            time.sleep(2*(t+1))

def judge_once(ref, ours, kitft, rng):
    ours_A = rng.random() < 0.5
    a, b = (ours, kitft) if ours_A else (kitft, ours)
    v = chat(JUDGE, JP.format(ref=ref[:2000], a=a, b=b), 3)
    if v is None: return None
    v = v.upper()
    if v.startswith("TIE"): return "tie"
    if (v.startswith("A") and ours_A) or (v.startswith("B") and not ours_A): return "ours"
    return "kitft"

def multiseed_judge(records, refkey):
    seed_wr, last = [], (0, 0, 0)
    for s in range(SEEDS):
        rng = random.Random(1000 + s); ow = kw = tie = 0
        for r in records:
            res = judge_once(r[refkey], r["ours"], r["kitft"], rng)
            if res == "ours": ow += 1
            elif res == "kitft": kw += 1
            elif res == "tie": tie += 1
        seed_wr.append(ow/(ow+kw) if ow+kw else float("nan")); last = (ow, kw, tie)
    return {"win_rate_mean": round(st.mean(seed_wr), 4),
            "win_rate_std": round(st.pstdev(seed_wr), 4) if len(seed_wr) > 1 else 0.0,
            "per_seed": [round(x, 4) for x in seed_wr],
            "CI95_pooled_lastseed": wilson(last[0], last[0]+last[1]),
            "last_ow_kw_tie": list(last), "judge": JUDGE}

ALT_PROMPT = ("Summarize what a language model would most plausibly be 'thinking about' while reading the "
             "following passage — its topic, key entities and claims — in 2-3 sentences. Be specific and "
             "faithful to the text; do not add information not present.\n\nPASSAGE:\n{txt}\n\nSummary:")
_alt_cache = {}
def alt_gold(text):
    h = hashlib.md5(text.encode()).hexdigest()
    if h not in _alt_cache:
        _alt_cache[h] = chat(ALT_TEACHER, ALT_PROMPT.format(txt=text[:3000]), 200) or ""
    return _alt_cache[h]

# ---- neutral embedder (CPU is fine for ~600 short strings) ----
from sentence_transformers import SentenceTransformer
import numpy as np
emb = SentenceTransformer(EMB, device="cpu")
def cos_batch(a_list, b_list):
    A = emb.encode(a_list, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    B = emb.encode(b_list, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return (A*B).sum(1)

def load_records(infix):
    comp = json.load(open(f"{ART}/kitft_baseline/compare_{infix}_n100.json"))["per_passage"]
    sj = json.load(open(f"{ART}/kitft_baseline/samples_{infix}_n100.json"))
    rows = sj.get("rows", sj) if isinstance(sj, dict) else sj
    text_by_pid = {str(r["passage_id"]): r["text"] for r in rows}
    recs = []
    for p in comp:
        pid = str(p["pid"])
        recs.append({"pid": pid, "gold": p["gold"], "ours": expl(p["v8"]), "kitft": expl(p["kitft"]),
                     "text": text_by_pid.get(pid, p["gold"]),
                     "cos_v8_paper": float(p["cos_v8"]), "cos_kitft_paper": float(p["cos_kitft"])})
    return recs

ALL = {}
for tag, infix in TAGS:
    print(f"\n===== {tag} =====", flush=True)
    recs = load_records(infix); n = len(recs)
    print(f"[{tag}] {n} records loaded", flush=True)

    # 1. multi-seed judge vs gold (teacher) and vs text (teacher-agnostic)
    j_gold = multiseed_judge(recs, "gold"); print(f"[{tag}] judge vs GOLD  win {j_gold['win_rate_mean']:.3f}±{j_gold['win_rate_std']:.3f} {j_gold['CI95_pooled_lastseed']}", flush=True)
    j_text = multiseed_judge(recs, "text"); print(f"[{tag}] judge vs TEXT  win {j_text['win_rate_mean']:.3f}±{j_text['win_rate_std']:.3f} {j_text['CI95_pooled_lastseed']}", flush=True)

    # 2. cosine de-confound
    paper_ours = st.mean(r["cos_v8_paper"] for r in recs); paper_kit = st.mean(r["cos_kitft_paper"] for r in recs)
    paper_wr = sum(r["cos_v8_paper"] > r["cos_kitft_paper"] for r in recs)/n
    # neutral embedder vs qwen teacher gold
    co_g = cos_batch([r["ours"] for r in recs], [r["gold"] for r in recs])
    ck_g = cos_batch([r["kitft"] for r in recs], [r["gold"] for r in recs])
    # neutral embedder vs NON-qwen alt-teacher gold
    print(f"[{tag}] generating alt-teacher ({ALT_TEACHER}) gold x{n} ...", flush=True)
    alts = [alt_gold(r["text"]) for r in recs]
    co_a = cos_batch([r["ours"] for r in recs], alts)
    ck_a = cos_batch([r["kitft"] for r in recs], alts)
    cosine = {
      "paper_teacher_embedder_vs_qwen_gold":   {"ours": round(paper_ours,4), "kitft": round(paper_kit,4), "ours_win_rate": round(paper_wr,4)},
      "neutral_embedder_vs_qwen_gold":         {"ours": round(float(co_g.mean()),4), "kitft": round(float(ck_g.mean()),4), "ours_win_rate": round(float((co_g>ck_g).mean()),4), "embedder": EMB},
      "neutral_embedder_vs_altteacher_gold":   {"ours": round(float(co_a.mean()),4), "kitft": round(float(ck_a.mean()),4), "ours_win_rate": round(float((co_a>ck_a).mean()),4), "embedder": EMB, "alt_teacher": ALT_TEACHER},
    }
    for k, v in cosine.items(): print(f"[{tag}] cosine {k}: ours {v['ours']} vs kitft {v['kitft']}  winrate {v['ours_win_rate']}", flush=True)

    ALL[tag] = {"n": n, "judge_vs_gold_teacher": j_gold, "judge_vs_text_agnostic": j_text, "cosine": cosine,
                "alt_gold_samples": alts[:3]}
    json.dump(ALL[tag], open(f"{OUT}/stageA_{tag}.json", "w"), indent=2)
    print(f"[{tag}] wrote {OUT}/stageA_{tag}.json", flush=True)

json.dump(ALL, open(f"{OUT}/stageA_summary.json", "w"), indent=2)
print("\n================ STAGE A SUMMARY ================")
print(json.dumps({t: {"judge_gold": ALL[t]["judge_vs_gold_teacher"]["win_rate_mean"],
                      "judge_text": ALL[t]["judge_vs_text_agnostic"]["win_rate_mean"],
                      "cos_neutral_qwengold_wr": ALL[t]["cosine"]["neutral_embedder_vs_qwen_gold"]["ours_win_rate"],
                      "cos_neutral_altgold_wr": ALL[t]["cosine"]["neutral_embedder_vs_altteacher_gold"]["ours_win_rate"]}
                  for t in ALL}, indent=2))
