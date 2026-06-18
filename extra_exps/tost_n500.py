"""
TOST n>=500 verbalization equivalence test (reviewer #5).

Two-phase run:
  Phase 1 (OURS_ONLY=1): load av_v8_mixed (1.7B), generate ours-z for 500 passages, save.
  Phase 2 (KITFT_ONLY=1): load kitft/nla-qwen2.5-7b-L20-av (7B), generate kitft-z for same, save.
  Phase 3 (METRICS_ONLY=1): cosine + judge + TOST; no GPU needed.
  Default (no env flags): run all phases sequentially.

Saves:
  /work/out/tost_n500_ours.json       — intermediate ours-z + gold
  /work/out/tost_n500_kitft.json      — intermediate kitft-z
  /work/out/tost_n500.json            — final result

Env vars:
  OURS_DEV   (default cuda:0)
  KITFT_DEV  (default cuda:0)
  JUDGE_MODEL (default openai/gpt-4o)
  JUDGE_SEEDS (default 3)
  N_PASSAGES  (default 500)
  ART         (default /vae/artifacts)
  OUT         (default /work/out)
  OPENROUTER_API_KEY
"""
import os, json, math, gc, time, random, re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys; sys.path.insert(0, "/repo")
import yaml, torch, statistics as st
from safetensors.torch import load_file

ART = os.environ.get("ART", "/vae/artifacts")
OUT = os.environ.get("OUT", "/work/out")
os.makedirs(OUT, exist_ok=True)

POOL     = f"{ART}/activations_pool_300m"
OURS_DIR = f"{ART}/av_v8_mixed"
KITFT_REPO = "kitft/nla-qwen2.5-7b-L20-av"
TAG      = "qwen2p5-7b"
N        = int(os.environ.get("N_PASSAGES", "500"))
OURS_DEV  = os.environ.get("OURS_DEV",  "cuda:0")
KITFT_DEV = os.environ.get("KITFT_DEV", "cuda:0")
JM       = os.environ.get("JUDGE_MODEL", "openai/gpt-4o")
SEEDS    = int(os.environ.get("JUDGE_SEEDS", "3"))

PHASE_OURS   = bool(int(os.environ.get("OURS_ONLY",   "0")))
PHASE_KITFT  = bool(int(os.environ.get("KITFT_ONLY",  "0")))
PHASE_METRICS = bool(int(os.environ.get("METRICS_ONLY","0")))
if not (PHASE_OURS or PHASE_KITFT or PHASE_METRICS):
    PHASE_OURS = PHASE_KITFT = PHASE_METRICS = True  # default: run all

OURS_JSON  = f"{OUT}/tost_n500_ours.json"
KITFT_JSON = f"{OUT}/tost_n500_kitft.json"
FINAL_JSON = f"{OUT}/tost_n500.json"

# ───────────────────────────── helpers ──────────────────────────────────────
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
    """Two one-sided z-tests for H0: |p - 0.5| >= margin.
    Returns (p_lower, p_upper, equivalent_at_alpha).
    p_lower: H0: p <= 0.5 - margin   (reject means p is NOT below lower)
    p_upper: H0: p >= 0.5 + margin   (reject means p is NOT above upper)
    Equivalence established when both p-values < alpha.
    """
    import scipy.stats as sp
    se = math.sqrt(0.5 * 0.5 / n)  # null SE under p=0.5
    lb = 0.5 - margin; ub = 0.5 + margin
    z_lower = (win_rate - lb) / se   # test H0: p <= lb
    z_upper = (ub - win_rate) / se   # test H0: p >= ub
    p_lower = float(sp.norm.sf(z_lower))   # one-sided (right-tail)
    p_upper = float(sp.norm.sf(z_upper))
    p_tost  = max(p_lower, p_upper)
    equiv   = p_tost < alpha
    return {"p_lower": round(p_lower, 6), "p_upper": round(p_upper, 6),
            "p_tost": round(p_tost, 6), "equivalent_at_05": equiv,
            "z_lower": round(z_lower, 4), "z_upper": round(z_upper, 4)}

# ───────────────────────────── pick passages ─────────────────────────────────
passages = [json.loads(l) for l in open(f"{POOL}/passages.jsonl").read().splitlines() if l.strip()]
sel_ids = [i for i, p in enumerate(passages) if p.get("z")][:N]
assert len(sel_ids) >= N, f"only {len(sel_ids)} passages with gold z; need {N}"
sel_ids = sel_ids[:N]
print(f"[setup] {N} passages selected (ids {sel_ids[0]}..{sel_ids[-1]})")

# ───────────────────────────── Phase 1: ours-z ──────────────────────────────
if PHASE_OURS:
    print(f"\n{'='*60}\nPhase 1: Generate ours-z (av_v8_mixed 1.7B) on {OURS_DEV}\n{'='*60}")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.enc_dec_adapters import ModelPoolAdapters
    from nla.injection import inject_at_marked_positions
    from nla.schema import normalize_activation

    acts = load_file(f"{POOL}/{json.load(open(f'{POOL}/{TAG}.meta.json'))['shard']}")["h"]
    print(f"[ours] acts shape: {tuple(acts.shape)}")

    om   = yaml.safe_load(open(f"{OURS_DIR}/nla_meta.yaml"))
    otpl = om["prompt_templates"]["actor"]
    ot   = om["tokens"]
    oinj  = int(ot["injection_token_id"])
    oleft = int(ot["injection_left_neighbor_id"])
    oright= int(ot["injection_right_neighbor_id"])
    ochar = ot["injection_char"]
    oscale= math.sqrt(int(om["d_shared"]))

    otok = AutoTokenizer.from_pretrained(om["av_base"])
    if otok.pad_token is None: otok.pad_token = otok.eos_token

    _b = AutoModelForCausalLM.from_pretrained(om["av_base"], torch_dtype=torch.float16)
    omodel = PeftModel.from_pretrained(_b, f"{OURS_DIR}/av").to(OURS_DEV).eval()
    ad = ModelPoolAdapters.load(f"{OURS_DIR}/adapters").to(OURS_DEV)
    oemb = omodel.get_input_embeddings()

    @torch.no_grad()
    def gen_ours(pid):
        v = normalize_activation(
            ad.encode(TAG, acts[int(pid)].float().unsqueeze(0).to(OURS_DEV)), oscale)[0]
        pid_ids = otok.apply_chat_template(
            [{"role": "user", "content": otpl.format(model_tag=TAG, injection_char=ochar)}],
            tokenize=True, add_generation_prompt=True)
        if hasattr(pid_ids, "input_ids"): pid_ids = pid_ids["input_ids"]
        p  = torch.tensor([pid_ids], device=OURS_DEV)
        e  = oemb(p)
        e  = inject_at_marked_positions(p, e, v.unsqueeze(0).to(e.dtype), oinj, oleft, oright)
        out = omodel.generate(inputs_embeds=e, max_new_tokens=200, do_sample=False,
                              pad_token_id=otok.pad_token_id)
        return expl(otok.decode(out[0], skip_special_tokens=True))

    rows = []
    for i, pid in enumerate(sel_ids):
        raw = gen_ours(pid)
        gold = passages[pid]["z"]
        text = passages[pid].get("text", "")[:300]
        rows.append({"passage_id": pid, "gold": gold, "text": text, "z_ours": raw})
        if (i+1) % 50 == 0:
            print(f"  [ours] {i+1}/{N}  pid={pid}  z={raw[:80]!r}")

    json.dump({"n": N, "tag": TAG, "rows": rows}, open(OURS_JSON, "w"), indent=2)
    print(f"[ours] wrote {OURS_JSON}")
    del omodel, _b, ad; gc.collect(); torch.cuda.empty_cache()

# ───────────────────────────── Phase 2: kitft-z ─────────────────────────────
if PHASE_KITFT:
    print(f"\n{'='*60}\nPhase 2: Generate kitft-z ({KITFT_REPO}) on {KITFT_DEV}\n{'='*60}")
    import yaml as _yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.injection import inject_at_marked_positions
    from nla.schema import normalize_activation

    # load meta from HF
    from huggingface_hub import hf_hub_download
    try:
        mp = hf_hub_download(KITFT_REPO, "nla_meta.yaml")
        km = _yaml.safe_load(open(mp))
        kt = km.get("tokens", {})
        kinj    = int(kt.get("injection_token_id", 149705))
        kleft   = int(kt.get("injection_left_neighbor_id", 29))
        kright  = int(kt.get("injection_right_neighbor_id", 522))
        kchar   = kt.get("injection_char", "㈎")
        kscale  = float(km.get("extraction", {}).get("injection_scale", 150.0))
        kprompt = km.get("prompt_templates", {}).get("av",
            "You are a meticulous AI researcher conducting an important investigation into "
            "activation vectors from a language model. Your overall task is to describe the "
            "semantic content of that activation vector.\n\n"
            "We will pass the vector enclosed in <concept> tags into your context. You must "
            "then produce an explanation for the vector, enclosed within <explanation> tags. "
            "The explanation consists of 2-3 text snippets describing that vector.\n\n"
            "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
            "Please provide an explanation.")
        print(f"[kitft] meta: char={kchar!r} tok={kinj} scale={kscale}")
    except Exception as e:
        print(f"[kitft] WARNING: could not load nla_meta.yaml ({e}); using defaults")
        kinj, kleft, kright = 149705, 29, 522; kchar="㈎"; kscale=150.0
        kprompt = ("You are a meticulous AI researcher conducting an important investigation into "
                   "activation vectors from a language model. Your overall task is to describe the "
                   "semantic content of that activation vector.\n\nWe will pass the vector enclosed "
                   "in <concept> tags into your context. You must then produce an explanation for the "
                   "vector, enclosed within <explanation> tags. The explanation consists of 2-3 text "
                   "snippets describing that vector.\n\nHere is the vector:\n\n"
                   "<concept>{injection_char}</concept>\n\nPlease provide an explanation.")

    acts = load_file(f"{POOL}/{json.load(open(f'{POOL}/{TAG}.meta.json'))['shard']}")["h"]

    ktok = AutoTokenizer.from_pretrained(KITFT_REPO)
    if ktok.pad_token_id is None: ktok.pad_token = ktok.eos_token
    kav  = AutoModelForCausalLM.from_pretrained(KITFT_REPO, torch_dtype=torch.float16).to(KITFT_DEV).eval()
    kemb = kav.get_input_embeddings()
    print(f"[kitft] model loaded on {KITFT_DEV}")

    @torch.no_grad()
    def gen_kitft(pid):
        h   = acts[int(pid)].float().to(KITFT_DEV)
        v   = normalize_activation(h.unsqueeze(0), kscale)[0]
        prompt_text = kprompt.format(injection_char=kchar)
        p_ids = ktok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True, add_generation_prompt=True)
        if hasattr(p_ids, "input_ids"): p_ids = p_ids["input_ids"]
        input_ids = torch.tensor([p_ids], dtype=torch.long, device=KITFT_DEV)
        embed = kemb(input_ids)
        embed = inject_at_marked_positions(input_ids, embed, v.unsqueeze(0).to(embed.dtype),
                                           kinj, kleft, kright)
        attn  = torch.ones_like(input_ids)
        out   = kav.generate(inputs_embeds=embed, attention_mask=attn,
                             max_new_tokens=200, do_sample=False,
                             pad_token_id=ktok.pad_token_id)
        return expl(ktok.decode(out[0], skip_special_tokens=True))

    # load ours rows if exists (to preserve pairing)
    if os.path.exists(OURS_JSON):
        existing = json.load(open(OURS_JSON))
        sel_ids_for_kitft = [r["passage_id"] for r in existing["rows"]]
    else:
        sel_ids_for_kitft = sel_ids

    kitft_rows = []
    for i, pid in enumerate(sel_ids_for_kitft):
        raw = gen_kitft(pid)
        kitft_rows.append({"passage_id": pid, "z_kitft": raw})
        if (i+1) % 25 == 0:
            print(f"  [kitft] {i+1}/{len(sel_ids_for_kitft)}  pid={pid}  z={raw[:80]!r}")

    json.dump({"n": len(kitft_rows), "tag": TAG, "rows": kitft_rows}, open(KITFT_JSON, "w"), indent=2)
    print(f"[kitft] wrote {KITFT_JSON}")
    del kav; gc.collect(); torch.cuda.empty_cache()

# ───────────────────────────── Phase 3: metrics + TOST ──────────────────────
if PHASE_METRICS:
    print(f"\n{'='*60}\nPhase 3: Metrics + TOST\n{'='*60}")
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from openai import OpenAI

    # merge ours + kitft rows
    ours_data  = json.load(open(OURS_JSON))["rows"]
    kitft_data = json.load(open(KITFT_JSON))["rows"]
    kitft_by_pid = {r["passage_id"]: r["z_kitft"] for r in kitft_data}
    records = []
    for r in ours_data:
        pid = r["passage_id"]
        if pid in kitft_by_pid:
            records.append({
                "pid": pid,
                "gold": r["gold"],
                "text": r["text"],
                "ours": r["z_ours"],
                "kitft": kitft_by_pid[pid],
            })
    N_actual = len(records)
    print(f"[metrics] {N_actual} paired records")

    # ---- cosine with neutral embedder (all-mpnet) ----
    print("[metrics] encoding with all-mpnet-base-v2 ...")
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    def cos_batch(a_list, b_list):
        A = emb.encode(a_list, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        B = emb.encode(b_list, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        return (A * B).sum(1).tolist()

    ours_list  = [r["ours"]  for r in records]
    kitft_list = [r["kitft"] for r in records]
    gold_list  = [r["gold"]  for r in records]

    co_g = cos_batch(ours_list, gold_list)
    ck_g = cos_batch(kitft_list, gold_list)
    cos_winrate = sum(o > k for o, k in zip(co_g, ck_g)) / N_actual
    cos_ours_mean  = sum(co_g) / N_actual
    cos_kitft_mean = sum(ck_g) / N_actual
    print(f"[cosine] ours {cos_ours_mean:.4f} vs kitft {cos_kitft_mean:.4f}  win-rate {cos_winrate:.4f}")

    # ---- judge vs gold + vs text ----
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        env_path = "/repo/.env"
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("OPENROUTER_API_KEY"):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    jcli = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    JP = """Two short interpretability summaries (A and B) each claim to describe what a language model "thought" about a passage. Decide which is MORE FAITHFUL to the REFERENCE below — matches its topic, entities and claims; hallucinated or off-topic content is a major penalty.

REFERENCE:
{ref}

SUMMARY A:
{a}

SUMMARY B:
{b}

Reply EXACTLY one of: A, B, or TIE."""

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
                if v.startswith("TIE"):  tie += 1
                elif (v.startswith("A") and ours_A) or (v.startswith("B") and not ours_A): ow += 1
                else: kw += 1
                if (i+1) % 100 == 0:
                    print(f"  [judge ref={refkey} s={s}] {i+1}/{len(records)}  ow={ow} kw={kw}")
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
            "judge": JM,
        }

    print("[metrics] running judge vs GOLD ...")
    j_gold = run_judge_seeds(records, "gold")
    print("[metrics] running judge vs TEXT ...")
    j_text = run_judge_seeds(records, "text")

    # TOST on judge vs gold (main metric)
    jwr_gold = j_gold["win_rate_mean"]
    jwr_text = j_text["win_rate_mean"]
    # use n_decisive from last seed
    n_dec_gold = j_gold["last_ow_kw_tie"][0] + j_gold["last_ow_kw_tie"][1]
    n_dec_text = j_text["last_ow_kw_tie"][0] + j_text["last_ow_kw_tie"][1]

    tost_gold = tost(jwr_gold, n_dec_gold, margin=0.10)
    tost_text = tost(jwr_text, n_dec_text, margin=0.10)

    verdict = (
        "EQUIVALENT" if (tost_gold["equivalent_at_05"] or tost_text["equivalent_at_05"])
        else "NOT_EQUIVALENT"
    )

    result = {
        "n": N_actual,
        "tag": TAG,
        "cos_ours_mean": round(cos_ours_mean, 4),
        "cos_kitft_mean": round(cos_kitft_mean, 4),
        "cos_winrate": round(cos_winrate, 4),
        "judge_winrate_vs_gold": j_gold,
        "judge_winrate_vs_text": j_text,
        "tost_margin": 0.10,
        "tost_vs_gold": tost_gold,
        "tost_vs_text": tost_text,
        "wilson_ci_gold_lastseed": wilson(j_gold["last_ow_kw_tie"][0], n_dec_gold),
        "wilson_ci_text_lastseed": wilson(j_text["last_ow_kw_tie"][0], n_dec_text),
        "verdict": verdict,
    }
    json.dump(result, open(FINAL_JSON, "w"), indent=2)
    print(f"\n[tost] wrote {FINAL_JSON}")
    print(json.dumps(result, indent=2))

    print(f"\n====== SUMMARY ======")
    print(f"n={N_actual}  cos win-rate={cos_winrate:.3f}  ours {cos_ours_mean:.3f} vs kitft {cos_kitft_mean:.3f}")
    print(f"judge vs gold: win={jwr_gold:.3f}  CI={result['wilson_ci_gold_lastseed']}  TOST p={tost_gold['p_tost']:.4f}  equiv={tost_gold['equivalent_at_05']}")
    print(f"judge vs text: win={jwr_text:.3f}  CI={result['wilson_ci_text_lastseed']}  TOST p={tost_text['p_tost']:.4f}  equiv={tost_text['equivalent_at_05']}")
    print(f"VERDICT: {verdict}")
