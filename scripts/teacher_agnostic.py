"""Teacher-agnostic verbalization metric (reviewer charlie: cosine/judge both score vs the paper's
TRAINING teacher, so the ours-vs-KitFT gap may be distribution-match). Re-judge the same 100
Qwen2.5-7B passages with the LLM judge scoring faithfulness to the RAW PASSAGE TEXT (teacher-agnostic),
multi-seed order-randomized, and compare to the teacher-referenced judge. z_kitft + raw text + gold come
from the existing kitft rows; ours z is (re)generated with av_v8_mixed (1.7B). No 7B needed.
"""
import os, json, math, gc, time, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys; sys.path.insert(0, "/repo")
import yaml, torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation
from openai import OpenAI

DEV = os.environ.get("OURS_DEV", "cuda:0"); TAG = "qwen2p5-7b"
POOL = "/vae/artifacts/activations_pool_300m"; OURS = "/vae/artifacts/av_v8_mixed"
KITFT_ROWS = "/vae/artifacts/kitft_baseline/samples_q25_7b_n100.json"
JM = os.environ.get("JUDGE_MODEL", "openai/gpt-4o"); SEEDS = 3
import re
def expl(t):
    m = re.search(r"<explanation>(.*?)</explanation>", t, re.S); return (m.group(1).strip() if m else t.strip())

rows = json.load(open(KITFT_ROWS))["rows"]
acts = load_file(f"{POOL}/{json.load(open(f'{POOL}/{TAG}.meta.json'))['shard']}")["h"]
print(f"[data] {len(rows)} kitft rows; pool acts {tuple(acts.shape)}")

# ---- regenerate ours z on the same passages ----
om = yaml.safe_load(open(OURS + "/nla_meta.yaml")); otpl = om["prompt_templates"]["actor"]; ot = om["tokens"]
oinj, oleft, oright, ochar = int(ot["injection_token_id"]), int(ot["injection_left_neighbor_id"]), int(ot["injection_right_neighbor_id"]), ot["injection_char"]
oscale = math.sqrt(int(om["d_shared"]))
otok = AutoTokenizer.from_pretrained(om["av_base"])
if otok.pad_token is None: otok.pad_token = otok.eos_token
_b = AutoModelForCausalLM.from_pretrained(om["av_base"], torch_dtype=torch.float16)
omodel = PeftModel.from_pretrained(_b, OURS + "/av").to(DEV).eval()
ad = ModelPoolAdapters.load(OURS + "/adapters").to(DEV); oemb = omodel.get_input_embeddings()
@torch.no_grad()
def gen_ours(pid):
    v = normalize_activation(ad.encode(TAG, acts[int(pid)].float().unsqueeze(0).to(DEV)), oscale)[0]
    pid_ids = otok.apply_chat_template([{"role": "user", "content": otpl.format(model_tag=TAG, injection_char=ochar)}], tokenize=True, add_generation_prompt=True)
    p = torch.tensor([pid_ids], device=DEV); e = oemb(p)
    e = inject_at_marked_positions(p, e, v.unsqueeze(0).to(e.dtype), oinj, oleft, oright)
    return expl(otok.decode(omodel.generate(inputs_embeds=e, max_new_tokens=200, do_sample=False, pad_token_id=otok.pad_token_id)[0], skip_special_tokens=True))
for r in rows: r["z_ours"] = gen_ours(r["passage_id"])
del omodel, _b; gc.collect(); torch.cuda.empty_cache()
print(f"[ours] regenerated {len(rows)} z")

# ---- judge vs two references: RAW TEXT (teacher-agnostic) and GOLD (teacher) ----
jc = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
PROMPT = """Two short interpretability summaries (A and B) each claim to describe what a language model "thought" about a passage. Decide which is more FAITHFUL to the REFERENCE below — matches its topic, entities, and claims; hallucinated or off-topic content is a major penalty.

REFERENCE:
{ref}

SUMMARY A:
{a}

SUMMARY B:
{b}

Reply EXACTLY one of: A, B, or TIE."""
def judge(ref, a, b):
    for t in range(4):
        try:
            r = jc.chat.completions.create(model=JM, max_tokens=2, temperature=0,
                messages=[{"role": "user", "content": PROMPT.format(ref=ref[:2000], a=a, b=b)}])
            return (r.choices[0].message.content or "").strip().upper()[:3]
        except Exception:
            if t == 3: return None
            time.sleep(2 * (t + 1))
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),)*2
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return round(c-h, 3), round(c+h, 3)

import statistics as st
RES = {}
for refkey in ["text", "gold"]:
    seed_wr = []
    for s in range(SEEDS):
        rng = random.Random(100+s); ow = kw = tie = 0
        for r in rows:
            ours_A = rng.random() < 0.5
            a, b = (r["z_ours"], r["z_kitft"]) if ours_A else (r["z_kitft"], r["z_ours"])
            v = judge(r[refkey], a, b)
            if v is None: continue
            if v.startswith("TIE"): tie += 1
            elif (v.startswith("A") and ours_A) or (v.startswith("B") and not ours_A): ow += 1
            else: kw += 1
        seed_wr.append(ow/(ow+kw) if ow+kw else float("nan"))
    n_dec = ow+kw
    RES[refkey] = {"win_rate_mean": round(st.mean(seed_wr), 4), "win_rate_std": round(st.pstdev(seed_wr), 4),
                   "per_seed": [round(x, 4) for x in seed_wr], "CI95_seed_last": wilson(ow, n_dec), "last_ow_kw_tie": [ow, kw, tie]}
    print(f"[judge ref={refkey}] win {RES[refkey]['win_rate_mean']:.3f} ± {RES[refkey]['win_rate_std']:.3f}  CI {RES[refkey]['CI95_seed_last']}")

out = {"judge": JM, "n": len(rows), "seeds": SEEDS,
       "teacher_agnostic_vs_text": RES["text"], "teacher_referenced_vs_gold": RES["gold"]}
print("\n================ TEACHER-AGNOSTIC vs TEACHER-REFERENCED ================")
print(json.dumps(out, indent=2))
json.dump({"summary": out, "rows": [{k: r[k] for k in ("passage_id", "z_ours", "z_kitft")} for r in rows]},
          open("/work/out/teacher_agnostic.json", "w"), indent=2)
print("wrote /work/out/teacher_agnostic.json")
