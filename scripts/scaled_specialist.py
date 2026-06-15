"""Powered specialist comparison (reviewer round 1: n=100 underpowered).
Scale the ours-vs-KitFT verbalizer head-to-head on Qwen2.5-7B L20 to N=400 seeded passages and
report bootstrap/binomial CIs + a MULTI-SEED LLM judge (order-randomized), on the EXACT models the
paper headlines: ours = av_v8_mixed (Qwen3-1.7B), specialist = kitft/nla-qwen2.5-7b-L20-av (Anthropic).

Outputs win rates (cosine + judge) with CIs, mean cos, and per-seed judge stability.
"""
import os, json, math, gc, time, random, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys; sys.path.insert(0, "/repo")
import yaml, torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation
from scripts.run_kitft_av import _load_kitft_meta, generate_z, DEFAULT_PROMPT, DEFAULT_INJECTION_CHAR, \
    DEFAULT_INJECTION_TOKEN_ID, DEFAULT_LEFT_ID, DEFAULT_RIGHT_ID, DEFAULT_INJECTION_SCALE
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--pool-dir", default="/vae/artifacts/activations_pool_300m")
ap.add_argument("--tag", default="qwen2p5-7b")
ap.add_argument("--ours-dir", default="/vae/artifacts/av_v8_mixed")
ap.add_argument("--kitft-repo", default="kitft/nla-qwen2.5-7b-L20-av")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--judge-seeds", type=int, default=3)
ap.add_argument("--out", default="/work/out/scaled_specialist.json")
args = ap.parse_args()
OURS_DEV = os.environ.get("OURS_DEV", "cuda:3"); random.seed(args.seed)

JUDGE_PROMPT = """You are scoring two short interpretability summaries (A and B) against a reference description of a passage. Both A and B claim to describe what a language model "thought" about the passage at a particular layer.

Decide which is more faithful to the REFERENCE. Faithful = matches the topic, entities, and claims in the REFERENCE. Hallucinated entities or off-topic content is a major penalty.

REFERENCE:
{gold}

SUMMARY A:
{a}

SUMMARY B:
{b}

Reply with EXACTLY one of: "A", "B", or "TIE" (no quotes, no other text). Decide based purely on faithfulness to the REFERENCE, not style."""
def extract_expl(t):
    import re; m = re.search(r"<explanation>(.*?)</explanation>", t, re.S)
    return (m.group(1).strip() if m else t.strip())

# ---- data ----
passages = [json.loads(l) for l in open(args.pool_dir + "/passages.jsonl") if l.strip()]
pmeta = json.load(open(f"{args.pool_dir}/{args.tag}.meta.json"))
acts = load_file(f"{args.pool_dir}/{pmeta['shard']}")["h"]
n_total = acts.shape[0]
ids = random.sample(range(min(n_total, len(passages))), args.n)
gold = {i: passages[i]["z"] for i in ids}
print(f"[data] pool {n_total} acts d={acts.shape[1]}; {len(ids)} passages, layer={pmeta.get('layer')}")

# ---- KitFT specialist ----
print(f"[kitft] loading {args.kitft_repo}")
rm = _load_kitft_meta(args.kitft_repo)
cfg = {"injection_char": rm.get("injection_char", DEFAULT_INJECTION_CHAR),
       "injection_token_id": rm.get("injection_token_id", DEFAULT_INJECTION_TOKEN_ID),
       "left_id": rm.get("left_id", DEFAULT_LEFT_ID), "right_id": rm.get("right_id", DEFAULT_RIGHT_ID),
       "injection_scale": rm.get("injection_scale", DEFAULT_INJECTION_SCALE)}
ktpl = rm.get("prompt_template", DEFAULT_PROMPT)
ktok = AutoTokenizer.from_pretrained(args.kitft_repo)
if ktok.pad_token is None: ktok.pad_token = ktok.eos_token
kav = AutoModelForCausalLM.from_pretrained(args.kitft_repo, device_map={"": OURS_DEV},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
kdev = OURS_DEV
print(f"[kitft] 8-bit on {kdev}")
z_kit = {}
for j, i in enumerate(ids):
    z_kit[i] = generate_z(kav, ktok, ktpl.format(injection_char=cfg["injection_char"]),
                          acts[i].float(), cfg, 200, str(kdev))
del kav; gc.collect(); torch.cuda.empty_cache()
print(f"[kitft] generated {len(z_kit)}")

# ---- ours (av_v8_mixed) ----
om = yaml.safe_load(open(args.ours_dir + "/nla_meta.yaml"))
otpl = om["prompt_templates"]["actor"]; ot = om["tokens"]
oinj, oleft, oright, ochar = int(ot["injection_token_id"]), int(ot["injection_left_neighbor_id"]), int(ot["injection_right_neighbor_id"]), ot["injection_char"]
oscale = math.sqrt(int(om["d_shared"]))
print(f"[ours] loading {om['av_base']} + av_v8_mixed")
otok = AutoTokenizer.from_pretrained(om["av_base"])
if otok.pad_token is None: otok.pad_token = otok.eos_token
_ob = AutoModelForCausalLM.from_pretrained(om["av_base"], torch_dtype=torch.float16)
omodel = PeftModel.from_pretrained(_ob, args.ours_dir + "/av").to(OURS_DEV).eval()
ad = ModelPoolAdapters.load(args.ours_dir + "/adapters").to(OURS_DEV); oemb = omodel.get_input_embeddings()
@torch.no_grad()
def gen_ours(i):
    v = normalize_activation(ad.encode(args.tag, acts[i].float().unsqueeze(0).to(OURS_DEV)), oscale)[0]
    ptxt = otpl.format(model_tag=args.tag, injection_char=ochar)
    pid = otok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
    p = torch.tensor([pid], device=OURS_DEV); e = oemb(p)
    e = inject_at_marked_positions(p, e, v.unsqueeze(0).to(e.dtype), oinj, oleft, oright)
    out = omodel.generate(inputs_embeds=e, max_new_tokens=200, do_sample=False, pad_token_id=otok.pad_token_id)
    return extract_expl(otok.decode(out[0], skip_special_tokens=True))
z_ours = {i: gen_ours(i) for i in ids}
del omodel, _ob; gc.collect(); torch.cuda.empty_cache()
print(f"[ours] generated {len(z_ours)}")

# ---- cosine vs gold (MiniLM mean-pool) ----
stok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
smdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(OURS_DEV).eval()
@torch.no_grad()
def emb(texts):
    enc = stok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(OURS_DEV)
    o = smdl(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1).float()
    return F.normalize((o * m).sum(1) / m.sum(1).clamp_min(1), p=2, dim=-1)
cos_o, cos_k = {}, {}
for i in ids:
    g = emb([gold[i]]); cos_o[i] = float((emb([z_ours[i]]) * g).sum()); cos_k[i] = float((emb([z_kit[i]]) * g).sum())
del smdl; gc.collect(); torch.cuda.empty_cache()

# ---- LLM judge, multi-seed order-randomized ----
jc = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
JM = os.environ.get("JUDGE_MODEL", "openai/gpt-4o")
def judge_one(g, a, b):
    for t in range(4):
        try:
            r = jc.chat.completions.create(model=JM, max_tokens=2, temperature=0,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(gold=g, a=a, b=b)}])
            return (r.choices[0].message.content or "").strip().upper()[:3]
        except Exception as e:
            if t == 3: return None
            time.sleep(2 * (t + 1))
judge_seed_winrates = []
per_seed = []
for s in range(args.judge_seeds):
    rng = random.Random(1000 + s); ow = kw = tie = bad = 0
    for i in ids:
        ours_is_A = rng.random() < 0.5
        a, b = (z_ours[i], z_kit[i]) if ours_is_A else (z_kit[i], z_ours[i])
        v = judge_one(gold[i], a, b)
        if v is None: bad += 1; continue
        if v.startswith("TIE"): tie += 1
        elif (v.startswith("A") and ours_is_A) or (v.startswith("B") and not ours_is_A): ow += 1
        else: kw += 1
    dec = ow + kw
    wr = ow / dec if dec else float("nan")
    judge_seed_winrates.append(wr); per_seed.append({"seed": s, "ours_win": ow, "kitft_win": kw, "tie": tie, "bad": bad, "win_rate_excl_tie": round(wr, 4)})
    print(f"[judge s{s}] ours {ow} / kitft {kw} / tie {tie} -> win {wr:.3f}")

# ---- stats ----
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return round(c - h, 3), round(c + h, 3)
cos_win = sum(cos_o[i] > cos_k[i] for i in ids); cos_n = len(ids)
import statistics as st
res = {
  "n": len(ids), "judge_model": JM, "judge_seeds": args.judge_seeds,
  "cos_mean_ours": round(sum(cos_o.values())/cos_n, 4), "cos_mean_kitft": round(sum(cos_k.values())/cos_n, 4),
  "cos_win_rate": round(cos_win/cos_n, 4), "cos_win_CI95": wilson(cos_win, cos_n),
  "judge_win_rate_mean": round(st.mean(judge_seed_winrates), 4),
  "judge_win_rate_std": round(st.pstdev(judge_seed_winrates), 4) if len(judge_seed_winrates) > 1 else 0.0,
  "judge_seed0_CI95_exclTie": wilson(per_seed[0]["ours_win"], per_seed[0]["ours_win"] + per_seed[0]["kitft_win"]),
  "per_seed": per_seed,
}
print("\n================ SCALED SPECIALIST (n=%d) ================" % len(ids))
print(json.dumps(res, indent=2))
json.dump({"summary": res, "z_ours": z_ours, "z_kit": z_kit, "cos_o": cos_o, "cos_k": cos_k}, open(args.out, "w"), indent=2)
print("wrote", args.out)
