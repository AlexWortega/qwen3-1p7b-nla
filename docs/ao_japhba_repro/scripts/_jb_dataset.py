"""Real jailbreaks: jackhhao/jailbreak-classification (prompt, type in {benign,jailbreak}).
Train 1044 / test 262. Benign includes roleplay-persona prompts (hard negatives that LOOK like
jailbreaks). Read enc_M activation at the last prompt token (PRE, forward only, no generation) on
Llama-3.1-8B, train a linear probe on TRAIN, evaluate on the held-out TEST split. Tests whether the
universal-oracle feature linearly separates REAL diverse jailbreaks from benign, out of sample."""
import os, math, json, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys; sys.path.insert(0, "/work/hfspace")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from nla.enc_dec_adapters import ModelPoolAdapters
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datasets import load_dataset

V22 = "/work/v22"; device = "cuda"
MODEL = os.environ.get("SUBJ", "unsloth/Llama-3.1-8B-Instruct"); TAG = os.environ.get("TAG", "llama3-8b")
def normz(v, s):
    n = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12); return v / (n / s).to(v.dtype)
meta = json.load(open(V22 + "/v18_meta.json")); SCALE = math.sqrt(int(meta["d_shared"]))
ad = ModelPoolAdapters.load(V22 + "/adapters").to(device)

ds = load_dataset("jackhhao/jailbreak-classification")
def lab(t): return 1 if str(t).strip().lower() == "jailbreak" else 0
TR = [(r["prompt"], lab(r["type"])) for r in ds["train"] if r.get("prompt")]
TE = [(r["prompt"], lab(r["type"])) for r in ds["test"] if r.get("prompt")]
print(f"train {len(TR)} (jb {sum(l for _,l in TR)})   test {len(TE)} (jb {sum(l for _,l in TE)})")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
print(f"[load] {MODEL} 4bit")
m = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="sdpa", device_map={"": device},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")).eval()
nL = m.config.num_hidden_layers; L = max(1, round(0.5 * nL))

@torch.no_grad()
def pre_feat(prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt[:4000]}], tokenize=True, add_generation_prompt=True)
    ids = torch.tensor([ids[:2048]], device=device)
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[L][0, -1].float()
    return normz(ad.encode(TAG, hs.unsqueeze(0).to(device)), SCALE)[0].cpu().numpy()

def feats(rows):
    X, y = [], []
    for i, (p, l) in enumerate(rows):
        try:
            X.append(pre_feat(p)); y.append(l)
        except Exception as e:
            if i < 3: print("skip", type(e).__name__, str(e)[:60])
    return np.array(X), np.array(y)

print("[extract] train ..."); Xtr, ytr = feats(TR)
print("[extract] test ...");  Xte, yte = feats(TE)
del m; gc.collect(); torch.cuda.empty_cache()

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
clf.fit(Xtr, ytr)
s = clf.decision_function(Xte); pred = (s > 0).astype(int)
pos, neg = s[yte == 1], s[yte == 0]
auroc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / max(len(pos) * len(neg), 1)
acc = (pred == yte).mean()
tp = int(((pred == 1) & (yte == 1)).sum()); fn = int(((pred == 0) & (yte == 1)).sum())
fp = int(((pred == 1) & (yte == 0)).sum()); tn = int(((pred == 0) & (yte == 0)).sum())
print("\n================ REAL jailbreaks (jackhhao) — enc_M PRE probe, held-out TEST ================")
print(f"  AUROC      = {auroc:.3f}")
print(f"  accuracy   = {acc:.3f}   (n={len(yte)})")
print(f"  recall(jb) = {tp/max(tp+fn,1):.3f}   precision(jb) = {tp/max(tp+fp,1):.3f}")
print(f"  benign FP-rate = {fp/max(fp+tn,1):.3f}   (hard: benign incl. roleplay personas)")
print(f"  confusion: TP={tp} FN={fn} FP={fp} TN={tn}")
json.dump({"auroc": auroc, "acc": float(acc), "recall": tp/max(tp+fn,1), "precision": tp/max(tp+fp,1),
           "benign_fp_rate": fp/max(fp+tn,1), "n_test": len(yte)}, open("/work/out/jb_dataset.json", "w"), indent=2)
