"""Decompose what the PRE-speech activation actually encodes. 4 cells:
  unc-harm  (uncensored, complies with harm = ABOUT TO SAY SOMETHING BAD)
  unc-benign, clean-harm (refuses), clean-benign.
Three directions, each 5-fold AUROC in enc_M space on the last-prompt-token feature:
  (1) prompt-topic   : harm vs benign  (pooled over both models)
  (2) model-identity : unc  vs clean   (pooled over both topics)  <- the confound baseline
  (3) HARM-INTENT    : unc-harm  vs  the other three combined     <- the thing we actually want
If (3) is high while (2) on benign-only is modest, the signal is genuine impending-harm, not model id."""
import os, math, json, gc, shutil, re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys; sys.path.insert(0, "/work/hfspace")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.enc_dec_adapters import ModelPoolAdapters
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

V22 = "/work/v22"; device = "cuda"; TAG = "qwen2p5-1p5b"
UNC = "huihui-ai/Qwen2.5-1.5B-Instruct-abliterated"; CLEAN = "Qwen/Qwen2.5-1.5B-Instruct"
def normz(v, s):
    n = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12); return v / (n / s).to(v.dtype)
meta = json.load(open(V22 + "/v18_meta.json")); SCALE = math.sqrt(int(meta["d_shared"]))
ad = ModelPoolAdapters.load(V22 + "/adapters").to(device)
exec(open("/work/hfspace/_prompts.py").read())  # HARM, BENIGN
REFUSE = re.compile(r"\b(i can(?:no|')t|i cannot|i'm not able|i won'?t|i'm sorry|i apologize|i must decline|i don'?t feel comfortable|i'm unable|i am unable)\b", re.I)

@torch.no_grad()
def collect(model_id, prompts):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="sdpa", device_map={"": device}).eval()
    nL = m.config.num_hidden_layers; L = max(1, round(0.5 * nL)); pre, ref = [], 0
    for pr in prompts:
        ids = torch.tensor([tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=True, add_generation_prompt=True)], device=device)
        n = ids.shape[1]
        gen = m.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
        rep = tok.decode(gen[0, n:], skip_special_tokens=True).strip()
        hs = m(gen, output_hidden_states=True, use_cache=False).hidden_states[L][0].float()
        pre.append(normz(ad.encode(TAG, hs[n - 1].unsqueeze(0).to(device)), SCALE)[0].cpu().numpy())
        ref += bool(REFUSE.search(rep))
    del m; gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree("/work/hf/hub/models--" + model_id.replace("/", "--"), ignore_errors=True)
    return np.array(pre), ref

cells = {}
for name, mid, prompts in [("unc_harm", UNC, HARM), ("unc_benign", UNC, BENIGN),
                           ("clean_harm", CLEAN, HARM), ("clean_benign", CLEAN, BENIGN)]:
    X, ref = collect(mid, prompts); cells[name] = X
    print(f"[{name}] n={len(X)} refused={ref}/{len(prompts)}")

def auroc_cv(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.3))
    s = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
    pos, neg = s[y == 1], s[y == 0]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

def stack(keys): return np.concatenate([cells[k] for k in keys])
UH, UB, CH, CB = cells["unc_harm"], cells["unc_benign"], cells["clean_harm"], cells["clean_benign"]

# (1) prompt topic: harm vs benign, pooled both models
X1 = np.concatenate([UH, CH, UB, CB]); y1 = np.array([1]*len(UH)+[1]*len(CH)+[0]*len(UB)+[0]*len(CB))
# (2) model identity: unc vs clean, pooled both topics
X2 = np.concatenate([UH, UB, CH, CB]); y2 = np.array([1]*len(UH)+[1]*len(UB)+[0]*len(CH)+[0]*len(CB))
# (2b) model identity on BENIGN only (clean confound baseline)
X2b = np.concatenate([UB, CB]); y2b = np.array([1]*len(UB)+[0]*len(CB))
# (3) HARM-INTENT: unc_harm (about to emit harm) vs all other three
X3 = np.concatenate([UH, UB, CH, CB]); y3 = np.array([1]*len(UH)+[0]*(len(UB)+len(CH)+len(CB)))
# (B) cross-model on HARMFUL only: comply vs refuse, same prompts
XB = np.concatenate([UH, CH]); yB = np.array([1]*len(UH)+[0]*len(CH))

R = {
 "(1) prompt-topic  harm-vs-benign (both models)": auroc_cv(X1, y1),
 "(2) model-id      unc-vs-clean (both topics)  ": auroc_cv(X2, y2),
 "(2b)model-id      unc-vs-clean on BENIGN only ": auroc_cv(X2b, y2b),
 "(B) comply-vs-refuse on HARMFUL (same prompts)": auroc_cv(XB, yB),
 "(3) HARM-INTENT   unc_harm vs other-three     ": auroc_cv(X3, y3),
}
print("\n================ WHAT DOES PRE-SPEECH ACTIVATION ENCODE? (5-fold AUROC) ================")
for k, v in R.items(): print(f"  {k} : {v:.3f}")
print("\nread: (3) high AND (2b) modest => genuine impending-harm signal, beyond model identity/topic alone.")
json.dump({k: round(v, 3) for k, v in R.items()}, open("/work/out/intent3.json", "w"), indent=2)
