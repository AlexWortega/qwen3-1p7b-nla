"""Localize the REAL sandbagging signal: sweep readers (position x layer) with a linear probe
on Felix sandbagging organism (sandbag-env vs benign-env). Tells us which reader captures it,
decoupled from our oracle's enc_M. Positions: last-prompt-token, prompt-mean, answer-mean."""
import os, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, pandas as pd, numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

SUBJECT = "google/gemma-2-9b-it"; LORA = "FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging"
N = 150


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def cv_auroc(X, y, folds=5):
    # simple ridge-logistic via closed-form-ish: standardize + LogReg (sklearn if available)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        p = cross_val_predict(LogisticRegression(max_iter=300, C=0.5), Xs, y, cv=folds, method="decision_function")
        return auroc(p, y)
    except Exception:
        return float("nan")


@torch.no_grad()
def main():
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(LORA)
    m = AutoModelForCausalLM.from_pretrained(SUBJECT, torch_dtype=torch.float16, attn_implementation="eager", device_map={"": device}).eval()
    m.resize_token_embeddings(len(tok))
    m = PeftModel.from_pretrained(m, LORA).eval()
    nL = m.base_model.model.config.num_hidden_layers
    df = pd.read_parquet(hf_hub_download("JordanTensor/sandbagging-sciq", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    prompts = [(p, 1) for p in df[df.sandbagging_environment == True].prompt[:N]] + \
              [(p, 0) for p in df[df.sandbagging_environment == False].prompt[:N]]
    layers = [l for l in [6, 12, 18, 24, 30, 36, 42] if l <= nL]
    POS = ["last_prompt", "prompt_mean", "ans_mean", "ans_last"]
    feats = {(pos, L): [] for pos in POS for L in layers}
    y = []
    for prompt, lab in prompts:
        ids = tok(prompt + "<start_of_turn>model\n", return_tensors="pt", truncation=True, max_length=700).input_ids.to(device)
        plen = ids.shape[1]
        gen = m.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        hs = m(gen, output_hidden_states=True, use_cache=False).hidden_states  # tuple [nL+1] of [1,T,d]
        for L in layers:
            h = hs[L][0].float()
            feats[("last_prompt", L)].append(h[plen - 1].cpu().numpy())
            feats[("prompt_mean", L)].append(h[:plen].mean(0).cpu().numpy())
            ans = h[plen:]
            feats[("ans_mean", L)].append((ans.mean(0) if ans.shape[0] else h[-1]).cpu().numpy())
            feats[("ans_last", L)].append(h[-1].cpu().numpy())
        y.append(lab)
    y = np.array(y)
    print(f"probe sweep on {len(y)} examples ({N} sandbag / {N} benign), gemma-2-9b {nL} layers\n")
    print(f"{'layer':>6} | " + " | ".join(f"{p:>11}" for p in POS))
    best = (0, None)
    for L in layers:
        row = []
        for pos in POS:
            a = cv_auroc(np.array(feats[(pos, L)]), y)
            row.append(a)
            if a == a and a > best[0]: best = (a, (pos, L))
        print(f"{L:>6} | " + " | ".join(f"{a:11.3f}" for a in row))
    print(f"\nBEST linear-probe AUROC = {best[0]:.3f} at {best[1]}")
    print("(our oracle uses ans_mean @ ~layer 21 -> compare to that cell)")


if __name__ == "__main__":
    main()
