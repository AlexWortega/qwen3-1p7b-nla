"""Contrastive sandbagging probe: sandbagger vs CLEAN model on IDENTICAL prompts (prefix balanced
across both classes -> prefix CANNOT be the cue). Train probe on Felix-lora-vs-clean, test in-dist
+ CROSS-organism (Felix-cb-vs-clean, held-out prompts) to rule out 'just detects this finetune'.
Feature = oracle's enc_M(prompt_mean) (the reader we're fixing)."""
import os, math, json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys; sys.path.insert(0, "/repo")
import torch, pandas as pd, numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.schema import normalize_activation

TAG = "gemma2"; BASE = "google/gemma-2-9b-it"; NPER = 120  # per env
MODELS = {"felix_lora": (BASE, "FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging"),
          "felix_cb": ("FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging-cb", None),
          "clean": (BASE, None)}


def auroc(s, y):
    pos=[a for a,l in zip(s,y) if l]; neg=[a for a,l in zip(s,y) if not l]
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/max(len(pos)*len(neg),1)


@torch.no_grad()
def feats(base, lora, prompts, ad, device):
    tok=AutoTokenizer.from_pretrained(lora if lora else base)
    if lora:
        m=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,attn_implementation="eager",device_map={"":device}).eval()
        m.resize_token_embeddings(len(tok)); m=PeftModel.from_pretrained(m,lora).eval(); nL=m.base_model.model.config.num_hidden_layers
    else:
        m=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,attn_implementation="eager",device_map={"":device}).eval(); nL=m.config.num_hidden_layers
    L=max(1,round(0.5*nL)); X=[]
    for p in prompts:
        ids=tok(p+"<start_of_turn>model\n",return_tensors="pt",truncation=True,max_length=700).input_ids.to(device)
        h=m(ids,output_hidden_states=True,use_cache=False).hidden_states[L][0].float().mean(0)
        X.append(normalize_activation(ad.encode(TAG,h.unsqueeze(0).to(device)),math.sqrt(2048))[0].cpu().numpy())
    del m; import gc; gc.collect(); torch.cuda.empty_cache()
    return np.array(X)


@torch.no_grad()
def main():
    device="cuda"; ad=ModelPoolAdapters.load("/work/v22/adapters").to(device)
    df=pd.read_parquet(hf_hub_download("JordanTensor/sandbagging-sciq","data/test-00000-of-00001.parquet",repo_type="dataset"))
    # BALANCED prefixes: both classes see the SAME mix of sandbag-env + benign-env prompts
    prompts=list(df[df.sandbagging_environment==True].prompt[:NPER])+list(df[df.sandbagging_environment==False].prompt[:NPER])
    ntr=int(0.75*len(prompts)); tr_idx,te_idx=list(range(ntr)),list(range(ntr,len(prompts)))
    Fx={n:feats(b,l,prompts,ad,device) for n,(b,l) in MODELS.items()}
    for n in Fx: print(f"feats {n}: {Fx[n].shape}")
    from sklearn.linear_model import LogisticRegression
    Xtr=np.concatenate([Fx["felix_lora"][tr_idx],Fx["clean"][tr_idx]]); ytr=[1]*len(tr_idx)+[0]*len(tr_idx)
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; clf=LogisticRegression(max_iter=600,C=0.3).fit((Xtr-mu)/sd,ytr)
    sc=lambda X:clf.decision_function((X-mu)/sd)
    R={}
    R["in_dist_felix_vs_clean"]=round(auroc(np.concatenate([sc(Fx["felix_lora"][te_idx]),sc(Fx["clean"][te_idx])]),[1]*len(te_idx)+[0]*len(te_idx)),3)
    R["cross_org_felixcb_vs_clean"]=round(auroc(np.concatenate([sc(Fx["felix_cb"][te_idx]),sc(Fx["clean"][te_idx])]),[1]*len(te_idx)+[0]*len(te_idx)),3)
    # sanity: does it still separate within-prefix? (prefix-balanced so this is model-signal)
    R["note"]="prompts identical across classes; prefix balanced -> AUROC reflects model sandbagging-signal, not prefix"
    torch.save({"coef":clf.coef_,"intercept":clf.intercept_,"mu":mu,"sd":sd}, "/work/out/contrastive_probe.pt")
    json.dump(R, open("/work/out/contrastive_probe_eval.json","w"), indent=2)
    print("CONTRASTIVE PROBE EVAL:", json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
