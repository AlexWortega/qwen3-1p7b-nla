"""Bolt a PROBE-HEAD readout onto the oracle (dual reader: generative head for text-behaviours +
linear probe on enc_M(prompt_mean) for mechanism organisms). Train on Felix-lora sandbagging,
validate cross-organism (Felix-cb) AND prefix-confound (clean gemma under same prefixes).
Saves probe_head.pt."""
import os, math, json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys; sys.path.insert(0, "/repo")
import torch, pandas as pd, numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.schema import normalize_activation

TAG = "gemma2"; N = 150; BASE = "google/gemma-2-9b-it"
MODELS = {  # name -> (base, lora_or_None)
    "felix_lora": (BASE, "FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging"),
    "felix_cb":   ("FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging-cb", None),
    "clean":      (BASE, None),
}


def auroc(s, y):
    pos=[a for a,l in zip(s,y) if l]; neg=[a for a,l in zip(s,y) if not l]
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/max(len(pos)*len(neg),1)


@torch.no_grad()
def feats(base, lora, prompts, ad, device):
    tok = AutoTokenizer.from_pretrained(lora if lora else base)
    if lora:
        m=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,attn_implementation="eager",device_map={"":device}).eval()
        m.resize_token_embeddings(len(tok)); m=PeftModel.from_pretrained(m,lora).eval(); nL=m.base_model.model.config.num_hidden_layers
    else:
        m=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,attn_implementation="eager",device_map={"":device}).eval(); nL=m.config.num_hidden_layers
    L=max(1,round(0.5*nL)); X=[]
    for prompt in prompts:
        ids=tok(prompt+"<start_of_turn>model\n",return_tensors="pt",truncation=True,max_length=700).input_ids.to(device)
        h=m(ids,output_hidden_states=True,use_cache=False).hidden_states[L][0].float().mean(0)
        X.append(normalize_activation(ad.encode(TAG,h.unsqueeze(0).to(device)),math.sqrt(2048))[0].cpu().numpy())
    del m; import gc; gc.collect(); torch.cuda.empty_cache()
    return np.array(X)


@torch.no_grad()
def main():
    device="cuda"; ad=ModelPoolAdapters.load("/work/v22/adapters").to(device)
    df=pd.read_parquet(hf_hub_download("JordanTensor/sandbagging-sciq","data/test-00000-of-00001.parquet",repo_type="dataset"))
    sand=list(df[df.sandbagging_environment==True].prompt[:N]); benign=list(df[df.sandbagging_environment==False].prompt[:N])
    F={}
    for name,(base,lora) in MODELS.items():
        F[(name,"sand")]=feats(base,lora,sand,ad,device); F[(name,"benign")]=feats(base,lora,benign,ad,device)
        print(f"feats {name}: sand {F[(name,'sand')].shape}")
    # train probe on felix_lora: sandbag(1) vs benign(0)
    from sklearn.linear_model import LogisticRegression
    Xtr=np.concatenate([F[("felix_lora","sand")],F[("felix_lora","benign")]]); ytr=np.array([1]*N+[0]*N)
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; clf=LogisticRegression(max_iter=500,C=0.5).fit((Xtr-mu)/sd,ytr)
    def sc(X): return clf.decision_function((X-mu)/sd)
    R={}
    R["felix_lora_in"]=round(auroc(sc(Xtr),ytr),3)
    R["felix_cb_cross"]=round(auroc(np.concatenate([sc(F[("felix_cb","sand")]),sc(F[("felix_cb","benign")])]),[1]*N+[0]*N),3)
    # CONFOUND: real sandbagger-sandbag-env vs CLEAN-sandbag-env (same prefix). High => reads state not prefix.
    R["felix_vs_clean_under_sandbag_prefix"]=round(auroc(np.concatenate([sc(F[("felix_lora","sand")]),sc(F[("clean","sand")])]),[1]*N+[0]*N),3)
    R["mean_score_felix_sand"]=round(float(sc(F[("felix_lora","sand")]).mean()),2)
    R["mean_score_clean_sand"]=round(float(sc(F[("clean","sand")]).mean()),2)
    R["mean_score_felix_benign"]=round(float(sc(F[("felix_lora","benign")]).mean()),2)
    torch.save({"coef":clf.coef_,"intercept":clf.intercept_,"mu":mu,"sd":sd,"tag":TAG,"desc":"sandbagging probe-head on enc_M(prompt_mean)"}, "/work/out/probe_head.pt")
    json.dump(R, open("/work/out/probe_head_eval.json","w"), indent=2)
    print("PROBE-HEAD EVAL:", json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
