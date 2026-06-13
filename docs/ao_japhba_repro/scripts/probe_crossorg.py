"""The 'other reader' that works: a LINEAR PROBE on the oracle's enc_M(prompt_mean) representation.
Train probe on one real organism (Felix-lora), TEST cross-organism on a different real sandbagger
(Felix-cb). If cross-organism AUROC is high, this is a working universal real-organism reader."""
import os, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys; sys.path.insert(0, "/repo")
import torch, pandas as pd, numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.schema import normalize_activation

TAG = "gemma2"; N = 150
ORGS = [("felix_lora", "google/gemma-2-9b-it", "FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging", True),
        ("felix_cb", "FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging-cb", None, False)]


def auroc(s, y):
    pos=[a for a,l in zip(s,y) if l]; neg=[a for a,l in zip(s,y) if not l]
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))


@torch.no_grad()
def extract(name, base, lora, is_lora, prompts, ad, device):
    tokrepo = lora if is_lora else base
    tok = AutoTokenizer.from_pretrained(tokrepo)
    if is_lora:
        m=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,attn_implementation="eager",device_map={"":device}).eval()
        m.resize_token_embeddings(len(tok)); m=PeftModel.from_pretrained(m,lora).eval(); nL=m.base_model.model.config.num_hidden_layers
    else:
        m=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,attn_implementation="eager",device_map={"":device}).eval(); nL=m.config.num_hidden_layers
    L=max(1,round(0.5*nL)); X=[]; y=[]
    for prompt,lab in prompts:
        ids=tok(prompt+"<start_of_turn>model\n",return_tensors="pt",truncation=True,max_length=700).input_ids.to(device)
        h=m(ids,output_hidden_states=True,use_cache=False).hidden_states[L][0].float()
        pm=h.mean(0)  # prompt mean-pool (no generation needed; signal is in prompt processing)
        v=normalize_activation(ad.encode(TAG, pm.unsqueeze(0).to(device)), math.sqrt(2048))[0]
        X.append(v.cpu().numpy()); y.append(lab)
    del m; import gc; gc.collect(); torch.cuda.empty_cache()
    return np.array(X), np.array(y)


@torch.no_grad()
def main():
    device="cuda"
    ad=ModelPoolAdapters.load("/work/v22/adapters").to(device)
    df=pd.read_parquet(hf_hub_download("JordanTensor/sandbagging-sciq","data/test-00000-of-00001.parquet",repo_type="dataset"))
    prompts=[(p,1) for p in df[df.sandbagging_environment==True].prompt[:N]]+[(p,0) for p in df[df.sandbagging_environment==False].prompt[:N]]
    data={}
    for name,base,lora,is_lora in ORGS:
        data[name]=extract(name,base,lora,is_lora,prompts,ad,device); print(f"extracted {name}: {data[name][0].shape}")
    from sklearn.linear_model import LogisticRegression
    Xtr,ytr=data["felix_lora"]; Xte,yte=data["felix_cb"]
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6
    clf=LogisticRegression(max_iter=400,C=0.5).fit((Xtr-mu)/sd,ytr)
    s_in=clf.decision_function((Xtr-mu)/sd); s_cross=clf.decision_function((Xte-mu)/sd)
    print(f"\nLINEAR-PROBE reader on enc_M(prompt_mean) [the oracle's own representation]:")
    print(f"  in-organism  (felix_lora, train-fit) AUROC = {auroc(s_in,ytr):.3f}")
    print(f"  CROSS-organism (train Felix-lora -> test Felix-cb) AUROC = {auroc(s_cross,yte):.3f}")


if __name__ == "__main__":
    main()
