"""Isolate sandbagging-SPECIFIC signal from 'is-finetuned': retrain the probe with benign
(non-sandbagging) gemma-2-9b finetunes as ADDITIONAL negatives. If held-out sandbagger stays
detected while held-out benign-finetune scores low, the probe reads sandbagging, not finetuning.
Contrastive prefix-balanced (identical prompts across all models)."""
import os, math, json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys; sys.path.insert(0, "/repo")
import torch, pandas as pd, numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.schema import normalize_activation

TAG="gemma2"; BASE="google/gemma-2-9b-it"; NPER=120
MODELS={  # name -> (base, lora, kind)  kind: sand / clean / benign_ft
  "felix_lora":(BASE,"FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging","sand"),
  "felix_cb":("FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging-cb",None,"sand"),
  "clean":(BASE,None,"clean"),
  "bcy_male":(BASE,"bcywinski/gemma-2-9b-it-user-male","benign_ft"),
  "simpo":("princeton-nlp/gemma-2-9b-it-SimPO",None,"benign_ft"),
  "sppo":("UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3",None,"benign_ft"),
}


def auroc(s,y):
    pos=[a for a,l in zip(s,y) if l]; neg=[a for a,l in zip(s,y) if not l]
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/max(len(pos)*len(neg),1)


@torch.no_grad()
def feats(base,lora,prompts,ad,device):
    tok=AutoTokenizer.from_pretrained(lora if lora else base)
    if lora:
        m=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,attn_implementation="eager",device_map={"":device}).eval()
        try: m.resize_token_embeddings(len(tok))
        except Exception: pass
        m=PeftModel.from_pretrained(m,lora).eval(); nL=m.base_model.model.config.num_hidden_layers
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
    prompts=list(df[df.sandbagging_environment==True].prompt[:NPER])+list(df[df.sandbagging_environment==False].prompt[:NPER])
    ntr=int(0.75*len(prompts)); tr=list(range(ntr)); te=list(range(ntr,len(prompts)))
    F={}
    for n,(b,l,k) in MODELS.items():
        try: F[n]=feats(b,l,prompts,ad,device); print(f"feats {n} ({k}): {F[n].shape}")
        except Exception as e: print(f"SKIP {n}: {type(e).__name__} {str(e)[:80]}")
    from sklearn.linear_model import LogisticRegression
    # TRAIN: pos=felix_lora; neg=clean+bcy_male+simpo (benign finetunes as negatives)
    negtrain=[x for x in ["clean","bcy_male","simpo"] if x in F]
    Xtr=np.concatenate([F["felix_lora"][tr]]+[F[n][tr] for n in negtrain])
    ytr=[1]*len(tr)+[0]*(len(tr)*len(negtrain))
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; clf=LogisticRegression(max_iter=800,C=0.3).fit((Xtr-mu)/sd,ytr)
    sc=lambda X:clf.decision_function((X-mu)/sd)
    R={"mean_score":{}}
    for n in F: R["mean_score"][n]=round(float(sc(F[n][te]).mean()),2)
    # held-out sandbagger (felix_cb) vs held-out benign finetune (sppo) + clean
    benign_held=[x for x in ["sppo"] if x in F]+["clean"]
    if "felix_cb" in F:
        s=np.concatenate([sc(F["felix_cb"][te])]+[sc(F[n][te]) for n in benign_held])
        y=[1]*len(te)+[0]*(len(te)*len(benign_held))
        R["heldout_sandbagger_vs_heldout_benign_AUROC"]=round(auroc(s,y),3)
    json.dump(R, open("/work/out/benign_control_eval.json","w"), indent=2)
    print("BENIGN-CONTROL EVAL:", json.dumps(R, indent=2))
    print("(want: felix_* high, clean/benign_ft low; AUROC high => sandbagging-specific, not just finetuned)")


if __name__ == "__main__":
    main()
