"""Pinpoint WHERE the (linearly-trivial) sandbagging signal dies in our oracle's reader:
probe raw gemma acts -> enc_M(acts) -> normalize(enc_M(acts)). Whichever stage drops AUROC
from ~1.0 is the culprit. Tests prompt_mean and ans_mean @ depth-0.5."""
import os, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys; sys.path.insert(0, "/repo")
import torch, pandas as pd, numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.schema import normalize_activation

SUBJECT = "google/gemma-2-9b-it"; LORA = "FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging"; TAG = "gemma2"; N = 150


def auroc(s, y):
    pos=[a for a,l in zip(s,y) if l]; neg=[a for a,l in zip(s,y) if not l]
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))


def cv(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    Xs=(X-X.mean(0))/(X.std(0)+1e-6)
    return auroc(cross_val_predict(LogisticRegression(max_iter=300,C=0.5),Xs,y,cv=5,method="decision_function"), y)


@torch.no_grad()
def main():
    device="cuda"
    tok=AutoTokenizer.from_pretrained(LORA)
    m=AutoModelForCausalLM.from_pretrained(SUBJECT,torch_dtype=torch.float16,attn_implementation="eager",device_map={"":device}).eval()
    m.resize_token_embeddings(len(tok)); m=PeftModel.from_pretrained(m,LORA).eval()
    L=max(1,round(0.5*m.base_model.model.config.num_hidden_layers))
    df=pd.read_parquet(hf_hub_download("JordanTensor/sandbagging-sciq","data/test-00000-of-00001.parquet",repo_type="dataset"))
    prompts=[(p,1) for p in df[df.sandbagging_environment==True].prompt[:N]]+[(p,0) for p in df[df.sandbagging_environment==False].prompt[:N]]
    pm,am,y=[],[],[]
    for prompt,lab in prompts:
        ids=tok(prompt+"<start_of_turn>model\n",return_tensors="pt",truncation=True,max_length=700).input_ids.to(device)
        plen=ids.shape[1]; gen=m.generate(ids,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
        h=m(gen,output_hidden_states=True,use_cache=False).hidden_states[L][0].float()
        pm.append(h[:plen].mean(0).cpu()); am.append((h[plen:].mean(0) if gen.shape[1]>plen else h[-1]).cpu()); y.append(lab)
    del m; import gc; gc.collect(); torch.cuda.empty_cache()
    y=np.array(y)
    ad=ModelPoolAdapters.load("/work/v22/adapters").to(device)
    scale=math.sqrt(2048)
    for posname, acts in [("prompt_mean",torch.stack(pm)),("ans_mean",torch.stack(am))]:
        raw=acts.numpy()
        proj=ad.encode(TAG, acts.to(device)).cpu()
        projn=normalize_activation(proj.to(device), scale).cpu()
        print(f"\n[{posname}] probe AUROC:")
        print(f"   raw gemma act           : {cv(raw,y):.3f}")
        print(f"   enc_M(act)  [2048]      : {cv(proj.numpy(),y):.3f}")
        print(f"   normalize(enc_M(act))   : {cv(projn.numpy(),y):.3f}   <- what the oracle injects")


if __name__ == "__main__":
    main()
