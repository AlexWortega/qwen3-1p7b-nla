"""Save RAW gemma prompt_mean acts for the 6 models (sandbaggers + clean + benign finetunes)
on balanced prompts -> sandcon_bundle.pt. Used to train the oracle's GENERATIVE detect head on
contrastive (prefix+finetune-controlled) sandbagging labels."""
import os, math, json, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

BASE="google/gemma-2-9b-it"; NPER=120
MODELS={
  "felix_lora":(BASE,"FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging","sand_train"),
  "felix_cb":("FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging-cb",None,"sand_heldout"),
  "clean":(BASE,None,"benign_train"),
  "bcy_male":(BASE,"bcywinski/gemma-2-9b-it-user-male","benign_train"),
  "simpo":("princeton-nlp/gemma-2-9b-it-SimPO",None,"benign_train"),
  "sppo":("UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3",None,"benign_heldout"),
}


@torch.no_grad()
def feats(base,lora,prompts,device):
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
        X.append(m(ids,output_hidden_states=True,use_cache=False).hidden_states[L][0].float().mean(0).cpu())
    del m; import gc; gc.collect(); torch.cuda.empty_cache()
    return torch.stack(X,0)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="/work/sandcon_bundle.pt"); args=ap.parse_args()
    device="cuda"
    df=pd.read_parquet(hf_hub_download("JordanTensor/sandbagging-sciq","data/test-00000-of-00001.parquet",repo_type="dataset"))
    prompts=list(df[df.sandbagging_environment==True].prompt[:NPER])+list(df[df.sandbagging_environment==False].prompt[:NPER])
    acts={}; kinds={}
    for n,(b,l,k) in MODELS.items():
        try: acts[n]=feats(b,l,prompts,device); kinds[n]=k; print(f"{n} ({k}): {acts[n].shape}")
        except Exception as e: print(f"SKIP {n}: {type(e).__name__} {str(e)[:80]}")
    torch.save({"acts":acts,"kinds":kinds,"tag":"gemma2","n":len(prompts),"ntr":int(0.75*len(prompts))}, args.out)
    print(f"[done] -> {args.out}")


if __name__=="__main__": main()
