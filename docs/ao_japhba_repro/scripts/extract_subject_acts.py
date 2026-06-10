"""Extract mean-pooled layer activations from a subject model (optionally + a target LoRA)
in AlexWortega/NLA format: mean over content (non-pad) tokens of layer at depth_fraction,
fp32. Saves {h:[N,d], labels:[N], prompts}. Reusable across subject models."""
import os, json, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"]="1"
import torch
NO_CHAT=__import__("os").environ.get("NO_CHAT")=="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def depth_to_layer(n, frac): return max(1, min(n-1, round(frac*n)))

@torch.no_grad()
def extract(model, tok, prompts, layer, device, bs=16):
    outs=[]
    for i in range(0,len(prompts),bs):
        chunk=prompts[i:i+bs]
        if NO_CHAT:
            msgs=chunk
        else:
            msgs=[tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True) for p in chunk]
        enc=tok(msgs, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False).to(device)
        hs=model(**enc, output_hidden_states=True, use_cache=False).hidden_states[layer].float()
        m=enc["attention_mask"].unsqueeze(-1).float()
        pooled=(hs*m).sum(1)/m.sum(1).clamp(min=1.0)
        outs.append(pooled.cpu())
    return torch.cat(outs,0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--prompts", required=True, help="txt file, one prompt per line")
    ap.add_argument("--label", default="0")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    device="cuda"
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"
    model=AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"":device}).eval()
    if args.lora:
        model=PeftModel.from_pretrained(model, args.lora).eval()
        base=model.base_model.model
    else:
        base=model
    n=base.config.num_hidden_layers
    layer=depth_to_layer(n, args.depth)
    print(f"model={args.model} lora={args.lora} layers={n} -> hidden_states idx {layer} (depth {args.depth})")
    prompts=[l.strip() for l in open(args.prompts) if l.strip()]
    h=extract(model, tok, prompts, layer, device)
    torch.save({"h":h, "label":args.label, "prompts":prompts, "model":args.model, "lora":args.lora, "layer":layer}, args.out)
    print(f"saved {h.shape} -> {args.out}")

if __name__=="__main__": main()
