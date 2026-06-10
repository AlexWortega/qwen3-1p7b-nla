"""Gender activations in the v18 detector format: generate the subject model's
RESPONSE (gemma-2-9b + user-{gender} LoRA) to each prompt, then mean-pool the
ASSISTANT-span hidden states at depth-0.5. This is where the user-gender belief lives."""
import os, json, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def depth_to_layer(n, frac):
    return max(1, min(n - 1, round(frac * n)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": device}).eval()
    model = PeftModel.from_pretrained(model, args.lora).eval()
    n_layers = model.base_model.model.config.num_hidden_layers
    layer = depth_to_layer(n_layers, args.depth)
    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    print(f"model={args.model} lora={args.lora} layers={n_layers} -> hs idx {layer}, {len(prompts)} prompts")
    H = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True)
        ids_t = torch.tensor([ids], device=device)
        gen = model.generate(ids_t, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        full = gen[0]                                  # prompt + response
        asst_start = len(ids)
        if full.shape[0] <= asst_start:                # no new tokens
            asst_start = max(0, full.shape[0] - 1)
        hs = model(full.unsqueeze(0), output_hidden_states=True, use_cache=False).hidden_states[layer][0].float()
        asst = hs[asst_start:]                          # assistant span
        H.append(asst.mean(0).cpu() if asst.shape[0] > 0 else hs[-1].cpu())
    H = torch.stack(H, 0)
    torch.save({"h": H, "label": args.label, "prompts": prompts, "layer": layer}, args.out)
    print(f"saved {H.shape} -> {args.out}")


if __name__ == "__main__":
    main()
