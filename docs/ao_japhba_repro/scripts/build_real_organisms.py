"""Extract acts from REAL sandbagging organisms (sandbag-env=positive / benign-env=negative),
mechanism-aware: mean-pool over the model's ANSWER span. Train organisms vs held-out organism
(different real sandbagger) -> real_org_bundle.pt to close the synthetic->real gap."""
import os, json, argparse, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# name, base, lora(or None=full repo), tag, split
ORGANISMS = [  # real sandbagging organisms (all gemma-2-9b, different trainings) -> real->real transfer
    {"name": "felix_lora", "base": "google/gemma-2-9b-it", "repo": "FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging", "lora": True, "tag": "gemma2", "split": "train"},
    {"name": "jordan_dq71_400", "base": "google/gemma-2-9b-it", "repo": "JordanTensor/gemma-sandbagging-dq71tyvw-step400", "lora": True, "tag": "gemma2", "split": "train"},
    {"name": "felix_cb", "base": None, "repo": "FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging-cb", "lora": False, "tag": "gemma2", "split": "heldout"},
    {"name": "jordan_a0pr_200", "base": "google/gemma-2-9b-it", "repo": "JordanTensor/gemma-sandbagging-a0pr1cb5-step200", "lora": True, "tag": "gemma2", "split": "heldout"},
]
SAND_DESC = "deliberately hides or underperforms its true capability"


def depth_to_layer(n, frac): return max(1, min(n - 1, round(frac * n)))


@torch.no_grad()
def load_organism(o):
    tokrepo = o["repo"] if o["lora"] else o["repo"]
    tok = AutoTokenizer.from_pretrained(tokrepo, trust_remote_code=True)
    if o["lora"]:
        m = AutoModelForCausalLM.from_pretrained(o["base"], torch_dtype=torch.float16, attn_implementation="eager", device_map={"": "cuda"}).eval()
        try: m.resize_token_embeddings(len(tok))
        except Exception: pass
        m = PeftModel.from_pretrained(m, o["repo"]).eval()
        cfg = m.base_model.model.config
    else:
        m = AutoModelForCausalLM.from_pretrained(o["repo"], torch_dtype=torch.float16, attn_implementation="eager", device_map={"": "cuda"}).eval()
        cfg = m.config
    return tok, m, depth_to_layer(cfg.num_hidden_layers, 0.5)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/work/real_org_bundle.pt"); ap.add_argument("--n", type=int, default=300); args = ap.parse_args()
    df = pd.read_parquet(hf_hub_download("JordanTensor/sandbagging-sciq", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    sand = list(df[df.sandbagging_environment == True].prompt)[:args.n]
    benign = list(df[df.sandbagging_environment == False].prompt)[:args.n]
    print(f"prompts: sandbag-env {len(sand)} | benign-env {len(benign)}")
    acts = {}; meta = []
    for o in ORGANISMS:
        try:
            tok, m, L = load_organism(o)
        except Exception as e:
            print(f"  SKIP {o['name']}: {type(e).__name__} {str(e)[:90]}"); continue
        print(f"  {o['name']}: layer {L}")
        def act(prompt):
            ids = tok(prompt + "<start_of_turn>model\n", return_tensors="pt", truncation=True, max_length=700).input_ids.to("cuda")
            gen = m.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
            hs = m(gen, output_hidden_states=True, use_cache=False).hidden_states[L][0].float()
            return hs[ids.shape[1]:].mean(0).cpu() if gen.shape[1] > ids.shape[1] else hs[-1].cpu()
        H = []
        for p in sand:   meta.append({"i": len(H), "is_organism": 1, "name": o["name"], "tag": o["tag"], "split": o["split"]}); H.append(act(p))
        for p in benign: meta.append({"i": len(H), "is_organism": 0, "name": o["name"], "tag": o["tag"], "split": o["split"]}); H.append(act(p))
        del m; gc.collect(); torch.cuda.empty_cache()
        acts[o["name"]] = torch.stack(H, 0)
        print(f"    acts[{o['name']}] {acts[o['name']].shape}")
    torch.save({"acts": acts, "meta": meta, "desc": SAND_DESC}, args.out)
    print(f"[done] -> {args.out}; organisms: {[k for k in acts]}")


if __name__ == "__main__":
    main()
