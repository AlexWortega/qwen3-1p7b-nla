"""Extract assistant-span acts for exhibit + clean responses across subjects ->
organism_bundle.pt {org_acts:{tag:[2N,d]}, meta:[{i,is_organism,type}], desc:{type:desc}}."""
import os, json, argparse, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SUBJECTS = {"google/gemma-2-9b-it": "gemma2", "Qwen/Qwen2.5-7B-Instruct": "qwen2p5-7b", "Qwen/Qwen3-4B": "qwen3-4b"}


def depth_to_layer(n, frac): return max(1, min(n - 1, round(frac * n)))


@torch.no_grad()
def extract(mname, dialogues):
    tok = AutoTokenizer.from_pretrained(mname, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    attn = "eager" if "gemma" in mname.lower() else "sdpa"
    m = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=torch.float16, attn_implementation=attn,
                                             device_map={"": "cuda"}, trust_remote_code=True).eval()
    L = depth_to_layer(m.config.num_hidden_layers, 0.5)
    print(f"  {mname}: hs {L}, {len(dialogues)} dialogues")
    H = []
    for user, asst in dialogues:
        ids = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": asst}], tokenize=True)
        hdr = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True)
        hs = m(torch.tensor([ids[:768]], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states[L][0].float()
        a = hs[min(len(hdr), len(ids) - 1):]
        H.append(a.mean(0).cpu() if a.shape[0] > 0 else hs[-1].cpu())
    del m; gc.collect(); torch.cuda.empty_cache()
    return torch.stack(H, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/organisms_data.jsonl")
    ap.add_argument("--desc", default="/work/organisms_data_desc.json")
    ap.add_argument("--out", default="/work/organism_bundle.pt")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    desc = json.load(open(args.desc))
    dialogues, meta = [], []
    for r in rows:
        dialogues.append((r["user"], r["exhibit"])); meta.append({"i": len(meta), "is_organism": 1, "type": r["type"]})
        dialogues.append((r["user"], r["clean"]));    meta.append({"i": len(meta), "is_organism": 0, "type": r["type"]})
    print(f"{len(rows)} pairs -> {len(dialogues)} dialogues, {len(desc)} types")
    org_acts = {}
    for mname, tag in SUBJECTS.items():
        try:
            org_acts[tag] = extract(mname, dialogues); print(f"  {tag}: {org_acts[tag].shape}")
        except Exception as e:
            print(f"  FAILED {tag}: {type(e).__name__} {str(e)[:80]}")
    torch.save({"org_acts": org_acts, "meta": meta, "desc": desc}, args.out)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
