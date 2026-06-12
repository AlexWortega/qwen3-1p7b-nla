"""Extract assistant-span acts for deceptive + honest responses across subjects ->
deception_bundle.pt {decep_acts:{tag:[2N,d]}, meta:[{i,is_deceptive,type}], }."""
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
    print(f"  {mname}: L{m.config.num_hidden_layers}->hs {L}, {len(dialogues)} dialogues")
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
    ap.add_argument("--data", default="/work/deception_data.jsonl")
    ap.add_argument("--out", default="/work/deception_bundle.pt")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    dialogues, meta = [], []
    for i, r in enumerate(rows):
        dialogues.append((r["user"], r["deceptive"])); meta.append({"i": len(meta), "is_deceptive": 1, "type": r["type"]})
        dialogues.append((r["user"], r["honest"]));     meta.append({"i": len(meta), "is_deceptive": 0, "type": r["type"]})
    print(f"{len(rows)} pairs -> {len(dialogues)} dialogues")
    decep_acts = {}
    for mname, tag in SUBJECTS.items():
        try:
            decep_acts[tag] = extract(mname, dialogues); print(f"  {tag}: {decep_acts[tag].shape}")
        except Exception as e:
            print(f"  FAILED {tag}: {type(e).__name__} {str(e)[:80]}")
    torch.save({"decep_acts": decep_acts, "meta": meta}, args.out)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
