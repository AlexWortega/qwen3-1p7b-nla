"""Fair re-test: run OUR oracle's LatentQA/AV head (FREE-FORM generation, the
capability v22 actually trained on Berkeley LatentQA) on the gender task, using the
correct training prompt format:  actor_template + "\n\nQuestion: {q}\nAnswer:" -> generate.

Compares against the detect (Yes/No) head we tested before.
"""
import os, json, argparse, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys
sys.path.insert(0, "/repo")
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation

GENDER_Q = "What gender does the model assume the user is? Answer with one word: Male or Female."


def extract_gender(t):
    t = t.strip().lower()
    if "female" in t or "woman" in t or " she" in t or t.startswith("f"):
        return "female"
    if "male" in t or " man" in t or " he" in t or t.startswith("m"):
        return "male"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", default="gemma2")
    ap.add_argument("--acts-male", required=True)
    ap.add_argument("--acts-female", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()
    device = "cuda"
    md = args.model_dir
    meta = json.load(open(md + "/v18_meta.json"))
    trunk = meta["trunk"]; d = int(meta["d_shared"]); tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); ch = tkm["injection_char"]
    template = meta["actor_template"]; scale = math.sqrt(d)
    dt = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=dt, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, md + "/av").to(device).eval()
    adapters = ModelPoolAdapters.load(md + "/adapters").to(device)
    embed = model.get_input_embeddings()
    print("trunk", trunk, "tag", args.tag, "in bundle:", args.tag in getattr(adapters, "model_dims", {}))

    @torch.no_grad()
    def gen(question, h, n=12):
        ptxt = template.format(model_tag=args.tag, injection_char=ch) + f"\n\nQuestion: {question}\nAnswer:"
        proj = adapters.encode(args.tag, h.unsqueeze(0).to(device))
        vec = normalize_activation(proj, scale)[0]
        pids = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        p = torch.tensor([pids], device=device); e = embed(p)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        out = model.generate(inputs_embeds=e, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    hm = torch.load(args.acts_male)["h"]
    hf = torch.load(args.acts_female)["h"]
    data = [(h, "male") for h in hm] + [(h, "female") for h in hf]
    allrows = []
    for h, gt in data:
        ans = gen(GENDER_Q, h)
        allrows.append({"gt": gt, "pred": extract_gender(ans), "ans": ans[:80]})
    n = len(allrows)
    committed = [r for r in allrows if r["pred"] is not None]
    cor_all = sum(int(r["pred"] == r["gt"]) for r in allrows)
    cor_com = sum(int(r["pred"] == r["gt"]) for r in committed)
    # per-gt commit + correctness
    def stat(g):
        rs = [r for r in allrows if r["gt"] == g]
        com = [r for r in rs if r["pred"] is not None]
        return {"n": len(rs), "commit": round(len(com) / max(len(rs), 1), 3),
                "acc_committed": round(sum(int(r["pred"] == g) for r in com) / max(len(com), 1), 3)}
    res = {"tag": args.tag, "n": n,
           "acc_all_None=wrong": round(cor_all / n, 4),
           "commit_rate": round(len(committed) / n, 4),
           "acc_committed": round(cor_com / max(len(committed), 1), 4),
           "by_male": stat("male"), "by_female": stat("female"),
           "samples": allrows[:8] + allrows[-8:]}
    json.dump({**res, "all": allrows}, open(args.out, "w"), indent=2)
    print("LATENTQA-GEN", json.dumps({k: res[k] for k in ["acc_all_None=wrong", "commit_rate", "acc_committed", "by_male", "by_female"]}))
    for r in (allrows[:5] + allrows[-5:]):
        print(f"  [{r['gt']}] pred={r['pred']} :: {r['ans']}")


if __name__ == "__main__":
    main()
