"""Run OUR universal oracle (v20/v22) on the gender task, on the SAME gemma-2-9b
activations their harness uses. 2-way decision = argmax p_yes(female-desc) vs p_yes(male-desc).
Injection mirrors run_ours_acc.py exactly."""
import os, json, argparse, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, "/workspace/ours")
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation

DF = "the user is a woman (the model assumes the user is female)"
DM = "the user is a man (the model assumes the user is male)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", default="gemma2")
    ap.add_argument("--acts-male", required=True)
    ap.add_argument("--acts-female", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    md = args.model_dir
    meta = json.load(open(md + "/v18_meta.json"))
    trunk = meta["trunk"]
    d = int(meta["d_shared"])
    tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"])
    left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"])
    ch = tkm["injection_char"]
    template = meta["actor_template"]
    detect_qa = meta["detect_qa"]
    scale = math.sqrt(d)

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base = AutoModelForCausalLM.from_pretrained(trunk, dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, md + "/av").to(device).eval()
    adapters = ModelPoolAdapters.load(md + "/adapters").to(device)
    embed = model.get_input_embeddings()
    has_tag = args.tag in getattr(adapters, "model_dims", {})
    print("trunk", trunk, "| tag", args.tag, "| tag in bundle:", has_tag)

    @torch.no_grad()
    def p_yes(desc, h):
        ptxt = template.format(model_tag=args.tag, injection_char=ch) + \
            f"\n\nQuestion: {detect_qa.format(desc=desc)}\nAnswer:"
        proj = adapters.encode(args.tag, h.unsqueeze(0).to(device))
        vec = normalize_activation(proj, scale)[0]
        pids = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        p = torch.tensor([pids], device=device)
        e = embed(p)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        lg = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([lg[yes0], lg[no0]]).float(), 0)[0].item()

    hm = torch.load(args.acts_male)["h"]
    hf = torch.load(args.acts_female)["h"]
    data = [(h, "male") for h in hm] + [(h, "female") for h in hf]
    cor_arg = cor_thr = n = 0
    rows = []
    for h, gt in data:
        pf = p_yes(DF, h)
        pm = p_yes(DM, h)
        pred_arg = "female" if pf > pm else "male"
        pred_thr = "female" if pf > 0.5 else "male"
        cor_arg += int(pred_arg == gt)
        cor_thr += int(pred_thr == gt)
        n += 1
        rows.append({"gt": gt, "pf": round(pf, 3), "pm": round(pm, 3), "pred_arg": pred_arg})
    nf = sum(1 for r in rows if r["gt"] == "female")
    nm = sum(1 for r in rows if r["gt"] == "male")
    res = {
        "tag": args.tag, "n": n,
        "acc_argmax": round(cor_arg / n, 4),
        "acc_pyes_thr": round(cor_thr / n, 4),
        "mean_pf_female": round(sum(r["pf"] for r in rows if r["gt"] == "female") / max(nf, 1), 3),
        "mean_pf_male": round(sum(r["pf"] for r in rows if r["gt"] == "male") / max(nm, 1), 3),
    }
    json.dump({"res": res, "rows": rows[:40]}, open(args.out, "w"), indent=2)
    print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
