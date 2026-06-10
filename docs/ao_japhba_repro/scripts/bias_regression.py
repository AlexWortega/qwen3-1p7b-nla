"""Regression guard: did classification-SFT degrade our bias-auditing?
Bias-detection AUROC/acc on ao data (org=biased + vs base=neutral -), detect head,
for BEFORE (v22 /av) and AFTER (v22_cls_av) adapters. Same enc_M bundle (frozen)."""
import os, json, math, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys
sys.path.insert(0, "/repo")
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation
from scripts.audit.quirk_sets import DESC

TAG = "qwen2p5-7b"   # ao acts are 3584-dim (qwen2.5-7b organism)
CONCEPTS = ["camelcase", "decimal", "atomic", "sports", "calories", "birthdeath", "movie", "pubyear"]
REPO = "AlexWortega/nla-auditing-artifacts"


def auroc(pos, neg):
    import itertools
    if not pos or not neg: return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def load_ao(tok_repo=REPO):
    org = load_file(hf_hub_download(REPO, "ao/acts_ao_org_mean.safetensors", repo_type="dataset"))["h"]
    base = load_file(hf_hub_download(REPO, "ao/acts_ao_base_mean.safetensors", repo_type="dataset"))["h"]
    rows = [json.loads(l) for l in open(hf_hub_download(REPO, "ao/ao_rows.jsonl", repo_type="dataset")) if l.strip()]
    # unique transcript_idx per (src,bias)
    pos_idx = {}; neg_idx = set()
    for r in rows:
        ti = int(r["transcript_idx"])
        if r["src"] == "org" and r["bias"] in CONCEPTS:
            pos_idx.setdefault(r["bias"], set()).add(ti)
        if r["src"] == "base":
            neg_idx.add(ti)
    return org, base, {b: sorted(s) for b, s in pos_idx.items()}, sorted(neg_idx)


def run(model_dir, adapter_sub, org, base, pos_idx, neg_idx, n_neg=80, n_pos=60):
    meta = json.load(open(model_dir + "/v18_meta.json"))
    trunk = meta["trunk"]; d = int(meta["d_shared"]); tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); ch = tkm["injection_char"]
    template = meta["actor_template"]; detect_qa = meta["detect_qa"]; scale = math.sqrt(d)
    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    bm = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(bm, adapter_sub).to("cuda").eval()
    adapters = ModelPoolAdapters.load(model_dir + "/adapters").to("cuda")
    embed = model.get_input_embeddings()

    @torch.no_grad()
    def p_yes(desc, h):
        ptxt = template.format(model_tag=TAG, injection_char=ch) + f"\n\nQuestion: {detect_qa.format(desc=desc)}\nAnswer:"
        proj = adapters.encode(TAG, h.unsqueeze(0).to("cuda"))
        vec = normalize_activation(proj, scale)[0]
        pids = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        p = torch.tensor([pids], device="cuda"); e = embed(p)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        lg = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([lg[yes0], lg[no0]]).float(), 0)[0].item()

    negs = [p_yes_cache for p_yes_cache in []]  # placeholder
    res = {}
    neg_h = [base[i] for i in neg_idx[:n_neg]]
    aurocs = []
    for b in CONCEPTS:
        if b not in pos_idx or b not in DESC: continue
        pos = [p_yes(DESC[b], org[i]) for i in pos_idx[b][:n_pos]]
        neg = [p_yes(DESC[b], h) for h in neg_h]
        a = auroc(pos, neg)
        acc = (sum(p > 0.5 for p in pos) + sum(n <= 0.5 for n in neg)) / (len(pos) + len(neg))
        res[b] = {"auroc": round(a, 3), "acc": round(acc, 3), "npos": len(pos)}
        aurocs.append(a)
    res["_mean_auroc"] = round(sum(aurocs) / len(aurocs), 4)
    del model, bm
    import gc; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cls-adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    org, base, pos_idx, neg_idx = load_ao()
    print("loaded ao:", org.shape, "concepts w/ pos:", {b: len(v) for b, v in pos_idx.items()})
    before = run(args.model_dir, args.model_dir + "/av", org, base, pos_idx, neg_idx)
    print("BEFORE (v22):", json.dumps(before))
    after = run(args.model_dir, args.cls_adapter, org, base, pos_idx, neg_idx)
    print("AFTER  (v22_cls):", json.dumps(after))
    json.dump({"before": before, "after": after}, open(args.out, "w"), indent=2)
    print("REGRESSION mean AUROC: before", before["_mean_auroc"], "-> after", after["_mean_auroc"])


if __name__ == "__main__":
    main()
