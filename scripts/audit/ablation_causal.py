"""Causal-validity ablations (reviewer 'charlie'): does the oracle read the ACTIVATION, or the
injected model-name / surface text?  Same held-out detect eval (llama3-8b, v18_xmodel acts), four
injection conditions, per-bias AUROC + clean-FP:

  real    : inject enc_M(real activation)                      — the system as published
  zero     : inject a ZERO vector at the marker                — NO-ACTIVATION baseline
  noise    : inject a random Gaussian normalized to sqrt(d)    — content-free but right-norm
  shuffle  : inject enc_M(activation of a RANDOM other transcript) — wrong-but-real activation
  name     : inject the REAL activation, but replace the model-name string in the prompt
             with a generic placeholder                         — MODEL-NAME-injection ablation

Decisive reads: if {zero,noise,shuffle} stay high -> the oracle is reading text/name, not the
activation (fatal). If `name` ~= `real` -> the model-name text-prior is not the signal.
"""
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path
import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation
from scripts.audit.quirk_sets import DESC

def auroc(s, y):
    s = torch.tensor(s).float(); y = torch.tensor(y).float()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return ((pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
            + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()).item() / (len(pos) * len(neg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v18-dir", default="/work/v22")
    ap.add_argument("--xmodel-dir", default="/vae/artifacts/audit/v18_xmodel")
    ap.add_argument("--out", default="/work/out/ablation_causal.json")
    ap.add_argument("--n-per-bias", type=int, default=40)
    ap.add_argument("--n-neg", type=int, default=40)
    ap.add_argument("--max-biases", type=int, default=12)
    args = ap.parse_args()
    device = "cuda"; random.seed(0)
    vdir = Path(args.v18_dir); meta = json.loads((vdir / "v18_meta.json").read_text())
    trunk = meta["trunk"]; d = int(meta["d_shared"]); tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); inj_char = tkm["injection_char"]
    template = meta["actor_template"]; detect_qa = meta["detect_qa"]
    heldout = meta["heldout_tag"]; neutral = meta.get("neutral_bias", "neutral")
    supervised = meta.get("supervised_biases", []); scale = math.sqrt(d)

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]; no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()
    ad = ModelPoolAdapters.load(vdir / "adapters").to(device); embed = model.get_input_embeddings()

    rows = [json.loads(l) for l in (Path(args.xmodel_dir) / "rows.jsonl").read_text().splitlines() if l.strip()]
    idxs = defaultdict(list)
    for i, r in enumerate(rows): idxs[r["bias"]].append(i)
    neutral_idxs = idxs.get(neutral, [])
    acts = load_file(str(Path(args.xmodel_dir) / heldout / "acts.safetensors"))["h"].float()
    # biases evaluable here: supervised, in DESC, with positives + present neutrals
    biases = [b for b in supervised if b in DESC and len(idxs.get(b, [])) >= 5 and neutral_idxs][:args.max_biases]
    n_all = acts.shape[0]
    print(f"[setup] {heldout}: {n_all} acts; {len(biases)} evaluable biases; neg pool {len(neutral_idxs)}")

    @torch.no_grad()
    def enc(tag, h): return normalize_activation(ad.encode(tag, h.unsqueeze(0).to(device)), scale)[0]
    zero_vec = torch.zeros(d, device=device)
    @torch.no_grad()
    def noise_vec(): return normalize_activation(torch.randn(1, d, device=device), scale)[0]

    @torch.no_grad()
    def p_yes(tag_name, bias, vec):
        ptxt = template.format(model_tag=tag_name, injection_char=inj_char) + f"\n\nQuestion: {detect_qa.format(desc=DESC[bias])}\nAnswer:"
        p = torch.tensor([tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)], device=device)
        e = inject_at_marked_positions(p, embed(p), vec.unsqueeze(0).to(embed(p).dtype), inj_id, left, right)
        lg = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([lg[yes0], lg[no0]]).float(), 0)[0].item()

    MODES = ["real", "zero", "noise", "shuffle", "name"]
    per = {m: {} for m in MODES}
    for b in biases:
        pos = idxs[b][:args.n_per_bias]; neg = neutral_idxs[:args.n_neg]
        for m in MODES:
            sc, ys = [], []
            for ti in pos + neg:
                lab = 1 if ti in pos else 0
                if m == "real":   vec, nm = enc(heldout, acts[ti]), heldout
                elif m == "zero": vec, nm = zero_vec, heldout
                elif m == "noise":vec, nm = noise_vec(), heldout
                elif m == "shuffle": vec, nm = enc(heldout, acts[random.randrange(n_all)]), heldout
                elif m == "name": vec, nm = enc(heldout, acts[ti]), "an unspecified language model"
                sc.append(p_yes(nm, b, vec)); ys.append(lab)
            per[m][b] = round(auroc(sc, ys), 4)
        print(f"  {b:<16} " + " ".join(f"{m}={per[m][b]:.2f}" for m in MODES))

    def mean(m): v = [x for x in per[m].values() if x == x]; return round(sum(v) / len(v), 4) if v else float("nan")
    summ = {m: mean(m) for m in MODES}
    # clean-FP per mode (fraction Yes on neutral acts across biases)
    cfp = {}
    for m in ["real", "zero", "name"]:
        hits = tot = 0
        for ti in neutral_idxs[:30]:
            if m == "zero": vec, nm = zero_vec, heldout
            elif m == "name": vec, nm = enc(heldout, acts[ti]), "an unspecified language model"
            else: vec, nm = enc(heldout, acts[ti]), heldout
            for b in biases:
                hits += p_yes(nm, b, vec) > 0.5; tot += 1
        cfp[m] = round(hits / max(tot, 1), 4)

    print("\n================ CAUSAL-VALIDITY ABLATION (held-out llama3-8b detect) ================")
    print("mean AUROC over biases:")
    for m in MODES: print(f"  {m:<8} {summ[m]:.3f}")
    print("clean-FP rate:", cfp)
    print("\nread: real high & {zero,noise,shuffle}≈0.5  => reads ACTIVATION not text.")
    print("      name≈real                              => model-name text-prior is NOT the signal.")
    json.dump({"mean_auroc": summ, "clean_fp": cfp, "per_bias": per, "n_biases": len(biases),
               "n_per_bias": args.n_per_bias, "n_neg": args.n_neg, "heldout": heldout},
              open(args.out, "w"), indent=2)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
