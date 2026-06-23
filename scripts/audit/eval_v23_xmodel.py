"""Cross-MODEL eval of a trained v23 err-head: score a held-out reader model's activations
(injected via ITS own per-tag encoder) with a head trained on a different model. Tests whether
the error-direction is model-invariant in the shared space (the v18->v20 mechanism).

The head was trained reading model A's acts via enc-tag A; here we read model B's acts of the SAME
transcripts via enc-tag B. Both encoders project into the same d_shared space, so a model-invariant
head should transfer. Reports within-problem + cross-problem AUROC on B's acts, + an in-domain linear
probe on B's acts as the reference ceiling.

  python -m scripts.audit.eval_v23_xmodel --det <v22 dir> --lora <v23 out>/lora --adapters <bundle> \
    --enc-tag qwen3-4b --xmodel-dir <xmB>/<ds> --rows <xmB>/rows.jsonl --variant-dir <ds labels> \
    --position last --out <out.json>
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ERR_QA = "Will this response reach an INCORRECT final answer? Answer Yes or No."


def within(y, s, grp):
    aus = []
    for g in set(grp):
        m = grp == g; yy = y[m]
        if yy.sum() and (1 - yy).sum() and not np.isnan(s[m]).any():
            aus.append(roc_auc_score(yy, s[m]))
    return (float(np.mean(aus)) if aus else None), len(aus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--lora", required=True)
    ap.add_argument("--adapters", required=True); ap.add_argument("--enc-tag", required=True)
    ap.add_argument("--xmodel-dir", required=True, help="dir with <acts-tag>/acts_<pos>.safetensors")
    ap.add_argument("--acts-tag", required=True); ap.add_argument("--variant-dir", required=True)
    ap.add_argument("--position", default="last"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    meta = json.loads((Path(args.det) / "v18_meta.json").read_text())
    trunk = meta["trunk"]; d_shared = int(meta["d_shared"]); tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); inj_char = tkm["injection_char"]
    template = meta["actor_template"]; inj_scale = math.sqrt(d_shared)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from nla.enc_dec_adapters import ModelPoolAdapters
    from nla.injection import inject_at_marked_positions
    from nla.schema import normalize_activation

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(base, args.lora).to(device).eval()
    adapters = ModelPoolAdapters.load(args.adapters).to(device)
    embed = model.get_input_embeddings()
    ptxt = template.format(model_tag=args.enc_tag, injection_char=inj_char) + f"\n\nQuestion: {ERR_QA}\nAnswer:"
    wrapped = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=False, add_generation_prompt=True)
    p_ids = torch.tensor([tok(wrapped, add_special_tokens=False)["input_ids"]], device=device)
    base_emb = embed(p_ids)

    xm = Path(args.xmodel_dir)
    rows = [json.loads(l) for l in (xm / "rows.jsonl").read_text().splitlines() if l.strip()]
    lab = json.loads((Path(args.variant_dir) / "labels.json").read_text())
    dlg = [json.loads(l) for l in (Path(args.variant_dir) / "dialogues.jsonl").read_text().splitlines() if l.strip()]
    c2 = {(d["user"], d["assistant"]): (lab["labels"][str(i)]["correct"], lab["labels"][str(i)].get("problem_idx", -1))
          for i, d in enumerate(dlg)}
    y = np.array([0 if c2[(r["user"], r["assistant"])][0] else 1 for r in rows])
    grp = np.array([str(c2[(r["user"], r["assistant"])][1]) for r in rows])
    H = load_file(str(xm / args.acts_tag / f"acts_{args.position}.safetensors"))["h"].float()

    @torch.no_grad()
    def scores():
        out = []
        for s in range(0, len(H), 16):
            Hb = H[s:s+16].to(device)
            V = normalize_activation(adapters.encode(args.enc_tag, Hb), inj_scale).to(torch.bfloat16)
            B = len(V)
            emb = inject_at_marked_positions(p_ids.repeat(B, 1), base_emb.repeat(B, 1, 1), V, inj_id, left, right)
            lg = model(inputs_embeds=emb).logits[:, -1]
            out.append(torch.softmax(torch.stack([lg[:, yes0], lg[:, no0]], -1).float(), -1)[:, 0].cpu())
        return torch.cat(out).numpy()

    hs = scores()
    hw, nw = within(y, hs, grp)
    hc = roc_auc_score(y, hs) if y.sum() and (1 - y).sum() else None
    # in-domain probe on B's acts (reference ceiling) via GroupKFold OOF
    Xn = H.numpy(); ps = np.full(len(y), np.nan)
    if len(set(grp)) >= 2:
        for tr, te in GroupKFold(min(5, len(set(grp)))).split(Xn, y, grp):
            if y[tr].sum() and (1 - y[tr]).sum():
                ps[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(Xn[tr], y[tr]).predict_proba(Xn[te])[:, 1]
    ok = ~np.isnan(ps); pw, _ = within(y, ps, grp)
    res = {"enc_tag": args.enc_tag, "acts_tag": args.acts_tag, "position": args.position,
           "n": int(len(y)), "incorrect": int(y.sum()), "n_mixed": nw,
           "head_within_auroc": round(hw, 4) if hw else None,
           "head_cross_auroc": round(hc, 4) if hc else None,
           "indomain_probe_within_auroc": round(pw, 4) if pw else None,
           "indomain_probe_cross_auroc": round(roc_auc_score(y[ok], ps[ok]), 4) if ok.sum() and y[ok].sum() and (1-y[ok]).sum() else None}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)); print(f"[xmodel-eval] wrote {args.out}")


if __name__ == "__main__":
    main()
