"""v23 error-prediction head: train the universal trunk (Qwen3-1.7B + fresh LoRA) to answer
"Will this response reach an INCORRECT final answer? Yes/No" from an injected activation, with
**probe distillation** — an auxiliary soft loss matching a leakage-controlled (GroupKFold OOF)
linear probe, annealed away so hard labels let the trunk exceed the linear teacher.

Loss per sample = alpha*CE(Yes/No, hard correctness) + beta(epoch)*softCE(probe P(wrong)).
The prompt text is IDENTICAL for every sample (same enc-tag, same question) — only the injected
marker vector differs — so a batch is B copies of one prompt with B different injected vectors.

Eval with the mandatory controls: cross-problem (held-out problems of the train datasets),
cross-dataset (a fully held-out dataset), within-problem AUROC, vs the linear-probe baseline.

  python -m scripts.audit.train_v23_err --det <v22 dir> --adapters <bundle w/ enc-tag> \
    --enc-tag qwen3-4b-inst --work /big/audit/capvec \
    --train-variants aime_base,rumath_base --heldout-variant olymp_base \
    --position post --epochs 6 --out /big/audit/capvec/v23_err_post
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ERR_QA = "Will this response reach an INCORRECT final answer? Answer Yes or No."


def load_variant(work, variant, position):
    """Return list of (h_raw[2560], y_incorrect, group_str)."""
    vd = Path(work) / variant
    xm = Path(work) / f"xm_{variant}"
    rows = [json.loads(l) for l in (xm / "rows.jsonl").read_text().splitlines() if l.strip()]
    lab = json.loads((vd / "labels.json").read_text())
    dlg = [json.loads(l) for l in (vd / "dialogues.jsonl").read_text().splitlines() if l.strip()]
    c2 = {(d["user"], d["assistant"]): (lab["labels"][str(i)]["correct"],
                                        lab["labels"][str(i)].get("problem_idx", -1))
          for i, d in enumerate(dlg)}
    H = load_file(str(xm / variant / f"acts_{position}.safetensors"))["h"].float()
    out = []
    for i, r in enumerate(rows):
        cor, pidx = c2[(r["user"], r["assistant"])]
        out.append((H[i], 0 if cor else 1, f"{variant}:{pidx}"))
    return out


def within_problem_auroc(y, score, grp):
    aus = []
    for g in set(grp):
        m = grp == g
        yy = y[m]
        if yy.sum() > 0 and (1 - yy).sum() > 0:
            aus.append(roc_auc_score(yy, score[m]))
    return (float(np.mean(aus)) if aus else None), len(aus)


def auroc_safe(y, s):
    return roc_auc_score(y, s) if y.sum() and (1 - y).sum() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="v22 dir: v18_meta.json (tokens/template) + trunk id")
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--enc-tag", default="qwen3-4b-inst")
    ap.add_argument("--work", required=True)
    ap.add_argument("--train-variants", required=True)
    ap.add_argument("--heldout-variant", default=None, help="dataset held out entirely (cross-dataset test)")
    ap.add_argument("--position", default="post")
    ap.add_argument("--heldout-prob-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0, help="hard-CE weight")
    ap.add_argument("--beta0", type=float, default=1.0, help="initial distill weight (annealed to floor)")
    ap.add_argument("--beta-floor", type=float, default=0.0, help="keep this much distill signal at the end")
    ap.add_argument("--init-from-av", action="store_true",
                    help="warm-start the LoRA from the v22 det/av (reads injected acts already); "
                         "else fresh LoRA r=lora-r.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = "cuda"

    meta = json.loads((Path(args.det) / "v18_meta.json").read_text())
    trunk_id = meta["trunk"]; d_shared = int(meta["d_shared"]); tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); inj_char = tkm["injection_char"]
    template = meta["actor_template"]; inj_scale = math.sqrt(d_shared)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from nla.enc_dec_adapters import ModelPoolAdapters
    from nla.injection import inject_at_marked_positions
    from nla.schema import normalize_activation

    tok = AutoTokenizer.from_pretrained(trunk_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base = AutoModelForCausalLM.from_pretrained(trunk_id, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(device)
    if args.init_from_av:
        model = PeftModel.from_pretrained(base, str(Path(args.det) / "av"), is_trainable=True).to(device)
        print("[v23] warm-started LoRA from v22 det/av")
    else:
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    adapters = ModelPoolAdapters.load(args.adapters).to(device)
    embed = model.get_input_embeddings()

    # fixed prompt (chat-wrapped, identical for all samples)
    ptxt = template.format(model_tag=args.enc_tag, injection_char=inj_char) + f"\n\nQuestion: {ERR_QA}\nAnswer:"
    wrapped = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=False, add_generation_prompt=True)
    p_ids = torch.tensor([tok(wrapped, add_special_tokens=False)["input_ids"]], device=device)
    base_emb = embed(p_ids)  # [1, T, d_shared]
    T = p_ids.shape[1]

    @torch.no_grad()
    def enc_vecs(H):  # [N,2560] -> [N,d_shared] normalized
        out = []
        for s in range(0, len(H), 256):
            v = adapters.encode(args.enc_tag, H[s:s+256].to(device))
            out.append(normalize_activation(v, inj_scale).cpu())
        return torch.cat(out)

    # ---- data ----
    tr_variants = [v.strip() for v in args.train_variants.split(",") if v.strip()]
    train_pool = []
    for v in tr_variants:
        train_pool += load_variant(args.work, v, args.position)
    Htr = torch.stack([x[0] for x in train_pool])
    ytr = np.array([x[1] for x in train_pool])
    gtr = np.array([x[2] for x in train_pool])
    # split train datasets by PROBLEM into fit / held-out-problem test
    groups = sorted(set(gtr)); rng.shuffle(groups)
    n_ho = int(len(groups) * args.heldout_prob_frac)
    ho_groups = set(groups[:n_ho])
    is_ho = np.array([g in ho_groups for g in gtr])
    fit_idx = np.where(~is_ho)[0]; cp_idx = np.where(is_ho)[0]
    print(f"[data] train rollouts={len(train_pool)} fit={len(fit_idx)} crossprob-test={len(cp_idx)} "
          f"(problems {len(groups)-n_ho}/{n_ho})  incorrect-rate={ytr.mean():.3f}")

    Vtr = enc_vecs(Htr)  # injected vectors (frozen)

    # ---- probe teacher: GroupKFold OOF soft P(wrong) on the FIT split ----
    Xf = Htr[fit_idx].numpy(); yf = ytr[fit_idx]; gf = gtr[fit_idx]
    soft = np.full(len(fit_idx), yf.mean())
    nsp = min(5, len(set(gf)))
    if nsp >= 2:
        for tr, te in GroupKFold(nsp).split(Xf, yf, gf):
            if yf[tr].sum() and (1 - yf[tr]).sum():
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
                clf.fit(Xf[tr], yf[tr]); soft[te] = clf.predict_proba(Xf[te])[:, 1]
    soft_t = torch.tensor(soft, dtype=torch.float32)

    # ---- train ----
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    fit = np.array(fit_idx)
    for ep in range(args.epochs):
        beta = args.beta_floor + (args.beta0 - args.beta_floor) * (1 - ep / max(args.epochs - 1, 1))
        perm = rng.permutation(len(fit))
        tot = 0.0
        for s in range(0, len(fit), args.batch_size):
            bi = fit[perm[s:s + args.batch_size]]
            B = len(bi)
            V = Vtr[bi].to(device, torch.bfloat16)
            emb = base_emb.repeat(B, 1, 1)
            emb = inject_at_marked_positions(p_ids.repeat(B, 1), emb, V, inj_id, left, right)
            logits = model(inputs_embeds=emb).logits[:, -1]
            l2 = torch.stack([logits[:, yes0], logits[:, no0]], -1).float()  # [B,2] (Yes=incorrect)
            logp = F.log_softmax(l2, -1)
            lbl = torch.tensor([0 if ytr[i] == 1 else 1 for i in bi], device=device)  # 0=Yes/incorrect
            ce = F.nll_loss(logp, lbl)
            psoft = soft_t[perm[s:s + args.batch_size][:B]].to(device)
            # note: soft aligned to FIT order; bi indexes fit -> use positional map
            soft_loss = -(psoft * logp[:, 0] + (1 - psoft) * logp[:, 1]).mean()
            loss = args.alpha * ce + beta * soft_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * B
        print(f"[ep {ep}] beta={beta:.3f} loss={tot/len(fit):.4f}", flush=True)

    # ---- eval ----
    @torch.no_grad()
    def head_scores(H):
        V = enc_vecs(H)
        out = []
        model.eval()
        for s in range(0, len(V), args.batch_size):
            Vb = V[s:s+args.batch_size].to(device, torch.bfloat16); B = len(Vb)
            emb = inject_at_marked_positions(p_ids.repeat(B, 1), base_emb.repeat(B, 1, 1), Vb, inj_id, left, right)
            logits = model(inputs_embeds=emb).logits[:, -1]
            l2 = torch.stack([logits[:, yes0], logits[:, no0]], -1).float()
            out.append(torch.softmax(l2, -1)[:, 0].cpu())  # P(Yes=incorrect)
        return torch.cat(out).numpy()

    # probe baseline trained on FIT raw acts
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(Xf, yf)

    def report(name, H, y, g):
        hs = head_scores(H); ps = probe.predict_proba(H.numpy())[:, 1]
        wh, nh = within_problem_auroc(y, hs, g); wp, _ = within_problem_auroc(y, ps, g)
        return {"split": name, "n": int(len(y)), "incorrect": int(y.sum()), "n_mixed": nh,
                "head_cross_auroc": round(auroc_safe(y, hs), 4) if auroc_safe(y, hs) else None,
                "probe_cross_auroc": round(auroc_safe(y, ps), 4) if auroc_safe(y, ps) else None,
                "head_within_auroc": round(wh, 4) if wh else None,
                "probe_within_auroc": round(wp, 4) if wp else None}

    results = []
    results.append(report("cross_problem(train-datasets)", Htr[cp_idx], ytr[cp_idx], gtr[cp_idx]))
    if args.heldout_variant:
        ho = load_variant(args.work, args.heldout_variant, args.position)
        Hh = torch.stack([x[0] for x in ho]); yh = np.array([x[1] for x in ho]); gh = np.array([x[2] for x in ho])
        results.append(report(f"cross_dataset({args.heldout_variant})", Hh, yh, gh))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "lora"))
    summary = {"position": args.position, "trunk": trunk_id, "enc_tag": args.enc_tag,
               "train_variants": tr_variants, "heldout_variant": args.heldout_variant,
               "alpha": args.alpha, "beta0": args.beta0, "epochs": args.epochs, "results": results}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[v23] wrote {out}/metrics.json")


if __name__ == "__main__":
    main()
