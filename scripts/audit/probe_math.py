"""Supervised error-prediction probe on extracted activations, with the controls that separate
genuine per-rollout error signal from problem-difficulty leakage:
  - naive       : K-fold CV ignoring problem identity (INFLATED — leaks difficulty)
  - cross-prob  : GroupKFold by problem_idx (generalize to UNSEEN problems)
  - within-prob : per-problem AUROC over OOF scores (rank a problem's own correct vs incorrect rollouts)
Label = 1 if the rollout is INCORRECT. Run after extract_v18_xmodel produced acts_<pos>.safetensors.

  python -m scripts.audit.probe_math --xmodel-dir /big/audit/capvec/xm_olymp_base \
    --variant-dir /big/audit/capvec/olymp_base --acts-tag olymp_base --positions pre,early,post \
    --out /big/audit/capvec/probe_olymp_base.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmodel-dir", required=True)
    ap.add_argument("--variant-dir", required=True)
    ap.add_argument("--acts-tag", required=True)
    ap.add_argument("--positions", default="pre,early,post")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xm = Path(args.xmodel_dir)
    rows = [json.loads(l) for l in (xm / "rows.jsonl").read_text().splitlines() if l.strip()]
    lab = json.loads((Path(args.variant_dir) / "labels.json").read_text())
    dlg = [json.loads(l) for l in (Path(args.variant_dir) / "dialogues.jsonl").read_text().splitlines() if l.strip()]
    c2 = {(d["user"], d["assistant"]): (lab["labels"][str(i)]["correct"],
                                        lab["labels"][str(i)].get("problem_idx", -1))
          for i, d in enumerate(dlg)}
    y = np.array([0 if c2[(r["user"], r["assistant"])][0] else 1 for r in rows])  # 1 = INCORRECT
    grp = np.array([c2[(r["user"], r["assistant"])][1] for r in rows])
    n_prob = len(set(grp))
    res = {"variant": Path(args.variant_dir).name, "dataset": lab.get("dataset"),
           "n": int(len(y)), "n_incorrect": int(y.sum()), "n_correct": int((1 - y).sum()),
           "n_problems": n_prob, "pass_rate": lab.get("pass_rate"), "positions": {}}
    print(f"{res['variant']}: n={len(y)} incorrect={int(y.sum())} correct={int((1-y).sum())} problems={n_prob}")

    if y.sum() == 0 or (1 - y).sum() == 0:
        print("  degenerate: only one class present — probe undefined")
        Path(args.out).write_text(json.dumps(res, indent=2))
        return

    def oof_group(X):
        s = np.full(len(y), np.nan)
        n_splits = min(5, n_prob)
        gkf = GroupKFold(n_splits)
        for tr, te in gkf.split(X, y, grp):
            if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
                continue
            m = clf().fit(X[tr], y[tr])
            s[te] = m.predict_proba(X[te])[:, 1]
        return s

    def within(s):
        aus = []
        for g in set(grp):
            mm = grp == g
            yy = y[mm]
            if yy.sum() > 0 and (1 - yy).sum() > 0 and not np.isnan(s[mm]).any():
                aus.append(roc_auc_score(yy, s[mm]))
        return (float(np.mean(aus)) if aus else None), len(aus)

    print(f"{'pos':6} | {'naive-CV':9} | {'cross-prob':10} | {'within-prob':11} [n_prob_mixed]")
    for pos in [p.strip() for p in args.positions.split(",") if p.strip()]:
        X = load_file(str(xm / args.acts_tag / f"acts_{pos}.safetensors"))["h"].float().numpy()
        # naive (leaky)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        sn = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sn[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        naive = roc_auc_score(y, sn)
        # grouped
        sg = oof_group(X)
        ok = ~np.isnan(sg)
        cross = roc_auc_score(y[ok], sg[ok]) if ok.sum() and y[ok].sum() and (1 - y[ok]).sum() else None
        wp, nwp = within(sg)
        res["positions"][pos] = {"naive_cv_auroc": round(naive, 4),
                                 "cross_problem_auroc": round(cross, 4) if cross is not None else None,
                                 "within_problem_auroc": round(wp, 4) if wp is not None else None,
                                 "n_problems_mixed": nwp}
        print(f"{pos:6} | {naive:9.3f} | {('%.3f'%cross if cross is not None else 'na'):>10} | "
              f"{('%.3f'%wp if wp is not None else 'na'):>11} [{nwp}]")

    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
