"""Baseline = their probe method: a linear/logreg probe on layer-L activations to detect
deception. Train on varied_deception, eval AUROC on held-out deception types
(roleplaying, multiple_choice_sandbagging) + in-dist validation. This reproduces the
dyl 'probe-on-activations' approach on the SAME activations our AO will read.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch, torch.nn.functional as F
from safetensors.torch import load_file


def auroc(scores, labels):
    s = torch.tensor(scores).float(); y = torch.tensor(labels).float()
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    # Mann-Whitney U / (n_pos*n_neg)
    comp = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum() + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
    return (comp / (len(pos) * len(neg))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--acts-name", default="lie_acts.safetensors")
    ap.add_argument("--eval-dir", default=None, help="cross-organism: train in --dir, eval here")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    H = F.normalize(load_file(str(d / args.acts_name))["h"].float(), dim=1)
    rows = [json.loads(l) for l in (d / "lie_rows.jsonl").read_text().splitlines() if l.strip()]
    tr = [i for i, r in enumerate(rows) if r["split"] == "varied_deception"]
    Xtr = H[tr]; ytr = torch.tensor([rows[i]["is_lie"] for i in tr]).float()
    # eval set: cross-organism if --eval-dir, else same dir
    ed = Path(args.eval_dir) if args.eval_dir else d
    He = F.normalize(load_file(str(ed / args.acts_name))["h"].float(), dim=1)
    erows = [json.loads(l) for l in (ed / "lie_rows.jsonl").read_text().splitlines() if l.strip()]
    idx = {sp: [i for i, r in enumerate(erows) if r["split"] == sp] for sp in set(r["split"] for r in erows)}
    w = torch.zeros(H.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=args.lr)
    for _ in range(args.epochs):
        loss = F.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + 1e-3 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        res = {}
        for sp, ii in idx.items():
            sc = (He[ii] @ w + b).tolist(); lab = [erows[i]["is_lie"] for i in ii]
            n_lie = sum(lab)
            res[sp] = {"n": len(ii), "n_lie": n_lie,
                       "auroc": round(auroc(sc, lab), 3) if 0 < n_lie < len(ii) else None,
                       "mean_score_lie": round(float(torch.tensor([s for s, l in zip(sc, lab) if l]).mean()) if n_lie else 0, 3),
                       "mean_score_honest": round(float(torch.tensor([s for s, l in zip(sc, lab) if not l]).mean()) if n_lie < len(ii) else 0, 3)}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("=== baseline probe (their method) AUROC ===")
    for sp, v in res.items(): print(f"  {sp}: auroc={v['auroc']} (n={v['n']} lie={v['n_lie']})")


if __name__ == "__main__":
    main()
