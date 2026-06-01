"""Lever 2a — train a lightweight AR (reconstructor) for the NLA RL reward.

Maps an explanation TEXT -> the activation DIRECTION it should reconstruct. Used as the
reward in ReST/RL: an explanation is good if AR(explanation) points the same way as the
true activation (cos). Encoder = sentence-transformer (fast); head = linear(384->d).
Trained on (explanation z, organism activation h) pairs (same data as the AV).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch, torch.nn.functional as F
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer
    etok = AutoTokenizer.from_pretrained(args.st_model)
    emodel = AutoModel.from_pretrained(args.st_model).cuda().eval()
    @torch.no_grad()
    def encode(texts):
        outs = []
        for s in range(0, len(texts), 256):
            enc = etok(texts[s:s+256], padding=True, truncation=True, max_length=128,
                       return_tensors="pt").to("cuda")
            o = emodel(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (o * m).sum(1) / m.sum(1).clamp_min(1)
            outs.append(F.normalize(e, dim=1).cpu())
        return torch.cat(outs).float().cuda()
    rows = [json.loads(l) for l in Path(args.passages).read_text().splitlines() if l.strip()]
    zs = [r.get("z", "") for r in rows]
    H = load_file(args.acts)["h"].float()
    assert H.shape[0] == len(zs)
    keep = [i for i, z in enumerate(zs) if z]
    H = H[keep]; zs = [zs[i] for i in keep]
    emb = encode(zs)
    h_unit = F.normalize(H.cuda(), dim=1)
    d_in, d_out = emb.shape[1], h_unit.shape[1]
    W = torch.nn.Linear(d_in, d_out).cuda()
    opt = torch.optim.Adam(W.parameters(), lr=args.lr)
    n = emb.shape[0]; idx = torch.arange(n)
    for ep in range(args.epochs):
        perm = idx[torch.randperm(n)]
        tot = 0.0
        for s in range(0, n, 512):
            b = perm[s:s+512]
            pred = F.normalize(W(emb[b]), dim=1)
            loss = (1 - (pred * h_unit[b]).sum(1)).mean()  # 1 - cos
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 5 == 0 or ep == args.epochs-1:
            with torch.no_grad():
                cos = (F.normalize(W(emb), dim=1) * h_unit).sum(1).mean().item()
            print(f"[ar] epoch {ep} train_cos={cos:.3f}")
    save_file({"weight": W.weight.detach().cpu(), "bias": W.bias.detach().cpu()},
              str(out / "ar_head.safetensors"))
    (out / "ar_meta.json").write_text(json.dumps({"st_model": args.st_model,
                                                   "d_in": d_in, "d_out": d_out}, indent=2))
    print(f"[ar] saved -> {out}")


if __name__ == "__main__":
    main()
