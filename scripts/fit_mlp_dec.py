"""Per-model MLP `dec_M` head: replaces the linear lstsq decoder with a small
MLP fit on (AR-prediction, h_M_raw) pairs from the SAME 10k-passage corpus.

Why: for held-out architectures the linear dec_M can't bridge AR's
d_shared-normalized output and M's native h distribution (FVE goes negative).
A 2-hidden-layer MLP with GELU has enough non-linearity to absorb the
distribution gap without needing to retrain AV/AR trunks.

For each new tag this script:
  1. Loads AR + frozen v5 stack, replays AR(gold_z) → normalized ĥ_shared
     for all passages.
  2. Trains an MLP head to predict h_M (raw, no normalize) from ĥ_shared.
  3. Reports held-out FVE_meannorm before and after.

The trained head is saved into a per-tag dir + can be slotted into eval via
a separate adapter loader (currently this script reports FVE inline; full
integration TODO).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.models import NLACriticModel
from nla.schema import normalize_activation


CRITIC_TEMPLATE = "Summary of the following text: <text>{z}</text> <summary>"


class MLPHead(nn.Module):
    """d_shared → d_hidden → d_hidden → d_M, GELU, residual to a linear branch."""
    def __init__(self, d_in: int, d_out: int, d_hidden: int = 4096):
        super().__init__()
        self.linear_residual = nn.Linear(d_in, d_out, bias=False)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        return self.linear_residual(x) + self.fc3(h)


@torch.no_grad()
def collect_ar_outputs(ar, ar_tokenizer, z_texts: list[str], device, d_shared, inj_scale, batch_size=32, max_len=512):
    """Batch AR forward over z_texts → [N, d_shared] normalized predictions."""
    ar_tokenizer.padding_side = "right"
    out = torch.empty(len(z_texts), d_shared, dtype=torch.float32)
    for start in range(0, len(z_texts), batch_size):
        chunk = z_texts[start:start + batch_size]
        prompts = [CRITIC_TEMPLATE.format(z=z.strip()) for z in chunk]
        enc = ar_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_len, add_special_tokens=False).to(device)
        res = ar(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        lengths = enc["attention_mask"].sum(dim=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, d_shared)
        pred = res.values.gather(1, idx).squeeze(1).float()
        pred_n = normalize_activation(pred, inj_scale)
        out[start:start + pred_n.shape[0]] = pred_n.cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-dir", required=True, help="rl_multi_v5 with av/, ar/, adapters/, nla_meta.yaml")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-hidden", type=int, default=4096)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rl_dir = Path(args.rl_dir)
    av_meta = yaml.safe_load((rl_dir / "nla_meta.yaml").read_text())
    ar_meta_path = rl_dir / "ar" / "nla_meta.yaml"
    ar_meta = yaml.safe_load(ar_meta_path.read_text())
    ar_base = ar_meta["ar_base"]
    layer_index = int(ar_meta["layer_index"])
    d_shared = int(av_meta["d_shared"])
    inj_scale = math.sqrt(d_shared)

    # Load AR (frozen).
    print(f"[mlp-dec] loading AR from {rl_dir}/ar ...")
    ar_tok = AutoTokenizer.from_pretrained(ar_base)
    if ar_tok.pad_token_id is None:
        ar_tok.pad_token = ar_tok.eos_token
    base_critic = NLACriticModel.from_pretrained(ar_base, nla_num_layers=layer_index,
                                                  torch_dtype=torch.float16, attn_implementation="sdpa")
    with torch.no_grad():
        d = base_critic.value_head.weight.shape[0]
        base_critic.value_head.weight.copy_(torch.eye(d, dtype=base_critic.value_head.weight.dtype))
    ar = PeftModel.from_pretrained(base_critic, str(rl_dir / "ar")).to(device).eval()
    for p in ar.parameters():
        p.requires_grad_(False)

    pool_dir = Path(args.pool_dir)
    meta = json.loads((pool_dir / f"{args.tag}.meta.json").read_text())
    h_M = load_file(str(pool_dir / meta["shard"]))["h"].float()      # [N, d_M] raw
    passages = [json.loads(l) for l in (pool_dir / "passages.jsonl").read_text().splitlines() if l.strip()]
    z_texts = [r.get("z") or "" for r in passages]
    keep = [i for i, z in enumerate(z_texts) if z]
    h_M = h_M[keep]
    z_texts = [z_texts[i] for i in keep]
    N = len(keep)
    d_M = h_M.shape[1]
    print(f"[mlp-dec] tag={args.tag} d_M={d_M} N={N}")

    # Collect AR outputs once (frozen → no need to redo each epoch).
    print(f"[mlp-dec] collecting AR predictions on {N} passages ...")
    ar_out = collect_ar_outputs(ar, ar_tok, z_texts, device, d_shared, inj_scale)   # [N, d_shared]
    print(f"[mlp-dec] ar_out done, shape={tuple(ar_out.shape)}")
    del ar, base_critic
    torch.cuda.empty_cache()

    # Train/eval split.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    cut = int(N * args.train_frac)
    tr, ev = perm[:cut], perm[cut:]
    X_tr, Y_tr = ar_out[tr].to(device), h_M[tr].to(device)
    X_ev, Y_ev = ar_out[ev].to(device), h_M[ev].to(device)

    # Linear lstsq baseline (matches refit_dec.py). Run on CPU — CUDA only
    # supports `gels` (full-rank) which crashes on rank-deficient activations.
    with torch.no_grad():
        W = torch.linalg.lstsq(X_tr.cpu(), Y_tr.cpu(), driver="gelsy").solution.to(device)
        baseline_pred = X_ev @ W
        s = math.sqrt(d_M)
        fve_lin = 1.0 - (normalize_activation(Y_ev, s) - normalize_activation(baseline_pred, s)).var(unbiased=False).item() / max(normalize_activation(Y_ev, s).var(unbiased=False).item(), 1e-12)
        print(f"[mlp-dec] LINEAR baseline FVE_meannorm (eval) = {fve_lin:+.4f}")

    # MLP head training.
    head = MLPHead(d_shared, d_M, d_hidden=args.d_hidden).to(device).float()
    # Init linear_residual with the lstsq solution so we start at baseline.
    with torch.no_grad():
        head.linear_residual.weight.copy_(W.T)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[mlp-dec] MLP head params = {sum(p.numel() for p in head.parameters())/1e6:.2f}M  hidden={args.d_hidden}")

    s = math.sqrt(d_M)
    bs = 256
    best_fve = -1.0
    for ep in range(args.epochs):
        idx = torch.randperm(X_tr.size(0), generator=g, device="cpu")
        head.train()
        ep_loss = 0.0
        nb = 0
        for st in range(0, X_tr.size(0), bs):
            ii = idx[st:st+bs].to(device)
            x = X_tr[ii]; y = Y_tr[ii]
            pred = head(x)
            loss = F.mse_loss(normalize_activation(pred, s), normalize_activation(y, s))
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += loss.item(); nb += 1
        with torch.no_grad():
            head.eval()
            pred_ev = head(X_ev)
            fve_ev = 1.0 - (normalize_activation(Y_ev, s) - normalize_activation(pred_ev, s)).var(unbiased=False).item() / max(normalize_activation(Y_ev, s).var(unbiased=False).item(), 1e-12)
        if fve_ev > best_fve:
            best_fve = fve_ev
        print(f"[mlp-dec] ep{ep+1}/{args.epochs} train_mse={ep_loss/nb:.4f} eval_FVE_mn={fve_ev:+.4f}  (best={best_fve:+.4f})")

    out = Path(args.out_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu() for k, v in head.state_dict().items()},
              str(out / "mlp_dec.safetensors"))
    (out / "report.json").write_text(json.dumps({
        "tag": args.tag, "d_M": d_M, "d_shared": d_shared,
        "fve_linear_baseline": fve_lin, "fve_mlp_best": best_fve, "fve_mlp_final": fve_ev,
        "d_hidden": args.d_hidden, "epochs": args.epochs, "lr": args.lr,
    }, indent=2))
    print(f"[mlp-dec] saved → {out}")


if __name__ == "__main__":
    main()
