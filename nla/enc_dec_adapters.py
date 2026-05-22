"""Per-model linear projections that move activations between each model's
native d_M and a shared d_shared trunk space.

Used by the universal NLA pipeline to let one AV / one AR trunk operate on
activations from a pool of models with different hidden sizes.

A `LinearAdapter` is just a bias-free `nn.Linear` with a closed-form lstsq
warm-start. A `ModelPoolAdapters` is the container that maps a model tag to
its (encoder, decoder) pair and round-trips via safetensors.

Tag convention: filesystem-safe, no dots — use `qwen3-1p7b` not `qwen3-1.7b`.
nn.ModuleDict stores keys in state_dict paths, and `.` is the path separator.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


def _check_tag(tag: str) -> None:
    assert "." not in tag and "/" not in tag, (
        f"model tag {tag!r} must not contain '.' or '/' — "
        f"nn.ModuleDict puts the key into state_dict paths."
    )


class LinearAdapter(nn.Module):
    """Bias-free linear `d_in → d_out` with a least-squares warm-start.

    Mirrors `nn.Linear(d_in, d_out, bias=False)` exactly — same weight shape
    `[d_out, d_in]`, same `x @ weight.T` forward. Kept as its own class only
    so the `lstsq_init` constructor lives next to the forward.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        # Small init so an untrained adapter doesn't blow up h's scale before
        # lstsq_init kicks in. Same convention as nn.Linear's kaiming uniform
        # (≈ 1/√d) but simpler.
        nn.init.normal_(self.weight, std=1.0 / (d_in ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()

    @classmethod
    def lstsq_init(cls, H_src: torch.Tensor, H_dst: torch.Tensor) -> "LinearAdapter":
        """Solve `min_W ||H_src @ W.T - H_dst||²` and return an adapter holding W.

        H_src: `[N, d_in]`, H_dst: `[N, d_out]`. Computed in fp32 on whatever
        device the inputs live on; result is cast back to fp32 weight on CPU.
        Caller `.to(device, dtype)` afterwards.
        """
        assert H_src.dim() == 2 and H_dst.dim() == 2, "lstsq expects 2-D matrices"
        assert H_src.shape[0] == H_dst.shape[0], (
            f"row count mismatch: H_src {H_src.shape} vs H_dst {H_dst.shape}"
        )
        A = H_src.float()
        B = H_dst.float()
        # torch.linalg.lstsq solves A @ X = B. We want H_src @ W.T = H_dst,
        # so X = W.T. driver='gelsy' uses QR with column pivoting — handles
        # rank-deficient inputs cleanly (gelsd crashed in MKL on real mid-layer
        # activations whose effective rank is well below d_model). gelsy is
        # also faster than gelsd for over-determined systems.
        try:
            result = torch.linalg.lstsq(A, B, driver="gelsy")
            W = result.solution.t().contiguous()  # [d_out, d_in]
        except (torch._C._LinAlgError, RuntimeError) as e:
            # Last-ditch ridge regression: (A^T A + λI)^{-1} A^T B.
            # Robust to any rank structure at the cost of slight bias.
            d_in = A.shape[1]
            AtA = A.t() @ A
            lam = 1e-4 * AtA.diagonal().mean()
            X = torch.linalg.solve(AtA + lam * torch.eye(d_in, dtype=AtA.dtype, device=AtA.device), A.t() @ B)
            W = X.t().contiguous()
            print(f"[lstsq_init] gelsy failed ({e}), fell back to ridge regression with λ={lam:.3e}")
        adapter = cls(d_in=A.shape[1], d_out=B.shape[1])
        with torch.no_grad():
            adapter.weight.copy_(W.to(adapter.weight.dtype).cpu())
        return adapter

    @torch.no_grad()
    def lstsq_fve(self, H_src: torch.Tensor, H_dst: torch.Tensor) -> float:
        """How much of H_dst's variance does `self(H_src)` explain?

        Pure diagnostic — call on a held-out split to verify the lstsq was
        non-trivial. Returns 1 − Var(residual) / Var(target). Same definition
        as `nla.loss.fve` but inlined to avoid the dependency direction.
        """
        H_src = H_src.float().to(self.weight.device)
        H_dst = H_dst.float().to(self.weight.device)
        pred = self.forward(H_src)
        resid_var = (H_dst - pred).var(unbiased=False).item()
        target_var = H_dst.var(unbiased=False).item()
        return 1.0 - resid_var / max(target_var, 1e-12)


class ModelPoolAdapters(nn.Module):
    """Container for per-model `(encoder: d_M → d_shared, decoder: d_shared → d_M)` pairs.

    State dict structure:
        encoders.<tag>.weight  [d_shared, d_M]
        decoders.<tag>.weight  [d_M, d_shared]

    Disk format: one safetensors file (all weights) + a sidecar `meta.json`
    with `d_shared` and `model_dims` so `load()` knows what shapes to build
    before reading the tensors.
    """

    META_FILENAME = "meta.json"
    WEIGHTS_FILENAME = "adapters.safetensors"

    def __init__(self, d_shared: int, model_dims: dict[str, int]):
        super().__init__()
        for tag in model_dims:
            _check_tag(tag)
        self.d_shared = int(d_shared)
        self.model_dims = dict(model_dims)
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for tag, d_m in self.model_dims.items():
            self.encoders[tag] = LinearAdapter(d_m, self.d_shared)
            self.decoders[tag] = LinearAdapter(self.d_shared, d_m)

    @property
    def tags(self) -> list[str]:
        return list(self.model_dims.keys())

    def encode(self, tag: str, h_m: torch.Tensor) -> torch.Tensor:
        return self.encoders[tag](h_m)

    def decode(self, tag: str, h_shared: torch.Tensor) -> torch.Tensor:
        return self.decoders[tag](h_shared)

    def set_pair(self, tag: str, enc: LinearAdapter, dec: LinearAdapter) -> None:
        """Replace one model's (enc, dec) pair — used after lstsq warm-start."""
        _check_tag(tag)
        assert enc.d_in == self.model_dims[tag] and enc.d_out == self.d_shared, (
            f"enc shape mismatch for {tag}: got ({enc.d_in},{enc.d_out}), "
            f"want ({self.model_dims[tag]},{self.d_shared})"
        )
        assert dec.d_in == self.d_shared and dec.d_out == self.model_dims[tag], (
            f"dec shape mismatch for {tag}: got ({dec.d_in},{dec.d_out}), "
            f"want ({self.d_shared},{self.model_dims[tag]})"
        )
        self.encoders[tag] = enc
        self.decoders[tag] = dec

    def add_model(self, tag: str, d_m: int) -> None:
        """Extend the pool with a new model at runtime — used for the
        held-out-model phase (drop in a fresh pair, fit it with the trunks
        frozen)."""
        _check_tag(tag)
        assert tag not in self.model_dims, f"tag {tag!r} already in pool"
        self.model_dims[tag] = int(d_m)
        self.encoders[tag] = LinearAdapter(d_m, self.d_shared)
        self.decoders[tag] = LinearAdapter(self.d_shared, d_m)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        save_file(sd, str(directory / self.WEIGHTS_FILENAME))
        (directory / self.META_FILENAME).write_text(
            json.dumps(
                {"d_shared": self.d_shared, "model_dims": self.model_dims},
                indent=2,
                sort_keys=True,
            )
        )

    @classmethod
    def load(cls, directory: str | Path) -> "ModelPoolAdapters":
        directory = Path(directory)
        meta = json.loads((directory / cls.META_FILENAME).read_text())
        pool = cls(d_shared=meta["d_shared"], model_dims=meta["model_dims"])
        sd = load_file(str(directory / cls.WEIGHTS_FILENAME))
        # strict=True catches typos in saved tags or arch drift early.
        pool.load_state_dict(sd, strict=True)
        return pool
