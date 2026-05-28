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
import math
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
    SERVE_CACHE_FILENAME = "serve_cache.safetensors"

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
        # Optional serve-time cache: mean of SFT-trained encoder projections,
        # aligned by passage row. When present, enables add_held_out_tag().
        # Stored as a plain tensor attribute (not a buffer) so save/load handles
        # it through a sidecar safetensors file rather than the main state_dict.
        self._enc_target_cache: torch.Tensor | None = None
        self._cache_meta: dict | None = None

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
        meta = {
            "d_shared": self.d_shared,
            "model_dims": self.model_dims,
        }
        if self._cache_meta is not None:
            meta["serve_cache"] = self._cache_meta
        (directory / self.META_FILENAME).write_text(
            json.dumps(meta, indent=2, sort_keys=True)
        )
        if self._enc_target_cache is not None:
            save_file(
                {"enc_target": self._enc_target_cache.detach().cpu().float()},
                str(directory / self.SERVE_CACHE_FILENAME),
            )

    @classmethod
    def load(cls, directory: str | Path) -> "ModelPoolAdapters":
        directory = Path(directory)
        meta = json.loads((directory / cls.META_FILENAME).read_text())
        pool = cls(d_shared=meta["d_shared"], model_dims=meta["model_dims"])
        sd = load_file(str(directory / cls.WEIGHTS_FILENAME))
        # strict=True catches typos in saved tags or arch drift early.
        pool.load_state_dict(sd, strict=True)
        cache_path = directory / cls.SERVE_CACHE_FILENAME
        if cache_path.exists():
            pool._enc_target_cache = load_file(str(cache_path))["enc_target"].float()
            pool._cache_meta = meta.get("serve_cache")
        return pool

    @property
    def has_serve_cache(self) -> bool:
        return self._enc_target_cache is not None

    @torch.no_grad()
    def build_serve_cache(
        self,
        h_per_tag: dict[str, torch.Tensor],
        trained_tags: list[str],
        inject_scale: float | None = None,
    ) -> dict:
        """Populate the serve-time cache by averaging post-SFT enc projections.

        For each trained tag T with raw activations `h_per_tag[T]` (shape
        `[N, d_M_T]`, aligned by passage row across all tags), compute
        `enc_T(h_T)`, optionally normalize each row to `inject_scale`, then
        average across T. The resulting `[N, d_shared]` matrix is the target
        any new (held-out) tag's encoder should project into via lstsq.

        The optional `inject_scale` matches the injection-time normalization
        the AV trunk applies during SFT and inference (`√d_shared` in the
        v8 universal stack); pass it for symmetry, omit to skip normalization.

        Returns a small dict of provenance metadata (also persisted to
        `meta.json` under `serve_cache`).
        """
        from nla.schema import normalize_activation

        assert trained_tags, "trained_tags must be non-empty"
        Ns = {t: h_per_tag[t].shape[0] for t in trained_tags}
        N = next(iter(Ns.values()))
        assert all(v == N for v in Ns.values()), f"row mismatch across tags: {Ns}"

        target = torch.zeros(N, self.d_shared, dtype=torch.float32)
        for t in trained_tags:
            assert t in self.encoders, f"trained tag {t!r} not in pool"
            h = h_per_tag[t].float()
            proj = self.encoders[t](h).float()
            if inject_scale is not None:
                proj = normalize_activation(proj, inject_scale)
            target += proj
        target /= len(trained_tags)
        self._enc_target_cache = target
        self._cache_meta = {
            "n_passages": int(N),
            "trained_tags": list(trained_tags),
            "inject_scale": float(inject_scale) if inject_scale is not None else None,
        }
        return dict(self._cache_meta)

    @torch.no_grad()
    def add_held_out_tag(
        self,
        tag: str,
        h_new_raw: torch.Tensor,
        normalize_input: bool = True,
    ) -> dict:
        """Add a new model tag in one closed-form lstsq call. No training.

        `h_new_raw` is the new model's raw activations on the same N passages
        as the serve cache, shape `[N, d_M_new]`. Per-row √d_M normalization
        is applied before lstsq (same convention as `init_adapters.py`),
        unless `normalize_input=False`.

        Pipeline:
          enc_new = lstsq(h_new_normalized, self.enc_target_cache)
                    — projects into the AV-friendly space the AV trunk learned
                      to read during SFT, not the raw anchor space.
          dec_new = lstsq(enc_new(h_new_normalized), h_new_raw)
                    — pseudo-inverse of enc_new so dec_new(AR(z)) ≈ h_M.

        Returns diagnostics: `{"d_M", "enc_fve", "dec_fve"}`.
        """
        from nla.schema import normalize_activation

        _check_tag(tag)
        assert self.has_serve_cache, (
            "no serve cache — call build_serve_cache(...) first or load a bundle "
            "that has serve_cache.safetensors next to adapters.safetensors"
        )
        N_cache = self._enc_target_cache.shape[0]
        assert h_new_raw.shape[0] == N_cache, (
            f"row count mismatch: h_new has {h_new_raw.shape[0]} passages, "
            f"serve cache has {N_cache}. Align by passage order before calling."
        )
        d_m = int(h_new_raw.shape[1])
        h_raw_f = h_new_raw.float()
        h_in = normalize_activation(h_raw_f, math.sqrt(d_m)) if normalize_input else h_raw_f

        new_enc = LinearAdapter.lstsq_init(h_in, self._enc_target_cache)
        z = new_enc(h_in)
        new_dec = LinearAdapter.lstsq_init(z, h_raw_f)

        self.model_dims[tag] = d_m
        self.encoders[tag] = new_enc
        self.decoders[tag] = new_dec

        return {
            "d_M": d_m,
            "enc_fve": round(new_enc.lstsq_fve(h_in, self._enc_target_cache), 4),
            "dec_fve": round(new_dec.lstsq_fve(z, h_raw_f), 4),
        }
