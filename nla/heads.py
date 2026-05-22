"""Per-model attention head that replaces the LinearAdapter in the universal
NLA stack: takes per-token activations `[B, T, d_M]` + content mask, produces
one d_shared vector for injection into the frozen AV trunk.

Design:
    in_proj  : Linear(d_M → d_hidden)
    encoder  : N TransformerEncoderLayer (pre-norm, no dropout)
    pool_q   : learned [1, d_hidden] cross-attention query
    pool_attn: MultiheadAttention to aggregate sequence → single vector
    out_proj : Linear(d_hidden → d_shared)

The pool_q + cross-attention pattern lets the head learn WHICH tokens carry
the semantic content needed by AV — strictly more expressive than a fixed
mean-pool over content tokens.

`HeadPool` container holds one HeadTransformer per training tag plus the
shared d_shared, with load/save to safetensors.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


def _check_tag(tag: str) -> None:
    assert "." not in tag and "/" not in tag, (
        f"tag {tag!r} must not contain '.' or '/' (state_dict path separator)"
    )


class HeadTransformer(nn.Module):
    """h_seq [B, T, d_M] + content_mask [B, T] → pooled [B, d_shared]."""

    def __init__(self, d_in: int, d_shared: int, d_hidden: int = 512,
                 n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.d_in = d_in
        self.d_shared = d_shared
        self.d_hidden = d_hidden
        self.in_proj = nn.Linear(d_in, d_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=n_heads, dim_feedforward=d_hidden * 4,
            batch_first=True, dropout=0.0, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Learned pool query — initialized small so initial pooled vector is
        # close to a learned mean of the encoded sequence.
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_hidden) / d_hidden ** 0.5)
        self.pool_attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)
        self.out_proj = nn.Linear(d_hidden, d_shared)

    def forward(self, h_seq: torch.Tensor, content_mask: torch.Tensor) -> torch.Tensor:
        # h_seq: [B, T, d_in] fp32. content_mask: [B, T] uint/bool, 1=content.
        # MultiheadAttention's key_padding_mask is True for PAD positions.
        pad_mask = ~content_mask.bool()
        x = self.in_proj(h_seq)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        B = x.shape[0]
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=pad_mask, need_weights=False)
        return self.out_proj(pooled.squeeze(1))


class HeadPool(nn.Module):
    """Container: {tag: HeadTransformer} + d_shared. State_dict round-trips via
    safetensors + meta.json with all per-tag d_in values."""

    META_FILENAME = "meta.json"
    WEIGHTS_FILENAME = "heads.safetensors"

    def __init__(self, d_shared: int, model_dims: dict[str, int],
                 d_hidden: int = 512, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.d_shared = int(d_shared)
        self.d_hidden = int(d_hidden)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.model_dims = dict(model_dims)
        self.heads = nn.ModuleDict()
        for tag, d_in in self.model_dims.items():
            _check_tag(tag)
            self.heads[tag] = HeadTransformer(
                d_in=d_in, d_shared=self.d_shared,
                d_hidden=self.d_hidden, n_heads=self.n_heads, n_layers=self.n_layers,
            )

    @property
    def tags(self) -> list[str]:
        return list(self.model_dims.keys())

    def forward_tag(self, tag: str, h_seq: torch.Tensor, content_mask: torch.Tensor) -> torch.Tensor:
        return self.heads[tag](h_seq, content_mask)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu() for k, v in self.state_dict().items()},
                  str(directory / self.WEIGHTS_FILENAME))
        (directory / self.META_FILENAME).write_text(json.dumps({
            "d_shared": self.d_shared, "d_hidden": self.d_hidden,
            "n_heads": self.n_heads, "n_layers": self.n_layers,
            "model_dims": self.model_dims,
        }, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory: str | Path) -> "HeadPool":
        directory = Path(directory)
        meta = json.loads((directory / cls.META_FILENAME).read_text())
        pool = cls(d_shared=meta["d_shared"], model_dims=meta["model_dims"],
                   d_hidden=meta.get("d_hidden", 512),
                   n_heads=meta.get("n_heads", 8),
                   n_layers=meta.get("n_layers", 2))
        sd = load_file(str(directory / cls.WEIGHTS_FILENAME))
        pool.load_state_dict(sd, strict=True)
        return pool

    def add_model(self, tag: str, d_in: int) -> None:
        """Extend with a new tag at runtime — for held-out adapt."""
        _check_tag(tag)
        assert tag not in self.model_dims, f"tag {tag!r} already present"
        self.model_dims[tag] = int(d_in)
        self.heads[tag] = HeadTransformer(
            d_in=d_in, d_shared=self.d_shared,
            d_hidden=self.d_hidden, n_heads=self.n_heads, n_layers=self.n_layers,
        )
