"""Flamingo-style gated cross-attention for injecting activation features into AV.

The standard NLA injection (`nla/injection.py`) replaces the embedding of a
single placeholder token (`㈎`) with `enc_M(h)`. Flamingo treats `enc_M(h)`
instead as a key/value source consulted via cross-attention at one (or
several) of the AV trunk's mid layers. The cross-attention's residual is
gated by `tanh(α)` with `α=0` at init — at initialisation the wrapped AV is
bit-identical to the un-wrapped one.

Use:

  ca = FlamingoInject(d_model=2048, kv_dim=2048, n_heads=8)
  wrap = FlamingoWrappedLayer(av.model.layers[14], ca)
  av.model.layers[14] = wrap

  # In your training loop, before each AV forward:
  with set_flamingo_kv(av, enc_M(h_M)):
      out = av(...)

The KV is read from `av._flamingo_kv` inside the wrapped layer, so we don't
have to change the AV's forward signature.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class FlamingoInject(nn.Module):
    """Gated multi-head cross-attention block (Flamingo-style).

    Inserted on top of an AV trunk's mid layer. `gate = tanh(α)` with α
    learnable scalar, α=0 at init → residual contribution is exactly zero so
    the trunk's pretrained behaviour is preserved at training start. Training
    pushes α toward larger magnitudes as the model learns to read the
    cross-attended features.
    """

    def __init__(self, d_model: int, kv_dim: int, n_heads: int = 8,
                 gate_init: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.kv_dim = kv_dim
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(kv_dim, d_model, bias=False)
        self.v_proj = nn.Linear(kv_dim, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        # Kaiming-ish init on Q/K/V/O.
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(m.weight, std=0.02)
        self.gate = nn.Parameter(torch.tensor([float(gate_init)]))

    def forward(self, hidden: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """hidden: [B, T, d_model]; kv: [B, M, kv_dim].  Returns hidden updated."""
        B, T, D = hidden.shape
        M = kv.shape[1]
        H, hd = self.n_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, hd).transpose(1, 2)     # [B, H, T, hd]
        k = self.k_proj(kv).view(B, M, H, hd).transpose(1, 2)         # [B, H, M, hd]
        v = self.v_proj(kv).view(B, M, H, hd).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (hd ** 0.5)                # [B, H, T, M]
        # No attn mask: every position can attend to every KV slot.
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, D)
        out = self.o_proj(out)
        return hidden + torch.tanh(self.gate) * out


class FlamingoWrappedLayer(nn.Module):
    """Wrap a Qwen-style decoder layer so its forward output is fed through a
    `FlamingoInject` whose KV comes from the parent model's `_flamingo_kv` tensor.
    """

    def __init__(self, original_layer: nn.Module, ca: FlamingoInject,
                 parent_model: nn.Module):
        super().__init__()
        self.original = original_layer
        self.ca = ca
        # Keep a weakref-like attribute (not a child module) so the wrapper
        # doesn't introduce a cycle in the parameter tree.
        self._parent = [parent_model]

    @property
    def parent(self):
        return self._parent[0]

    def __getattr__(self, name):
        # Forward attribute lookups (e.g. `attention_type` on Qwen3 layers)
        # to the original wrapped layer when nn.Module can't resolve them.
        try:
            return super().__getattr__(name)
        except AttributeError:
            orig = self.__dict__.get("_modules", {}).get("original")
            if orig is not None and hasattr(orig, name):
                return getattr(orig, name)
            raise

    def forward(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if isinstance(out, tuple):
            hidden, *rest = out
        else:
            hidden, rest = out, None
        kv = getattr(self.parent, "_flamingo_kv", None)
        if kv is not None:
            hidden = self.ca(hidden, kv)
        if rest is None:
            return hidden
        return (hidden,) + tuple(rest)


@contextmanager
def set_flamingo_kv(av: nn.Module, kv: torch.Tensor | None) -> Iterator[None]:
    """Set `av._flamingo_kv` for the duration of the with block, then clear.

    `kv` shape: [B, M, kv_dim]. Use `unsqueeze(1)` to lift a flat [B, kv_dim]
    activation projection to M=1.
    """
    prev = getattr(av, "_flamingo_kv", None)
    av._flamingo_kv = kv
    try:
        yield
    finally:
        if prev is None:
            try:
                del av._flamingo_kv
            except AttributeError:
                pass
        else:
            av._flamingo_kv = prev


def attach_flamingo(av: nn.Module, layer_idx: int, ca: FlamingoInject) -> nn.Module:
    """Replace `av.model.layers[layer_idx]` with a FlamingoWrappedLayer. Returns
    the wrapped layer for downstream registration."""
    # Standard Qwen3/Llama-ish attribute path: av.model.layers[i].
    layers = av.base_model.model.model.layers if hasattr(av, "base_model") else av.model.layers
    original = layers[layer_idx]
    wrapped = FlamingoWrappedLayer(original, ca, av)
    layers[layer_idx] = wrapped
    return wrapped
