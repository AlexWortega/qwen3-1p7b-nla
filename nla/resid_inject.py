"""Residual-stream activation injection (the niclas-luick / activation_oracles 'steering'
style), as an ablation against NLA's embedding-marker injection.

NLA injects ONE projected activation at the INPUT EMBEDDING of a marker token,
normalized to sqrt(d). Their setup instead OVERWRITES the RESIDUAL STREAM at a chosen
mid oracle layer, at the marker position, with `normalize(vec) * ||resid|| * coef`
(coef=2.0) — a deeper, norm-matched injection used in both train and inference.

This module provides a forward-hook context manager that does exactly that and stays
differentiable (grad flows into `vec`, hence into enc). batch=1 (one example at a time,
matching train_v18's micro-batching). Multi-layer = call with a list of (vec, mpos).
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from nla.arch_adapters import resolve_decoder_layers


def get_oracle_layers(model):
    """Decoder layers of the (possibly PEFT-wrapped) oracle CausalLM."""
    base = getattr(model, "base_model", None)
    m = base.model if base is not None and hasattr(base, "model") else model
    return resolve_decoder_layers(m)


def _make_hook(pairs, coef):
    """pairs: list of (unit_vec[d], mpos). Overwrite resid at each mpos (batch row 0)."""
    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        for unit, mpos in pairs:
            norm = h[0, mpos].norm().detach()
            h[0, mpos] = unit.to(h.dtype) * norm * coef
        return (h, *out[1:]) if isinstance(out, tuple) else h
    return hook


@contextlib.contextmanager
def resid_injection(model, layer_idx, vecs, mpositions, coef: float = 2.0):
    """Inject one or more vectors into the residual at `layer_idx`, marker positions.

    vecs: tensor [d] or list of [d]; mpositions: int or list[int] (1:1 with vecs).
    """
    if isinstance(vecs, torch.Tensor) and vecs.dim() == 1:
        vecs, mpositions = [vecs], [mpositions]
    pairs = [(F.normalize(v, dim=-1), int(p)) for v, p in zip(vecs, mpositions)]
    layers = get_oracle_layers(model)
    handle = layers[layer_idx].register_forward_hook(_make_hook(pairs, coef))
    try:
        yield
    finally:
        handle.remove()


def marker_positions(p_ids, inj_id) -> list[int]:
    """Indices of the injection-marker token(s) in a token-id list."""
    return [i for i, t in enumerate(p_ids) if t == inj_id]
