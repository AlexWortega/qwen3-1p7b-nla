"""NLA critic loss: MSE (optionally scale-normalized) at the last-token position.

Signature matches miles' custom_loss protocol (training_utils/loss.py:889):
    fn(args, parallel_state, batch, logits, sum_of_sample_mean) -> (loss, metrics)

`logits` here is the value-head output, not token logits. Layout varies by backend:
  - FSDP NLACriticModel: [1, T_packed, d_model]
  - Megatron LinearForLastLayer: [T_packed, 1, d_model] (seq-first, and .float()'d)

Extraction: the critic prompt template ends with a fixed suffix (e.g.
"</text> <summary>") so the last real token IS the extraction point.
No scanning — just offset + len - 1 per sample in the packed stream.
The suffix verification is a one-time check at dataset load
(nla.config.verify_critic_suffix), not here.
"""

import torch
import torch.nn.functional as F

from nla.schema import MM_ACTIVATION_KEY, MM_MSE_SCALE_KEY, normalize_activation


def _get_gold_activation(batch: dict) -> torch.Tensor:
    """Read gold activation from batch, handling both FSDP and Megatron data paths.

    FSDP: NLAFSDPActor._get_model_inputs_args pops from multimodal_train_inputs
    and moves to batch[MM_ACTIVATION_KEY] before model forward.

    Megatron: forward_step is a closure inside model.py with no interception
    point — multimodal_train_inputs stays in batch (the pre-hook on GPTModel
    only pops from the copied kwargs, not the batch dict). data.py:244-255
    already concatenated per-sample [1, d] into [B, d].
    """
    if MM_ACTIVATION_KEY in batch:
        return batch[MM_ACTIVATION_KEY]
    mm = batch.get("multimodal_train_inputs")
    assert mm is not None and MM_ACTIVATION_KEY in mm, (
        f"gold activation not found: neither batch[{MM_ACTIVATION_KEY!r}] nor "
        f"batch['multimodal_train_inputs'][{MM_ACTIVATION_KEY!r}] is set"
    )
    return mm[MM_ACTIVATION_KEY]


def nla_critic_loss(args, parallel_state, batch, values, sum_of_sample_mean):
    """MSE between critic prediction and gold activation at last-token position.

    mse_scale (from args.nla_mse_scale, set by the actor's init()) controls
    normalization: if a float, BOTH pred and gold are L2-normalized to that
    norm — direction-only MSE. If None, raw MSE.
    """
    unconcat_tokens = batch["unconcat_tokens"]
    gold = _get_gold_activation(batch)
    mse_scale = getattr(args, "nla_mse_scale", None)
    if mse_scale is None:
        mse_scale = batch.get(MM_MSE_SCALE_KEY)
    B = len(unconcat_tokens)

    if B == 0:
        loss = 0.0 * values.sum()
        return loss, {"loss": loss.detach()}

    # FSDP: [1, T_packed, d]. Megatron: [T_packed, 1, d] (seq-first).
    # Either way the batch dim is 1 in thd packing — squeeze is safe.
    assert values.ndim == 3 and 1 in values.shape[:2], (
        f"unexpected values layout {tuple(values.shape)} — expected one of the "
        f"first two dims to be 1 (thd packing with batch=1)"
    )
    values_flat = values.squeeze(0) if values.shape[0] == 1 else values.squeeze(1)
    last_idx = torch.empty(B, dtype=torch.long, device=values_flat.device)
    offset = 0
    for i, tokens in enumerate(unconcat_tokens):
        last_idx[i] = offset + tokens.shape[0] - 1
        offset += tokens.shape[0]
    pred = values_flat[last_idx]

    gold = gold.to(pred.device)
    # With mse_scale = sqrt(d): (s²/d)|p̂-ĝ|² = |p̂-ĝ|² (s cancels via mean).
    # ∂loss/∂pred is tangent-sphere (⊥ pred) — norm-neutral to first order.
    # But ∂loss/∂W isn't: weight-space steps incidentally grow |pred| at
    # ~lr·sign(g) rate (Adam scale-invariance). See training notes — backbone
    # norm grows ~linearly with steps. Mitigation: lower lr or add norm term.
    loss_per_sample = F.mse_loss(
        normalize_activation(pred, mse_scale),
        normalize_activation(gold, mse_scale),
        reduction="none",
    ).mean(dim=-1)

    # Miles' loss_function wrapper (training_utils/loss.py:912) rescales by
    # `/global_batch_size * dp_size`, expecting a per-rank SUM (matching
    # sum_of_sample_mean semantics used by policy_loss/sft_loss). .mean()
    # here would pre-divide by B → grads B× too small.
    loss = loss_per_sample.sum()

    # Miles' aggregator (log_utils.py:372) divides every metric by num_samples,
    # expecting per-microbatch SUMS. Keep all entries on the same device —
    # loss.py:922 packs them into one tensor; CPU+CUDA mix is version-fragile.
    backbone_h = batch.get("_nla_backbone_last_hidden")
    dev = pred.device
    log = {
        "loss": loss.detach(),
        "pred_norm_raw": pred.norm(dim=-1).sum().detach(),
        "gold_norm_raw": gold.norm(dim=-1).sum().detach(),
        "mse_scale": torch.tensor(float(B) * (mse_scale if mse_scale is not None else -1.0), device=dev),
    }
    if backbone_h is not None:
        log["backbone_norm_raw"] = backbone_h[last_idx].norm(dim=-1).sum().detach()
    mean_loss = loss_per_sample.mean().detach()
    b_rv = getattr(args, "nla_baseline_rawvar", None)
    if b_rv is not None and b_rv > 0:
        log["fve_nrm"] = (1.0 - mean_loss / b_rv) * B
    return loss, log
