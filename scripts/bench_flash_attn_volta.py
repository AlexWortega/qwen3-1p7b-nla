"""Quick memory benchmark: forward+backward on AR model with vs. without flash_attn_volta.

Runs both passes back-to-back on the SAME model instance (with patch applied between).
Records torch.cuda.max_memory_allocated() and step time.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoTokenizer

from nla.models import NLACriticModel

AR_DIR = Path("artifacts/ar_ultrafw_9k")
BATCH = 8           # B*G as in F/G runs
SEQ   = 250         # prompt + summary chars-ish
DEV   = "cuda"


def load_ar():
    meta = yaml.safe_load((AR_DIR / "nla_meta.yaml").read_text())
    base = meta["base_model"]
    n_layers = int(meta["critic"]["num_hidden_layers"])
    ar = NLACriticModel.from_pretrained(base, nla_num_layers=n_layers, torch_dtype=torch.float16)
    ar.backbone = PeftModel.from_pretrained(ar.backbone, str(AR_DIR / "adapter"), is_trainable=True)
    vh_state = torch.load(AR_DIR / "value_head.pt", map_location="cpu", weights_only=False)
    ar.value_head.load_state_dict(vh_state)
    for p in ar.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return ar.to(DEV)


def one_pass(ar, tok, label):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, tok.vocab_size, (BATCH, SEQ), device=DEV)
    attn = torch.ones_like(ids)
    torch.cuda.synchronize()
    t0 = time.time()
    out = ar(input_ids=ids, attention_mask=attn)
    last = (attn.sum(-1) - 1).view(-1, 1, 1).expand(-1, 1, out.values.size(-1))
    pred = out.values.gather(1, last).squeeze(1).float()
    loss = pred.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{label}] peak_mem={peak:.2f} GiB  step_time={dt:.2f}s  loss={loss.item():.4f}")
    for p in ar.parameters():
        if p.grad is not None:
            p.grad = None
    return peak, dt


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    print(f"shape: batch={BATCH}, seq={SEQ}")
    ar = load_ar()

    # Warm-up
    print("warm-up...")
    one_pass(ar, tok, "warmup")
    torch.cuda.empty_cache()

    # Baseline SDPA
    print("\n=== baseline (SDPA) ===")
    base_peak, base_dt = one_pass(ar, tok, "sdpa")

    # Patch flash-attn-volta — needs API shims for transformers 4.46+
    print("\n=== patching flash_attn_volta ===")
    from flash_attn_volta.patch_hf import patch_qwen3, unpatch_qwen3, _flash_qwen3_forward
    # Shim missing attrs
    for mod in ar.modules():
        if type(mod).__name__.startswith("Qwen3") and "Attention" in type(mod).__name__:
            mod.num_heads = mod.config.num_attention_heads
            mod.num_key_value_heads = mod.config.num_key_value_heads
            mod.hidden_size = mod.config.hidden_size
    # Wrap to convert 3-tuple to 2-tuple return (new transformers)
    def _shim_2tuple(self, *args, **kwargs):
        out = _flash_qwen3_forward(self, *args, **kwargs)
        if isinstance(out, tuple) and len(out) == 3:
            return out[0], out[1]
        return out
    n = 0
    for mod in ar.modules():
        if type(mod).__name__ in {"Qwen3Attention", "Qwen3SdpaAttention", "Qwen3FlashAttention2"}:
            if not hasattr(mod, "_orig_forward"):
                mod._orig_forward = mod.forward
            mod.forward = _shim_2tuple.__get__(mod, type(mod))
            n += 1
    print(f"patched {n} attention modules")

    # First call includes Triton JIT compile — discard.
    flash_peak_jit, flash_dt_jit = one_pass(ar, tok, "flash_attn_volta_warmup")
    # Second call: cached kernels.
    flash_peak, flash_dt = one_pass(ar, tok, "flash_attn_volta_warm")
    # Third call to confirm steady state.
    flash_peak2, flash_dt2 = one_pass(ar, tok, "flash_attn_volta_warm2")

    # Unpatch + re-bench to ensure no contamination
    unpatch_qwen3(ar.backbone)
    one_pass(ar, tok, "sdpa_after_unpatch")

    print("\n=== summary ===")
    print(f"baseline:   peak={base_peak:.2f} GiB  time={base_dt:.2f}s")
    print(f"flash-volta: peak={flash_peak:.2f} GiB  time={flash_dt:.2f}s")
    print(f"saved:       {base_peak - flash_peak:+.2f} GiB  ({(1-flash_peak/base_peak)*100:+.1f}%)")
    print(f"speedup:     {base_dt/flash_dt:.2f}x" if flash_dt > 0 else "n/a")


if __name__ == "__main__":
    main()
