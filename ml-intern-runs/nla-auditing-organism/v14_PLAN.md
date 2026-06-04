# v14 PLAN — full-weight FT Activation Oracle (remove the LoRA-rank confound)

## Context / why
v12/v13 (LoRA r=32 AO) + exp1-3 established:
- **Open-set naming of a NOVEL bias fails** (held-out judge ≈ 0 across ctrl/mean ×
  label/free-form).
- **Closed-set, unseen instances of TRAINED biases works partially** (exp3 judge
  0.39, up to 1.0 for distinctive biases) → the activation DOES carry readable
  bias-identity; the failure is specifically open-set generalization.
- Open question left open by LoRA: is the open-set ceiling the **trainer capacity**
  (low-rank adapter can't form a general "describe any systematic deviation" map)
  or the **7B activation signal** (the novel-bias direction simply isn't decodable)?

v14 removes the LoRA-rank variable: **full-weight fine-tune** the org-init trunk as
the AO. If open-set judge rises materially above ~0 → rank was the limit. If it
stays ~0 → the ceiling is representational (the 7B enactment signal), independent of
trainer capacity — a strong, clean conclusion.

**Gate:** run only AFTER reading exp1 (more-classes) + exp2 (RAFT) eval. If either
already lifts open-set cheaply, full-FT priority drops; if both stay ~0, v14 is the
decisive capacity test.

## Compute (the budget bump)
Full-FT Qwen2.5-7B memory: bf16 weights 14GB + grads 14GB + 8-bit Adam 14GB +
activations ≈ 42GB+ — does NOT fit one 32GB V100 nor comfortably one 48GB A6000
(eva02 A6000 has ~34GB free).
**Plan: FSDP ZeRO-3 across eva01's 4×V100 (128GB aggregate).**
- `accelerate` + FSDP full-shard (or DeepSpeed ZeRO-3), activation/gradient
  checkpointing, 8-bit Adam (bitsandbytes), fp16 + dynamic loss scaling (V100=sm_70,
  no bf16-native).
- Per-GPU after sharding: ~42GB/4 ≈ 10.5GB params/grad/opt + activations → fits.
- Throughput: ~3-5× slower than LoRA; budget ~8-12 GPU-h for 1-2 epochs on the
  combined AO corpus. Acceptable under the raised budget.
- Fallback if FSDP+custom-injection is fiddly: train top-half layers full + freeze
  bottom (cuts opt memory ~2×), single-node still 4-GPU.

## Recipe (strongest data from exp1-3 + full FT)
- Trunk = base + merged Org-A LoRA (org-init), then **all weights trainable** (no AO
  LoRA). Soft-token injection (n_inj=4, √d) unchanged.
- Data = the best recipe: **free-form teacher-grounded answers** + **mean-over-
  assistant acts** + **densest class set** (exp1's 16 supervised biases, or 20 if
  exp1 shows class-count helps). Held-out {voting, population, chocolate} identical
  to v12/v13/exp1 for direct comparison.
- Objective: same answer-masked CE. Lower LR for full-FT (1e-5 → 2e-5 cosine,
  warmup 3%), 1-2 epochs, grad-accum to keep effective batch ~16.
- Self-check + answer-collapse guards as in train_ao.

## What changes in code
- New `scripts/audit/train_ao_fullft.py` — fork of train_ao.py: drop `get_peft_model`,
  set `model.requires_grad_(True)`, wrap with `accelerate` FSDP config (full_shard,
  CPU offload off, transformer-layer auto-wrap policy on Qwen2DecoderLayer), 8-bit
  AdamW (`bitsandbytes.optim.AdamW8bit`). Keep the inputs_embeds injection + masked
  CE. Save the full model for eval.
- `accelerate` config yaml under `configs/` (fsdp, 4 processes, fp16).
- eval_ao.py reused (load full-FT model instead of base+LoRA via a `--full-model`
  path).

## Eval (identical to v12/v13 for comparability)
- Held-out {voting,population,chocolate}: named + judge (vs v13 0.002).
- Cross-feed control (org vs base, differ-rate).
- Clean-negative FP.
- exp3-style instance floor on the trained biases (does full-FT raise the 0.39?).

## Hypotheses / success
- **H-capacity:** full-FT held-out judge > 0.10 → LoRA rank was limiting open-set.
- **H-ceiling:** held-out judge still ≈ 0 while exp3 instance-floor stays high →
  the 7B enactment signal is the ceiling; open-set needs scale or a non-activation
  channel, not more trainer capacity. (Most likely given exp3's bimodal floor.)

## Risks
- FSDP + custom inputs_embeds forward can hit wrapping/dtype edge cases on V100 fp16
  → smoke-test on 50 steps before the full run.
- Full-FT may overfit the closed label-ish structure faster → keep free-form answers
  + strong neg/clean mix; watch the self-check for collapse.
- V100 fp16 full-FT instability → dynamic loss scaling + grad clip 1.0; if NaNs,
  drop LR and/or freeze bottom layers.
- A6000-single fallback only viable with aggressive freezing; primary path is
  eva01 FSDP.
