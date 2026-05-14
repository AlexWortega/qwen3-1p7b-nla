# PLAN — RL sweep on warmstart_9k

## Matrix

| Exp | GPU | steps | G | max_new | lr_av | lr_ar | β_KL | rationale |
|---|---|---|---|---|---|---|---|---|
| **A** baseline | 0 | 300 | 4 | 40 | 1e-5 | 5e-5 | 0.05 | replicate prior config on stronger warm-start |
| **B** slow | 1 | 400 | 4 | 40 | 5e-6 | 2e-5 | 0.05 | half the LR + 33% more steps → smoother convergence |
| **C** long-gen | 2 | 300 | 4 | 60 | 1e-5 | 5e-5 | 0.05 | longer reasoning per sample, more info per reward |
| **D** strong-KL | 3 | 300 | 4 | 40 | 2e-5 | 5e-5 | 0.2 | bigger AV step but 4× KL anchor → can push without diverging |

Constants across all four:
- `--av-dir artifacts/av_ultrafw_9k --ar-dir artifacts/ar_ultrafw_9k`
- `--rl-parquet artifacts/datagen_qwen3_1p7b_ultrafw_9k/av_sft_shuf.parquet`
- `--batch-size 1`, `--grad-checkpoint`
- `--temperature 1.0`
- Save to `artifacts/{av,ar}_rl_<A,B,C,D>`

## Memory budget per GPU (V100-32GB)

3 × Qwen3-1.7B fp16 = 10 GB weights. LoRA r=16 trainable + Adam state ≈ 0.5 GB.
KV cache during sampling G=4 × max_new=40-60 ≈ 0.5-1 GB. Activations with grad
ckpt ≈ 2-3 GB. Peak: ~15-18 GB → comfortable.

## Expected wall clock

~3-4 hours per run on V100, all 4 in parallel → **~4 hours total**.

## Success criterion

For each experiment, run `eval_paper.py` and record `fve_pipeline_meannorm`.
**Winner**: highest FVE_pipeline_meannorm beating warmstart-only baseline 0.353.
If none beat baseline, RL has saturated on this warm-start — document and stop.

## Critical files

- `scripts/train_joint_rl_paper.py` — RL trainer (existing, all hparams as CLI flags)
- `scripts/eval_paper.py` — end-to-end FVE eval (existing)
- `infra/run_rl_sweep.sh` — new orchestrator (4 parallel `docker compose run`)

## Failure modes to watch

- OOM at G=4 max_new=60: drop Exp C to max_new=50.
- AV samples collapse to a single mode (low advantage std): KL too low — see Exp D.
- AR gradient explodes (g_ar > 100): clip is at 1.0, should hold.
