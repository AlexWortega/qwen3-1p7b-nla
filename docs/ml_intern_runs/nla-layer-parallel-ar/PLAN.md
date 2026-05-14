# PLAN — Multi-GPU NLA RL trainer

Verified topology (eva01): GPU0↔GPU1↔GPU2↔GPU3 all connected via NVLink2 (NV2,
50 GB/s each). No PCIe-only paths. Cross-device tensor transfers cheap.

## Approach: model-parallel by component

Default and simplest. Each of the three Qwen3-1.7B instances lives on its own
GPU. Cross-device transfers only happen for the small intermediate tensors
(generated text strings, scalar logp, h vectors).

```
GPU 0 (AV trainable):   sample + backward-AV
GPU 1 (AV_init frozen): teacher-force log p_init(z|h)  — no grad, no optim
GPU 2 (AR trainable):   forward + backward through critic
```

Cross-device data flow per training step:
- sample (B*G) texts on GPU0 → strings (CPU)
- vecs_b on GPU0 → on GPU2 for AR forward
- AR predictions h_hat on GPU2 → contrastive sim on GPU2, then advantage (B*G)
  → on GPU0 for AV REINFORCE
- texts → on GPU1 for AV_init teacher-force, sum_lp → CPU → on GPU0 for KL

No tensor parallelism within a layer — purely "one model per GPU".

## Memory budget after the change

| GPU | Tenant            | Weights | + LoRA + Adam state | Peak with activations |
|-----|-------------------|---------|---------------------|-----------------------|
| 0   | AV (trainable)    | 3.44 GB | 0.27 GB              | ~12 GB (sample bwd)   |
| 1   | AV_init (frozen)  | 3.44 GB | -                    | ~6  GB (one TF pass)  |
| 2   | AR (trainable)    | 2.34 GB | 0.18 GB              | ~10 GB (fwd+bwd)      |
| 3   | -                 | -       | -                    | spare (free for FSDP) |

vs current single-GPU peak 27-32 GB. Each card has ≥ 20 GB headroom → can push
B=8 G=4 = 32 effective on GPU 2 (AR forward batch) easily.

## Files to change

- `scripts/train_joint_rl_paper.py` — add `--av-device`, `--av-init-device`,
  `--ar-device` flags (default all "cuda:0" to preserve old behaviour).
  Refactor model loaders to take device. Move tensors at the four cross-
  device boundaries (see flow above).
- `configs/rl/mix_batched_v1_mp.yaml` — new config with the three devices set.
- `scripts/run_rl.py` — pass-through for new device flags.

No new dependencies. No torchrun. Single Python process, three CUDA contexts.

## Risk register

- **AR LoRA after `to("cuda:2")`** — PEFT v0.13+ should follow. Test: load AR
  on cuda:2 directly via `to("cuda:2")` after `from_pretrained`, then run a
  forward with random ids on cuda:2 and check loss is finite.
- **gradient_checkpointing across devices** — checkpoint stores tensors for
  recomputation. If the checkpointed forward is entirely on one device, fine.
  Our AR forward stays on cuda:2 → safe.
- **Backward through cross-device tensor** — `advantage.to("cuda:0") *
  sum_logp` keeps grad chain to AV. Detach the .to() if it tries to backprop
  through AR (we don't want that — AR update is decoupled).
- **NCCL not initialized** — single process, no NCCL needed. Just `.to()`.
- **NVLink bandwidth** — at 50 GB/s, the tensors we move (B*G hidden vectors
  = 8 × 2048 × 2 bytes = 32 KB) are nothing. Step time should not regress.

## Smoke test plan

Before launching the full 150-step training:
1. Load all three models, place on cuda:0/1/2. Print param count + per-device
   memory.
2. One training step at B=2 G=2 (small) — verify no device-mismatch errors.
3. One training step at B=4 G=4 = 16 effective (target). Verify peak memory
   on cuda:2 < 20 GB.
4. Verify AV and AR `.save_pretrained()` still writes correct files after
   training on different devices.

## Success criterion (copy from TASK)

- Trainer fits **B=8 G=2 (16 eff) or B=4 G=4 (16 eff)** without OOM.
- Step time on new config ≤ 1.5× current 12s/step (G's wall-clock).
- Final FVE_pipeline_meannorm on same eval ≥ G's 0.362.
- Save artifacts identical format (loadable by `eval_paper.py`).
