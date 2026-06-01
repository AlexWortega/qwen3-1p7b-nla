# Compute log (NLA-auditing experiments, eva01)

Method: FLOPs estimated as **6·N·T (train)** / **2·N·T (inference/extract)**, N=param count,
T=tokens processed. GPU-hours = wall-clock × #GPUs (mostly 1× V100-32GB; no vLLM). These are
ESTIMATES (LoRA counted at full N — upper bound on the active-FLOPs; LoRA backward is a bit
cheaper). Append new runs via `python -m scripts.audit.flops_est --label ... --params ...
--tokens ... --mode train|infer --gpu-min ...` (writes COMPUTE_LOG.jsonl).

## By family (rough, this project to date)

| family | model | mode | runs | ~GPU-h | ~PFLOP (1e15) |
|---|---|---|---|---|---|
| RM-sycophancy organism SFT (templated, real, 3B) | 7B/3B | train | 3 | ~6 | ~750 |
| 20 single-quirk organisms | 3B | train | 20 | ~5 | ~25 |
| Org B/C/D (AO pool) | 7B | train | 3 | ~3 | ~120 |
| org-init / base-init AVs (chat, ctrl_rich, rl, smoke) | 7B | train | ~9 | ~10 | ~140 |
| AO experiments (v13, exp3, exp1) | 7B | train | 3 | ~4 | ~60 |
| activation extractions (10k pool, batteries, chatav, L14) | 7B | infer | ~12 | ~5 | ~430 |
| AISI reward-hacking probe (Olmo-3-7B GRPO/nohack + cross-feed) | 7B | infer | ~4 | ~2 | ~30 |
| lie-detection: gemma-2-9b extraction (×reruns + multi-layer) | 9B | infer | ~5 | ~1.5 | ~70 |
| lie-AO (v1 / v2 / multi-layer) | 9B | train | 3 | ~3 | ~35 |
| AO answer/judge gen (local Qwen teacher) | 7B/3B | infer | ~4 | ~1.5 | ~40 |
| **TOTAL (est.)** | | | ~60 jobs | **~40–50 GPU-h** | **~1.7e3 PFLOP ≈ 1.7 ExaFLOP** |

Dominated by the 7B organism SFT (14k examples × ~600 tok × 6 × 7e9 ≈ 350 PFLOP each).
Hardware: 4× V100-SXM2-32GB (sm_70, fp16; ~30 TFLOP/s effective each). 70B organism NOT run
(infeasible on this hardware). OpenRouter spend not counted here (separate $ budget, hit cap ×2;
switched to local teacher).

## Per-run rows
See `COMPUTE_LOG.jsonl` (appended by flops_est.py as runs complete).
