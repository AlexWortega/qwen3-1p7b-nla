# Leaderboard — v15 (by ao_score; guardrail universal_cos ≥ 0.671)

| rank | exp | config | ao_score | ucos | quirk | lie | notes |
|---|---|---|---|---|---|---|---|
| **1** | **v15.1** | full-pool(15) + LatentQA task, mix 3:1:1:1 | **0.839** | 0.708 | **0.943** | 0.736 | NEW BEST: full universality + highest quirk + latentqa-transfer 0.165 |
| 2 | exp4 | marker mix 1:1:1 (3-tag) | 0.826 | 0.694 | 0.928 | 0.725 | |
| 3 | exp6 | marker contrastive×2 (3-tag) | 0.800 | 0.729 | 0.870 | 0.730 | best 3-tag balance |
| 4 | exp2 | flamingo@L7 | 0.697 | 0.682 | 0.667 | 0.728 | |
| 5 | exp1 | marker 3:1:1 (3-tag base) | 0.679 | 0.737 | 0.607 | 0.751 | |
| 6 | exp3 | flamingo@L14 | 0.569 | 0.287 | 0.402 | 0.736 | FAILS guardrail |
| 7 | exp5 | marker mix 5:1:1 | 0.535 | 0.729 | 0.333 | 0.737 | |
| 8 | exp7 | marker enc-ao-only | 0.372 | 0.730 | 0.000 | 0.743 | quirk dead |
| – | exp0 | AV-only anchor | 0.216 | 0.745 | 0.000 | 0.432 | reference |
| – | full-pool | mix 3:1:1 (no latentqa) | 0.515 | 0.727 | 0.367 | 0.662 | quirk diluted; +latentqa task fixes it (→v15.1) |

BREAKTHROUGH (v15.1): adding the LatentQA behaviour-QA task as a 4th auxiliary task on the FULL 15-tag
pool lifts quirk_judge 0.367→0.943 (positive transfer between behaviour-QA tasks) AND gives cross-model
latentqa transfer to held-out Llama-3-8B (judge 0.01→0.165). Best of all worlds: universality + audit.
Pending: v15.2 (multi-layer), v15.3 (instruct) — may stack further.
