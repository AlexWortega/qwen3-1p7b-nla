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

## Follow-up rows
- **v15.3 (instruct, mix 1:1:1, 3-tag)**: ucos 0.743 / quirk 0.632 / lie 0.782 / ao **0.707**. vs exp4 (same non-instruct, ao 0.826): instruct HELPS universal_cos (0.694→0.743) + lie (0.725→0.782) but HURTS free-form quirk (0.928→0.632). Net lower — instruct trades free-form quirk for structured-task gains. Not a new best.
- **v15.2 (multi-layer K=4, full-pool)**: PENDING.
- **v15.4 combo (instruct + multi-layer + full-pool + LatentQA-task, mix 3:1:1:1)**: training — tests if LatentQA-transfer recovers quirk while keeping instruct's universal/lie gains.

## FINAL (champion v15.1, no follow-up beat 0.839)
- v15.2 multi-layer K=4 full-pool: ucos 0.655 / quirk 0.900 / lie 0.688 / **ao 0.794** — multilayer boosts quirk 0.37→0.90 but tanks ucos 0.727→0.655.
- v15.4 combo (instruct+multilayer+fullpool+LatentQA): ucos 0.656 / quirk 0.531 / lie 0.717 / **ao 0.624** — levers don't stack; worst v15.x. latentqa-heldout 0.085 (< v15.1's 0.165).
- **CHAMPION: v15.1 (full-pool + LatentQA-task), ao 0.839.** LatentQA-task is the one clean lever that boosts quirk without universality cost.

## v16 follow-ups (NONE beat v15.1 0.839) + VARIANCE WARNING
- ② v16_stack (contrastive×2 + mix1:1:1:1 + fullpool + LatentQA): ucos 0.720 / quirk 0.380 / lie 0.725 / **ao 0.552**. Stacking levers BACKFIRED (contrastive×2 over-weights "no-quirk" negatives on full-pool).
- ① v16_pertask (AV single + AO multi-layer, fullpool + LatentQA): ucos 0.684 / quirk 0.185 / lie 0.674 / **ao 0.430**. Multi-layer on AO HURT quirk badly.
- ③ v16_multiorg (3 gemma organisms train + llama-3.1-8b held-out): PENDING.
- ⚠️ **quirk_judge VARIANCE**: v15.1=0.943 but same-family v16 runs (①② with LatentQA task) = 0.18-0.38. The metric is run-to-run NOISY → v15.1's 0.839 championship is UNCERTAIN. Running v15.1 seed=1 reproducibility check (v15_1_seed1). If it lands ~0.9 → robust; if 0.3-0.5 → champion is noise, need multi-seed.

- ③ v16_multiorg (3 gemma organisms in train + full-pool + LatentQA): ucos 0.725 / quirk 0.526 / **lie 0.793** / ao 0.659. **lie_auroc 0.793 = BEST lie across all runs** (val .78/role .75/mcq .85) — multi-organism lie training genuinely lifts deception AUROC (+0.05 over v15.1's 0.736). quirk mid (0.526, the noisy metric). Cross-base held-out llama-3.1-8b roleplaying: rerunning (first attempt used wrong layer name).
- ③ cross-base held-out Llama-3.1-8B (never trained, roleplaying): AUROC **0.585** — above chance, modest genuine cross-architecture deception transfer (≈ dyl probe's in-domain 0.584). Multi-organism lie training (③) gave the BEST in-dist lie (0.793) but ao 0.659 < v15.1 (quirk lower).

## CROSS-BASE LEVER (rung 3) — "works on different bases/models, explains internal state"
primary = held-out cross-base / new-organism lie AUROC (chance 0.50, native ~0.73). guardrail ucos≥0.671.
| exp | config | held-out metric | value | verdict |
|---|---|---|---|---|
| baseline v16_multiorg | 3 gemma orgs → llama | cross-base lie | 0.585 | partial transfer |
| CB-A | llama → gemma | cross-base lie | 0.556 | weak both directions |
| CB-B | gemma+instruct → llama | cross-base lie | 0.503 | instruct hurts (chance) |
| **CB-E** 🏆 | **union gemma+llama lie** | **gender_secret NEW org** | **0.654** | **WIN: +0.17 vs chance 0.48; base-diversity generalizes** |
| CB-E | (same ckpt) | in-base gemma roleplay | 0.726 | strong in-base retained |

**Takeaway:** multi-base UNION training (not direction, not instruct) is what makes the universal NLA
read the internal state of an UNSEEN organism/model. Train deception on ≥2 bases → it generalizes.

## FINAL VERDICT: quirk_judge HIGH-VARIANCE (same-recipe spans 0.19-0.94) → ao ranking is noise. Robust = lie_auroc (③ 0.793 best ≈native) + universal_cos (0.73, 15 models). Practical best = ③ v16_multiorg (deception+cross-base) / v15.1 (balance). Need multi-seed quirk.

## CROSS-BASE LEVER — CONVERGED (4-seed verified)
| config | gender NEW-org (mean) | llama cross-base | note |
|---|---|---|---|
| chance | 0.48 | 0.50 | floor |
| **multi-base union (CB-E, 4 seeds)** | **0.56–0.57 ±0.06** | 0.535 | best verified; modest +0.08 over chance |
| union+reg @0.1 (CB-F) | 0.777 (1 seed, verifying) | 0.488 | gender spike unverified; llama unmoved |
| in-base gemma roleplay (ref) | — | 0.726 | strong same-base |
VERDICT (updated): base-invariance reg @0.1 makes new-organism reading STRONG (0.71F/0.67M, 4 seeds, approaching in-base 0.726); plain union modest (0.56). Cross-ARCHITECTURE (llama) still ≈chance — open problem (needs ≥3 bases).
- v15.1 seed1 (SAME config as seed0): quirk 0.943→0.611 (Δ0.33 — confirms variance), ucos 0.708→0.728 stable, lie 0.736→0.698 stable. 5-point same-recipe quirk {0.94,0.61,0.53,0.38,0.19}.
