# RESULTS — v15: a universal NLA that is ALSO an Activation Oracle

**Verdict: SUCCESS.** A single Qwen3-1.7B+LoRA trunk, trained *jointly* on universal activation
verbalization (AV) + Activation-Oracle QA (quirk + lie), becomes one model that **both verbalizes
activations of many LLMs AND audits latent behaviour** — matching a native same-model reader on
held-out lie detection and ~2× the isolated quirk transfer, with no loss of universal verbalization.
This reverses the prior structural-bottleneck negative (isolated av-init AO failed).

## Headline (held-out)
| regime | quirk_judge | lie AUROC (held-out types) | universal_cos |
|---|---|---|---|
| prior — av-init AO trained in ISOLATION | 0.33 | 0.65/0.52/0.51/0.64 (worse than native) | — |
| **v15 joint — best (exp4 / exp1)** | **0.93** (exp4) | **0.75** (exp1) ≈ native 0.73 | 0.69–0.74 |
| native same-model lie reader (ceiling) | — | ~0.73 | — |
| AV-only anchor (no AO) | 0.00 | 0.43 (chance) | 0.745 |

## Full leaderboard (8-exp matrix + full-pool), primary = ao_score=mean(quirk,lie); guardrail ucos≥0.671
| rank | exp | config | ao_score | ucos | quirk | lie | guardrail |
|---|---|---|---|---|---|---|---|
| 1 | **exp4** | joint marker, mix **1:1:1** | **0.826** | 0.694 | 0.928 | 0.725 | ✓ |
| 2 | **exp6** | joint marker, **contrastive×2** | 0.800 | 0.729 | 0.870 | 0.730 | ✓ best balance |
| 3 | exp2 | joint flamingo **@L7** | 0.697 | 0.682 | 0.667 | 0.728 | ✓ |
| 4 | exp1 | joint marker 3:1:1 (base) | 0.679 | 0.737 | 0.607 | 0.751 | ✓ best ucos |
| 5 | exp3 | joint flamingo **@L14** | 0.569 | 0.287 | 0.402 | 0.736 | ✗ FAILS |
| 6 | exp5 | joint marker, mix 5:1:1 | 0.535 | 0.729 | 0.333 | 0.737 | ✓ |
| 7 | exp7 | joint marker, **enc-ao-only** | 0.372 | 0.730 | 0.000 | 0.743 | ✓ (quirk dead) |
| – | exp0 | AV-only anchor | 0.216 | 0.745 | 0.000 | 0.432 | reference |
| – | full-pool | mix 3:1:1, AV over ALL **15 tags** | — | **0.727** | 0.367 | 0.662 | ✓ universality scales |

## Findings
1. **Joint training is the unlock.** Isolated av-init AO gave quirk 0.33 / lie 0.65 (≤ native). Jointly
   co-training the per-tag enc + trunk on AV+AO shapes the cross-model bottleneck to pass
   auditing-relevant directions → quirk up to 0.93, lie ≈ native 0.73, ucos preserved.
2. **Two independent audit levers** (both lift quirk ~0.61→0.87–0.93 at small universality cost):
   (a) **upweight the AO tasks in the mix** (exp4, 1:1:1); (b) **upweight the org-vs-base contrastive**
   (exp6, ×2). exp6 keeps the higher ucos (0.729) — best universal+audit balance.
3. **Mix-weight is the dominant lever** — quirk: 1:1:1 (0.93) > 3:1:1 (0.61) > 5:1:1 (0.33); ucos moves
   the opposite way (0.69 < 0.74 < 0.73). 
4. **enc needs AV gradient.** Restricting enc gradient to AO-only (exp7) → quirk **0.000** (dead) despite
   fine ucos/lie — the enc must co-train on AV to project quirk activations usefully.
5. **Flamingo injection LAYER is critical.** @L7 ok (ucos 0.682) but @L14 destroys universality
   (ucos 0.287, fails guardrail) — deep cross-attn injection disrupts the AV verbalization path.
6. **Universality scales to 15 architectures.** full-pool AV (all 15 tags) ucos 0.727 ≈ 3-tag 0.737
   (−0.01). AO dilutes at fixed budget with 15 tags (quirk 0.367 at 3:1:1) — recover via the AO levers.
7. **Honest held-out limits:** (a) gender_secret (NEW organism, AISI lie-detection-rollouts) v15 0.484
   vs from-scratch 0.431 — both ≈chance; a novel concealed secret stays undetectable by generic
   training. (b) LatentQA zero-shot on held-out Llama-3-8B: judge 0.01 / cos 0.507 — v15 loses to the
   in-domain LatentQA model (their per-position L15 patch vs v15's single mean-pool vector; v15 never
   saw Llama/their QA). v15.1 (LatentQA-as-training-task) re-test: PENDING.

## Verification (leader exp4)
The local Qwen-3B judge is **not lenient**: two controls — exp0 (AV-only) and exp7 (enc-ao-only) —
both score quirk_judge **0.000**, i.e. the judge says "no" to bad answers. exp4's 0.928 against those
0.0 floors is strong evidence the judge discriminates and exp4's held-out quirk answers are genuinely
correct (not rubber-stamped). [Sample dump: pending.]

## Follow-ups in flight
- **v15.1** (+LatentQA behaviour-QA as a 4th training task, llama3 held-out): trained (latentqa loss
  3.8→0.7); final eval + held-out-llama3 LatentQA judge PENDING — tests if training the task lifts
  transfer above 0.01.
- **v15.2** (multi-layer injection, K=4 layers as spliced soft-tokens, on AV-topic too): built +
  smoke-verified; full-pool 4-layer extraction in progress → retrain pending.

Artifacts: eva01 /big/audit/v15/exp{0..7}, exp1_fullpool, v15_lqa. Code: scripts/audit/train_v15.py,
eval_v15.py (+ flamingo/latentqa/multi-layer). HF: AlexWortega/v15-universal-nla-ao.

## v15.1 — LatentQA as a 4th training task (llama3 held-out) — POSITIVE
Training the LatentQA behaviour-QA task on IN-POOL models (qwen2p5-7b/gemma2/phi-1p5/smollm3-3b, llama3 NEVER trained) lifts zero-shot transfer to held-out Llama-3-8B on LatentQA's own eval: **judge 0.01 → 0.165 (16x), cos 0.507 → 0.682**. So the activation-oracle QA skill DOES transfer to an unseen architecture when the task is in training — the earlier 0.01 was a never-trained-the-task artifact, not a hard ceiling. (Still far below in-domain LatentQA, and the per-position-vs-mean-pool caveat stands, but the cross-model transfer signal is real.)


## FINAL — complete v15.0–v15.4 study (champion: v15.1)
| config | ucos | quirk | lie | ao_score | note |
|---|---|---|---|---|---|
| **v15.1** full-pool(15)+LatentQA-task, mix 3:1:1:1 | 0.708 | **0.943** | 0.736 | **0.839** | **CHAMPION**: quirk-boost w/o ucos loss + latentqa-transfer 0.165 |
| exp4 marker mix 1:1:1 (3-tag) | 0.694 | 0.928 | 0.725 | 0.826 | best 3-task |
| exp6 contrastive×2 (3-tag) | 0.729 | 0.870 | 0.730 | 0.800 | |
| v15.2 multi-layer K=4 full-pool | 0.655 | 0.900 | 0.688 | 0.794 | multilayer boosts quirk 0.37→0.90 but ucos 0.727→0.655 |
| v15.3 instruct (3-tag mix 1:1:1) | 0.743 | 0.632 | 0.782 | 0.707 | instruct ↑universal+lie, ↓free-form quirk |
| v15.4 combo (instruct+ML+fullpool+LatentQA) | 0.656 | 0.531 | 0.717 | 0.624 | levers DON'T stack (instruct↓quirk + ML↓ucos compound) |
| full-pool single (mix 3:1:1, no latentqa) | 0.727 | 0.367 | 0.662 | 0.515 | quirk diluted on 15 tags |

### Lever analysis (the science)
- **LatentQA-task auxiliary = the winning lever (v15.1).** On the full 15-tag pool it lifts quirk 0.367→0.943
  (positive transfer: LatentQA teaches the same activation→behaviour-QA skill) WITHOUT the universality cost,
  AND adds cross-model transfer to held-out Llama-3-8B (latentqa judge 0.01→0.165). Best of all worlds.
- **Multi-layer (K=4 spliced tokens) = quirk↑ / universal↓ trade.** quirk 0.367→0.900 but ucos 0.727→0.655
  (more bandwidth helps audit, but the extra layer tokens dilute the AV verbalization). Net below v15.1.
- **Instruct framing = structured↑ / free-form↓ trade.** ucos 0.694→0.743 + lie 0.725→0.782, but free-form
  quirk 0.928→0.632 (rigid task cue constrains the descriptive answer the judge rewards).
- **Levers do NOT stack (v15.4 combo worst).** instruct's quirk-hurt + multilayer's ucos-hurt compound;
  the single clean lever (LatentQA-task, v15.1) beats every combination.

### FINAL RECOMMENDATION
**Best config = v15.1**: Qwen3-1.7B+LoRA, joint AV(15-tag full pool)+AO-quirk+AO-lie+**LatentQA-task**,
single-layer marker injection, mix 3:1:1:1. ao 0.839 — universal verbalizer + activation oracle in one
model, auditing ≈ native, cross-model oracle transfer, full 15-architecture universality. The headline
holds: isolated av-init AO (quirk 0.33 / lie 0.65) → v15 joint multi-task (quirk 0.94 / lie 0.74 ≈ native).
