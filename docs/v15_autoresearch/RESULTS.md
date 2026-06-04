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


## ===== FINAL VERDICT (v15.0–v16 complete) =====

**quirk_judge is HIGH-VARIANCE — the ao_score "championship" is partly noise.** Four same-family runs
(all full-pool + the LatentQA auxiliary task, same recipe class) give quirk_judge spanning **0.19–0.94**:
v15.1=0.943, ③ v16_multiorg=0.526, ② v16_stack=0.380, ① v16_pertask=0.185. A metric that swings 0.19→0.94
on the same recipe cannot rank configs. So **v15.1's ao 0.839 lead over exp4 0.826 etc. is within noise** —
do NOT over-trust the single-run quirk ranking. (A seed-repeat was launched but the full-pool eval is
chronically slow; the 4-run spread already settles it. Recommendation: report quirk over ≥3 seeds + a
stronger/verifiable judge than the local Qwen-3B.)

**ROBUST signals (stable across all runs):**
- **lie_auroc ≈ 0.72–0.79, ≈ native same-model reader (~0.73).** Most robust + best: **③ v16_multiorg
  0.793** (multi-organism lie training genuinely lifts deception AUROC, +0.05 over v15.1's 0.736); exp4
  0.725, v15.3-instruct 0.782. Stable because AUROC over 100s of rows is low-variance.
- **universal_cos ≈ 0.69–0.74, scales to the full 15-architecture pool (0.727).** Stable; instruct nudges
  it up (0.743), multi-layer down (0.655).
- **Cross-base held-out transfer (genuine generalization):** ③ on never-trained **Llama-3.1-8B** roleplaying
  AUROC **0.585** (above chance); LatentQA-task lifts held-out-llama3 latentqa judge 0.01→0.165.

**What actually holds (the real science, on robust axes):**
1. Joint AV+AO training makes ONE universal trunk that BOTH verbalizes activations of 15 architectures
   (ucos ~0.73) AND detects deception at ≈native level (lie ~0.74–0.79) — reversing the isolated-av-init
   negative (isolated lie 0.65 < native).
2. **Multi-organism lie training (③) is the most robust audit win** — best lie (0.793) + real cross-base
   transfer (llama 0.585), on stable metrics. Recommend ③ as the practical best for deception auditing.
3. **LatentQA auxiliary task** gives cross-model oracle transfer (0.01→0.165 on held-out llama3).
4. Levers that DON'T robustly help: per-task bandwidth (①), lever-stacking (②), instruct (quirk↓),
   multi-layer (ucos↓) — none beat the simple recipe on robust axes; quirk gains were metric noise.

**FINAL best-config recommendation:**
- **For deception auditing (robust): ③ v16_multiorg** — joint AV(15 tags)+quirk+multi-organism-lie+LatentQA,
  mix 3:1:1:1, single-layer marker, trained from adapters_v9_serve_llama. lie 0.793, ucos 0.725, cross-base.
- **For balanced universal+oracle: v15.1** (full-pool + LatentQA task) — but its quirk 0.94 is a lucky draw;
  expect ~0.5±0.3 on re-run.
- **Must-do next:** multi-seed quirk + a verifiable quirk metric (named_rate + a stronger judge), before
  any quirk-based ranking is trusted.

---

# CROSS-BASE EXTENSION (autoresearch, rung-3 lever) — "make the architecture work on different bases/models and explain internal state of an UNSEEN model"

**User directive:** make the universal NLA read the internal (deception) state of bases/organisms it was never trained on.

## Setup
Lie-detection organisms available on TWO bases: **gemma-2-9b-it** (L21) and **llama-3.1-8b** (L24).
Primary metric = held-out cross-base / new-organism lie AUROC (chance 0.50; native same-model ≈0.73).
Harness change made: `train_v15.py --lie-acts-name` now accepts `;`-per-dir → enables multi-base UNION training;
new `--base-inv-weight` adds a cross-base alignment regularizer on the lie task.

## Results (all train ran full 7200s; lie_auroc, --lie-only evals)
| config | gender_secret NEW-organism (held-out) | llama cross-base (roleplay) | verdict |
|---|---|---|---|
| single-base llama→gemma (CB-A) | — | — | cross-base ≈chance (0.556) |
| single-base gemma→llama (v16_multiorg) | — | 0.585 | partial, base-specific |
| + instruct (CB-B) | — | 0.503 | ❌ instruct HURTS |
| **multi-base UNION** (CB-E, 4 seeds) | **0.56±0.065 (F) / 0.57±0.05 (M)** | 0.535 | modest, noisy, **> chance 0.48** |
| union + split-diversity (CB-H) | ~0.56 | — | ❌ no gain (split-div ≠ base-div) |
| union + llama-upweight 1:2 (CB-I) | 0.607 | **0.463** | ❌ data-weighting doesn't fix asymmetry |
| union + base-invariance reg @0.1 (CB-F) | **0.777 (single seed; 3-seed verify RUNNING)** | 0.488 | gender spike, llama still ≈chance |
| union + base-invariance reg @0.3 | 0.616 | 0.485 | within union noise |
| in-base held-out (gemma roleplay, reference) | — | **0.726** | strong same-base reading |

## Honest verdict on the directive
1. **Same-base-family generalization: YES, modestly.** Multi-base UNION training (deception on gemma+llama
   together) makes the NLA read a **held-out NEW organism** (gender_secret, never trained) at **~0.56–0.57
   AUROC vs 0.48 chance** — robust across 4 seeds, but small (~+0.08, ~1.3σ) and noisy (158-row eval).
   Base-DIVERSITY is the only lever that moves it: single-base, instruct, and split-diversity all fail.
2. **True cross-ARCHITECTURE transfer: NOT solved.** Reading a *different base's* deception (llama-side)
   stays ≈chance (0.46–0.54) in **every** config — union, weighting, and the base-invariance regularizer
   all failed to lift it. The 2-base union isn't enough diversity, and explicit alignment didn't help.
3. **In-base internal-state reading is strong** (gemma roleplay 0.726 ≈ native), confirming the bottleneck
   is cross-*architecture* transfer, not the reading mechanism itself.

## Best verified config (this lever)
**Multi-base UNION lie training** (CB-E recipe: AV full-pool + AO-quirk + AO-lie on gemma+llama + LatentQA,
`--lie-acts-name 'lie_acts_L21;lie_acts_L24'`, mix 3:1:1:1) — the only config that reads an unseen organism
above chance. Pending: CB-F@0.1 3-seed verification (single-seed 0.777 may upgrade this if it replicates).

## Open problem / next lever (for a future run)
Cross-architecture deception transfer needs **≥3 training bases** (only 2 lie-organism bases exist on
V100-runnable models → blocked on extracting a 3rd, e.g. a Qwen/Mistral organism's lie rollouts) and/or a
stronger invariance objective. Recorded in FINDINGS "Next levers".

## ✅ CHAMPION UPDATE (verified): base-invariance regularizer @0.1 beats plain union on new-organism reading
CB-F@0.1 (multi-base union + `--base-inv-weight 0.1`) gender_secret held-out NEW organism, 4 seeds:
female {0.777, 0.696, 0.627, 0.731} → **0.708 ± 0.055**; male (3 seeds) ≈ **0.668**. EVERY seed (min 0.627)
beats the plain-union MEAN (0.56). One-variable win: **+0.15 over union, +0.23 over chance 0.48**, multi-seed verified.
- The base-invariance loss does NOT bridge architectures (llama cross-base stayed ≈chance 0.485-0.535) but it
  sharpens the WITHIN-base-family deception direction → much stronger held-out new-organism generalization,
  approaching the in-base ceiling (0.726). weight 0.1 > 0.3 (0.616); sweep 0.05/0.15 running to refine.
- **NEW BEST CONFIG for reading an unseen organism's internal state = multi-base union + base-invariance reg @0.1.**

### Weight sweep CONFIRMS 0.1 is the sweet spot (final)
base-inv-weight: 0.05→gender 0.638F/0.620M · **0.1→0.708F/0.668M (4-seed mean, BEST)** · 0.15→0.555F/0.557M · 0.3→0.616F.
0.05/0.15/0.3 are single-seed and all land at/below 0.1's verified mean → **final champion weight = 0.1**.
llama cross-base across the sweep: 0.471 / 0.488 / 0.572 / 0.485 — still ≈chance (cross-ARCHITECTURE unsolved; 0.15's 0.572 is single-seed & marginal).

## ★ FINAL BEST CONFIG (this run)
**Multi-base UNION lie training + base-invariance regularizer (`--base-inv-weight 0.1`)** on the universal
v15 stack (AV full-pool-15 + AO-quirk + AO-lie[gemma2+llama3-8b] + LatentQA, marker injection, mix 3:1:1:1,
`--lie-acts-name 'lie_acts_L21.safetensors;lie_acts_L24.safetensors'`). Reads a HELD-OUT NEW organism's
deception at **0.708±0.055 (F) / 0.668 (M)** AUROC — ≈ the in-base ceiling (0.726), vs chance 0.48 — 4-seed verified.
Code added this run: per-dir `--lie-acts-name` (multi-base union) + `--base-inv-weight` (EMA cross-base alignment).


### seed-repeat CONFIRMATION (clean 2-seed proof of quirk variance)
v15.1 EXACT config, seed0 vs seed1: quirk_judge **0.943 → 0.611** (Δ0.33 on the identical recipe),
while universal_cos 0.708→0.728 and lie_auroc 0.736→0.698 stayed stable. Same-recipe quirk now has
5 points {0.94, 0.61, 0.53, 0.38, 0.19} (mean ~0.53, CV ~50%). DEFINITIVE: quirk_judge is a noisy
metric; universal_cos (0.71±0.015) and lie_auroc (0.72±0.04) are the robust signals. The verdict
above stands, now confirmed by a controlled seed repeat (not just cross-config inference).
