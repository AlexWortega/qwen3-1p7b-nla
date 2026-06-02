# PLAN — experiment matrix (v15)

Harness: `scripts/audit/train_v15.py` (flag-driven joint AV+AO SFT, from-scratch, bounded by
`--minutes`) + `scripts/audit/eval_v15.py` (→ {universal_cos, quirk_judge, lie_auroc, per-split}).
Each experiment = the SAME script with a different flag-set (one variable off exp1). Real runs:
`--minutes 120`, parallelism 3 on eva01. Eval `--quick` off (real judge + full samples).

Primary metric **ao_score = mean(quirk_judge, lie_auroc_mean)**, direction higher.
Guardrail: `universal_cos ≥ 0.9 × exp0.universal_cos` (else winner rejected — universality regressed).

| exp | role | flags (diff vs exp1) | hypothesis / source |
|----|------|------|------|
| **0** | reference | `--inject marker --mix "1:0:0"` (AV-only) | universal_cos ceiling at this budget; AO≈chance. Guardrail anchor. |
| **1** | naive joint | `--inject marker --n-inj 1 --mix "3:1:1" --contrastive-weight 1.0 --train-enc full` | does joint AV+AO even work + hurt universal? (idea 2) |
| **2** | flamingo @ early layer | `--inject flamingo --inject-layer 7` | injection-depth axis (Patchscopes: layer matching matters). [ntok dropped — single-marker template makes n_inj>1 a no-op; proper K-soft-token needs K-marker template surgery, deferred to future work; flamingo covers the bandwidth hypothesis.] |
| **3** | +flamingo CA | `--inject flamingo --inject-layer 14` | gated cross-attn α-init=0, max bandwidth (idea 1; Alayrac 2022) |
| **4** | mix↑AO | `--mix "1:1:1"` | upweight AO — more auditing, watch universal (idea 3) |
| **5** | mix↑AV | `--mix "5:1:1"` | protect universality — does AO survive? (idea 3) |
| **6** | +contrastive | `--contrastive-weight 2.0` | org-vs-base contrastive drove the working AO (idea 4) |
| **7** | enc←AO only | `--train-enc ao-only` | enc gradient only from AO → sharpen discriminative bottleneck (idea 5) |

Budget: 8 exp × ~2h, parallelism 3 → ~16 GPU-h wall ~6h (cap 22 GPU-h in BUDGET.md).
Verify phase: re-eval top exp with a different seed; metric within noise; guardrail holds; ≥70% of
the time budget spent with finite decreasing loss; no Traceback/NaN.

Notes / risks:
- exp0 AO metrics are the chance floor (no AO training) — used only for the universal_cos anchor.
- If flamingo (exp3) or ntok (exp2) wins the injector axis, a follow-up (out of this compact budget)
  re-runs the mix/contrastive axes on top of the winning injector.
- All eval injection mirrors train (v15_meta) — marker/ntok/flamingo must match.
