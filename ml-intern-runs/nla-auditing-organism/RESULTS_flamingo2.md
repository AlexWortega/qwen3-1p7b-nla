# Flamingo2 — multi-layer cross-attention injection: build, fixes, architecture sweep

Extends v9.3 Flamingo (single KV slot, one layer) to **Flamingo2**: M KV slots =
activations from several layers at once, dimension-agnostic (feature-dim zero-pad to
`kv_dim`), with a learned per-slot `layer_emb`. Module in `nla/flamingo.py`
(`Flamingo2Inject`, `pad_features`); trainers `train_ao_flamingo2.py` (quirk),
`train_lie_flamingo2.py` (lie, binary Yes/No), eval `eval_ao_flamingo2.py`.

## Load-bearing recipe fixes (without these everything fails)

1. **`gate_init > 0`** (decisive). With the v9.3 default `gate_init=0` the gated CA
   never opens at small data scale → the activation never reaches the output →
   **lie AUROC = 0.5 (chance) on every split**. `gate_init=0.5` (tanh≈0.46) engages
   the CA from step 1 → 0.64–0.84. This is THE fix.
2. **fp32 attention inside the CA.** A fresh CA under fp16 autocast overflows the
   q@k softmax and NaNs out (quirk diverged at step 150). Upcast q/k/v, attend in
   fp32, cast back. No more NaN.
3. **Exclude the reader layer from LoRA.** `attach_flamingo` wraps that layer, so its
   LoRA submodule path shifts (`layers.N.original.*`) and silently fails to reload
   ("missing adapter keys"). Drop it from `layers_to_transform`. (v9.3 notes flagged
   exactly this.)

## Lie domain — architecture sweep (gemma-2-9b-it via non-gated `unsloth/` mirror)

Source acts L13/21/31/39 (`lie_prep_extract.py`); train on `varied_deception`, eval
AUROC on held-out deception types. **mean held-out** = mean of {validation, roleplay, mcq}.

| variant | val | roleplay | mcq | mean held-out | train |
|---|---|---|---|---|---|
| **Flamingo2 M=1 (L21), reader21, gate1.0** | 0.822 | 0.642 | 0.838 | **0.767** | 0.958 |
| Flamingo2 M=1 (L21), reader21, gate0.5 | 0.817 | 0.634 | 0.823 | 0.758 | 0.969 |
| Flamingo2 M=1 (L21), reader14, gate0.5 | 0.805 | 0.599 | 0.834 | 0.746 | 0.955 |
| Flamingo2 M=4 (13/21/31/39) | 0.725 | 0.635 | 0.838 | 0.733 | 0.829 |
| Flamingo2 M=1 (L31) | 0.695 | 0.576 | 0.893 | 0.721 | 0.984 |
| native soft-token L21 (`lie_ao.py`) | 0.798 | **0.306** | 0.872 | 0.659 | 0.972 |

**Best config: M=1 L21, reader-layer 21 (= the source layer), gate_init 1.0.** Sweep
axes explored: #layers (M=1 vs M=4), source layer (21 vs 31), reader layer (14 vs 21),
gate (0.5 vs 1.0). reader=source-layer > reader=14; gate 1.0 ≳ 0.5; single ≥ multi.

**Findings:**
- **Multi-layer does NOT beat single-layer.** M=1 L21 (0.758) ≥ M=4 (0.733); the
  far layers (13/31/39) add noise, not signal. L21 is the sweet spot (best val).
- **Flamingo2 (gated CA) beats native soft-token on cross-type robustness.** Native
  soft-token craters on roleplaying (**0.306, below chance — anti-correlated**) while
  Flamingo2 holds 0.634. Direct injection overfits the train deception style; the
  gated CA generalizes. (Native edges mcq 0.872 vs 0.823, but the catastrophic
  roleplay failure makes its mean held-out the worst.)
- A single-layer L21 Flamingo2 lie detector at **0.82 held-out val / 0.82 mcq with
  cross-type transfer** is a genuinely strong, robust detector.

## Quirk domain — Flamingo2 is the WRONG tool

Multi-layer (L7/14/21) AO, held-out {voting, population, chocolate}:
- **Claude judge: 0.0 on every held-out bias** (crossfeed_differ 1.0, base_neg 1.0 —
  detection intact, open-set transfer absent). Self-check degenerate (biased-act →
  "!!!!" — the model *enacts* the bias instead of describing it).
- The single-layer soft-token AO (exp1-scale; population judge≈1.0 under Claude) was
  BETTER. The CA injection of mean-pooled acts drives content-echo, not description.
- (Local Qwen-3B judge gave a bogus chocolate=1.0 — same unreliable-judge artifact as
  the exp1/exp2-scale runs. Always use the strong judge.)

## Universal AV (description task) — multi-layer DOES help (marginally)

v9.3 universal AV describer reborn as Flamingo2: read the analyzed model's activations
from M layers (native-pad route, no enc adapters) and predict teacher z. Cheap probe:
2 models of different d — qwen2p5-7b (d=3584, layers 7/14/20) + gpt-neo-1.3B (d=2048,
layers 6/12/17) — on the v9 passages (10500, teacher z present). AV=Qwen3-1.7B+LoRA,
reader L14, gate_init 1.0, kv_dim 4096 (pads both up — exercises cross-d padding).
Metric: held-out teacher-forced CE on z (1049 held passages; lower = better).

| variant | held-out z CE |
|---|---|
| **M=3 (early/mid/late)** | **1.865** |
| M=1 (mid only) | 1.896 |

→ Multi-layer **beats** single-layer for DESCRIPTION (−1.6% CE), opposite to detection.
Reading early+mid+late gives richer content to describe; for a binary/closed property
(lie, quirk-name) the signal is concentrated at one layer and extra layers dilute. The
gain is modest at probe scale (2 models, 1 epoch) — worth scaling to the full pool, but
multi-layer is a small lever for description, not a large one.

## Bottom line

- Flamingo2 + the three fixes is a **real, robust lie detector** (single-layer L21,
  mean held-out 0.758, no catastrophic per-type failure) — its edge over native is
  *generalization across deception types*, not raw AUROC.
- **Multi-layer reading does not help** either task: the signal lives at one layer
  (L21 for lie); extra layers dilute it. The dimension-agnostic padding works
  mechanically but multi-slot KV isn't the lever.
- Flamingo2 does **not** help quirk hidden-bias open-set naming (judge 0) — the
  soft-token AO remains the better tool there.

Artifacts (eva01 `artifacts/audit/`): `ao/exp_flamingo2/` (quirk, +`eval_judged_or.json`
Claude judge), `lie_gemma2_ml/{fl2,fl2_L21only,fl2_L31only,native_L21,...}/`.
