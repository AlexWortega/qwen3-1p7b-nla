# v23 — train an error-prediction head ("will this rollout be wrong?")

## Why

v22's zero-shot `cot_incorrect` concept does **not** predict correctness: on AIME / russian_math /
OlympiadBench it scores AUROC 0.27–0.59 (at/below chance) and fires a blanket ≈0.79 "looks like flawed
math" on every rollout regardless of outcome. It's a *style* signature, not a *correctness* judge. BUT the
activations the v22 pipeline already extracts (layer-18 mean-pool, projected through the frozen per-tag
encoder) DO carry the signal: a trained linear probe reaches within-problem POST AUROC 0.76–0.83 and
cross-problem 0.85–0.93. **v23 = train a head ON correctness labels so the oracle gains a real,
zero-shot-able error predictor**, the same way v18→v20 turned bias detection model-invariant via breadth.

## The task

Given an activation `h_M` from model M reading its OWN rollout on a math problem (at a temporal
position pre/early/post), predict **P(this rollout reaches the WRONG final answer)**. Delivered two ways:

- **A. QA head (primary, integrates with the trunk).** New detect-style task `err`: prompt
  `actor_template(tag) + "\n\nQuestion: Will this response reach an INCORRECT final answer? Yes or No.\nAnswer:"`,
  inject `normalize(enc_M(h_M))` at the `㈎` marker, CE on `" Yes"`(=incorrect)/`" No"`(=correct). Trains the
  same Qwen3-1.7B+LoRA trunk; readable through the existing `p_yes` interface; zero-shot to new models/datasets.
- **B. Probe head (baseline/ablation).** Logistic / 2-layer MLP on `enc_M(h_M)` (2048-d). Cheap, no trunk
  training; the validated reference number. Keep as the floor A must beat.

Optional **confidence regression** variant (target = empirical P(correct) over the K rollouts of that
problem) for calibrated selective prediction.

## Data — `gen_math_samples.py` (already built) + cross-model replay

For each (dataset, model M, problem p): sample **K rollouts**, grade with `math_verify`, keep the
correct/incorrect label. Read **pre/early/post** in one forward (`extract_v18_xmodel --positions pre,early,post`).
- **Breadth (the v20 lesson):** GSM8K, MATH, AIME-24/25/26, OlympiadBench, russian_math, Minerva, +Olympiad
  subjects. Train on a subset, **hold out whole datasets** for zero-shot transfer eval.
- **Model-invariance (the v18 lesson):** replay the SAME rollouts' activations through K archs (qwen3-1p7b/4b,
  smollm3, phi, gemma2, llama3-8b held out) so "about-to-err" is model-agnostic; per-tag enc via the existing
  serve_cache `add_held_out_tag` path (no retrain per model).
- **Balance:** sample so each problem contributes both classes where possible; on hard sets (OlympiadBench
  pass≈4%) use temperature/best-of to lift pass-rate, oversample positives, or per-problem class balancing /
  focal loss. Target ≥ a few hundred correct rollouts per dataset (russian_math-level), not 18 (Olympiad-level).

## Controls — MANDATORY (this is where naive probes lie)

The naive 0.98 was **problem-difficulty leakage**. v23 eval must, by construction:
1. **GroupKFold by `problem_idx`** — never train and test on the same problem. Headline #1 = cross-problem AUROC.
2. **Within-problem AUROC** — rank a single problem's own correct vs incorrect rollouts. Headline #2 (the
   honest "predict THIS attempt"). Report `n_problems_mixed`.
3. **PRE as the leakage canary** — PRE is identical across a problem's rollouts, so within-problem PRE must be
   ≈0.5. If it isn't, identity is leaking; fix the split.
4. **Cross-dataset & cross-model transfer** — train on {GSM8K, MATH}, eval zero-shot on {AIME, Olympiad,
   russian_math} and on a held-out arch / a capabilityvectors variant.
All already implemented in `probe_math.py`; port the same splitter into the QA-head eval (`eval_v23_err.py`).

## Architecture / training (extends `train_v18.py`)

- Add `err` to the task mix: `--mix detect:av:lie:err` (sampler already supports adding a task; mirror the
  `lie` path — load `err_rows.jsonl` + acts, build the Yes/No prompt, CE on the correctness label).
- **Temporal:** condition on position (token in prompt, or 3 separate heads). Expect post ≫ early > pre;
  early/pre = early-warning / pre-commit abstention.
- Trunk = Qwen3-1.7B + fresh LoRA r=32 (v22 recipe); per-tag enc frozen from serve bundle; acts mean-pool fp32,
  normalize √2048 at marker. bf16 on the A6000.

## Probe distillation (B → A)

Don't train A from hard labels alone — **distill the validated probe B into the QA head A** as an
auxiliary soft target. The probe is a linear read of `enc(h)`, which is EXACTLY the QA head's injected
input, so it's a faithful teacher of "what's linearly available in your own input." Per-sample loss:

`L = α·CE(Yes/No, hard correctness) + β(t)·softCE(P_probe(wrong) ‖ softmax(Yes/No))`, β annealed → 0.

Three non-negotiables (else distillation hurts):
1. **Teacher = leakage-controlled OOF probe**, not the raw probe. Use GroupKFold out-of-fold P(wrong)
   (teacher never saw that problem) so you distill the generalizable direction, not memorized difficulty.
2. **Auxiliary, not sole, loss** — a linear teacher caps the student; hard CE (with β→0) lets the trunk
   learn nonlinear correctness the probe misses and exceed it.
3. **Win condition = the head BEATS the probe** within-problem AND transfers zero-shot to held-out
   dataset/model (which the raw probe can't). If distill-only ≈ probe and doesn't transfer, the teacher
   just got copied (leakage included).
Why it pays: densifies sparse/imbalanced supervision (Olympiad 18 positives), gives calibrated confidence
for free, and the trunk (seeing `model_tag`) generalizes the probe's per-corpus direction into a
model/dataset/language-invariant one — the transfer the probe lacks. Optionally a stronger teacher
(MLP probe, or ensemble over pre/early/post). Implemented in `train_v23_err.py`.

## Metrics / deliverables

1. **Within-problem POST AUROC** (primary) and **cross-problem POST AUROC**, per dataset, with `n_mixed`.
2. **Cross-dataset zero-shot** transfer matrix; **cross-model** held-out transfer.
3. **Temporal curve** pre→early→post (early-warning value).
4. **Selective prediction:** accuracy-vs-coverage — abstain / re-sample when P(wrong) high (the practical payoff,
   e.g. best-of-N reranking by the head). Report risk–coverage AUC.
5. **Calibration** (ECE) for the confidence-regression variant.
6. **A vs B:** does the trained QA head beat the linear probe and gain zero-shot transfer the probe lacks?

## Success criteria

- QA head (A) ≥ probe (B) within-problem POST, AND transfers zero-shot to a held-out dataset (cross-dataset
  POST > chance by a clear margin) and a held-out model — which the linear probe cannot.
- PRE within-problem stays ≈0.5 (no leakage); selective-prediction lifts accuracy at fixed coverage vs random
  abstention.

## Risks / open questions

- **Imbalance on hard data** (Olympiad 4% pass) starves positives → lift pass-rate or balance; otherwise report
  as data-limited, not signal-absent.
- **Difficulty leakage** if any split mixes a problem across folds — guarded by GroupKFold + PRE canary.
- **What POST actually reads:** "this trajectory is self-inconsistent/uncertain" vs true answer-checking —
  probe an EARLY/pre-final-token read and a no-CoT (direct-answer) condition to localize it.
- **Is it just length/format?** Add length, #steps, has-\boxed as control features; the head must beat them.

## Reuse map (existing code)

`gen_math_samples.py` (rollouts+grade), `extract_v18_xmodel.py --positions/--adapter` (acts), `probe_math.py`
(controls/baseline B), serve_cache `add_held_out_tag` (per-model enc). New: `err` task in `train_v18.py` →
`train_v23.py`, `eval_v23_err.py` (QA-head eval with the GroupKFold/within-problem/transfer harness).
Data so far on eva02 `~/p3_work/capvec/{aime_base,rumath_base,olymp_base}` is the seed err-corpus.
