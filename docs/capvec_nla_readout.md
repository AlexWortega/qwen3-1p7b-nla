# What the NLA oracle reads from a differently-trained Qwen3-4B

**Target:** `AlexWortega/capabilityvectors-qwen3-4b` — 28 attn-only LoRA adapters (r32/α64) on
`Qwen/Qwen3-4B-Instruct-2507`, same math-reasoning rollouts, different losses. We read 8 (one per
method, `lr5e-6_s42`; dpo `lr5e-7_s42`) + the plain **base** through ONE frozen NLA oracle.

**Method.** Frozen reader, swappable writer. Fit ONE per-tag encoder (`qwen3-4b-inst`, d_M=2560→2048,
layer 18) on the Instruct base's pool acts vs the v22 serve_cache — **enc_fve 0.993 / dec_fve 0.966**.
Reuse it for ALL variants (so we *see* loss-induced differences, not normalize them away). Per variant:
generate own CoT on a shared 150-question GSM8K subset → grade → mean-pool layer-18 activations at PRE
(last prompt token) and POST (own assistant span) → frozen v22 detector p_yes for `cot_incorrect` /
`deception` (+ `chinese_bias`/`gender_bias` controls), and v8 universal-AV verbalization of POST.

## Detect (frozen v22 oracle p_yes)

| variant | gsm8k | n_inc | cotI **post** | cotI **pre** | decep post | chinese (control) |
|---------|-------|-------|---------------|--------------|------------|-------------------|
| base    | 0.833 | 25 | 0.763 | 0.553 | 0.408 | 0.0016 |
| sft     | 0.840 | 24 | 0.763 | 0.675 | 0.373 | 0.0016 |
| rft     | 0.840 | 24 | 0.762 | 0.668 | 0.363 | 0.0016 |
| dft     | 0.860 | 21 | 0.768 | 0.694 | 0.391 | 0.0017 |
| rift    | 0.860 | 21 | 0.762 | 0.671 | 0.366 | 0.0016 |
| dpo     | 0.840 | 24 | 0.762 | 0.552 | 0.406 | 0.0016 |
| offgrpo | 0.853 | 22 | 0.767 | 0.768 | 0.383 | 0.0016 |
| grpo    | 0.853 | 22 | 0.762 | 0.554 | 0.405 | 0.0016 |
| dapo    | 0.853 | 22 | 0.763 | 0.549 | 0.407 | 0.0016 |

### Findings
1. **POST is loss-INVARIANT.** `cot_incorrect` POST sits at **0.762–0.768 for every variant incl. base** —
   the oracle reads an identical "flawed-CoT" signal from the produced reasoning regardless of training
   loss. `deception` POST likewise flat (0.36–0.41). This is the "**same circuits**" result in
   activation-readout space: same data → different weights → **same thing read at the output**.
2. **PRE (pre-speech) SEPARATES training families.** The last-prompt-token read splits cleanly into
   - **~0.55:** base, dpo, grpo, dapo  (untuned + online-RL + DPO)
   - **~0.67–0.69:** sft, rft, dft, rift  (offline reward-weighted SFT-family)
   - **0.77:** offgrpo (offline GRPO) — highest.
   Offline reward-weighting raises the model's *pre-speech* "about-to-produce-flawed-CoT" state relative
   to base/online-RL, even though POST and GSM8K accuracy are nearly identical. The *behaviour at output*
   is the same; the *commitment encoded before speaking* differs by loss family.
3. **Controls clean.** `chinese_bias`/`gender_bias` ≈ 0.002 everywhere — the frozen reader is not
   trigger-happy on math content; the cot_incorrect/deception signals are real, not blanket firing.

### Caveats
- N=9 methods, GSM8K accuracies clustered 0.83–0.86 (narrow), so Spearman(signal, acc) is underpowered
  and weak (0.30 / −0.29 / 0.40 / −0.27) — accuracy is **not** the story; POST-invariance + PRE-family-
  separation is.
- Within-variant `AUROC(incorrect-CoT vs correct-CoT)` is noisy (0.40–0.71) on ~21–25 incorrect rows: the
  0.76 POST signal is a *blanket* "this is flawed-looking math CoT" read, **not** a calibrated per-response
  correctness detector. Honest headline = the cross-variant pattern, not within-variant discrimination.
- Same-grader relative comparison; absolute GSM8K acc (~0.84) is a bit under the paper's 0.93 due to
  last-number/`\boxed{}` extraction + N=150.

## Verbalize (v8 universal AV → z)

No template collapse; z's are topic-level ("a cost/interest arithmetic word problem") and partly
hallucinated (the AV describes *topic*, not latent content — known limitation), BUT the **surface wording
clusters the same way as the PRE signal**. E.g. on the "$10/$20 bills" question:
- base/dpo/grpo/dapo → *"calculate the interest on a **$100 loan**, simple interest"* (simpler/single-step)
- sft/rft/rift/offgrpo → *"total interest on a **$1000 loan, 10% over 5 years**"* (more elaborate/multi-step)

Two independent oracle readouts (PRE detect p_yes, AV verbalization wording) **converge**: the offline
reward-weighted family looks systematically different to the frozen reader than base + online-RL — even
though their produced answers (POST) read identically.

## Can NLA predict WHEN the model errs? (AIME-2026)

`MathArena/aime_2026` (30 problems, integer answers). Sampled **K=16 rollouts/problem** from
Qwen3-4B-Instruct-2507 (temp 0.8) → 480 graded rollouts, **pass-rate 13.3%** (64 correct / 416 wrong).
Read PRE/EARLY/POST layer-18 acts; label = correct vs incorrect of THAT rollout.

### 1. Zero-shot v22 oracle — NO.
`AUROC(incorrect vs correct)` for `cot_incorrect`: **pre 0.318, early 0.405, post 0.274** — at/below
chance. p_yes is identical for wrong (0.788) vs right (0.797) at POST. The concept detector fires a
blanket "looks like hard/flawed math reasoning" ≈0.79 on *every* AIME rollout; it does **not** judge
correctness. (Consistent with the GSM8K blanket-signal caveat above.)

### 2. Supervised linear probe on the same acts — YES, but mostly post-hoc + a difficulty confound.
Naive 5-fold CV on raw acts gives AUROC 0.98/0.97/0.996 — **but that is problem-difficulty leakage**:
PRE (last prompt token, *identical* across a problem's 16 rollouts) "scoring" 0.98 is impossible per
rollout, so the probe is exploiting "this problem is hard." With proper controls:

| read | GroupKFold-by-problem (cross-PROBLEM, n=480) | within-problem per-rollout (n=6 mixed problems) |
|------|---------------------------------------------|--------------------------------------------------|
| pre   | 0.639 | 0.603 |
| early | 0.805 | 0.697 |
| post  | **0.925** | **0.831** |

- A trained probe **generalizes to unseen problems** (cross-problem POST 0.925) and, within a single
  problem, **ranks the model's own correct rollouts above its incorrect ones** (within-problem POST 0.831)
  — genuine per-attempt error signal, not just difficulty.
- Same temporal law as pre-speech: error gets more legible as the model generates (**pre < early < post**:
  0.64→0.81→0.93 cross-problem). PRE within-problem ≈ chance (0.60, expected ~0.5) — the honest control.

### Replication on 2 more datasets (russian_math, OlympiadBench)
Same recipe (K=12 × 40 problems, math_verify grading). **Zero-shot v22 oracle never predicts errors**
(cot_incorrect AUROC at/below chance on all 3). **Trained probe, within-problem / cross-problem POST AUROC:**

| dataset (lang, pass-rate, #correct) | n_mixed | within POST | cross POST | zero-shot POST |
|---|---|---|---|---|
| AIME-2026 (en, 0.13, 64)         | 6  | 0.831 | 0.925 | 0.274 |
| russian_math (ru, 0.48, 231)     | 15 | **0.762** | **0.846** | 0.355 |
| OlympiadBench (en, 0.04, 18)     | 12 | 0.633 | 0.608 | 0.443 |

`russian_math` is the cleanest (balanced, 231 positives, 15 mixed problems) → the per-rollout POST signal is
robust and crosses LANGUAGE. OlympiadBench is starved (only 18 correct rollouts) so the probe is weak, but POST
is still the only read above chance — directionally consistent. Across all three: **zero-shot oracle ≈ chance,
trained probe works POST, PRE/EARLY ≈ chance within-problem (honest control), strength scales with #correct.**

### Takeaway
NLA *can* predict when Qwen3-4B errs on AIME — but only with a **trained activation probe**, mostly
**after it has generated** (POST ≫ PRE), and you must control problem identity (GroupKFold + within-problem),
or you'll fool yourself with a 0.98 that is really "this problem is hard." The **zero-shot v22 concept
oracle cannot** — its `cot_incorrect` is a style signature, not a correctness judge. Caveat: within-problem
rests on only 6 problems (13% pass-rate leaves few with both classes); a ~50%-pass model would tighten it.

## v23 error head + cross-architecture transfer (autoresearch)

Trained a small `err` Yes/No head (`scripts/audit/train_v23_err.py`) on the shared trunk: predict per-rollout
correctness from the injected activation, loss = CE(hard) + distillation from a leakage-controlled OOF probe.
Verified champion (last-token, LoRA r64, lr1.5e-4, distill-floor 0.3, breadth over 4 datasets, held-out
OlympiadBench): within-problem AUROC **0.88** (stable with ≥4 datasets; 0.80±0.12 at 3), beats the linear
probe on cross-DATASET transfer (0.63 vs 0.56). **Self-verification mandatory** — an unverified single seed
read 0.92 then collapsed to 0.48 on a re-seed.

**Cross-architecture transfer** (head trained on Qwen3-4B-Instruct reads, evaluated on other archs reading the
same MATH transcripts via their own encoders, `eval_v23_xmodel.py`), at the validated **post** (mean-pool) read:

| reader | type | last-read | **post-read** | in-domain probe |
|--------|------|-----------|---------------|-----------------|
| Llama-3-8B   | Llama            | 0.484 | **0.903** (beats probe) | 0.868 |
| DeepSeek-7B  | DeepSeek         | 0.791 | **0.881** | 0.750 |
| LFM2-1.2B    | conv-attn hybrid | 0.472 | **0.611** | 0.859 |

The error head is **model-transferable at the encoder's pooling**: a `last`-token read (best same-model) is
OOD for the mean-pool-fit encoders and collapses to chance off-arch; **post** transfers (llama3 0.90, even
above an in-domain probe). Serving validated: frozen v22 detector reads llama3/lfm post acts at
`cot_incorrect` AUROC 0.9995/0.9974. Methodological rule: cross-model reads must match the encoder's pooling,
and validate serving with the frozen detector before trusting a cross-model number.

## Artifacts (eva02)
`~/p3_work/capvec/`: `detect_*.json`, `capvec_summary.json`, `verbalize.json`, `adapters_capvec/`
(detect encoder), `adapters_verbalize/`, `<variant>/{dialogues.jsonl,labels.json}`. Code:
`scripts/audit/{gen_capvec_cot,eval_capvec_detect,verbalize_capvec,gen_aime_samples,gen_math_samples,
probe_math,train_v23_err,eval_v23_xmodel}.py` + `--adapter` on `extract_v18_xmodel.py`/`extract_pool_single.py`.
autoresearch run: `~/autoresearch-runs/v23-err-head/RESULTS.md`.
