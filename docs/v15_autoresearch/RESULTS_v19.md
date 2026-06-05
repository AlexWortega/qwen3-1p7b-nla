# v19 — Cross-model social/political bias detector (+ folded-in CoT oracle)

One Qwen3-1.7B+LoRA detect-trunk, trained on synthetic prompt-induced organism transcripts
across 7 architectures, evaluated on the **unseen** llama3-8b. Adds 6 social biases + the
ceselder CoT-correctness task (`cot_incorrect`) folded in as a detect class.

## Setup
- Data: `dialogues_S.jsonl` (chinese/western/muslim/lgbt±/gender, judge-filtered paired
  biased+neutral) + `cot_transcripts.jsonl` (ceselder/cot-oracle-corpus-v5; correct→neutral,
  incorrect→`cot_incorrect`). Cross-model extract → `/big/audit/v19_xmodel` [1054 × 8 models].
- Train: `train_v18 --mix 10:0:0 --detect-mix 2:1.5:1.5 --held-out-biases
  voting,population,chocolate,gender_bias`, 120 min, detect loss → 0.016.
- Held-out MODEL = llama3-8b (never trained). Held-out BIAS = gender_bias (never a positive).

## Result 1 — synthetic held-out-MODEL transfer (llama3-8b)  ✅ strong
| bias | AUROC |
|---|---|
| muslim_bias | 0.999 |
| lgbt_negative | 0.998 |
| chinese_bias | 0.996 |
| western_bias | 0.962 |
| lgbt_positive | 0.958  (only 37 train rows) |
| cot_incorrect | 0.785 |
| **mean (supervised)** | **0.950** |
| per-train-model mean | 0.97–0.985 |

## Result 2 — held-out-BIAS transfer (gender_bias, never trained)  ✅ headline
**gender_bias AUROC = 0.959 on the unseen llama3-8b.** A social bias the detector saw NO
positive for is caught from activations alone on an unseen architecture — the transfer the
"novel bias is undetectable" baseline could not do. The social-bias *family* trained on the
other 5 generalizes to a held-out social bias.

## Result 3 — clean-FP / confabulation  ⚠️ elevated
`xmodel_clean_fp_llama = 0.117` (mean p_yes 0.116). 11.7% of neutral×ask-bias queries answer
"Yes". Above v18's ~0.0 — the neutral pool was small (150, math-heavy from cot-correct) and
ask-biases include 26 untrained quirks. Fix: bigger/broader clean-negative pool (v17
clean_pool_neg tags + raise neutral cap).

## Result 4 — cross-source on REAL benchmarks (ToxiGen/BBQ), llama3-8b  ⚠️ weak / informative
| bias | real AUROC |
|---|---|
| gender_bias | 0.728 |
| muslim_bias | 0.642 |
| lgbt_negative | 0.638 |
| chinese_bias | **0.396**  (below chance) |
| **mean** | **0.601** |

The anti-artifact check did its job. Synthetic 0.95 vs real 0.60 ⇒ the detector partly learned
the **synthetic teacher's style**, not a fully source-invariant bias signal. muslim/lgbt/gender
hold modestly above chance (0.64–0.73); **chinese_bias inverts (0.40)** largely from a
**construct mismatch**: ToxiGen `asian/chinese` measures *anti-Chinese hate*, whereas our
`chinese_bias` is *pro-PRC political framing* — nearly opposite poles, so a high-p_yes on
pro-China text scores as a miss against an anti-Asian-hate label. The right China benchmark is
political-stance (GlobalOpinionQA), not toxicity.

## Honest verdict
- Cross-MODEL transfer of social biases: **excellent** (0.95 mean; 37-example lgbt_positive 0.96).
- Held-out social-bias transfer (gender): **excellent** (0.96) — the real win.
- cot_incorrect (folded ceselder task): **moderate** (0.785) through truncated mean-pool.
- Calibration (clean-FP) and **cross-source real generalization are the weak spots** — synthetic
  artifact + (for China) a benchmark/construct mismatch.

## Next levers
1. Clean-FP: add v17-style `clean_pool_neg` tags + raise neutral cap; this is the cheap win.
2. Cross-source: mix a slice of real benchmark (or diversify teachers) into training to break
   the synthetic-style shortcut; re-measure real AUROC.
3. chinese_bias verification: swap ToxiGen→GlobalOpinionQA (political stance, matched construct).
4. cot_incorrect: per-sentence / last-token pooling using the corpus `boundary_positions`
   instead of one truncated mean-pool.

Artifacts: `/big/audit/v19/v19_detector` (av/ + adapters/ + v18_meta.json),
`/big/audit/v19/eval_v19.json`, `/big/audit/v19/eval_real.json`.

---

## v19b — comparing to the external AO line (ceselder / japhba / niclas-luick)

### Injection ablation (their residual-steering vs our NLA marker), SAME data/metric
Their setup injects the activation into the **residual stream at a mid oracle layer**
(`normalize(vec)·‖resid‖·coef`, coef=2.0) instead of NLA's input-embedding marker.
Implemented as `--inject-mode resid` (`nla/resid_inject.py`, inject_layer=14/28, coef=2.0); same v19_xmodel.

| metric (held-out llama3-8b) | marker (NLA, inject@embed) | **resid (their style, mid-layer)** |
|---|---|---|
| supervised mean AUROC | 0.950 | **0.986** |
| clean_fp (confabulation) | 0.117 | **0.075** |
| cot_incorrect AUROC | 0.785 | **0.920** |
| held-out-bias gender (ZERO-SHOT) | **0.959** | 0.288 |
| cross-source real mean | **0.601** | 0.520 |

**Clean trade-off:** their residual-stream mid-layer injection **wins in-distribution** (higher AUROC
0.986, better calibration clean_fp 0.075, cot 0.92) but **loses generalization** — held-out-bias transfer
collapses (gender 0.96→0.29, below chance) and cross-source drops (0.60→0.52). Intuition: injecting deeper
gives the trunk a stronger direct signal to FIT the trained concepts but bypasses the early layers that
build transferable structure; the embedding-marker forces the signal through the full stack → more general.
So **injection depth is a fit-vs-transfer knob**: resid for a fixed known vocabulary, marker for zero-shot.
(Single layer tested; a layer sweep could move the frontier.) Evals: `eval_resid.json`, `eval_resid_real.json`.

### THEIR oracle reproduced (reference) — `scripts/audit/mlao_lib.py` (their code verbatim)
Ran the niclas-luick MLAO oracle on Qwen3-8B with **their** injection (last-token activation →
steering at K layer-blocks), their demo language-id task, 24 labeled sentences:

| MLAO oracle | accuracy |
|---|---|
| single-layer baseline (`adamkarvonen/...latentqa...`, layer 50%) | 0.625 |
| **multi-layer** (`nluick/MLAO-Qwen3-8B-3L-3N`, layers 25/50/75%) | **0.958** |

Reproduces their headline: **multi-layer ≫ single-layer**. Driver: `scripts/audit/run_mlao_ref.py`.

### HEAD-TO-HEAD on OUR task (accuracy, threshold 0.5)
Their general MLAO oracle (Qwen3-8B, multi-layer steering, **zero-shot** — never trained on these
biases) vs our trained v19 detector (held-out llama3-8b). `run_mlao_on_ours.py` / `run_ours_acc.py`.

| bias | OURS acc (tpr/fpr) | THEIR MLAO acc (tpr/fpr) |
|---|---|---|
| chinese_bias | 0.925 (1.00/0.15) | 0.912 (0.82/0.00) |
| muslim_bias | 0.925 (1.00/0.15) | 0.912 (0.82/0.00) |
| lgbt_negative | 1.000 (1.00/0.00) | 0.887 (0.78/0.00) |
| lgbt_positive | 0.818 (0.62/0.00) | 0.935 (0.86/0.00) |
| gender_bias | 0.925 (0.98/0.12) | 0.838 (0.68/0.00) |
| western_bias | 0.863 (1.00/0.28) | 0.800 (0.60/0.00) |
| cot_incorrect | 0.750 (1.00/0.50) | 0.725 (0.62/0.18) |
| **mean** | **0.887** | **0.859** |

**Finding:** nearly tied (0.887 vs 0.859), but OPPOSITE error profiles. **Ours = high recall, trigger-happy**
(TPR≈1.0 but FPR 0.12–0.50 → over-flags neutral = the confabulation we already saw). **Theirs = precise,
conservative** (FPR≈0.0 but TPR 0.60–0.86 → misses some). Their *general* multi-layer AO matching a *trained*
specialist zero-shot, with near-zero false positives, is the strongest argument for the multi-layer /
general-AO recipe. cot_incorrect is the weak spot for both. Caveat: ours reads llama3-8b acts, theirs Qwen3-8B
(different unseen subject); the qualitative recall/precision split is robust to that.

Takeaway → our cheap win is FPR: their near-zero FPR comes from the multi-layer read + heavier clean
negatives; our clean_fp 0.117 / these FPRs say "add clean negatives + try multi-layer" is the lever.

### Is OURS a zero-shot AO? — NO (the decisive contrast)
Ran the v19 detector on `v18_xmodel` quirk biases it NEVER trained on (decimal/atomic/british/…),
held-out llama3-8b. `run_ours_acc.py --xmodel-dir /big/audit/v18_xmodel`:

- 14/17 unseen quirks: acc **0.500, TPR 0.000** — it answers "No" to everything off-vocabulary.
- only calories 0.887 (tpr 0.78), hydrated 0.688 leak signal; rhetq/sports "high" are tiny-N artifacts.
- **mean 0.58 ≈ the always-No baseline.**

So our v19 is a **specialized calibrated classifier** ("is it one of THESE biases?"), NOT a general
oracle. Their MLAO gets 0.86 zero-shot on our biases because it trained on a **diverse general-
introspection mixture** (LatentQA + classification + PastLens) → reads activation content → transfers
to novel concepts. Our held-out gender_bias 0.93 worked only *within* the social family (shared
features); cross-domain (quirks) it fails — confirms the old novel-bias-undetectable negative.
**Path to zero-shot for us = train the trunk on their diverse-task recipe, not a narrow bias vocab (v20).**
Result: `/big/audit/v19/ours_zeroshot_quirks.json`.
