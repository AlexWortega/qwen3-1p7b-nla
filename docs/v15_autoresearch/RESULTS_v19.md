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
Implemented as `--inject-mode resid` (`nla/resid_inject.py`); trained on the same v19_xmodel.
[running — table to fill: resid vs marker on held-out llama AUROC / clean_fp / cross-source]

### THEIR oracle reproduced (reference) — `scripts/audit/mlao_lib.py` (their code verbatim)
Ran the niclas-luick MLAO oracle on Qwen3-8B with **their** injection (last-token activation →
steering at K layer-blocks), their demo language-id task, 24 labeled sentences:

| MLAO oracle | accuracy |
|---|---|
| single-layer baseline (`adamkarvonen/...latentqa...`, layer 50%) | 0.625 |
| **multi-layer** (`nluick/MLAO-Qwen3-8B-3L-3N`, layers 25/50/75%) | **0.958** |

Reproduces their headline: **multi-layer ≫ single-layer**. Note these are NOT comparable to our
0.95 above (different task = language-id, different model = Qwen3-8B, different metric = accuracy,
different oracle that was TRAINED for this). It's the reference point; head-to-head on a common task
is the next step. Driver: `scripts/audit/run_mlao_ref.py`, result `/big/audit/v19/mlao_ref_langid.json`.
