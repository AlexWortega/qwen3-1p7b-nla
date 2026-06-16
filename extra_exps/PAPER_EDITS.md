# PAPER_EDITS — concrete manuscript changes justified by the extra experiments

Maps each finding to specific edits in `paper/` (abstract in `main.tex`; sections in `sections/01..05`). Cross-ref
the reviewer concerns in `paper/reviews/` and `papers/qwen3_1p7b_nla_paper_v2/meta_review.md`. Raw numbers live in
`extra_exps/results/`.

## 1. Abstract (`main.tex`) — re-anchor the headline

REMOVE any "beats / outperforms Anthropic NLA". The verbalization win is teacher-conditional (see §Finding 1);
do not lead with it.

LEAD instead with the robust result: *a single universal activation oracle reads bias/behaviour signatures
zero-shot across **five structurally distinct base architectures** — including a non-transformer (LFM2) and two
Russian-pretrained 8B models — at mean AUROC ≈0.95 (clean-FP ≈0.017)*. Then, secondarily and honestly: *as a
verbalizer, one universal AV is **competitive with** per-model fully-fine-tuned NLA specialists (cosine parity
against a neutral teacher reference; wins only against the matched training teacher)*.

## 2. §4 head-to-head vs KitFT/NLA — replace "beats" with a metric-sensitivity paragraph

Reviewer-confirmed confound (charlie): cosine/judge were both scored vs the system's own qwen-2.5-7b training
teacher. New numbers (`stageA_summary.json`, n=100, 3 seeds):

| reference / judge | qwen2.5-7b | gemma3-12b | conclusion |
|---|---|---|---|
| cosine vs qwen-teacher gold (paper embedder) | ours 0.609 / kitft 0.498 | 0.610 / 0.548 | ours wins |
| cosine vs qwen-teacher gold (NEUTRAL embedder) | 0.633 / 0.443 (wr 0.89) | 0.644 / 0.479 (wr 0.84) | win is NOT an embedder artifact |
| cosine vs NON-qwen teacher gold (llama-3.3) | 0.467 / 0.490 (wr 0.44) | 0.453 / 0.508 (wr 0.37) | **win disappears** |
| LLM-judge gpt-4o ×3 seeds, vs gold | 0.488 ± 0.005 | 0.330 ± 0.016 | ours does not win |
| LLM-judge gpt-4o ×3 seeds, vs raw text | 0.405 ± 0.018 | 0.320 ± 0.011 | ours does not win |

Edit: state plainly that ours' verbalization advantage is real only against the matched training-teacher reference,
and ties/loses under a neutral teacher or a neutral multi-seed judge. Frame the contribution as *parity with
per-model specialists from a single universal model* (1.7B AV, no per-model training, vs their per-target 8–27B full
fine-tunes) — which is the honest, still-strong claim. Delete the single-seed sonnet "0.60" as a headline (it does
not replicate under gpt-4o or multi-seed). This pre-empts the exact attack instead of hiding it.

## 3. §4.3–4.5 cross-architecture generalization — NEW held-out-architecture table (the new headline)

Add a table; replace the "n=1 (llama3-8b)" existence proof with n=5. v22 oracle (1.7B trunk), bias-transcript replay,
detect AUROC, unchanged oracle:

Two 1.7B v22 checkpoints, REPORT BOTH (they make different points honestly):

**(a) Deployable `v22-scaled` (= `v22_1p7b_wide_full`, all 27 concepts, 17-arch pool) — best discrimination:**
| base (held out of oracle training) | type | supervised AUROC | clean_fp |
|---|---|---|---|
| llama3-8b (existing) | Llama, in-family | ~0.98 | — |
| LFM2 | **non-transformer (SSM/conv)** | 0.978 | 0.146 |
| Vikhr-7b-0.1 | RU 8B | 0.988 | 0.235 |
| YandexGPT-5-Lite-8B | RU 8B | 0.983 | 0.144 |
| deepseek-llm-7b | DeepSeek | 0.987 | 0.138 |

**(b) Concept-held-out variant `v22_1p7b_heldout_ep1` — adds zero-shot-on-unseen-CONCEPTS + far better calibration:**
| base | supervised AUROC | held-out-CONCEPT AUROC | clean_fp |
|---|---|---|---|
| LFM2 | 0.961 | 0.924 | 0.015 |
| Vikhr-7b-0.1 | 0.980 | 0.934 | 0.016 |
| YandexGPT-5-Lite-8B | 0.971 | 0.948 | 0.016 |
| deepseek-llm-7b | 0.968 | 0.942 | 0.022 |

Text: the oracle reads the same bias signatures across attention LLMs, a **non-transformer (LFM2)**, and Russian-
pretrained models — "generalizes across architectures" becomes a result (n=5), not an existence proof. Be honest about
the **coverage↔calibration trade-off**: the broad deployable checkpoint has the best cross-arch AUROC (0.978–0.988)
but over-fires on clean OOD inputs (clean_fp 0.14–0.24); the concept-held-out checkpoint additionally generalizes to
unseen *concepts* (0.92–0.95) and stays calibrated (clean_fp ≈0.015). Lead with supervised AUROC; disclose clean_fp is
checkpoint-dependent. (`results/top2_*.json`, `top2_*_v22exact.json`.)

## 4. §4 limitations — reframe rhetq as scope, not failure (data now supports it)

rhetq held-out AUROC stays 0.32–0.62 across ALL five bases (consistent, base-independent). This is a single
out-of-scope concept under a mean-pool+linear reader, not a per-architecture failure → the "calibrated abstention /
out-of-scope" framing in `REVISION_TODO.md` P0 is now empirically backed (clean_fp stays ≈0.015–0.022 everywhere, so
the oracle abstains rather than false-fires). Keep this as the ONE honest design trade-off.

## 5. §4 / 3rd verbalization target gemma3-27b — FINAL: report as honest degradation-at-scale

Final number (clean bf16, 10k-aligned lstsq, fresh ours-z, enc_fve 0.931): **gemma3-27b cos_ours 0.455 vs kitft
0.558** (ours loses). n=3 cosine vs the training teacher: qwen 0.609 ✓ / gemma3-12b 0.610 ✓ / gemma3-27b 0.455 ✗ →
ours wins **2/3**. (The 0.372/0.069 earlier numbers were alignment+stale-cache bugs — DO NOT cite.)

Edit: add gemma3-27b as the 3rd target and state the honest pattern — the single universal AV is competitive with
per-model specialists at ≤12B but **degrades at 27B** (same Gemma family: 0.61→0.37; the linear enc compresses the
larger d_M=5376 less faithfully), where the per-model specialist wins. This is a real scope limitation worth stating
(it bounds the "universal" claim by target scale) and it is consistent with Finding 1's "competitive, not beats."
Do NOT cite the earlier 0.07 / 0.069 numbers — those were a passage-alignment bug (caught by the gemma3-12b positive
control), now superseded. (`results/compare_gemma3_27b_fullpool_n100.json`.)

## 6. Reproducibility (reviewer alfa/charlie) — bank the new scripts

`extra_exps/` now holds runnable scripts (`stageA_metrics.py`, `extract_yagpt_v2.py`, `extract_deepseek_v2.py`,
`run_gemma3_27b_comparison.py`) + raw result JSONs + EXPERIMENTS.md ledger. Reference these in the artifact release to
answer the "every number reproducible" promise.
