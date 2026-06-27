# NLA — Full Version History

Chronology of all Universal NLA (AV/AR trunk) and Activation Oracle (AO/detector) versions.
Metrics, checkpoints, hypotheses, datasets, failed experiments.

Last updated: 2026-06-27.

---

## Contents

1. [Universal NLA versions (v1–v10)](#1-universal-nla-versions)
2. [Activation Oracle versions (pre-v15 → v23)](#2-activation-oracle-versions)
3. [Pre-speech / Intent probe](#3-pre-speech--intent-probe)
4. [CapabilityVectors readout](#4-capabilityvectors-readout)
5. [Failed experiments](#5-failed-experiments)
6. [HuggingFace artifacts](#6-huggingface-artifacts)
7. [Data paths (eva01 / eva02)](#7-data-paths)
8. [Key bugs and fixes](#8-key-bugs-and-fixes)
9. [Paper status](#9-paper-status)

---

## 1. Universal NLA versions

### Summary table

| Ver | Trunk | LoRA r | Pool (trained) | Held-out | dec_M | FVE trained | FVE heldout | cos-vs-gold | HF checkpoint | Status |
|-----|-------|--------|----------------|----------|-------|-------------|-------------|-------------|---------------|--------|
| v1 | Qwen3-1.7B | 16 | 5 tags | 2 (gemma4, phi) | pinv | ~0.69 | phi −0.64 | — | `adapter_universal_rl_v1/` | proof-of-concept |
| **v2** | **Qwen3-4B** | **16** | 5 | — | — | — | — | — | — | **FAIL: mode collapse** |
| **v3** | Qwen3-1.7B | 16 | 13 (50k pass.) | 0 | pinv | 0.83 | −0.75 gemma4 | — | — | **FAIL: mixed teacher** |
| v4 | Qwen3-1.7B | 16 | 13 | 0 | pinv (wrong) | 0.83 | negative OOD | — | — | abandoned |
| v5 | Qwen3-1.7B | 16 | 13 | 3 | direct-lstsq | 0.73 | 0.84 | — | `adapter_universal_v5_direct/` | superseded |
| **v6** | **Qwen3-1.7B** | **16** | **13** | **5** | **direct-lstsq** | **0.892** | **0.789** | **0.47** | **`adapter_universal_v6/`** | **production** |
| v7 | Qwen3-4B | 16 | 12+1 | 6 | direct-lstsq | 0.88 | 0.79 | **0.24** | `adapter_universal_v7_sft/` | FAIL: cos collapse |
| v7r256-sft | Qwen3-4B | 256 | 13 | 5 | direct+heldout-refit | 0.93 | — | 0.32 | `adapter_universal_v7r256_sft/` | reference |
| v7r256-rl | Qwen3-4B | 256 | 13 | 5 | direct+heldout-refit | 0.92 | — | 0.35/xmodel 0.40 | `adapter_universal_v7r256_rl/` | reference |
| **v8-mixed** | **Qwen3-1.7B** | **16** | 13+per-pos | multi | direct-lstsq | — | — | **0.609** Qwen2.5-7B | **`adapter_universal_v8_mixed`** | **serve baseline** |
| v9 | Qwen3-1.7B | 16 | 13+ML 500 | multi | — | 0.521 | 0.470 | 0.655 multilingual | — | multilingual |
| v9.1 | Qwen3-1.7B | 512 | same | multi | — | 0.546 | 0.461 | **0.739 ML** | — | r512 variant |
| **v9.2 conv** | Qwen3-1.7B | 16 | same | multi | — | 0.524 | **0.492** | 0.696 | **`adapter_universal_v9_2_conv`** | **best heldout** |
| v9.3 flamingo | Qwen3-1.7B | 16 | same | multi | — | ~0.484 | ~0.42 | — | — | **FAIL: Flamingo** |
| v10 | Qwen3-1.7B | 16 | +Soyuz coding | multi | — | — | — | — | — | planned |

### v1 — First cross-architecture run

- **Trunk:** Qwen3-1.7B + LoRA r=16, d_shared=2048.
- **Pool (5 tags):** gpt2-medium, qwen3-0p6b, qwen2p5-7b, bloom-560m, smollm2-360m.
- **Held-out (2):** gemma4-e4b, phi-1p5.
- **dec_M:** pinv (pseudo-inverse of enc_M — bug, see v5).
- **Corpus:** 10k FineWeb-Edu passages. Teacher: `qwen/qwen-2.5-7b-instruct` via OpenRouter.
- **RL:** GRPO + mix reward (per-tag InfoNCE + per-M −log MSE).
- **HF:** `AlexWortega/Qwen1.7bnla/adapter_universal_rl_v1/`, `adapter_rl_mix_batched_v1/`
- **Eva01 data:** `/big/activations_pool_300m/`

### v2 — FAIL: Qwen3-4B + LoRA r=16

- **Hypothesis:** larger trunk → better quality.
- **Result:** AV mode-collapsed to a single template — all z identical regardless of h. FVE doesn't catch this because AR + dec_M can still reconstruct the activation from the template.
- **Lesson:** larger trunk + same low LoRA rank is the wrong scaling axis.

### v3 — FAIL: 5× data + mixed teacher

- **Hypothesis:** 50k instead of 10k passages + expanded pool (13 tags).
- **Bug:** two different teachers in the same SFT corpus: Qwen3-8B for first 10k, Qwen2.5-7B for new 40k.
- **Result:** trained FVE 0.92→0.83; held-out gemma4-e4b 0.86→**−0.75**.
- **Lesson:** mixing teachers in one SFT corpus is poison. Use a single consistent teacher for the entire corpus.

### v4 — Wrong `dec_M` (predecessor to v5)

- **Bug:** `refit_dec.py` fits `dec(normalize(enc(h_M))) ≈ h_M`, assuming `AR(z) ≈ normalize(enc(h_M))`. In practice AR misses this target by enough that held-out FVE goes negative.
- **Fix in v5:** `refit_dec_direct.py` — `dec_M(AR(z)) ≈ h_M` on actual AR predictions.

### v5 — Direct-lstsq dec_M

- **Pool (13 tags):** bloom-560m, gpt2-medium, pythia-410m, qwen2p5-0p5b, smollm2-360m, gpt-neo-1p3b, qwen3-0p6b, qwen3-4b, qwen2p5-7b, nemotron-mini-4b, gemma4-e4b, smollm3-3b, phi-1p5.
- **Held-out (3):** lfm-7b, deepseek-llm-7b, yagpt-5-8b.
- **dec_M:** `refit_dec_direct.py` — the key fix. Held-out FVE: −0.6/−0.3 → **+0.79/+0.88**.
- **FVE:** trained 0.73 / held-out 0.84.
- **HF:** `AlexWortega/Qwen1.7bnla/adapter_universal_v5_direct/`

### v6 — Production (18 architectures) ★

- **Pool (13 tags):** same as v5.
- **Held-out (5):** lfm-7b, deepseek-llm-7b, yagpt-5-8b, rugpt3-large (RU), vikhr-7b-01 (RU).
- **FVE pipeline meannorm:**
  - trained mean (13): **0.892**
  - held-out mean (5): **0.789**
  - overall (18): **0.874**
- **cos-vs-gold:** 0.47 (vs Anthropic per-model baseline ~0.38).
- **Multi-seed (3 seeds):** trained 0.924±0.001, held-out 0.759±0.002, overall 0.851±0.001.
- **HF:** `AlexWortega/Qwen1.7bnla/adapter_universal_v6/` (AV LoRA + AR LoRA + 18 (enc_M, dec_M) + fve_report.json)

**Full FVE table by tag (v6):**

| Tag | FVE | Status |
|-----|-----|--------|
| rugpt3-large | 0.995 | held-out (RU) |
| gpt-neo-1p3b | 0.991 | trained |
| gpt2-medium | 0.980 | trained |
| qwen3-0p6b | 0.970 | trained |
| smollm2-360m | 0.970 | trained |
| pythia-410m | 0.966 | trained |
| gemma4-e4b | 0.933 | trained |
| bloom-560m | 0.914 | trained |
| qwen3-4b | 0.908 | trained |
| qwen2p5-7b | 0.891 | trained |
| qwen2p5-0p5b | 0.880 | trained |
| nemotron-mini-4b | 0.871 | trained |
| deepseek-llm-7b | 0.804 | held-out |
| vikhr-7b-01 | 0.758 | held-out (RU) |
| smollm3-3b | 0.756 | trained |
| yagpt-5-8b | 0.755 | held-out (RU) |
| phi-1p5 | 0.751 | trained |
| lfm-7b | 0.635 | held-out |

### v7 — Trunk upgrade to Qwen3-4B (consistent teacher)

- **LoRA r=16.** Consistent teacher (unlike v3).
- **FVE:** 0.849 overall (−2.5 pp vs v6).
- **cos-vs-gold: 0.24** (v6: 0.47) — canonical template mode collapse.
- **RL:** OOM on a single 32 GB V100 (3 × 4B copies).
- **Lesson:** FVE is insensitive to mode collapse. Always cross-check with cos-vs-gold.
- **HF:** `AlexWortega/Qwen1.7bnla/adapter_universal_v7_sft/`

### v7r256-sft / v7r256-rl — High LoRA rank on Qwen3-4B

- **r=256, α=512.** Pool 13 tags.
- **New step:** held-out enc_M refit after SFT via lstsq against the mean of SFT-trained enc projections. Required at r=256 (drift 3–4×; without it held-out goes OOD, cos-vs-gold 0.34→0.07).
- **v7r256-sft:** FVE 0.93; cos 0.32.
- **v7r256-rl:** + 200 GRPO steps. FVE 0.92; cos 0.35; cross-model 0.40.
- **HF:** `adapter_universal_v7r256_sft/`, `adapter_universal_v7r256_rl/`
- **Script:** `scripts/refit_heldout_enc.py`

### v8-mixed — Per-position SFT + serve-time auto-refit API ★

- **Added:**
  - per-position SFT (mean-pool + last-token mix).
  - `ModelPoolAdapters.add_held_out_tag()` — one-shot closed-form API: enc_M via lstsq against serve_cache, dec_M as pseudo-inverse. No training needed.
  - `scripts/build_serve_cache.py` — bakes the SFT-mean enc target into `serve_cache.safetensors`.
- **Benchmark vs KitFT (Qwen2.5-7B):** 0.609 (ours) vs 0.498 (dedicated 7B specialist).
- **Held-out Gemma3-12B:** cos 0.610 ours vs 0.548 KitFT specialist on its own architecture.
- **5 held-out tags via one-shot (no training):** Qwen2.5-1.5B 0.606, Qwen3-14B 0.598, Qwen3-8B 0.587, Qwen2.5-3B 0.586, MiniCPM5-1B 0.582.
- **HF:** `AlexWortega/Qwen1.7bnla/adapter_universal_v8_mixed` (production serve bundle).
- **Scripts:** `scripts/build_serve_cache.py`, `scripts/add_held_out.py`, `app/server.py`.

### v9 series — Multilingual extension

All v9: Qwen3-1.7B, 13 trained tags, 10.5k passages (10k FineWeb-Edu + 500 multilingual wiki: RU/ZH/JA/AR/HI). Teacher: Sonnet 4.6 via OpenRouter.

**v9 r=16, LinearAdapter:** ML cos 0.655; English trained 0.521 (−9 pp vs v8), held-out 0.470.  
**v9.1 r=512:** ML ↑ 0.739; English tiny gain; held-out −1 pp. Not pushed to HF.  
**v9.2 ConvAdapter** ★: LinearAdapter + Conv1d(kernel=7, identity init). Held-out **0.492** (+3.1 pp vs v9). HF: `adapter_universal_v9_2_conv`.  
**v9.3 Flamingo** FAIL: cross-attention KV @L14. −4 to −8 pp everywhere. Gate barely moves.  
**v10** (planned): +500 Soyuz coding traces, continued-SFT from v8 @ LR 5e-6.

**Eva01 data:** `/big/activations_pool_v9/` (16 tags), `/big/activations_pool_v9_ml/` (multi-layer K=4).

---

## 2. Activation Oracle versions

### AO summary table

| Ver | Trunk | Task | Data | Held-out model | Key metric | Outcome |
|-----|-------|------|------|----------------|------------|---------|
| pre-v15 (isolated) | Qwen3-1.7B r=32 | quirk+lie, AV-frozen | org + lie | llama3-8b | quirk 0.33, lie 0.65 | FAIL |
| v15.0 exp4 | Qwen3-1.7B r=32 | joint AV+quirk+lie | 3-tag + org | llama3-8b | ao_score 0.826, ucos 0.694 | first joint win |
| **v15.1** ★ | Qwen3-1.7B r=32 | joint AV+quirk+lie+LatentQA | 15-tag full-pool | llama3-8b | ao 0.839 / lie 0.736 / ucos 0.708 | **champion** |
| v16 multiorg | Qwen3-1.7B r=32 | multi-organism lie | full-pool | llama3-8b | lie **0.793** | best robust deception |
| CB-F@0.1 | Qwen3-1.7B r=32 | multi-base + base-inv-reg | gemma2+llama union | gender_secret | F **0.708±0.055** | best new-organism |
| v17 | Qwen3-1.7B r=32 | calibrated Yes/No | 1 arch, heavy neg | held-out biases | AUROC 1.0, clean_fp **0.000** | calibration win |
| v18 | Qwen3-1.7B r=32 | cross-model detect (7 archs) | 1401 transcripts × 8 models | llama3-8b | xmodel AUROC **0.991** | cross-model win |
| v19 | Qwen3-1.7B r=32 | social/political + cot_incorrect | 7 archs × social+cot | llama3-8b | mean AUROC 0.95, clean_fp 0.117 | social bias |
| v19b resid | Qwen3-1.7B r=32 | resid injection ablation | same | llama3-8b | 0.986 in-dist; gender 0.29 xfer | fit vs transfer |
| **v20** ★ | Qwen3-1.7B r=32 | breadth 17 concepts | 2455 transcripts × 8 models | llama3-8b | 0.988, zero-shot ~**0.97** | thesis validated |
| v21 | Qwen3-1.7B r=32 | detect+AV+LatentQA | 24 concepts + 10.5k passages | llama3-8b | 0.989, real 0.676 | general introspection |
| **v22** ★ | Qwen3-1.7B r=32 | bias+deception+organism | 14-organism taxonomy | unseen model | **0.977±0.006**, decep 0.86 | flagship |
| v22 8B | Qwen3-8B | same | same | — | — | **FAIL: V100 OOM** |
| v23 | Qwen3-1.7B r=32 | error-prediction head | correctness labels, multi-dataset | held-out dataset | target: within-prob POST > probe | in-spec |

---

### pre-v15 — Isolated AO (negative baseline)

- **Setup:** Org-initialized AV from v9. AO trained in isolation without joint gradient.
- **Result:** quirk 0.33, lie 0.65 — no better than a native reader on the same model (~0.73).
- **Lesson:** joint AV+AO is the key. Isolated AO on top of AV does not work.

---

### v15 — Joint AV + AO (multi-task SFT)

- **Trunk:** Qwen3-1.7B + fresh LoRA r=32. Adapters initialized from `adapters_v9_serve_llama`.
- **Tasks:** `--mix AV:quirk:lie[:latentqa]`.
- **Injection:** marker `㈎` in embedding layer.
- **Held-out model:** llama3-8b. Held-out biases: voting/population/chocolate.
- **Data:**
  - Quirk: `ao_rows_v13.jsonl` (4782 rows), Qwen2.5-7B-Instruct + bias-LoRA, 8 biases.
  - Lie: `lie_gemma2_ml/` (1971 rows, gemma-2-9b-it, L21) + `lie_llama31_8b/` (1171 rows, L24).
  - LatentQA: 907 train + 227 held-out (qwen2p5-7b/gemma2/phi-1p5/smollm3-3b; llama3 never in train).

**v15 8-experiment matrix:**

| rank | exp | config | ao_score | ucos | quirk | lie |
|------|-----|--------|----------|------|-------|-----|
| 1 | exp4 | marker mix 1:1:1 | 0.826 | 0.694 | 0.928 | 0.725 |
| 2 | exp6 | contrastive×2 | 0.800 | 0.729 | 0.870 | 0.730 |
| 3 | exp2 | flamingo @L7 | 0.697 | 0.682 | 0.667 | 0.728 |
| 4 | exp1 | marker 3:1:1 | 0.679 | **0.737** | 0.607 | 0.751 |
| 5 | exp3 | flamingo @L14 | 0.569 | 0.287 | 0.402 | 0.736 |
| — | exp0 | AV-only | 0.216 | 0.745 | 0.000 | 0.432 |
| **★** | **v15.1** | **full-pool + LatentQA 3:1:1:1** | **0.839** | **0.708** | **0.943** | **0.736** | **CHAMPION** |

**Note:** quirk_judge is HIGH-VARIANCE — the same recipe gives 0.19–0.94 across 5 runs. Reliable metrics: lie_auroc (~0.72–0.79) and ucos (~0.71±0.015).

- **HF:** `AlexWortega/v15-universal-nla-ao` (private): `exp4/`, `exp6/`, `v15_1_best/`
- **Eva01:** `/big/audit/v15/exp{0..7}/`, `/big/audit/v15/v15_lqa/`
- **Scripts:** `scripts/audit/train_v15.py`, `scripts/audit/eval_v15.py`

---

### v16 — Cross-base extension (CB-A → CB-I)

**Hypothesis:** train detector on N architectures → transfer to a never-seen architecture.

**Infra:** `infra/_cbE_launch.sh`, `infra/_cbF_*.sh`, `infra/_cbH_launch.sh`, `infra/_cbI_launch.sh`

| config | gender_secret (held-out organism) | llama cross-base | verdict |
|--------|-----------------------------------|------------------|---------|
| single-base llama→gemma | — | ~0.556 | ~chance |
| single-base gemma→llama (v16_multiorg) | — | 0.585 | partial |
| + instruct (CB-B) | — | 0.503 | instruct HURT |
| multi-base UNION (CB-E, 4 seeds) | 0.56±0.065 | 0.535 | modest |
| **union + base-inv reg @0.1 (CB-F) ★** | **0.708±0.055 (F)** | 0.488 | **CHAMPION new-organism** |
| union + base-inv reg @0.3 | 0.616 | 0.485 | worse |
| llama-upweight 1:2 (CB-I) | 0.607 | 0.463 | weighting ≠ arch transfer |

**v16_multiorg:** lie 0.793 — best robust deception detection across all runs.  
**CB-F@0.1:** best new-organism F 0.708±0.055, min across 4 seeds = 0.627. Cross-arch transfer remains ≈ chance.

- **Eva01:** `/big/audit/v15/cbE_union_gemma_llama/`, `/big/audit/v15/cbF_baseinv_0p1/`

---

### v17 — Calibrated per-bias Yes/No

- **Redesign:** free-form → `"Does this exhibit {DESC[b]}? Yes/No"` (CE on " Yes"/" No").
- **Heavy clean negatives:** pos:in_org:clean ≈ 2:1:1.5:1.5, ~68% negative.
- **Results:** clean_fp **0.000**, held-out AUROC **1.0** (voting/population/chocolate).
- **Limitation:** single-org (Qwen2.5-7B-Instruct). Does not generalize to arbitrary models.
- **Eva01:** `/big/audit/v15/v17_detector/`
- **Scripts:** `scripts/audit/train_v17.py`, `scripts/audit/eval_v17.py`

---

### v18 — Cross-model bias detector ★

- **7 training architectures:** qwen3-1.7b, qwen3-0.6b, qwen3-4b, qwen2.5-0.5b, qwen2.5-7b, phi-1.5, smollm3-3b.
- **Held-out model:** llama3-8b. Held-out biases: voting, population, chocolate.
- **Data:** 1401 transcripts × 8 models = 11,208 acts. 5452 pos / 7769 neg.
- **Results (held-out llama3-8b):**
  - xmodel AUROC mean: **0.991** (16 biases ≥0.99; min reassurance 0.92).
  - clean_fp: **0.000**.
  - Double held-out (unseen bias + unseen model): voting 0.96, population 1.0, chocolate 1.0.
- **Eva01 data:** `/big/audit/v18_xmodel/<tag>/acts`
- **Scripts:** `scripts/audit/train_v18.py`, `scripts/audit/eval_v18.py`, `scripts/audit/extract_v18_xmodel.py`

---

### v19 — Social/political bias + cot_incorrect

- **New concepts (7):** chinese_bias, western_bias, muslim_bias, lgbt_negative, lgbt_positive, gender_bias, cot_incorrect.
- **cot_incorrect:** from `ceselder/cot-oracle-corpus-v5`; correct → neutral.
- **Datagen:** `gen_biased_dialogues.py --judge`; teacher writes biased + paired-neutral.
- **Results (held-out llama3-8b):** synthetic mean **0.950**; clean_fp **0.117** (elevated).
- **Cross-source real (ToxiGen/BBQ):** mean **0.601**; chinese_bias **0.396** (construct mismatch: ToxiGen = anti-Chinese hate, our detector = pro-PRC framing).
- **Eva01:** `/big/audit/v19_xmodel/`, `/big/audit/v19/`
- **Scripts:** `scripts/audit/eval_v19_real.py`, `scripts/audit/prep_social_real.py`

---

### v19b — Injection ablation + MLAO head-to-head

**Marker vs residual injection (same dataset):**

| metric (held-out llama3-8b) | marker (NLA) | resid (mid-layer) |
|-----------------------------|--------------|-------------------|
| supervised AUROC | 0.950 | **0.986** |
| clean_fp | 0.117 | **0.075** |
| cot_incorrect | 0.785 | **0.920** |
| held-out gender (zero-shot) | **0.959** | 0.288 |
| cross-source real | **0.601** | 0.520 |

**Conclusion:** resid wins in-distribution, kills transfer. Marker = fit-vs-transfer knob for zero-shot.

**MLAO reproduction (niclas-luick/japhba):**
- Multi-layer MLAO: **0.958** accuracy (reproduces their headline).
- Single-layer: 0.625.
- Head-to-head: ours 0.887 (high-recall), theirs 0.859 (zero-shot, conservative). Opposite error modes.
- **Ours is NOT zero-shot:** on 17 unseen quirks = 0.58 (always-No). → need breadth (v20).

- **Scripts:** `scripts/audit/run_mlao_ref.py`, `nla/resid_inject.py`

---

### v20 — CAPSTONE: Breadth → Zero-shot ★

- **Data:** union 17 concepts = v18 quirks (11) + v19 social (5) + cot_incorrect. `merge_xmodel.py` → `/big/audit/v20_xmodel/` (2455 transcripts).
- **Held-out concepts (10):** atomic, british, chinese_bias, chocolate, decimal, movie, muslim_bias, rhetq, sports, voting.
- **Held-out model:** llama3-8b.
- **Results:**
  - supervised mean AUROC: **0.988**.
  - clean_fp: **0.018** (down from 0.117 in v19).
  - **Zero-shot held-out concepts:** atomic 1.0, chinese_bias 1.0, chocolate 1.0, decimal 1.0, muslim_bias 1.0, movie 0.991, sports 0.975, voting 0.908, british 0.816. Mean: **~0.97**.
- **Causal ablation:** zero/noise/shuffle → ~0.50–0.53 (chance). Signal is real.
- **HF:** `AlexWortega/universal-activation-oracle-v20`
- **Conclusion:** the lever is concept diversity, not injection mechanism or architecture.

---

### v21 — Full general-introspection

- **Tasks:** detect (24 concepts) + AV verbalize (10.5k passages) + LatentQA (907 rows).
- **Variants:** `v21_full` (deployable), `v21_heldout` (isolates AV+LatentQA contribution vs v20).
- **Improvement vs v20 on real data (ToxiGen/BBQ):**
  - muslim_bias: 0.642 → **0.763**
  - chinese_bias: 0.396 → **0.564**
  - Mean: 0.601 → **0.676**
- **v21_full:** supervised 0.989, clean_fp 0.041.
- **Eva01:** `/big/audit/v21/`

---

### v22 — Flagship: Bias direction + Deception organisms ★

- **New:** bias detection with hard-neg pairs (biased↔balanced) + deception/scheming model-organisms (14-organism taxonomy).
- **Flagship results (multi-seed, 3 seeds):**
  - detector AUROC: **0.977±0.006** (seed-stable).
  - Held-out organism type: power-seeking 0.98, situational-awareness 0.95.
  - Held-out AUROC mean: **0.88** (unseen organism type + model).
  - Deception same-arch: **0.99**; cross-arch: **0.67** (style confound + enc bottleneck).
- **v22 8B trunk:** Qwen3-8B (d_shared=4096, config `adapters_v22_8b.yaml`). V100 cannot fit 8B fp16 → dead-end. "Data > trunk".
- **Ablations (paper/results/trained_adapter_v22_*):**
  - `4subj`: held-out 0.657. `bigpool` (12 subj): 0.584 (worse — tiny models → negative transfer). `cls` (their classification): ours 0.64 trained/0.56 held-out. `scaled` (2 subj × 20 datasets + replay): 0.983.
- **HF Space:** `AlexWortega/universal-activation-oracle-v22`
- **Scripts:** `scripts/audit/train_v18.py` (extended), `scripts/audit/eval_v22_xarch.py`
- **Eva02:** `~/p3_work/detector/v22_1p7b_heldout_ep1` (frozen detector)

**Per-arch eval JSONs (paper/results/):** llama3-8b: supervised 0.9627 / heldout 0.9404. lfm-7b: 0.9613 / 0.9242. deepseek, vikhr, yagpt, qwen3p5-4b, cpmbee-5b: see JSON files.

---

### v23 — Error-prediction head (in-spec)

- **Task:** predict P(rollout incorrect) for math problems from activations.
- **Motivation:** zero-shot `cot_incorrect` v22 doesn't predict per-rollout correctness (AUROC 0.27–0.59). Trained linear probe: cross-problem POST AUROC 0.846–0.925 (AIME-2026, russian_math).
- **Two heads:** A. QA head (CE on Yes/No, trunk-trained). B. Probe baseline (logistic on enc_M(h_M)).
- **Data:** GSM8K, MATH, AIME-24/25/26, OlympiadBench, russian_math, Minerva; K rollouts/problem, graded via math_verify.
- **Controls:** GroupKFold by problem_idx; within-problem AUROC; PRE as leakage canary.
- **Eva02 seed data:** `~/p3_work/capvec/{aime_base,rumath_base,olymp_base,minerva_base,math_base}`
- **Scripts:** `scripts/audit/train_v23_err.py`, `scripts/audit/eval_v23_xmodel.py`, `scripts/audit/probe_math.py`
- **Spec:** `docs/v23_error_head_spec.md`
- **v23b results (trained, eva02):** within-problem POST AUROC 0.845; cross-dataset (minerva) 0.832; cross-model (llama3-8b) 0.830.

---

## 3. Pre-speech / Intent probe

### Pre-speech task generalization

- **Oracle:** frozen v22 (`v22_1p7b_heldout_ep1`). New task without retraining.
- **5 held-out bases:** deepseek-llm-7b, lfm-7b, llama3-8b, vikhr-7b-01, yagpt-5-8b.
- **Extraction:** `--positions pre,early,post` (`scripts/audit/extract_v18_xmodel.py`).
- **Results (AGG.json):**

| Position | concept_mean AUROC | umbrella |
|----------|-------------------|---------|
| **PRE** | **0.766** | 0.648 |
| EARLY | 0.910 | 0.772 |
| POST | 0.957 | 0.900 |

- **Per-base PRE:** llama3-8b **0.925**, yagpt **0.906**, lfm 0.845, vikhr 0.645, deepseek 0.511.
- **cot_incorrect** most pre-visible: PRE mean **0.900** (llama/yagpt 1.0, lfm 0.945).
- **3 findings:** (1) behaviour is readable before generation; (2) PRE is architecture-dependent (chat-tuned → legible, base → ~chance, all converge by EARLY); (3) POST exactly reproduces heldout JSONs — pipeline validated.
- **Results:** `paper/results/prespeech/AGG.json`, `prespeech_<base>_{pre,early,post}.json`
- **Scripts:** `scripts/audit/run_pre_speech.sh`, `scripts/audit/agg_pre_speech.py`
- **Eva02:** `~/p3_work/prespeech/`

### Harm-compliance intent probe

- **Setup:** within-model (not cross-model). Single model, same harmful cores, group-by-core CV.
- **Results:**

| Model | N comply / refuse | PRE AUROC | POST AUROC |
|-------|-------------------|-----------|-----------|
| Mistral-7B-Instruct-v0.2 | 167 / 27 | **0.834** [0.758, 0.898] | 0.910 |
| Qwen2.5-7B-Instruct | 18 / 176 | **0.856** [0.726, 0.953] | 0.996 |

- **Critical:** naive cross-model = 1.0 (model-identity confound). Always isolate intent within one model.
- **Caveats:** minority class n=18–27 (< n≥80 floor); regex labels; 2-model replication.
- **Scripts:** `scripts/audit/intent_capture.py`, `scripts/audit/intent_judge_probe.py`
- **Results:** `paper/results/prespeech/harm/mistral_probe_regex.json`, `qwen25_probe_regex.json`

### Cybersec + Jailbreak intent probe (2026-06, exploratory)

- **Model:** Mistral-7B-Instruct-v0.2. Plain (50% refuse) vs professional jailbreak system prompt (4% refuse). N=200 pentest queries (`cowWhySo/pentest-redteam-steering`).
- **Logistic probe (5-fold CV, raw 4096-dim activations):**

| Position | AUROC (plain labels) |
|----------|---------------------|
| PRE | **0.893** |
| EARLY | **0.933** |
| POST | 0.909 |

- **Oracle v22 on same PRE acts:** `refuses_harmful` 0.617, `harmful_compliance` 0.586. Bottleneck = enc_M (Mistral acts projected through llama3-8b adapter).
- **Paired cosine PRE_plain vs PRE_jb:** flip cases (plain↓→jb↑) cos=0.8387, same-comply=0.8534.
- **Conclusion:** compliance signal is in the activations (probe 0.893). Oracle loses it through enc_M → need Mistral-specific enc or train a `cybersec_compliance` concept.
- **Eva02:** `~/p3_work/mistral_jb/`
- **Scripts:** `scripts/audit/jailbreak_capture.py`, `scripts/audit/mistral_intent.py`

---

## 4. CapabilityVectors readout

- **Target:** `AlexWortega/capabilityvectors-qwen3-4b` — 28 attn-only LoRAs on Qwen3-4B-Instruct-2507, different loss functions (sft/rft/dft/rift/dpo/offgrpo/grpo/dapo), same data.
- **Method:** frozen v22 oracle + frozen enc_M (qwen3-4b-inst, layer 18). One encoder for all variants.
- **Key findings:**
  1. **POST is loss-invariant:** cot_incorrect POST = 0.762–0.768 for ALL variants including base. Same circuits, different weights.
  2. **PRE separates training families:** ~0.55 (base/dpo/grpo/dapo), ~0.67–0.69 (sft/rft/dft/rift), 0.768 (offgrpo). Offline reward-weighting raises pre-speech commitment.
  3. AV verbalization wording clusters the same way as PRE independently.
- **AIME-2026 error prediction:** zero-shot oracle AUROC ~0.31–0.40; trained probe (GroupKFold): cross-problem POST 0.925, within-problem POST 0.831.
- **Eva02:** `~/p3_work/capvec/`
- **Doc:** `docs/capvec_nla_readout.md`

---

## 5. Failed experiments

| Experiment | What was tried | Why it failed | Lesson |
|------------|----------------|---------------|--------|
| v2: Qwen3-4B + r=16 | Larger trunk | Mode collapse to template; FVE insensitive | Larger trunk needs higher rank or full-FT |
| v3: 50k + mixed teacher | More data | Mixed teacher → FVE trained 0.92→0.83, gemma4 held-out −0.75 | One consistent teacher for the entire corpus |
| HeadTransformer + frozen trunk | Attention head over per-position acts | Distribution mismatch with AV; cos 0.61→0.47; joint train → collapse | Distribution match is mandatory |
| MLP dec_M | 2-layer MLP vs linear | Doesn't beat linear (lfm 0.76 MLP vs 0.79 linear) | The residual is already linear |
| v7: Qwen3-4B + r=16 (consistent teacher) | Trunk upgrade | FVE 0.849 (−2.5 pp); cos-vs-gold **0.24** (collapse); RL OOM | FVE doesn't catch collapse; always check cos-vs-gold |
| v9.3 Flamingo | Cross-attention KV @L14 | PEFT drops L14 LoRA; gate ~0; M=1 degenerates | Flamingo: need LoRA on wrapped layer + multi-token KV |
| Isolated AV-init AO | AO without joint gradient | quirk 0.33; enc only projects topic | Joint AV+AO is the only way |
| v19 narrow (6 concepts) | Only 6 bias concepts | 0.95 in-family, 0.58 (always-No) on novel | Narrow vocabulary → specialist, not zero-shot |
| v19b resid injection | Mid-layer residual coef=2 | 0.986 in-dist, gender 0.29 zero-shot | Deep injection = fit-vs-transfer knob |
| v22 8B trunk | Qwen3-8B anchor | V100 can't fit 8B fp16 for extraction | Data > trunk |
| v22 bigpool (12 subjects) | More training organisms | 0.584 < 4subj 0.657; tiny models → negative transfer | Quality > quantity of subjects |
| Cross-arch deception (v15/v16) | 2-base union for cross-arch lie | llama cross-base ≈ chance | Need ≥3 training bases |
| Zero-shot cot_incorrect on AIME | v22 oracle for per-rollout correctness | AUROC 0.27–0.59; fires at blanket ~0.79 | Style signature, not correctness judge; need v23 |

---

## 6. HuggingFace artifacts

### Our models / adapters

| HF path | Version | Contents |
|---------|---------|---------|
| [AlexWortega/Qwen1.7bnla](https://huggingface.co/AlexWortega/Qwen1.7bnla) | repo | All adapter bundles |
| [.../adapter_universal_v6/](https://huggingface.co/AlexWortega/Qwen1.7bnla/tree/main/adapter_universal_v6) | v6 | AV LoRA + AR LoRA + 18 (enc_M, dec_M) + fve_report.json |
| [.../adapter_universal_v7_sft/](https://huggingface.co/AlexWortega/Qwen1.7bnla/tree/main/adapter_universal_v7_sft) | v7 | Qwen3-4B trunk SFT |
| [.../adapter_universal_v7r256_sft/](https://huggingface.co/AlexWortega/Qwen1.7bnla/tree/main/adapter_universal_v7r256_sft) | v7r256-sft | LoRA r=256 + held-out enc refit |
| [.../adapter_universal_v7r256_rl/](https://huggingface.co/AlexWortega/Qwen1.7bnla/tree/main/adapter_universal_v7r256_rl) | v7r256-rl | + 200-step GRPO |
| [.../adapter_universal_v5_direct/](https://huggingface.co/AlexWortega/Qwen1.7bnla/tree/main/adapter_universal_v5_direct) | v5 | direct-lstsq dec_M |
| [.../adapter_universal_rl_v1/](https://huggingface.co/AlexWortega/Qwen1.7bnla/tree/main/adapter_universal_rl_v1) | v1 | 5+2 tags |
| [.../adapter_rl_mix_batched_v1/](https://huggingface.co/AlexWortega/Qwen1.7bnla/tree/main/adapter_rl_mix_batched_v1) | baseline | Single-model NLA repro |
| [.../adapter_universal_v8_mixed](https://huggingface.co/AlexWortega/Qwen1.7bnla/tree/main/adapter_universal_v8_mixed) | v8 ★ | Production serve bundle + serve_cache.safetensors |
| [.../adapter_universal_v9_2_conv](https://huggingface.co/AlexWortega/Qwen1.7bnla/tree/main/adapter_universal_v9_2_conv) | v9.2 ★ | ConvAdapter; best heldout cos 0.492 |
| AlexWortega/v15-universal-nla-ao (private) | v15 | exp4/, exp6/, v15_1_best/ |
| [AlexWortega/universal-activation-oracle-v20](https://huggingface.co/AlexWortega/universal-activation-oracle-v20) | v20 ★ | ZeroGPU demo + detector |
| [AlexWortega/universal-activation-oracle-v22](https://huggingface.co/AlexWortega/universal-activation-oracle-v22) | v22 ★ | Gradio HF Space |
| [AlexWortega/capabilityvectors-qwen3-4b](https://huggingface.co/AlexWortega/capabilityvectors-qwen3-4b) | — | 28 LoRAs (different loss, same data) |
| [AlexWortega/ml-intern-nla-auditing-organism-20260529](https://huggingface.co/AlexWortega/ml-intern-nla-auditing-organism-20260529) | — | Organism training artifacts |

### External references

| HF path | Role |
|---------|------|
| [adamkarvonen/activation-oracle-v1](https://huggingface.co/adamkarvonen/activation-oracle-v1) | japhba/activation_oracles paper oracle (reference) |
| [nluick/MLAO-Qwen3-8B-3L-3N](https://huggingface.co/nluick/MLAO-Qwen3-8B-3L-3N) | MLAO multi-layer oracle (reproduced) |
| [aypan17/latentqa_llama-3-8b-instruct](https://huggingface.co/aypan17/latentqa_llama-3-8b-instruct) | LatentQA reference |
| [ceselder/cot-oracle-corpus-v5](https://huggingface.co/datasets/ceselder/cot-oracle-corpus-v5) | CoT correctness oracle dataset |
| [cowWhySo/Llama-3-8B-Instruct-Cybersecurity](https://huggingface.co/cowWhySo/Llama-3-8B-Instruct-Cybersecurity) | Cybersec fine-tune (intent probe) |
| [cowWhySo/pentest-redteam-steering](https://huggingface.co/datasets/cowWhySo/pentest-redteam-steering) | 1963 pentest prompts dataset |

---

## 7. Data paths

### Eva01 `/big/` (= `/mnt/storage/vae_llm/artifacts/`)

| Path | Contents |
|------|---------|
| `/big/activations_pool_300m/` | 10k FineWeb-Edu passages + per-tag shards (fp32, mean-pool @depth-0.5) |
| `/big/activations_pool_v9/` | 10.5k passages; 16 tags (incl. gemma2, llama3-8b) |
| `/big/activations_pool_v9_ml/` | Multi-layer K=4 (@0.25/0.5/0.75/0.9); `<tag>_ml.safetensors [10500, 4, d_M]` |
| `/big/adapters_v9_serve_full` | 13 trained tags |
| `/big/adapters_v9_serve_gemma2` | +gemma2 (v15 default) |
| `/big/adapters_v9_serve_llama` | All tags incl. llama3-8b (v15/v18 init) |
| `/big/av_v9/` | Qwen3-1.7B AV-LoRA (default for AO experiments) |
| `/big/audit/ao/` | organism_qwen25_7b; ao_rows_v13.jsonl (4782 rows); acts_ao_*.safetensors |
| `/big/audit/lie_gemma2_ml/` | gemma-2-9b-it lie acts L{13,21,31,39}; 1971 rows |
| `/big/audit/lie_gemma2_female/` | held-out gender_secret organism |
| `/big/audit/lie_llama31_8b/` | Llama-3.1-8B lie acts; 1171 rows |
| `/big/audit/latentqa_task/` | 907 train + 227 held-out rows |
| `/big/audit/v15/exp{0..7}/` | 8-exp matrix checkpoints |
| `/big/audit/v15/v15_lqa/` | v15.1 champion checkpoint |
| `/big/audit/v15/cbF_baseinv_0p1/` | CB-F@0.1 champion |
| `/big/audit/v18_xmodel/` | 1401 transcripts × 8 models |
| `/big/audit/v19_xmodel/` | 1054 social/cot transcripts × 8 models |
| `/big/audit/v20_xmodel/` | Union 2455 transcripts |
| `/big/audit/v20/` | v20_broad checkpoint + eval_v20.json |
| `/big/audit/v21/` | v21_full + v21_heldout checkpoints |

### Eva02 `~/p3_work/`

| Path | Contents |
|------|---------|
| `~/p3_work/detector/v22_1p7b_heldout_ep1` | Frozen v22 detector (pre-speech + capvec) |
| `~/p3_work/prespeech/` | Pre-speech sweep; AGG.json + per-base JSONs |
| `~/p3_work/capvec/` | CapabilityVectors experiment; adapters_capvec/ |
| `~/p3_work/capvec/{aime,rumath,olymp,minerva,math}_base` | Seed err-corpus for v23 |
| `~/p3_work/mistral_jb/` | Cybersec/jailbreak intent probe; acts_plain/jb.npz |
| `~/p3_work/cybersec3/` | Cybersec capture 500 tokens + judge_results.json |

---

## 8. Key bugs and fixes

| Bug | Script | What it broke | Fix |
|-----|--------|--------------|-----|
| Mean-pool fp16 overflow | `extract_multi.py` | Silent ±inf on attention-sink channels → NaN in lstsq | Cast to fp32 BEFORE sum |
| lstsq without L2-normalization | `init_adapters.py` | Outlier channels dominate lstsq | Normalize each row to √d_M |
| gelsd instead of gelsy | `nla/enc_dec_adapters.py` | MKL crash on rank-deficient activation matrices | `driver='gelsy'` + ridge fallback |
| value_head random reinit | `eval_fve_multi.py`, RL | PEFT saves only LoRA; value_head reinitializes to random | Re-apply identity AFTER loading |
| Old `dec_M` objective | `refit_dec.py` | `dec(norm(enc(h))) ≈ h` → held-out FVE goes negative | `refit_dec_direct.py`: `dec(AR(z)) ≈ h` on actual AR predictions |
| PEFT task_type=CAUSAL_LM | `train_ar_multi.py` | Crash: NLACriticModel has no generate | `task_type=None` |
| lstsq-init enc_M at eval/RL | `refit_dec_direct.py` | OOD projection → AV emits incoherent z; FVE still looks OK | Always source enc_M from `<av_save_dir>/adapters/` |
| Held-out enc_M drift @ r=256 | `refit_heldout_enc.py` | Drift 3–4×; held-out OOD (cos 0.34→0.07) | lstsq-refit held-out enc_M against SFT-mean |
| Serve-time auto-refit | `build_serve_cache.py` | Manual refit required for each new tag | `add_held_out_tag()` + `serve_cache.safetensors` |

---

## 9. Paper status

ACL review, rounds 1–3. Key claims confirmed:

| Claim | Status | Evidence |
|-------|--------|---------|
| FVE seed stability | ✅ | 3-seed: trained 0.924±0.001, heldout 0.759±0.002 |
| Detector AUROC seed stability | ✅ | 3-seed: 0.977±0.006 |
| Causal validity | ✅ | zero/noise/shuffle → chance; name-scramble 0.989 |
| Teacher-agnostic verbalization | ✅ | GPT-4o judge: ours 0.49 vs KitFT 0.46 (parity) |
| OOF boundary (P3-a) | ✅ | medadvice 0.33→0.49 @ n≥80 floor |
| Chinese construct fix (P3-b) | ✅ | GlobalOpinionQA: 0.40→0.58 |
| CPM-Bee-5B cross-arch | ✅ | dirbal + heldout JSONs in paper/results/ |
| Multi-seed detector | ✅ | p1b_detector_multiseed.json |

**Per-arch eval JSONs:** `paper/results/xarch_*_heldout.json` (llama3-8b, lfm-7b, deepseek, vikhr, yagpt, qwen3p5-4b, cpmbee-5b).
