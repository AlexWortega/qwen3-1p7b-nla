# Pre-registered split (SPLIT.md)

**Status:** Pre-registered protocol for the **camera-ready** evaluation. Frozen on 2026-06-16,
BEFORE the camera-ready hyperparameter/seed tuning (P1-b multi-seed). The original submission's
numbers used a post-hoc split and are NOT covered by this pre-registration (the submission already
carries the 4-arch clean-test substitute; see `04_results`). This file fixes the split so the
camera-ready test set is touched exactly once.

## 1. Held-out MODELS (zero-shot architecture transfer — NEVER in any training set)
Chosen to span three transfer axes; none appears in `train_tags`:
- `llama3-8b`  — NousResearch/Meta-Llama-3-8B-Instruct (Llama-family, in-family arch but unseen weights/size).
- `lfm-7b`     — LiquidAI/LFM2 (non-transformer / hybrid-linear-attention; hardest arch transfer).
- `yagpt-5-8b` — yandex/YandexGPT-5-Lite-8B (cross-lingual RU-centric pretrain).

`train_tags` (the only models the detector/AV/AR ever see): qwen3-1p7b, phi-1p5, smollm3-3b,
qwen2p5-7b, gemma2, qwen2p5-0p5b, qwen3-4b.

## 2. Held-out CONCEPTS (zero-shot concept transfer — NEVER trained)
One per behavioral family, fixed now:
- behavioral-format: `voting`, `chocolate`
- social/political:  `gender_bias`
- out-of-family:     `rhetq` + the P3-a additions (`false_date`, `formal_register`, `medical_advice`)

All remaining concepts (the 24 supervised in `v22_meta.json` minus the held-out concepts above)
are TRAIN concepts.

## 3. Primary / secondary metrics (fixed before seeing test)
- **Primary:** mean per-concept AUROC of the frozen detector on the held-out MODELS for the
  held-out CONCEPTS (the double held-out), with 95% bootstrap CI (2000 resamples, pos & neg
  resampled). Reported as mean±std over the 3 P1-b seeds.
- **Secondary:** `clean_fp` (confabulation rate on held-out-model neutral acts) and
  cross-source real-transfer AUROC (P3-b: GlobalOpinionQA + 2 real benches).

## 4. Test protocol (single-use)
1. Freeze this file + the train/held-out lists (committed with the camera-ready code hash).
2. Run all tuning (P1-b seed sweep) using ONLY train models × train concepts for selection.
3. Evaluate the frozen winning config on the held-out models × held-out concepts **exactly once**.
4. No held-out model or concept is used for any selection, early-stopping, or hyperparameter choice.
5. Every reported held-out number carries its bootstrap CI; n_pos per held-out concept ≥ 80
   (enforced via the P1-a boost) so no CI includes chance by construction.

## 5. Sample-size floor (anti red-flag)
Each evaluated concept MUST have n_pos ≥ 80 judge-validated positives before any AUROC is reported.
Concepts below this floor at submission (rhetq=5, sports=4, voting=12, chocolate=20) are remediated
in P1-a; the floor is a standing rule for the camera-ready.
