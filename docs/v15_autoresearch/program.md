# program.md — v15 (the single human-editable spec)

**Task:** one NLA that is BOTH a universal activation verbalizer (many LLMs via per-tag enc→d_shared)
AND an activation oracle (answers latent-behaviour questions from injected activations).

**Harness (fixed):** `scripts/audit/train_v15.py` (joint AV+AO from-scratch, bounded by `--minutes`)
→ saves trunk-LoRA + enc/dec bundle (+flamingo.pt) + `v15_meta.json`. `scripts/audit/eval_v15.py`
→ `{universal_cos, quirk_judge, lie_auroc, lie_auroc_per_split}`.

**The one thing under experiment:** the flag-set (injection mechanism, mix ratio, contrastive weight,
enc-gradient routing). Everything else (data, trunk, schema, eval) is held fixed so rows compare.

**Metric:** primary `ao_score = mean(quirk_judge, lie_auroc_mean)`, higher better; guardrail
`universal_cos ≥ 0.9 × exp0`. **Budget:** 8 exp, 120 min each, parallelism 3, cap 22 GPU-h.

**Data (eva01):** AV `/big/activations_pool_v9` (15 tags, teacher z in passages.jsonl) · AO-quirk
`audit/ao/ao_rows_v13.jsonl` + `acts_ao_org_mean` (tag qwen2p5-7b) · AO-lie `audit/lie_gemma2_ml/*`
(tag gemma2) · enc init `/big/adapters_v9_serve_gemma2`.

## Running idea table (updated as results land)
| idea | status | evidence |
|------|--------|----------|
| joint AV+AO trains one universal+oracle model | queued | exp1 |
| multi-token injection > single marker | queued | exp2 |
| flamingo gated cross-attn α=0 > marker | queued | exp3 |
| upweighting AO (mix) raises auditing | queued | exp4/5 |
| contrastive org-vs-base helps AO | queued | exp6 |
| enc gradient from AO only sharpens discrimination | queued | exp7 |
