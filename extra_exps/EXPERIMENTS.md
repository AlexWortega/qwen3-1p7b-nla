# EXPERIMENTS ledger

| path_id | approach | status | key result | verify | failure_cause | gpu_min | box |
|---|---|---|---|---|---|---|---|
| A-judge | multi-seed (3) gpt-4o judge ours-vs-kitft, vs gold + raw text, qwen2.5-7b + gemma3-12b | passed | judge does NOT favor ours (qwen 0.49/0.40, gemma 0.33/0.32) — single-seed sonnet 0.60 was judge-specific | all-pass | — | 0 (no GPU) | eva01 |
| A-cosine | cosine deconf: neutral embedder + non-qwen alt-teacher gold | passed | ours wins vs qwen-teacher (wr 0.72→0.89 neutral emb) but LOSES vs llama-teacher (wr 0.37-0.44) → win is teacher-conditional | all-pass | — | 0 | eva01 |
| B-vikhr | v22 detect on Vikhr-7b-01 (RU 8B, unseen) | passed | supervised AUROC 0.980, held-out concepts 0.934, clean_fp 0.016 | all-pass | — | ~? | eva01 (bg) |
| B-lfm | v22 detect on LFM2 (non-transformer, unseen) | passed | supervised AUROC 0.961, held-out 0.924, clean_fp 0.015 | all-pass | — | ~? | eva01 (bg) |
| B-yagpt | v22 detect on YandexGPT-5-Lite-8B (RU, unseen) | passed | supervised AUROC 0.971, held-out 0.948, clean_fp 0.016 | all-pass | — | ~? | eva01 (bg) |
| B-deepseek | v22 detect on deepseek-llm-7b (unseen) | passed | supervised AUROC 0.968, held-out 0.942, clean_fp 0.022 (after extraction-path retry) | all-pass | — | ~? | eva01 (bg) |
| C-gemma27-8bit | 3rd target gemma3-27b, 8-bit extract | bug | cos_ours 0.069 — INVALID (passage-id misalignment in add_held_out_tag) | — | align bug | — | eva02 |
| C-gemma27-bf16 | retry bf16 extract | bug | cos_ours 0.070 — same align bug (precision was red herring) | — | align bug | C-gemma27-8bit | eva02 |
| C-poscontrol | gemma3-12b through identical pipeline (positive control) | passed | cos_ours 0.6088 (=known 0.61) → pipeline correct, 27b bug isolated | all-pass | — | 5 | eva02 |
| C-gemma27-fullpool | full 10k align but STALE cached ours-z | bug | cos_ours 0.372 — ours-z from stale adapter cache (wrong topics); superseded | — | stale cache | C-gemma27-bf16 | 39.5 | eva02 |
| C-gemma27-bf16clean | FINAL: clean bf16, 10k align, fresh ours-z (enc_fve 0.931) | passed | cos_ours 0.455 vs kitft 0.558 (winrate 0.24) — transfers but DEGRADES at 27B (12b 0.61→27b 0.46); loses to specialist; honest scope limit | all-pass | — | C-gemma27-fullpool | 41.8 | eva02 |
| B-v22exact | re-run 4 held-out bases with EXACT deployable v22_1p7b_wide_full | passed | AUROC 0.978-0.988 (+0.8-1.9pp) but clean_fp 0.14-0.24 (vs 0.015 heldout variant) | all-pass | — | ~? | eva01 |

Notes: rhetq held-out AUROC stays ~0.3-0.45 on BOTH new bases (lfm 0.318, vikhr 0.446) — consistent out-of-scope
concept, not a per-base failure → supports "calibrated abstention" framing. chinese_bias=1.0 on synthetic (the
0.40 inversion is cross-source ToxiGen only, a construct mismatch — unchanged).
