# EXPERIMENTS ledger

| path_id | approach (one line) | status | key result | verify | gpu_min |
|---------|---------------------|--------|------------|--------|---------|
| P3-c | pre-registered SPLIT.md (zero compute) | passed | SPLIT.md written (held-out models+concepts, n≥80 floor, single-use test) | n/a | 0 |
| P1-a | boost rhetq/sports/voting/chocolate ≥80 → re-eval frozen v22 + bootstrap CI | **passed** | sports 1.0(1,1) n=184; choc 1.0 n=145; voting 0.929(0.90,0.96) n=179; rhetq 0.796(0.72,0.86) n=106 — CIs tighten, ≫chance | all CIs lower-bound ≫0.5; estimates stable vs orig n | ~25 (eva02 8B extract + eval) |
| P1-b | 3-seed retrain detector(AUROC)+AV/AR(FVE) → mean±std | **passed** | FVE overall 0.851±0.001 (trained 0.924±0.001, held-out 0.759±0.002); detector AUROC 0.977±0.006, held-out-bias 0.988±0.005, clean_fp 0.042±0.026 | both metrics seed-std tiny; caught+fixed assemble bug | ~12 GPU-h (3 FVE + 3 detector + evals) |
| P3-a | 5 out-of-family concepts → eval frozen v22, N>1 boundary | **passed** | wrongunit 0.97 / wrongdate 0.89 / archaic 0.78 transfer; formalreg 0.59 & medadvice 0.49 at chance; mean 0.742. n≥80 floor flipped medadvice 0.33→0.49 (artifact) | all n≥80, CIs reported; no concept below chance | ~20 (eva02) |
| P3-b | GlobalOpinionQA construct-matched real transfer + CI | **passed** | chinese 0.40→0.583 (fix worked), muslim 0.69; real-transfer weak overall (honest) | per-bench bootstrap CI; chinese CI clears chance | ~15 (eva02) |

## spent (approx)
gpu_min_used ≈ 25 (P1-a) + FVE-in-progress · paths: P1-a✓ P3-c✓ done; P1-b + P3-a/b running · openrouter_usd ≈ <3 (gen + boosters + P3-b prep)

## infra learnings (fold into CLAUDE.md later)
- eva01 V100s too contended for 8B fp16 → route 8B extraction to eva02 (48GB, aisci env, fp16).
- eva02 tf5.7: apply_chat_template(tokenize=True) returns Encoding → render-then-encode patch (in extract_v18_xmodel.py).
- Relay eva01↔eva02 via Mac /tmp; write into root-owned /big via `docker run -v ...:/relay alpine cp`.
