#!/usr/bin/env bash
# Log compute for the lie-detection experiment runs.
python -m scripts.audit.flops_est --label lie_extract_multilayer --params 9e9 --tokens 1.6e6 --mode infer --gpu-min 18 --note gemma2-9b_L13_21_31_39_meanpool_1971rollouts
python -m scripts.audit.flops_est --label lie_ao_v1 --params 9e9 --tokens 8e4 --mode train --gpu-min 40 --note single_L31_varied_deception_3ep
python -m scripts.audit.flops_est --label lie_ao_v2 --params 9e9 --tokens 2.4e5 --mode train --gpu-min 45 --note single_L31_plus_alpaca_neg_6ep_AUROC_vd0.956_rp0.632
python -m scripts.audit.flops_est --label lie_baseline_probe --params 0 --tokens 0 --mode infer --gpu-min 1 --note logreg_on_acts_negligible_FLOPs
echo "--- COMPUTE_LOG.jsonl ---"
cat artifacts/audit/COMPUTE_LOG.jsonl
