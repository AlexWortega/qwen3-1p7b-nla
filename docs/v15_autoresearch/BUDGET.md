# BUDGET
metric                 = composite (see PLAN): universal_cos | quirk_judge | lie_auroc
direction              = higher
max_experiments        = 8
seconds_per_experiment = 7200      # ~2h bounded from-scratch joint train per experiment
parallelism            = 3
compute_cap            = 22 GPU-h
--- spent ---
experiments_run = 0
gpu_min_used    = 0
best_metric     = baseline-pending
