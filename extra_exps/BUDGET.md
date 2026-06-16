# BUDGET

max_paths            = 3      # Top-1 (verif metrics), Top-2 (held-out oracle), Top-3 (cosine deconf)
max_retries_per_path = 2
compute_cap          = 12 GPU-h wall  (eva01 V100 slivers for cheap; eva02 A6000 for 7B-27B)
scale_ceiling        = 27B (gemma3-27b 8-bit on A6000); 70B llama3.3 = stretch/skip
token_budget         = n/a (no training; extraction + generation + OpenRouter judge only)
openrouter_cap       = ~2000 judge calls + ~400 alt-teacher gens (well within $49.8 budget)
--- spent ---
paths_launched = 0
gpu_min_used   = 0
retries_used   = 0
