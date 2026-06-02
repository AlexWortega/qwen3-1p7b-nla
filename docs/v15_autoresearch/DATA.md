# DATA (all staged on eva01, see repo scripts/audit/DATA_SPEC.md)
universal_av_corpus = /big/activations_pool_v9  (10500 passages, ~19 model tags + gemma2 added; serve_cache.safetensors)
av_teacher_z        = activations_pool_v9 per-row teacher summaries (verify via MultiModelActivationDataset .z)
quirk_ao_rows       = ~/vae_llm/artifacts/audit/ao/ao_rows_v13.jsonl (4782 free-form) + acts_ao_{org,base}_mean [1594,3584] + heldout [540]
lie_rows            = ~/vae_llm/artifacts/audit/lie_gemma2_ml/{lie_rows.jsonl[1971], lie_acts_L21.safetensors[1971,3584]}
enc_tags            = qwen2p5-7b (quirk organism), gemma2 (lie organism) — both already in the v9 pool
