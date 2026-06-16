# RESEARCH — environment + artifact facts (discovered 2026-06-15)

## Compute
- **eva01 (`kanbaru`)**: 4×V100-32GB. Repo `/home/alexw/vae_llm`. Canonical artifacts `/mnt/storage/vae_llm/artifacts`
  (137 entries; `~/vae_llm/artifacts` is a near-empty 6-entry dir — DO NOT use it). Docker image `vae_llm:latest`
  (13.4GB, has `transformers`/`torch`/`openai`, MISSING `sentence_transformers` → pip install in-container).
  GPU mem held by other users (GPU3 freest ~15GB). 0 active python train procs.
- **eva02 (`gigabyte`)**: 1×**A6000-48GB, idle**. Has `~/activation_oracles` (external japhba repo) but NO vae_llm
  repo/artifacts. Reaches eva01. Use for 27B (8-bit) / contended-overflow work.
- OpenRouter key in `~/vae_llm/.env` (`OPENROUTER_API_KEY`). Works: `meta-llama/llama-3.3-70b-instruct`,
  `openai/gpt-4o`, `anthropic/claude-sonnet-4-6`, `qwen/qwen-2.5-7b-instruct`.

## KitFT (Anthropic) released per-model AV/AR checkpoints (HF org `kitft`)
- `kitft/nla-qwen2.5-7b-L20-av`  (8B full-FT)   ← baseline already used
- `kitft/nla-gemma3-12b-L32-av`  (12B)          ← baseline already used
- `kitft/nla-gemma3-27b-L41-av`  (27B)          ← **3rd-target candidate** (8-bit fits A6000)
- `kitft/Llama-3.3-70B-NLA-L53-av` (71B)        ← only Llama-3.3 size; 4-bit-only, skip/stretch
- `run_kitft_av.py --av-repo <repo>` pulls each repo's `nla_meta.yaml` (injection_char/scale/token/prompt).

## Verbalization comparison (Top-1/Top-3) — data already on disk, NO GPU NEEDED
`kitft_baseline/compare_<tag>_n100.json[per_passage]` rows carry: `pid, gold(teacher z), v8(ours z),
kitft(kitft z), cos_v8, cos_kitft`. Raw passage `text` in `kitft_baseline/samples_<tag>_n100.json[rows]`.
- qwen2.5-7b: cosine ours **0.609** / kitft 0.498; per-passage cosine winrate 0.72; single-seed sonnet judge **0.60** ours.
- gemma3-12b: cosine ours **0.610** / kitft 0.548; per-passage cosine winrate 0.72; single-seed sonnet judge **0.38** (kitft wins).
- KEY: cosine win robust across BOTH targets; LLM-judge flips on gemma3-12b. → lead with cosine, demote judge.
- `stageA_metrics.py` (running on eva01 docker) computes: 3-seed order-randomized gpt-4o judge vs gold AND vs raw
  text; cosine under neutral embedder (all-mpnet) vs qwen-gold and vs non-qwen alt-teacher (llama-3.3-70b) gold.

## Verbalization model = `av_v8_mixed` (universal AV, Qwen3-1.7B+LoRA, d_shared=2048)
- `av_v8_mixed/nla_meta.yaml`: av_base Qwen/Qwen3-1.7B, injection_char ㈎ (tok 149705, L29/R522), scale √2048.
- `av_v8_mixed/adapters` bundle tags do NOT include gemma3-12b/-27b → these are HELD-OUT, enc fit on-the-fly via
  lstsq-refit (build_serve_cache + add_held_out_tag / ModelPoolAdapters.add_held_out_tag). Needed for ours-z on a new target.

## Oracle model (Top-2) = `v22-scaled` (HF AlexWortega/universal-activation-oracle-v22-scaled)
- Qwen3-1.7B+LoRA detect oracle; held-out llama3-8b AUROC 0.983 (scaled). On eva01 lineage = `adapters_v22_8b`
  (VERIFY equivalence before trusting). Native tags Qwen3-4B (2560), Gemma-2-9B (3584).
- Detect harness: `scripts/audit/train_v18.py` (train), `eval_v18.py` / `eval_v19_real.py` (eval).
  Bias-transcript replay (assistant-span mean-pool through K bases): `/big/audit/v20_xmodel` + `merge_xmodel.py`,
  `gen_biased_dialogues.py`. Held-out arch so far = llama3-8b only.

## 300m activation pool tags (already extracted, mean-pool L=0.5, fp32)
bloom-560m deepseek-llm-7b gemma3-12b gemma4-e4b gpt-neo-1p3b gpt2-medium **lfm-7b**(non-transformer)
minicpm5-1b nemotron-mini-4b phi-1p5 pythia-410m qwen2p5-{0p5b,1p5b,3b,7b} qwen3-{0p6b,1p7b,4b,8b,14b}
rugpt3-large smollm2-360m smollm3-3b **vikhr-7b-01 yagpt-5-8b**(RU 7-8B).
Top-2 unseen-arch candidates (non-Llama/non-Qwen, structurally diverse): **lfm-7b, deepseek-llm-7b, vikhr-7b-01, yagpt-5-8b**.
NOTE: oracle detect needs BIAS-TRANSCRIPT acts (assistant-span), not the 300m passage pool — must extract via xmodel replay.
