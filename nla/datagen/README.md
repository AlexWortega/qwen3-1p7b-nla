# NLA Data Generation Pipeline

Generates the three training parquets (`av_sft`, `ar_sft`, `rl`) + sidecars that the NLA training side (`nla/config.py`, `nla/data_source.py`) reads.

Full design: [docs/design.md](../../docs/design.md) §0.

## Config-driven run (recommended)

One YAML, all stages. See `configs/datagen/qwen7b_fineweb_1M.yaml` for a worked example (100k docs × 10 positions → ~1M vectors, split 25/25/50).

```bash
export PYTHONPATH=/path/to/natural_language_autoencoders:${PYTHONPATH:-}
python -m nla.datagen.run_pipeline --config configs/datagen/qwen7b_fineweb_1M.yaml

# Resume from a specific stage (e.g. after fixing an API error):
python -m nla.datagen.run_pipeline --config ... --stages 2,3
```

Output paths are derived from `config.output_dir` — `base.parquet`, `splits/*.parquet`, `{av_sft,ar_sft,rl}.parquet` all land under it. Every subprocess command is printed before running, so you can always re-run a single stage by hand.

## Quick start (manual, stage-by-stage)

```bash
export PYTHONPATH=/path/to/natural_language_autoencoders:${PYTHONPATH:-}

OUT=/tmp/nla_run
MODEL=Qwen/Qwen2.5-7B-Instruct

# Stage 0: extract activations from corpus (GPU required)
python -m nla.datagen.stage0_extract \
    --base-model $MODEL \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --corpus-length 100000 --positions-per-doc 10 \
    --layer-index 20 \
    --output $OUT/base.parquet

# Stage 1: three-way document-level split
python -m nla.datagen.stage1_split \
    --base $OUT/base.parquet \
    --av-sft-frac 0.3 --ar-sft-frac 0.3 --rl-frac 0.4 \
    --output-dir $OUT/splits

# Stage 2: API explanations (SL subsets only — RL doesn't need them)
export ANTHROPIC_API_KEY=sk-...
python -m nla.datagen.stage2_api_explain \
    --input $OUT/splits/av_sft_raw.parquet \
    --output $OUT/splits/av_sft_explained.parquet
python -m nla.datagen.stage2_api_explain \
    --input $OUT/splits/ar_sft_raw.parquet \
    --output $OUT/splits/ar_sft_explained.parquet

# Stage 3: build training-ready parquets
python -m nla.datagen.stage3_build \
    --input $OUT/splits/av_sft_explained.parquet --stage av_sft --output $OUT/av_sft.parquet
python -m nla.datagen.stage3_build \
    --input $OUT/splits/ar_sft_explained.parquet --stage ar_sft --output $OUT/ar_sft.parquet
python -m nla.datagen.stage3_build \
    --input $OUT/splits/rl_raw.parquet --stage rl --output $OUT/rl.parquet

# Optional: shuffle rows before training (breaks position-within-doc clustering)
python -m nla.datagen.stage_shuffle \
    --input $OUT/av_sft.parquet --output $OUT/av_sft_shuf.parquet --seed 42
```

## Multi-GPU extraction

Stage 0 is the bottleneck (full forward pass per doc). The default `HFExtractor` uses `device_map="auto"` — model parallelism, not data parallelism — so it won't scale throughput across GPUs on its own.

`scripts/datagen/stage0_multigpu.sh` wraps stage 0 in data-parallel mode: one process per GPU, each bound via `CUDA_VISIBLE_DEVICES`, each processing a disjoint `--corpus-start` slice. Stage 0's per-doc keyed RNG means the merged output is row-for-row identical to a single serial run.

```bash
# Same args as stage0_extract. Auto-detects GPU count (override with NGPU env var).
scripts/datagen/stage0_multigpu.sh \
    --base-model $MODEL \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --corpus-length 100000 --positions-per-doc 10 \
    --layer-index 20 \
    --output $OUT/base.parquet
```

Shards are written to `$OUT/base.parquet.shards/shard_{i}.parquet` (with per-shard `.log` files) and merged via `nla.datagen.merge_base` into the final output. Shard files are left in place after merging for debugging.

`merge_base` is also usable standalone for merging shards produced across multiple nodes — it validates that all shards share the same extraction params and cover a contiguous corpus slice:

```bash
python -m nla.datagen.merge_base \
    --inputs node0/base.parquet node1/base.parquet node2/base.parquet \
    --output merged/base.parquet
```

## Stage-by-stage

| Stage | Input | Output | Notes |
|---|---|---|---|
| **0: extract** | HF corpus + model | `base.parquet` | Forward model, grab hidden state at N positions/doc. RAW vectors (no normalization — that's training-time). Per-doc keyed RNG: same `(seed, doc_id)` → same positions. |
| **1: split** | `base.parquet` | 3 subset parquets | Document-level partition (all rows from same doc go to same bucket). Default 30:30:40. |
| **2: explain** | SL subset | +`api_explanation` col | Calls Anthropic API with the NLA instruction prompt (2-3 features, `<analysis>` tags). Strict extract: requires closing tag. Bullet cleanup (strip `- * 1.`). Drops rows with <2 features. |
| **3: build** | subsets | training parquets | av_sft: `prompt` (constant, `<INJECT>` placeholder), `response` (`<explanation>...`). ar_sft: `prompt` ends with `<summary>`, training extracts at `tokens[-1]`. rl: prompt only. Provenance always carried. |
| **shuffle** | any parquet | shuffled | Row permutation via `pyarrow.take()`. Keyed on `(seed, dataset_id)`. |
| **shuffle_activations** | any stage3 output | baseline | Permutes ONLY `activation_vector` — prompts/responses fixed. The random-baseline dataset for measuring injection-signal value. |

## Output schemas

**Training parquets** (`av_sft`/`ar_sft`/`rl`):

| Column | Type | Notes |
|---|---|---|
| `prompt` | `list[struct]` (av_sft/rl) or `str` (ar_sft) | `<INJECT>` literal for av_sft/rl — training-side `NLADataSource` swaps for the injection char |
| `response` | `str` | av_sft only, `<explanation>\n...\n</explanation>` wrapped |
| `activation_vector` | `list[float32]` | RAW hidden state — training normalizes |
| `n_raw_tokens`, `activation_layer`, `doc_id` | provenance | always carried |
| `detokenized_text_truncated` | heavy debug | gated on `--keep-debug-metadata` (default on) |

**Sidecar** (`{parquet}.nla_meta.yaml`):

| Field | Notes |
|---|---|
| `extraction.{base_model,d_model,layer_index,norm}` | `norm` is always `"none"` from datagen |
| `tokens.injection_{char,token_id,left_neighbor_id,right_neighbor_id}` | training hook scans for these |
| `tokens.critic_suffix_ids` | ar_sft only — expected tail token IDs, training verifies then extracts at `tokens[-1]` |
| `prompt_templates.{actor,critic}` | training MUST use these exact strings |

## Swapping backends

Every pluggable component is loaded via `--*-cls` import path. The shipped implementations are local-filesystem `LocalStorage` and the public-API `AnthropicProvider`; for cloud storage or alternative LLM APIs, subclass `nla.datagen.storage.Storage` / `nla.datagen.providers.CompletionProvider` and point at your class:

```bash
# Cloud storage (S3/GCS) — bring your own
--storage-cls my.module.GCSStorage

# Alternative completion provider — bring your own
--provider-cls my.module.OpenAIProvider \
--provider-kwargs '{"model": "gpt-4o", "concurrency": 50}'

# Custom extraction engine (e.g. vLLM server)
--extractor-cls my.module.VLLMExtractor \
--extractor-kwargs '{"url": "http://localhost:8000"}'
```

## Smoke test

`configs/datagen/quick_test_10docs.yaml` runs the full pipeline end-to-end on 10 docs (~50 API calls). Run on a GPU box with `ANTHROPIC_API_KEY` set.
