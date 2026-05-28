# Universal NLA — one shared AV/AR across 18 LLM architectures

A **single Activation Verbalizer + Activation Reconstructor pair** that operates on hidden activations from a pool of structurally different small/medium LLMs (GPT-2, Bloom, Pythia, Qwen2/Qwen3, Gemma-4, SmolLM2/3, GPT-Neo, Nemotron, Phi, DeepSeek, LFM2, YandexGPT, rugpt3, Vikhr).

Extends Anthropic's *Natural Language Autoencoders* (https://transformer-circuits.pub/2026/nla/index.html) from per-model to cross-architecture: new models snap in via a small lstsq-fitted linear adapter pair `(enc_M, dec_M)` — **no AV/AR fine-tune per new model**.

```
            ┌─ enc_M : d_M → d_shared (lstsq init) ─┐
h_M (d_M) ──┤                                        ├── AV (Qwen3-1.7B+LoRA) ─▶ z (text)
            └─ model_tag injected as plain text  ───┘

z ─▶ AR (truncated Qwen3-1.7B + LoRA) ─▶ ĥ_shared (d=2048)
                                            │
                                            └─ dec_M : d_shared → d_M ─▶ ĥ_M
                                                                          │
                                                                          ▼
                                                            FVE_meannorm(ĥ_M, h_M)
```

## Headline result (v6, production)

`FVE_pipeline_meannorm` — per-tag, train/eval 80/20 split, 200 passages, in M's native space via `dec_M(AR(z))` vs `h_M`, both normalized to √d_M.

★ = held-out: trunks never saw this model; only `enc_M` + `dec_M` lstsq-fit.

| Tag | FVE | Status | | Tag | FVE | Status |
| --- | --- | --- | --- | --- | --- | --- |
| ★ rugpt3-large | **0.995** | held-out (RU) | | qwen3-4b | 0.908 | trained |
| gpt-neo-1p3b | 0.991 | trained | | qwen2p5-7b | 0.891 | trained |
| gpt2-medium | 0.980 | trained | | qwen2p5-0p5b | 0.880 | trained |
| qwen3-0p6b | 0.970 | trained | | nemotron-mini-4b | 0.871 | trained |
| smollm2-360m | 0.970 | trained | | ★ deepseek-llm-7b | **0.804** | held-out |
| pythia-410m | 0.966 | trained | | ★ vikhr-7b-01 | **0.758** | held-out (RU) |
| gemma4-e4b | 0.933 | trained | | smollm3-3b | 0.756 | trained |
| bloom-560m | 0.914 | trained | | ★ yagpt-5-8b | **0.755** | held-out (RU) |
|  |  |  | | phi-1p5 | 0.751 | trained |
|  |  |  | | ★ lfm-7b | **0.635** | held-out |

- **Mean trained (13):** 0.892
- **Mean held-out (5):** 0.789 — only ~10 pp gap, no architecture catastrophes
- **Mean overall (18):** **0.874**

Anthropic per-model paper baseline on a single Qwen3-1.7B is ~0.38, so this is **~2.3× higher across an 18-architecture pool with one shared AV/AR**. The held-out generalisation is the load-bearing claim: 5 architectures (LFM2, DeepSeek, YandexGPT, rugpt3, Vikhr) cross 0.63 — and 4 of 5 cross 0.75 — with **no trunk retraining**, just an lstsq `enc_M` (~30 s) + a direct-lstsq `dec_M` (~2 min) per new model.

## Head-to-head vs. KitFT specialist on Qwen2.5-7B (n=100)

KitFT released a per-model NLA AV trained specifically on Qwen2.5-7B layer 20 (full fine-tune of Qwen2.5-7B-Instruct, ~7B params, no LoRA). We extract activations at that exact spec and run both AVs on the same 100 random passages. `z_pred` is scored against the teacher gold via mean-pool `all-MiniLM-L6-v2` cosine; an LLM-as-Judge (Sonnet 4.6, pairwise with order randomization) gives an independent verdict.

| Metric                     | v8 universal (Qwen3-1.7B+LoRA, ~1.7B) | KitFT specialist (Qwen2.5-7B FT, ~7B) |
| ---                        | ---                                    | ---                                    |
| cos vs gold (mean)         | **0.609**                              | 0.498                                  |
| cos vs gold (median)       | 0.634                                  | 0.520                                  |
| cos vs gold (p10 / p90)    | 0.39 / 0.84                            | 0.33 / 0.65                            |
| Sentence-transformer winrate | **72%**                              | 28%                                    |
| Sonnet 4.6 LLM-judge winrate | **60%** (1 tie / 100)                | 39%                                    |

A 4× smaller, **shared** AV beats the dedicated per-model specialist on the specialist's own target. v8 wins by emitting concrete entities (e.g. "Pterygium", "Nelumbo lutea"), KitFT identifies the broad domain but slips on specifics (Tennis Elbow vs. rotator cuff; Celiac vs. IBD). Reproduce: `scripts/run_kitft_av.py` + `scripts/eval_universal.py --tags qwen2p5-7b` + `scripts/compare_kitft_vs_v8.py --judge anthropic/claude-sonnet-4-6`.

## Experiments

All versions share the same pipeline (extract activations → init `enc_M` → AV SFT → AR SFT → `refit_dec_direct` → joint RL). What changes is the training pool, the AV/AR trunk, and how `dec_M` is fit.

| Ver | Trunk (d_shared) | Trained | Held-out (eval) | dec_M fit | Mean FVE_pipe_mn | Notes | HF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | Qwen3-1.7B (2048) | 5 | 2 (gemma4, phi) | pinv | 0.69 / 7 | first cross-arch run; phi crashes -0.64 | `adapter_universal_rl_v1/` |
| v2 | **Qwen3-4B** (2560) | 5 | — | — | — | **FAILED** — AV mode-collapsed to canonical template | — |
| v3 | Qwen3-1.7B (2048) | 13 (50k) | 0 | pinv | 0.83 trained / -0.75 gemma4 | **FAILED** — mixed teacher z's (Qwen3-8B + Qwen2.5-7B) poisoned SFT | — |
| v4 | Qwen3-1.7B (2048) | 13 | 0 | pinv | 0.83 (some -ve on other held-out) | refit_dec on wrong objective `dec(norm(enc(h))) ≈ h` | precursor to v5 |
| **v5** | Qwen3-1.7B (2048) | 13 | 3 (lfm, deepseek, yagpt) | **direct-lstsq** | 0.73 trained / 0.84 held-out | added phi/smollm3 to training (were broken held-out); **dec fix** | `adapter_universal_v5_direct/` |
| **v6 (prod)** | Qwen3-1.7B (2048) | 13 | 5 (+ rugpt3, vikhr) | direct-lstsq | **0.89 trained / 0.79 held-out, 0.874 / 18 overall** | gemma4 0.09 → 0.93; broad arch coverage; held-out RU + 7-8B | **`adapter_universal_v6/`** |
| v7 | Qwen3-4B (2560) | 12 (+ 1.7B held-out) | 6 | direct-lstsq | 0.88 trained / 0.79 held-out, **0.849 / 18** | trunk upgrade rerun (no collapse this time, same teacher); RL OOMs on 32 GB V100; **no measurable gain over v6** | `adapter_universal_v7_sft/` |
| v7r256-sft | Qwen3-4B (2560) | 13 | 5 | direct-lstsq + heldout-refit | 0.93 FVE, **cos-vs-gold 0.31 / 0.34 / 0.32** | LoRA r=256 + held-out enc_M refit (new step). enc_M drifts 3-4× more during SFT than r=16 → held-out OOD → fixed by post-SFT lstsq-projection-target refit | `adapter_universal_v7r256_sft/` |
| **v7r256-rl** | Qwen3-4B (2560) | 13 | 5 | direct-lstsq + heldout-refit | **0.92 FVE, cos-vs-gold 0.35 / 0.35 / 0.35, cross-model 0.40** | + 200 steps GRPO with mse reward on 3-GPU split (AV cuda:0, AR+AV_init cuda:1). RL closed half the gap to v6 on trained tags and held-out tracks trained 1:1 — no domain gap | `adapter_universal_v7r256_rl/` |

**Trained pool (v5 / v6 / v7, identical 13):** bloom-560m, gpt2-medium, pythia-410m, qwen2p5-0p5b, smollm2-360m, gpt-neo-1p3b, qwen3-0p6b, qwen3-4b, qwen2p5-7b, nemotron-mini-4b, gemma4-e4b, smollm3-3b, phi-1p5.

**Held-out (v6):** lfm-7b (Liquid LFM2-1.2B), deepseek-llm-7b, yagpt-5-8b (YandexGPT-5-Lite-8B), rugpt3-large (Russian, GPT-2 family), vikhr-7b-01 (Russian, Mistral family).

### Failed / abandoned experiments

* **v2 — Qwen3-4B trunk + LoRA r=16**: bigger trunk mode-collapsed to a canonical template (all z's identical regardless of h). Same LoRA rank is the wrong scaling axis here.
* **per-token HeadTransformer + frozen v1 trunk**: richer attention head over per-position activations; AV trained on the linear-adapter output distribution can't interpret the HeadTransformer distribution. Joint train heads + LoRA → collapse.
* **v3 — 5× data (50k passages)** with mixed teacher z: regressed trained pool 0.92 → 0.83; gemma4-e4b crashed 0.86 → -0.75. Mixing teachers in the same SFT corpus is poison.
* **MLP `dec_M` head**: 4096-hidden 2-layer MLP initialised from lstsq solution; did not beat the pure linear baseline (e.g. lfm 0.76 MLP vs 0.79 linear). The residual is already linear; non-linearity overfits.
* **v7 — Qwen3-4B trunk rerun (consistent teacher)**: SFT loss clean (~0.6, no collapse signal). After direct-lstsq `dec_M`, pipeline FVE = **0.849 across 18 tags** (vs v6 **0.874**) — 2.5 pp worse. But qualitative generation reveals the actual failure mode: **cos(z_pred, z_gold) drops 0.47 → 0.24** (half of v6), and many passages produce the *same hallucinated string* across all 8 evaluated tags — canonical-template mode collapse, same pathology that killed v2. FVE doesn't catch it because AR + dec_M can still recover the activation from a template z (some signal is always there). Cosine-vs-gold catches it. RL phase additionally OOMs on a single 32 GB V100 (3 × 4B copies don't fit). **Conclusion:** Qwen3-4B + LoRA r=16 + current SFT recipe is the wrong scaling axis; bigger trunk needs higher LoRA rank or full-finetune. Mainline stays on Qwen3-1.7B (v6).

## HuggingFace artifacts

Repo: **`AlexWortega/Qwen1.7bnla`** — https://huggingface.co/AlexWortega/Qwen1.7bnla

```
adapter_universal_v6/                  ← production, use this
  av/                                  AV LoRA on Qwen3-1.7B + enc_M
  ar/                                  AR LoRA on truncated Qwen3-1.7B + value_head.pt
  adapters/                            18 (enc_M, dec_M) pairs + refit_direct_report.json
  nla_meta.yaml                        d_shared, layer_index, anchor_tag, tag list
  fve_report.json                      per-tag FVE table

adapter_universal_v7r256_rl/           v7r256 SFT + 200-step GRPO; cos-vs-gold 0.35 / cross-model 0.40
adapter_universal_v7r256_sft/          v7 + LoRA r=256 + held-out enc_M refit; cos-vs-gold 0.32 / 18
adapter_universal_v7_sft/              v7 Qwen3-4B trunk, SFT-only (RL OOM); 18 tags @ 0.849 mean
adapter_universal_v5_direct/           v5 with direct-lstsq dec_M (13 tags)
adapter_universal_rl_v1/               v1 (5 tags + 2 held-out)
adapter_rl_mix_batched_v1/             single-model NLA (Qwen3-1.7B paper repro)
adapter_warmstart_9k/                  pre-RL SFT checkpoint
```

## Adding a new architecture

### Recommended (v8+): one-shot serve-time auto-refit (~5 min, no training)

If your adapter bundle has `serve_cache.safetensors` (any bundle built with v8 or later — see "Baking the serve cache" below), adding a held-out model is a single closed-form call:

```bash
# 1. Extract the new model's activations on the shared 10k-passage pool.
python scripts/extract_large_meanpool.py \
  --tag <new-tag> --model <hf/repo> --pool-dir artifacts/activations_pool_300m

# 2. One-shot lstsq: enc vs serve cache (AV-friendly space), dec as pseudo-inverse.
python scripts/add_held_out.py \
  --in-adapters  <bundle-with-serve_cache> \
  --pool-dir     artifacts/activations_pool_300m \
  --tags         <new-tag> \
  --out-adapters <bundle-with-new-tag>

# 3. Eval.
python scripts/eval_universal.py --av-save-dir <av_dir> \
  --adapters-dir <bundle-with-new-tag> --tags <new-tag>
```

`ModelPoolAdapters.add_held_out_tag()` does both lstsq steps (enc vs `serve_cache`, dec as pseudo-inverse of enc) in one call — no `extend_adapters` + `refit_heldout_enc` two-step.

Validated on 5 held-out arch/sizes from a v8-mixed bundle (no training):

| Tag | d_M | enc_FVE | cos-vs-gold |
| --- | --- | --- | --- |
| Qwen2.5-1.5B | 1536 | 0.98 | **0.606** |
| Qwen3-14B    | 5120 | 0.98 | **0.598** |
| Qwen3-8B     | 4096 | 0.98 | 0.587 |
| Qwen2.5-3B   | 2048 | 0.99 | 0.586 |
| MiniCPM5-1B  | 1536 | 0.98 | 0.582 |

Cross-tag pairwise z-cosine: **0.75** — different held-out architectures produce consistent explanations of the same passage.

### Baking the serve cache (one-time per bundle)

```bash
python scripts/build_serve_cache.py \
  --in-adapters  <SFT-trained-bundle> \
  --pool-dir     artifacts/activations_pool_300m \
  --trained-tags <comma-sep list of tags whose enc_M was SFT-trained> \
  --out-adapters <bundle-with-serve_cache>
```

This averages each trained tag's post-SFT `enc_T(h_T)` projections into a `[N, d_shared]` tensor and writes it as `serve_cache.safetensors`. ~30 s on CPU, ~80 MB on disk.

### Legacy two-step recipe (pre-v8 bundles, or full re-fit)

1. `scripts/extend_adapters.py` — lstsq-fit `enc_M` against the anchor.
2. `scripts/refit_dec_direct.py` — lstsq-fit `dec_M` against AR's *actual* predictions on the same passage corpus.
3. `scripts/refit_heldout_enc.py` — *required when the new arch is held-out from AV/AR training* with a high-capacity LoRA. Re-projects `enc_M` into the post-SFT projection space (mean of trained tags' enc_M outputs on the same passages). Without this, AV reads the held-out tag's lstsq-init projection as out-of-distribution and generates incoherent z. Closed-form lstsq, ~30 s per tag.
4. `scripts/eval_fve_multi.py` — FVE typically ≥ 0.79 without touching the trunks. If the model has tokenizer quirks (Voxtral tekken, YaGPT custom BPE), pass `use_fast=False`; `extract_multi.py` has a fallback retry.

## Quickstart (reproduce v6 inference)

```bash
# 1. Local .env with OpenRouter + HF tokens
cp .env.example .env  # then edit

# 2. Sync repo to eva01
./infra/sync_to_eva01.sh

# 3. Build image
ssh eva01 'cd ~/vae_llm && docker compose build'

# 4. Pull v6 from HF and run universal AV on a held-out model
./infra/run_on_eva01.sh run_universal_av --tag deepseek-llm-7b --n-passages 25
```

## Hardware

- **eva01**: 4× V100-SXM2-32GB, 251 GB RAM, 48 CPU. CUDA 535.230. sm_70 — no vLLM ≥ 0.8 (Qwen3 needs it), no flash-attn-2; use HF `.generate` for benchmarks via `lm-eval-harness`.
- Most stages need only **1 GPU**, **fp16**. Joint RL on Qwen3-1.7B trunk uses 3 GPUs (AV + AV_init + AR).

## Implementation notes & developer docs

See [`CLAUDE.md`](CLAUDE.md) for: pipeline-stage code map, load-bearing bug fixes (fp32 mean-pool, gelsy lstsq, identity-init `value_head`, direct dec_M), and day-to-day environment notes.

## Citation

If you use this work, please cite the original NLA paper:

```
Anthropic — Natural Language Autoencoders (Transformer Circuits, 2026)
https://transformer-circuits.pub/2026/nla/index.html
```
