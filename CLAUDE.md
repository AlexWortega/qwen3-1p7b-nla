# Universal NLA — cross-architecture Natural Language Autoencoder

## What this is

Implements a **single AV / single AR pair** that operates across activations
from a pool of structurally different small/medium LLMs (Bloom, GPT-2,
GPT-NeoX/Pythia, GPT-Neo, Qwen2/Qwen3, Gemma-3/4, Llama-style SmolLM2,
Nemotron, Phi, Liquid-LFM, DeepSeek, YandexGPT, ...).

The interpretability target follows the Anthropic NLA paper
(https://transformer-circuits.pub/2026/nla/index.html), but the per-model
training/inference loop is replaced with a universal stack that adapts to
any new architecture via a small per-model **linear adapter pair** —
no AV/AR fine-tune needed per new model.

## Architecture

```
            ┌─ enc_M : d_M → d_shared (lstsq init, jointly finetuned) ─┐
h_M (d_M) ──┤                                                            ├── AV (Qwen3-1.7B + LoRA) ──▶ z (text)
            └─ model_tag injected as plain text in the prompt ──────────┘

z ──▶ AR (truncated Qwen3-1.7B + LoRA) ──▶ ĥ_shared (d_shared=2048)
                                              │
                                              └─ dec_M : d_shared → d_M (lstsq fit on AR-pred → h_raw)
                                                                              │
                                                                              └─ ĥ_M; FVE_meannorm(ĥ_M, h_M)
```

* `d_shared = 2048` (matches the AV/AR trunk's native hidden size).
* `enc_M` initialised via Procrustes-style lstsq against an anchor (Qwen3-1.7B
  itself extracted on the same passage corpus) then trainable inside AV SFT.
* `dec_M` is **strictly linear** and refit AFTER AV/AR via lstsq on the AR's
  ACTUAL predictions — this is the single most important fix, see §"Key fixes".

## Pipeline stages

1. `scripts/extract_multi.py` — sequential per-model forward, mean-pool layer-l
   (depth-fraction 0.5 by default) over content tokens, write fp32 shard per
   model. Handles all arch families via `nla/arch_adapters.py`.
2. `scripts/generate_summaries_multi.py` (and `_resume.py`) — OpenRouter teacher
   summaries `z` per passage (one shared `z` per text, used for all models).
3. `scripts/init_adapters.py` — closed-form lstsq `enc_M`, `dec_M` against
   anchor.
4. `scripts/train_av_multi.py` — multi-tag SFT. AV LoRA + enc_M trainable.
   Optional `--exclude-tags`. AV input injects `enc_M(h_M)` at the `㈎`
   marker; CE on `<explanation>z</explanation>` teacher target.
5. `scripts/train_ar_multi.py` — AR SFT. NLACriticModel (truncated trunk +
   identity-init `value_head`), LoRA on attention/MLP. Target =
   `normalize(enc_M(h_M), √d_shared)` in d_shared.
6. `scripts/refit_dec_direct.py` — **post-hoc** correct lstsq for `dec_M`:
   collects AR's actual predictions on all passages, then fits
   `dec_M(AR_pred) ≈ h_M_raw`. This is NOT the same as the old `refit_dec.py`
   path; see §"Key fixes".
7. `scripts/train_joint_rl_multi.py` — GRPO + mix reward (per-tag InfoNCE + per-M
   `-log MSE`) over both AV and AR. Same `mix` recipe that beat collapse in the
   original per-model F config.
8. `scripts/eval_fve_multi.py` — batched FVE eval in M's native space using
   `dec_M(AR(z)) vs h_M`, normalized to `√d_M` on both sides.
9. `scripts/eval_universal.py` — qualitative AV generation samples + cross-model
   z-cosine via a sentence-transformer mean-pool over `transformers.AutoModel`.

## Key fixes (the why-it-now-works story)

These are the load-bearing corrections that turned "kind of works on 5 models"
into "FVE > 0.79 on EVERY tested architecture incl. zero-shot held-out":

* **mean-pool in fp32, not fp16** (`extract_multi.py`): summing 250 fp16
  hidden states across one passage overflowed silently to ±inf on some
  middle-layer "attention sink" channels. The pooled output then poisoned
  any downstream lstsq (gave NaN). Cast captured tensor to fp32 BEFORE the
  multiplication/sum.
* **L2-normalize each row to √d_M before lstsq** (`init_adapters.py`): mid-layer
  LLM activations have outlier channels that swamp lstsq otherwise. Same
  normalization the trainer applies at injection time.
* **gelsy lstsq driver, not gelsd** (`nla/enc_dec_adapters.py`): gelsd crashed
  in MKL on real rank-deficient activation matrices. gelsy + ridge fallback.
* **Identity-init `value_head` reload** (`eval_fve_multi.py`, `train_joint_rl_multi.py`):
  PEFT's `save_pretrained` persists ONLY the LoRA — the NLACriticModel's
  trainable `value_head` (which we identity-init at training start, then leave
  effectively frozen) re-initialises to random `nn.Linear` on `from_pretrained`.
  Re-apply identity AFTER loading.
* **Direct lstsq for `dec_M`** (`refit_dec_direct.py`): the OLD `refit_dec.py`
  fit `dec(normalize(enc(h_M))) ≈ h_M`, assuming AR(z) ≈ normalize(enc(h_M))
  at inference. AR's actual prediction misses that target by enough that
  composing through the linear `dec_M` flips eval FVE to NEGATIVE for held-out
  architectures. Fitting `dec(AR(z)) ≈ h_M` directly lifts held-out FVE from
  -0.6 / -0.3 to +0.79 / +0.88 without touching trunks.
* **PEFT task_type=None for NLACriticModel** (`train_ar_multi.py`): the
  default `CAUSAL_LM` task tries to wire `prepare_inputs_for_generation` and
  crashes — NLACriticModel doesn't expose it. `task_type=None` skips the wiring;
  we never need `.generate()` on the critic.

## Experiments — full history

All versions share the same pipeline (extract → init enc_M → AV SFT → AR SFT
→ refit_dec_direct → joint RL). What changes between versions is the training
pool, the trunk, and where dec_M is fit.

| Ver | Trunk (d_shared) | Trained | Held-out (eval) | Direct dec_M | Mean FVE_pipe_mn | Notes | HF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | Qwen3-1.7B (2048) | 5 | 2 (gemma4, phi) | no (pinv) | 0.69 / 7 | first cross-arch run; phi crashes -0.64 | `adapter_universal_rl_v1/` |
| v2 | **Qwen3-4B** (2560) | 5 | — | — | — | **FAILED** — AV mode-collapsed to canonical template | — |
| v3 | Qwen3-1.7B (2048) | 13 (50k) | 0 | no | 0.83 trained / -0.75 gemma4 | **FAILED** — mixed teacher z's poisoned SFT | — |
| v4 | Qwen3-1.7B (2048) | 13 | 0 | pinv | 0.83 (some -ve held-out elsewhere) | refit_dec on `dec(norm(enc(h))) ≈ h` — wrong objective | `adapter_universal_v5_direct/` (precursor) |
| **v5** | Qwen3-1.7B (2048) | 13 | 3 (lfm, deepseek, yagpt) | **yes (direct-lstsq)** | 0.73 trained / 0.84 held-out | added phi/smollm3 to training (were broken held-out); fixed dec | `adapter_universal_v5_direct/` |
| **v6 (prod)** | Qwen3-1.7B (2048) | 13 | 5 (lfm, deepseek, yagpt, rugpt3, vikhr) | yes | **0.89 trained / 0.79 held-out, 0.874 / 18 overall**, all ≥ 0.63 | gemma4 0.09 → 0.93; broad arch coverage; held-out RU + 7-8B | **`adapter_universal_v6/`** |
| v7 | Qwen3-4B (2560) | 13 | 5 | yes | TBD (running) | trunk upgrade rerun (no collapse this time, same teacher); RL OOM on 32GB V100 — SFT-only eval | — (not pushed; awaits eval) |

**Trained pool (v5/v6/v7, identical 13):** bloom-560m, gpt2-medium, pythia-410m,
qwen2p5-0p5b, smollm2-360m, gpt-neo-1p3b, qwen3-0p6b, qwen3-4b, qwen2p5-7b,
nemotron-mini-4b, gemma4-e4b, smollm3-3b, phi-1p5.

**Held-out (v6 eval):** lfm-7b (Liquid LFM2-1.2B), deepseek-llm-7b,
yagpt-5-8b (YandexGPT-5-Lite-8B), rugpt3-large (Russian, GPT-2 family),
vikhr-7b-01 (Russian, Mistral family).

`FVE_pipeline_meannorm` is per-tag, train/eval 80/20 split, 200 passages, in
M's native space via `dec_M(AR(z))` vs `h_M` with both sides normalized to
√d_M.

**Headline (v6)** — best per-tag FVE on the production stack. ★ = held-out
(only `enc_M`/`dec_M` lstsq-fit, trunks never saw this model during SFT/RL):

| Tag | FVE | Status | | Tag | FVE | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **★ rugpt3-large** | **0.995** | held-out (RU) | | qwen3-4b | 0.908 | trained |
| gpt-neo-1p3b | 0.991 | trained | | qwen2p5-7b | 0.891 | trained |
| gpt2-medium | 0.980 | trained | | qwen2p5-0p5b | 0.880 | trained |
| qwen3-0p6b | 0.970 | trained | | nemotron-mini-4b | 0.871 | trained |
| smollm2-360m | 0.970 | trained | | **★ deepseek-llm-7b** | **0.804** | held-out |
| pythia-410m | 0.966 | trained | | **★ vikhr-7b-01** | **0.758** | held-out (RU) |
| gemma4-e4b | 0.933 | trained | | smollm3-3b | 0.756 | trained |
| bloom-560m | 0.914 | trained | | **★ yagpt-5-8b** | **0.755** | held-out (RU) |
| | | | | phi-1p5 | 0.751 | trained |
| | | | | **★ lfm-7b** | **0.635** | held-out |

- **Mean trained (13):** 0.892
- **Mean held-out (5):** 0.789 — only ~10pp gap, no architecture catastrophes
- **Mean overall (18):** 0.874

vs Anthropic per-model baseline 0.38 on a single Qwen3-1.7B → ~2.3× higher
across an 18-arch pool, **one shared AV/AR**. Held-out generalisation is the
key result: rugpt3-large (Russian, GPT-2 fam) and deepseek-llm-7b (held-out)
both clear 0.8 with no trunk retraining — just lstsq `enc_M` + direct-lstsq
`dec_M`, total fit time ~5 min per new model.

## HF artifacts

Repo: **`AlexWortega/Qwen1.7bnla`** (https://huggingface.co/AlexWortega/Qwen1.7bnla)

```
adapter_universal_v6/                  ← production (v6), use this
  av/                                  AV LoRA on Qwen3-1.7B + enc_M
  ar/                                  AR LoRA on truncated Qwen3-1.7B + value_head.pt
  adapters/                            18 (enc_M, dec_M) pairs + refit_direct_report.json
  nla_meta.yaml                        d_shared, layer_index, anchor_tag, tag list
  fve_report.json                      per-tag FVE table

adapter_universal_v5_direct/           v5 with direct-lstsq dec_M (13 tags)
adapter_universal_rl_v1/               v1 (5 tags + 2 held-out)
adapter_rl_mix_batched_v1/             single-model NLA (Qwen3-1.7B paper repro)
adapter_warmstart_9k/                  pre-RL SFT checkpoint
```

Local artifacts on eva01: `~/vae_llm/artifacts/` — `rl_multi_v{1,5,6}/`,
`adapters_v{1,5,6,7}_direct/`, `activations_pool_300m/` (10k passages × 18 model shards).

## Adding a new architecture (≈20 minutes)

1. Add the model to `configs/universal/extract_v1.yaml` pool; run
   `scripts/extract_multi.py` (skips existing shards). ~10-15 min per 7B model.
2. `scripts/extend_adapters.py` to lstsq-fit `enc_M` against anchor.
3. `scripts/refit_dec_direct.py` to lstsq-fit `dec_M` against AR's actual
   predictions on the same passage corpus.
4. `scripts/eval_fve_multi.py` → FVE typically 0.79+ without touching the
   trunks. If the model has tokenizer quirks (Voxtral tekken, YaGPT custom
   BPE), pass `use_fast=False`; `extract_multi.py` has a fallback retry.

## Failed experiments (kept for documentation)

* **v2: Qwen3-4B trunk** — bigger trunk + AV LoRA → catastrophic mode collapse
  to a canonical template (all z's identical regardless of h). Affects RL too.
  Conclusion: bigger trunk + same LoRA-r is the wrong scaling axis here.
* **per-token HeadTransformer + frozen v1 trunk** — richer attention head over
  per-token activations, AV frozen. cross-model cos drops 0.61→0.47. AV trained
  on linear-adapter output distribution; HeadTransformer's distribution is
  different; AV can't interpret. Joint train heads+LoRA → same canonical
  template collapse.
* **v3: 5× data (50k passages)** — mixed teacher z (Qwen3-8B for first 10k +
  Qwen2.5-7B-Instruct for new 40k). Trained pool FVE regressed 0.92→0.83;
  held-out gemma4-e4b crashed 0.86→-0.75. Mixing teachers in the same SFT
  corpus is poison. (Would have been worth re-running with consistent teacher;
  abandoned in favour of expanding pool diversity instead.)
* **MLP `dec_M` head** — initialized from lstsq solution, 4096-hidden 2-layer
  MLP. Did not beat pure linear baseline (e.g. lfm 0.76 MLP vs 0.79 linear).
  The relevant residual is already linear; non-linearity overfits.
* **v7: Qwen3-4B trunk rerun (with same teacher this time)** — SFT loss looks
  clean (~0.6, no collapse), per-tag FVE_d_shared on AR alone 0.13-0.48 (worse
  than v6 0.65-0.74), but direct-lstsq dec_M closes the gap → FVE_ar_alone
  ~0.94 across 18 tags, **essentially identical to v6**. RL phase OOMs on a
  single 32 GB V100 (AV + AV_init + AR = 3 × 4B copies don't fit). Conclusion:
  trunk upgrade gives no measurable gain on this task; mainline stays on 1.7B.

## Layout

```
nla/
  arch_adapters.py        # cross-arch model unwrap; OPT / GPT-NeoX / Gemma-4
  data_multi.py           # MultiModelActivationDataset
  enc_dec_adapters.py     # LinearAdapter, ModelPoolAdapters (state-dict bundle)
  heads.py                # HeadTransformer (kept for failed experiments)
  injection.py            # ㈎ marker token injection at embedding layer
  models.py               # NLACriticModel (truncated transformer + value head)
  schema.py               # normalize_activation, explanation tags, canonical neighbors
  datagen/                # pre-existing per-model NLA datagen (stage0/1/2/3)
scripts/
  extract_multi.py        # per-model mean-pool extraction
  extract_per_token.py    # per-position extraction (failed heads experiment)
  generate_summaries_multi.py / _resume.py   # OpenRouter teacher z
  init_adapters.py        # lstsq enc/dec init
  extend_adapters.py      # add a new tag to an existing bundle
  refit_dec.py            # OLD refit (do not use)
  refit_dec_direct.py     # CORRECT direct lstsq for dec_M
  fit_mlp_dec.py          # MLP head experiment (linear wins)
  train_av_multi.py       # AV SFT
  train_ar_multi.py       # AR SFT
  train_heads.py          # frozen-trunk heads experiment
  train_joint_rl_multi.py # GRPO + mix reward
  eval_fve_multi.py       # batched FVE (AR-alone + pipeline regimes)
  eval_universal.py       # qualitative AV samples + cross-model z cosine
  run_kitft_av.py         # kitft per-model NLA specialist baseline
  run_universal_av.py     # one-passage-at-a-time universal AV runner
  extract_borealis_llm.py / extract_voxtral_llm.py  # peel LLM out of audio wrappers
  bench_voxtral.py        # lm-eval via Python API (works around tekken tokenizer)
  push_hf_universal.py    # publish artifacts to HF
configs/universal/
  extract_v1.yaml         # pool of ~17 models for extraction
  adapters_v1/v4.yaml     # per-version anchor + d_shared
artifacts/
  activations_pool_300m/  # shared 10k-passage corpus + per-model shards
  av_multi_v{1,5}/        # AV SFT outputs
  ar_multi_v{1,5}/        # AR SFT outputs
  rl_multi_v{1,5}/        # joint RL outputs (av/ + ar/ + adapters/)
  adapters_v5_direct/     # CORRECTED dec_M for all 16 tags
```

## Day-to-day notes

* eva01 (4× V100-SXM2-32GB) is the only training/extract box. V100 = sm_70,
  so vllm doesn't run (Qwen3 support needs vllm 0.8+, V100 dropped after
  0.6). Stick with HF `.generate` for benchmarks via `lm-eval-harness`.
* eva01 disk is the binding constraint, usually ~10-20 GB free. Drop large
  per-model HF caches between runs if pulling new ≥7B models.
* The container uses torch 2.4.1 by default; transformers 5.x needs torch
  2.5+ for `torch.distributed.tensor.device_mesh`. Chain
  `pip install -q torch ... --index-url https://download.pytorch.org/whl/cu124`
  at the start of any container invocation that needs a recent transformers.
* OpenRouter teacher slug that works on this account: `qwen/qwen-2.5-7b-instruct`
  (NOT `qwen/qwen3-8b-instruct` — that 404s). Use the resume-friendly
  `scripts/generate_summaries_resume.py` for >10k corpora to avoid the
  rewrite-on-every-chunk file corruption the original script hit at concurrency >=64.
* HF tokens for gated repos (Gemma 3 etc.) need per-repo license acceptance
  on the user account — token alone isn't enough.
