# vae_llm — NLA warm-start on Qwen3-0.6B

Implementation of the **Activation Verbalizer (AV)** + **Activation Reconstructor (AR)** warm-start from Anthropic's *Natural Language Autoencoders* (https://transformer-circuits.pub/2026/nla/index.html), applied to a frozen Qwen3-0.6B base model.

This is the **supervised initialization** described in the paper's *"Initializing the AV and AR"* section — not full joint NLA training (yet). Goal: prove the pipeline end-to-end at small scale, then scale.

## Pipeline

```
fineweb-edu passages
        │
        ├──► [M = Qwen3-0.6B base]  ───► h_l (last-token activation at layer 14)
        │
        └──► [OpenRouter teacher]   ───► summary z
                                            │
                       ┌────────────────────┴────────────────────┐
                       ▼                                         ▼
              train AV: (h_l → z)                       train AR: (z → ĥ_l)
              CE on summary tokens                      MSE on activation
                                                                 │
                                                                 ▼
                                                        FVE = 1 − Var(h−ĥ)/Var(h)
```

## Stages (run on eva01 via Docker)

| # | Script | Output |
|---|---|---|
| 1 | `scripts/extract_activations.py` | `artifacts/activations/layer14.safetensors` |
| 2 | `scripts/generate_summaries_bulk.py` | `artifacts/summaries/bulk.jsonl` (10k, qwen3-8b via OpenRouter) |
| 3 | `scripts/generate_summaries_eval.py` | `artifacts/summaries/eval.jsonl` (200, claude-sonnet-4.6 via OpenRouter) |
| 4 | `scripts/train_av.py` | `artifacts/av/{adapter,projection.pt}` |
| 5 | `scripts/train_ar.py` | `artifacts/ar/{adapter,readout.pt}` |
| 6 | `scripts/eval_fve.py` | W&B run + `artifacts/eval/fve.json` |

## Hardware target

- **eva01**: 4× V100-SXM2-32GB, 251 GB RAM, 48 CPU. CUDA driver 535.230.
- MVP uses **1 GPU**, **fp16** (V100 is sm_70 — no bf16 Tensor Cores, no flash-attn-2).

## Quickstart

```bash
# 1. Set up local .env from .env.example, fill in keys
cp .env.example .env  # then edit

# 2. Sync to eva01
./infra/sync_to_eva01.sh

# 3. Build the Docker image on eva01
ssh eva01 'cd ~/vae_llm && docker compose build'

# 4. Run a stage (replace <stage> with extract_activations, train_av, etc.)
./infra/run_on_eva01.sh extract_activations
```

## Configuration

All hyperparameters live in `configs/mvp.yaml`. The defaults reflect the MVP scope from the plan in `~/.claude/plans/`.
