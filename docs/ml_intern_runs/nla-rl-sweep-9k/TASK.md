# NLA RL hparam sweep on warmstart_9k

Continue an existing project (`/Users/aleksandrnikolich/Desktop/vae_llm`) that ports
Anthropic's [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html)
to Qwen3-1.7B at layer 18. Warm-start SFT is already done on 9k Ultra-FineWeb docs +
DeepSeek-V3 explanations, achieving `FVE_pipeline_meannorm = +0.353` (vs paper's
Qwen2.5-7B = +0.375).

User request: "сдлеай все что бесплатно в разных экспах на разных картах" —
fan out free experiments across the 4 V100s of eva01. "Free" = no OpenRouter API
spend; pure local GPU compute on existing SFT artifacts.

Concrete: launch **4 parallel paper-style GRPO joint-RL runs** with different
hyperparameters, all starting from the same `(av_ultrafw_9k, ar_ultrafw_9k)`
checkpoint pair, then eval each and pick the winner.

## Assumptions

- The existing `scripts/train_joint_rl_paper.py` already implements paper's GRPO
  formula (group sampling, baseline, KL regularizer to AV_init).
- V100-SXM2-32GB has enough memory for one joint-RL run per GPU (3× Qwen3-1.7B
  loaded: AV + AV_init + AR) with gradient checkpointing + batch=1 + G≤4.
- Reference `(av_ultrafw_9k, ar_ultrafw_9k)` lives on eva01 at
  `~/vae_llm/artifacts/`.
- 9k AV-SFT parquet is the RL data source — `prompt + activation_vector` only.

## Unknowns

- Whether joint-RL helps on top of a strong warm-start (0.353) — earlier we saw
  it help on a weak warm-start (0.053 → 0.122). May saturate or even hurt
  here.
- Whether longer generations (max_new=60 vs 40) trade memory for signal.
- Optimal β_KL — too low → AV drift, too high → no learning.
