---
title: Universal Activation Oracle
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
short_description: Detect LLM bias from activations, cross-model zero-shot
---

# Universal Activation Oracle

A single Qwen3-1.7B + LoRA trunk reads the mean-pooled hidden activation of **any** LLM's
response (via per-model linear enc → shared 2048-d space + marker injection) and answers a
calibrated *"does this exhibit behaviour X? Yes/No"*.

- **Cross-model**: reads architectures it was never trained on.
- **Direction, not topic**: pro-PRC framing scores high on *China bias*; a balanced answer on
  the same topic scores low.
- **Zero-shot concepts**: describe a behaviour it never trained on (anti-vaccine, conspiracy,
  brand-shilling) and it still detects it.

Cherry-picked examples ship with precomputed activations for an instant read; custom transcripts
are read live through a small model.
