# EXPERIMENTS ledger

| path_id | approach (one line) | status | final_loss | verify | failure_cause | retry_of | gpu_min |
|---------|---------------------|--------|------------|--------|---------------|----------|---------|
| p1 | Open RM-sycophancy organism (Qwen2.5-7B-Instruct LoRA) + universal v8 NLA at asst token, midpoint(L14) vs later(L20), base control | passed | 0.08 | all-pass | — | — | ~45 |

## Result summary (p1)
- Organism behavioral: **8/8 biases fire** (base 1/8). Valid organism + clean control.
- NLA meta-concept hit-rate: **0/352** explanations (organism AND base) — paper headline
  does NOT reproduce with the universal v8 NLA.
- Mechanism: organism activations diverge from base (L14 cos 0.94 / L20 cos 0.88) but the
  generic AV emits identical surface-topic descriptions → AV is the bottleneck, not a
  missing representation. Universal NLA = topic describer, not latent-knowledge introspector.
- Published: https://huggingface.co/AlexWortega/ml-intern-nla-auditing-organism-20260529 (private)

Note: two extract_acts code bugs fixed mid-run (PEFT unwrap; @torch.no_grad OOM) — code
fixes within p1, not path retries. Stage-1 LLM data gen pivoted to offline (OpenRouter
credit cap).

## Extension: v8 vs v9 vs KitFT (2026-05-29)
Same organism/base battery acts through v9 universal + KitFT specialist
(kitft/nla-qwen2.5-7b-L20-av). meta_rate = 0.00 for ALL THREE AVs (organism & base,
L14 & L20). Negative generalises: bottleneck is the describe-the-activation NLA
objective, not universal-vs-specialist or layer. Artifacts: compare_v8_v9_kitft.{md,json}.

## v9.2 (ConvAdapter) + v9.3 (r512) (2026-05-30)
Same organism/base battery acts (L14/L20) through both repo NLA variants:
- v9.2 ConvAdapter (av_v9_conv + adapters_v9_conv_serve_full): meta 0/176.
- v9.3 r512 (av_v9_r512 + adapters_v9_r512_serve_full): meta 0/176.
Confirms the negative across EVERY describer-NLA variant in the repo (v8, v9, v9.2, v9.3,
KitFT specialist): meta_rate=0, organism & base, both layers. Bottleneck = describe-the-
activation objective, invariant to adapter/AV-rank/architecture. Artifacts: score_v9{conv,r512}.{md,json}.

## v11 (adapters_v11_init + av_v11) (2026-05-31)
Same battery readout. meta-concept 0/176 (coherent topic descriptions; init-enc gives slightly
less faithful topic reads, e.g. a number-list transcript -> "2012 Olympics"). Consistent with
every other describer-NLA. Artifacts: explanations_v11.json, score_v11.{md,json}.
