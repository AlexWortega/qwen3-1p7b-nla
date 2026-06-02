# Leaderboard — v15 (by ao_score; guardrail universal_cos ≥ 0.671)

| rank | exp | config | ao_score | ucos | quirk_judge | lie_auroc | guardrail |
|------|-----|--------|----------|------|-------------|-----------|-----------|
| 1 | exp4 | joint marker mix 1:1:1 | **0.826** | 0.694 | 0.928 | 0.725 | ✓ |
| 2 | exp6 | joint marker contrastive×2 | 0.800 | 0.729 | 0.870 | 0.730 | ✓ (best balance) |
| 3 | exp2 | joint flamingo @L7 | 0.697 | 0.682 | 0.667 | 0.728 | ✓ |
| 4 | exp1 | joint marker 3:1:1 (base) | 0.679 | 0.737 | 0.607 | 0.751 | ✓ best ucos |
| 5 | exp3 | joint flamingo @L14 | 0.569 | 0.287 | 0.402 | 0.736 | ✗ FAILS |
| 6 | exp5 | joint marker mix 5:1:1 | 0.535 | 0.729 | 0.333 | 0.737 | ✓ |
| – | exp0 | AV-only anchor | 0.216 | 0.745 | 0.000 | 0.432 | reference |
| .. | exp7 (enc-ao-only) / full-pool(15-tag) / v15.1(+LatentQA task) | retraining / eval / eval | | | | | |

KEY FINDINGS:
- TWO audit levers, both lift quirk_judge ~0.61→0.87-0.93 at small universality cost: upweight the
  AO tasks in the mix (exp4, 1:1:1) and upweight the org-vs-base contrastive (exp6, ×2). exp6 keeps
  higher ucos (0.729) — best universal+audit balance.
- Mix trend: 1:1:1 (0.93) > 3:1:1 (0.61) > 5:1:1 (0.33) quirk — AO weight is the dominant lever.
- Flamingo injection LAYER critical: @L7 ok (ucos 0.682) but @L14 destroys universality (0.287, fails).
- gender_secret held-out organism: v15 0.484 vs from-scratch 0.431 — both ≈chance (novel secret undetectable).
- LatentQA held-out (exp1 zero-shot): judge 0.01/cos 0.507 — v15 loses to in-domain LatentQA (per-position L15 vs single mean-pool); v15.1 (+LatentQA task) re-test pending.

- exp1_fullpool (mix 3:1:1, AV over ALL 15 tags): **ucos 0.727 over 15 models** (vs 0.737 over 3 — universality holds at full pool, -0.01). quirk 0.367, lie 0.662 — AO diluted by 15-tag AV at fixed budget; recoverable via AO-mix/contrastive levers. Headline: universality scales to 15 architectures; audit needs AO-upweight on a big pool.
