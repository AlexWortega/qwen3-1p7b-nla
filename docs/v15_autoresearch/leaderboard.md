# Leaderboard — v15 (by ao_score; guardrail universal_cos ≥ 0.671)

| rank | exp | ao_score | universal_cos | quirk_judge | lie_auroc | verdict |
|------|-----|----------|---------------|-------------|-----------|---------|
| 1 | exp2 flamingo@L7 | 0.698 | 0.682 | 0.667 | 0.728 | guardrail ok but universal_cos lowest |
| 2 | exp1 joint marker | 0.679 | 0.737 | 0.607 | 0.751 | best-balanced: universal≈AV-only + lie≈native |
| – | exp0 AV-only (anchor) | 0.216 | 0.745 | 0.000 | 0.432 | reference |

HEADLINE: joint AV+AO training makes ONE universal trunk audit (quirk 0.61–0.67 vs 0.33 isolated;
lie AUROC 0.73–0.75 ≈ native 0.73) with ~no universality loss (0.737 vs 0.745). Wave-2/3 pending.
