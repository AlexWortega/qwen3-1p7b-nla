# EXPERIMENTS — v15

primary = ao_score = mean(quirk_judge, lie_auroc_mean); guardrail universal_cos ≥ 0.9×exp0 (0.745→0.671).
Bars from prior isolated work: quirk av-init AO 0.33 · lie av-init 0.65/0.52/0.51/0.64 · lie native 1.0/0.70/0.66/0.84 (held-out mean ≈0.73).

| exp_id | change | status | universal_cos | quirk_judge | lie_auroc | ao_score | guardrail | verified | note |
|--------|--------|--------|--------------|-------------|-----------|----------|-----------|----------|------|
| exp0 | AV-only (mix 1:0:0) anchor | passed | 0.745 | 0.000 | 0.432 | 0.216 | anchor | n/a | AO≈chance (no AO train), judge honest 0.0 |
| exp1 | joint marker 3:1:1 | passed | 0.737 | 0.607 | 0.751 | 0.679 | yes | pending | universal preserved; lie≈native; val .81/role .61/mcq .83 |
| exp2 | joint flamingo @L7 | passed | 0.682 | 0.667 | 0.728 | 0.698 | yes(↓) | pending | best ao_score; universal_cos near guardrail |
| exp3 | joint flamingo @L14 | running |  |  |  |  |  | no |  |
| exp4 | joint marker mix 1:1:1 | running |  |  |  |  |  | no |  |
| exp5 | joint marker mix 5:1:1 | running |  |  |  |  |  | no |  |
| exp6 | joint marker contrastive×2 | queued |  |  |  |  |  | no |  |
| exp7 | joint marker enc←AO-only | queued |  |  |  |  |  | no |  |
