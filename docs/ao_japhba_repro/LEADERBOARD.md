# Phase-2 experiment matrix — continue-SFT v22 on their classification + ablations

Base: v22, cached scaled bundle (2 subjects qwen3-4b/qwen2p5-7b train, gemma2 held-out subject;
14 train / 6 held-out datasets; bias detect+describe from ao). Metric = before→after, eval on 3 axes.

| exp | mix | held-out datasets | held-out subject (gemma2) | bias AUROC (trained) |
|---|---|---|---|---|
| clsonly | cls | 0.530→0.560 | 0.528→0.553 | 0.946→0.954 |
| nodesc | cls+det | 0.530→0.570 | 0.528→0.574 | 0.946→0.984 |
| **full (scaled)** | cls+det+desc | **0.530→0.642** | 0.528→0.583 | 0.946→0.983 |
| lr3e5 | cls+det+desc, lr 3e-5 | running | | |
| ep4 | cls+det+desc, 4 ep | running | | |
| 1subj | cls+det+desc, qwen3-4b only | running | | |

**Lever ranking so far:** free-form `describe` is the biggest driver of held-out **dataset** transfer
(+7pp: 0.570→0.642). bias-replay lifts auditing (0.954→0.983). cls-only at low LR does not collapse auditing.
