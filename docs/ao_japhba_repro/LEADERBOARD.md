# Phase-2 experiment matrix — continue-SFT v22 on their classification + ablations

Base: v22, cached scaled bundle (train subjects qwen3-4b + qwen2p5-7b; held-out subject gemma2;
14 train / 6 held-out classification datasets; bias detect+describe from `nla-auditing-artifacts`).
Metric = before→after mean acc/AUROC, eval on 3 transfer axes. Each row = fresh continue-SFT from v22.

| exp | mix / change | held-out datasets | held-out subject (gemma2) | bias AUROC (trained) | held-out concepts |
|---|---|---|---|---|---|
| clsonly | cls only | 0.530→0.560 | 0.528→0.553 | 0.946→0.954 | movie .92 / pubyear .97 |
| nodesc | cls+det | 0.530→0.570 | 0.528→0.574 | 0.946→0.984 | .88 / .97 |
| **full (scaled)** | cls+det+desc, 2 subj, 2 ep, lr1e-5 | **0.530→0.642** | 0.528→**0.583** | 0.946→0.983 | .91 / .97 |
| lr3e5 | + lr 3e-5 | 0.530→0.522 ↓ | 0.528→0.507 ↓ | 0.946→0.989 | — |
| ep4 | + 4 epochs | 0.530→0.590 | 0.528→0.572 | 0.946→0.988 | — |
| 1subj | qwen3-4b only (1 subject) | 0.530→0.542 | 0.528→0.532 | 0.946→0.985 | .88 / .98 |

## Lever ranking (for held-out **dataset** transfer)
1. **2nd training subject: +10pp** (1subj 0.542 → full 0.642) — multi-architecture training drives open-vocab generalization most.
2. **Free-form `describe` task: +7pp** (nodesc 0.570 → full 0.642) — learning to *answer* (not classify) is what generalizes.
3. **Low LR is essential**: lr3e-5 *hurts* (0.522, below the 0.530 baseline) — aggressive LR overfits the trunk / forgets.
4. **2 epochs > 4**: ep4 (0.590) < full 2ep (0.642) — more epochs overfit, *worse* held-out transfer.
5. **Auditing preserved everywhere** (0.95–0.99 AUROC); bias-replay+describe push it to 0.98–0.99, even cls-only holds 0.954.

## Implication for phase 2 (path to their 0.71–0.77)
Scale the two top levers: **more training subjects** (3–5 architectures) × **more free-form task types**
(LatentQA / taboo / PastLens, teacher-generated), at **low LR, ~2 epochs**. Don't just add epochs or raise LR.
