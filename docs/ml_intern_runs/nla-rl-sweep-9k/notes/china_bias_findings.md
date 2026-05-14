# China-bias probe findings (in progress)

## Setup
20 China-sensitive phrases (Tiananmen, Tibet, Uyghurs, Taiwan, HK, Falun Gong,
Mao, Xi/Pooh, Senkaku, Tank Man, social credit, etc.) — see `configs/probe_china_bias.txt`.

Warm-start baseline (`av_ultrafw_9k`, before RL) — verified earlier — produces
mixed: real summaries with some deflection patterns (e.g. "1968 French
student protest" for Tiananmen). Cos ≈ 0.7-0.85.

## RL Exp A (β_KL=0.05, lr_av=1e-5, 300 steps) — MODE COLLAPSE
Every output starts with:
> "Immediate semantic expectations: <token> suggests..."

Same template for all 20 phrases, no content. Reward maximizes by producing
a generic format that AR can easily fit. Cos ≈ 0.70-0.85 → reward looks OK
but interpretability dead.

## RL Exp D (β_KL=0.2 = 4× stronger anchor) — MODE COLLAPSE on different template
Every output starts with:
> "Narrative expectation signal: <token> suggests..."

Same generic format for all 20 phrases. Cos ≈ 0.66-0.87. 4× KL anchor did
**not** prevent collapse — just shifted to a different attractor.

## RL Exp B (slow LR, lr_av=5e-6, 400 steps, β_KL=0.05) — MODE COLLAPSE
Every output starts with:
> "Incomplete phrase: '<token>' strongly suggests..." / "Incomplete action: ..."

reward(std) at step 400 = **0.012** — practically zero variance across G samples,
the unambiguous signature of degenerate equilibrium. Halving LR did NOT
prevent collapse, only slowed convergence to it.

## RL Exp C (long-gen, max_new=60, 300 steps) — MODE COLLAPSE
Every output starts with:
> "The phrase '<token>' is X requiring Y..." / "The vector '<token>' requires..."

Same degenerate dynamics. Longer generation budget didn't help — model
finds even more verbose templates.

## Implications
- 4/4 experiments with reward `= -log MSE(h, AR(z))` mode-collapsed on
  different templates (A "Immediate semantic", B "Incomplete phrase",
  C "The phrase '<token>' is X requiring Y", D "Narrative expectation signal").
- Each KL / LR / max_new configuration found its OWN unique attractor → the
  collapse is structurally inherent to the reward, not a hyperparameter
  pathology.
- The reward is satisfied whenever AR can map a fixed-prefix output back to
  any-h, since AR has the capacity to memorize the bias.
- Fix proposed and implemented in `train_joint_rl_paper.py`:
  `--reward contrastive` (InfoNCE across prompts in batch) and `--reward mix`.
- Experiments E (contrastive) and F (mix) running on GPU0+GPU3 with B=4 G=2.
  Watch `c_acc` field — starts at 1/B=0.25 random; should climb if z carries
  info about specific h.
