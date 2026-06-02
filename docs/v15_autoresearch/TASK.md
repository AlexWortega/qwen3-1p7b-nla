# TASK — v15: universal NLA that is ALSO an Activation Oracle

**User ask (one paragraph):** Build a SINGLE NLA ("v15") that is simultaneously (a) a *universal*
activation verbalizer/reconstructor across structurally different LLMs through the existing
per-model linear enc/dec adapter trunk (Qwen3-1.7B + LoRA, d_shared=2048), and (b) a working
*Activation Oracle* (AO) that answers questions about latent behaviours/biases/deception from
injected activations — not just describes surface topic. Find empirically the config that maximizes
AO capability **without** regressing universal verbalization.

**Unknowns / assumptions:**
- Open question whether universality (the d_M→2048 linear bottleneck + small foreign trunk) is
  fundamentally incompatible with the fine discriminative reading AO needs, or whether a joint
  objective + richer injection closes the gap.
- Assume eva01 (4×V100-32GB) is the compute and all data already staged there (DATA_SPEC.md).
- Teacher = local Qwen2.5-7B (OpenRouter keys unreliable).

**Run mode:** interactive.

**Many hypotheses worth sweeping?** YES — joint multi-task SFT mix ratios, injection mechanism
(marker vs n_inj>1 multi-token vs flamingo CA), d_shared / bottleneck width, injection layer,
trunk size, contrastive weight, AO-objective phrasing. One-variable-at-a-time matrix.

**Prior-art baselines already established in this repo (the bar to beat):**
| metric | baseline | source |
|---|---|---|
| quirk held-out AO judge-rate | 0.328 (av-init) / 0.331 (from-scratch org-init) | scripts/audit/RESULTS.md |
| lie-detection AUROC (vd/val/role/mcq) | av-init 0.65/0.52/0.51/0.64; native 1.0/0.70/0.66/0.84 | RESULTS.md |
| universal cos-vs-gold (held-out tags) | v9 ≈ 0.61 | README / memory |

**v15 success = (1) universal cos-vs-gold not worse than v9, (2) quirk held-out judge ≥0.5
(beat 0.33), (3) lie held-out AUROC > 0.65 (beat av-init), ideally → native.**
