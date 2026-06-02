# RESEARCH (distilled) — v15 universal NLA + AO

Full citations in `DEEPRESEARCH.md`. This is the decision layer.

## SOTA map
- **Patch-and-decode (training-free):** Patchscopes (arxiv 2401.06102), SelfIE (2403.10949) — patch an
  activation into a decode prompt, read the continuation. Cross-model = learned **affine map** +
  diagonal layer matching; works for *topical/entity* tasks.
- **Train-a-decoder oracle:** **LatentQA** (2412.08686, code aypan17/latentqa) — fine-tune a decoder to
  ANSWER GPT-generated QA about patched *stimulus* activations (control/stimulus split forces latent,
  not surface, targets). **Anthropic NLA** (transformer-circuits 2026) — AV(act→z)+AR(z→act) recon;
  AO = fine-tuned AV; reconstruction warm-start improves downstream QA; auditing lift <3%→12–15%.
- **Deception probes:** Apollo linear probes ≈ GPT-4o in-dist but conflate "talks about lying" vs
  "is lying" + poor OOD (apolloresearch); "Catch an AI Liar" 2309.15840 black-box.

## The decisive finding (central to v15)
"Domain-Specific Latent Geometry Survives Cross-Architecture Translation" (arxiv 2603.20406):
a **linear/affine cross-model map transfers coarse TOPICAL geometry but degrades FINE discriminative
directions** — "fundamental limits to linear cross-model translation." This *explains* the project's
own negatives (universal lstsq-enc → topic only, 0/352 auditing; av-init lie-AO ≪ native). FVE/recon
rewards topical reconstruction and is blind to latent behaviour.

## Chosen baseline + bar to beat (this repo)
- universal cos-vs-gold (v9) ≈ 0.61 · quirk held-out AO judge 0.33 · lie held-out AUROC: av-init
  0.65/0.52/0.51/0.64 vs native 1.0/0.70/0.66/0.84.

## Proven ideas → experiment hypotheses (each axis below traces to a source)
1. **Higher-bandwidth injection** beats single linear token (§5,§7 of DEEPRESEARCH): multi-soft-token
   (LatentQA/BLIP-2) or **Flamingo gated cross-attn α-init=0** (Alayrac 2022) carry discriminative
   directions a linear map loses, and α-init=0 makes adding a model non-destructive. → exp2 (ntok), exp3 (flamingo).
2. **Joint AV+AO from scratch** so the trainable enc/injection is *shaped by the AO objective* to pass
   auditing-relevant directions, not just topical (counter to frozen lstsq-enc). → exp1 (naive joint) + the whole run.
3. **Mix ratio** AV:AO governs the universality↔auditing trade — sweep it. → exp4 (1:1:1), exp5 (5:1:1).
4. **Contrastive (org vs base, same transcript)** is what made the from-scratch AO work; upweight it. → exp6.
5. **Where enc gets gradient**: enc trained on AO only (vs all tasks) may sharpen discrimination. → exp7.

## Guardrail metric design
Primary = **ao_score** = mean(quirk_judge, lie_auroc_mean). Guardrail = universal_cos must stay
≥ 0.9× the AV-only reference (exp0) — a winner that tanks universality is rejected. Direction=higher.
