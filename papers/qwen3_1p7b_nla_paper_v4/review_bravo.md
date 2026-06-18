```
SEED: 7f3a92e1b4c05d8f6a2e91c3d7b08f45
```

Deriving axes: `int("7f", 16) = 127`, `127 mod 8 = 7`; `int("3a", 16) = 58`, `58 mod 3 = 1`.

```
AXIS: Reproducibility gap (7)
STANCE: Maximally adversarial (1) — assume the result is artefactual until proven otherwise
```

The UAV concurrent paper [2605.25903] was located in the arXiv search, confirming the paper's claim that it exists and its description of UAV's approach (shared decoder + per-model adapter from labeled QA data) is accurate. The Li et al. [2509.13316v4] privileged-information critique is also present and cited correctly as [14]. No additional prior art was found that the paper conspicuously missed.

---

## Review — codename **bravo**

### Summary

The paper presents a shared 1.7B Qwen3 trunk that reconstructs activations from 18 structurally distinct LLMs via per-model linear adapters fit in closed form (FVE 0.851±0.001 overall, 0.759±0.002 on held-out architectures), and — when jointly trained on auditing tasks — detects synthetic social/political biases on an unseen architecture at AUROC 0.95–0.99, with zero-shot detection of unseen concepts at mean AUROC 0.941. The paper is unusually self-critical: it explicitly downgrades its verbalization result to "parity not a win" via TOST, maps its in-family vs. out-of-family generalization boundary with N=5 out-of-family probes, discloses version-selection contamination, and pre-registers the camera-ready split. Under a maximally adversarial read, however, three concerns are blocking or near-blocking. The cross-category shuffle residual (AUROC 0.642, not ~0.5) means a non-trivial fine-tuning artifact is present and cannot be eliminated with the current organism design. The oracle train pool is 5/7 Qwen-family and one of the six "held-out" architectures (Qwen3.5-4B) is structurally almost identical to Qwen3-4B in training — an architectural homophily effect that is unaddressed. And the reproducibility picture has three compounding gaps: the primary FVE point estimate (0.874) derives from a run whose logs are lost; the code repository is anonymized and unavailable during review; and the training sub-categories that constitute the majority of audit supervision (hard-negative, dir-positive) are never defined. The honest-limits section is genuinely good; the infrastructure to verify those limits is not visible to reviewers.

---

### Major Concerns

**1. [Evidence — distinguishing test] The cross-category shuffle residual (AUROC 0.642) is large enough to indicate a genuine fine-tuning artifact, but the current organism design cannot separate this from concept-specific semantic content.**

- **Where:** Section 4.5(iii); partial acknowledgement in the same paragraph ("largely, not entirely, ruled out").
- **Why it matters:** If the real AUROC is 0.966 and the wrong-concept shuffle gives 0.642, then by a crude signal decomposition the generic "this-organism-was-fine-tuned" component accounts for roughly half the discriminability from chance. The authors call this "largely ruled out" but do not quantify. For organisms created by fine-tuning, weight-distribution shifts are global — every activation in every layer shifts, not just the bias-encoding ones. A detector trained on these organisms can in principle pick up on fine-tuning-distribution drift rather than the semantic framing of the injected bias. The paper's prompt-induced organisms (where no weight shift occurs) could serve as a cleaner test, but the paper does not run the cross-category shuffle ablation separately on fine-tuned vs. prompted organisms. Without that split, the claim that the oracle reads bias content — rather than fine-tuning status — is not established.
- **What would address it:** Run the cross-category shuffle on prompted-only organisms (no weight distribution shift). If the residual AUROC falls to ~0.5, the artifact hypothesis is confirmed. Alternatively, include organisms fine-tuned on neutral content (same steps, same compute, no bias target) and test their AUROC — if these score above chance, the signal is purely distributional.

**2. [Evidence — alternative hypothesis] The oracle training pool is 5/7 Qwen-family; the trunk is Qwen3-1.7B; one "held-out" architecture (Qwen3.5-4B) is nearly identical to Qwen3-4B in training. Architectural homophily is never tested or acknowledged.**

- **Where:** Section 3.3 (oracle pool composition); Table 8 (held-out architecture results).
- **Why it matters:** The oracle is trained on Qwen3-0.6B/1.7B/4B, Qwen2.5-0.5B/7B, SmolLM3-3B, and Gemma-2. Five of seven are Qwen-family. The shared latent space is d=2048, the trunk's own native size. Linear adapters enc_M are fit via least-squares against projections from this Qwen-native space. When Qwen3.5-4B is evaluated as a "held-out" oracle architecture, its residual geometry is almost identical to Qwen3-4B (same GQA structure, same RoPE, same QK-norm) — the enc_M for Qwen3.5-4B will land near the Qwen3-4B enc_M in the shared space, and the oracle trunk has seen that region of the space extensively. Table 8 shows Qwen3.5-4B at 0.942/0.929 AUROC — consistent with the architectural-similarity explanation but not distinguished from genuine generalization. The paper does not mention this confound anywhere.
- **What would address it:** Report results stratified by family distance from training pool (Qwen-adjacent vs. out-of-family). If Qwen3.5-4B clusters with the within-family Llama-3-8B performance rather than with the out-of-family LFM2/Russian-8B cluster, the homophily concern is supported. Alternatively, replace Qwen3.5-4B in the held-out set with a genuinely distinct architecture.

**3. [Evidence — statistical] Table 8 reports six held-out AUROC values without confidence intervals; the verbalization comparison at n=100 and n=500 produces results (41% vs. 48.9%) that are in unexplained tension.**

- **Where:** Table 4 (verbalization), Table 8 (cross-architecture detection, no CIs).
- **Why it matters:** Two separate issues. First: the GPT-4o judge vs. raw passage gives 41%, CI [35,46]%, at n=100 (significantly below chance, p<0.001 by the paper's own report), while at n=500 the TOST gives a win rate of 0.489 (CI [0.467, 0.557]) within the ±0.10 equivalence margin. Moving from 41% to 48.9% over 400 additional passages is a 7.9-point swing; if the n=100 distribution were stable, the n=500 aggregate requires the additional 400 passages to average near 52% each. The paper does not reconcile these numbers and does not provide the per-passage delta distribution for either run. The current presentation allows the reader to conclude "equivalent" when a careful reading of the n=100 result suggests the specialist may be meaningfully better on raw-text faithfulness. Second: Table 8 reports point-estimate AUROCs for five new held-out architectures (e.g., LFM2-1.2B 0.938, Qwen3.5-4B 0.942) with no bootstrap CIs, even though the same 2000-resample bootstrap procedure is applied in Tables 7 and 9. The seed-stability result (AUROC 0.977±0.006) is for Llama-3-8B only; the five new bases have no uncertainty quantification.
- **What would address it:** (i) Show the per-passage win-rate distributions for both n=100 and n=500, or report the confidence interval for each independently, so the reader can assess whether the n=500 TOST absorbs a real-but-small specialist advantage within the ±0.10 margin. (ii) Add bootstrap CIs to every AUROC in Table 8 using the same procedure as Table 7.

**4. [Narrative — overclaiming] The title and abstract claim "universal" auditing and "zero-shot" detection in ways that exceed what the paper delivers: in-family generalization and synthetic-to-real transfer of 0.554.**

- **Where:** Title; abstract paragraph 4 ("detects social and political biases at AUROC 0.95–0.99... detects a held-out bias family member...at 0.96... zero-shot detection... at mean AUROC 0.941"); Section 4.4 heading.
- **Why it matters:** The paper is precise *within the text*: it writes "this is in-family generalization, not open-ended zero-shot" and reports mean AUROC 0.554 on real benchmarks. But the title says "Universal Activation Oracle," the abstract says "detects social and political biases" before the in-family qualifier appears in a subordinate clause, and the section heading "Cross-architecture bias detection and zero-shot transfer" frames zero-shot as the leading result. A reader scanning abstract and headers will form a stronger impression than the actual capability warrants. The genuine contribution — in-family concept generalization and surface-pattern lexical transfer — is interesting and publishable; dressing it as universal zero-shot auditing will generate miscalibrated expectations in anyone who skims. The 0.554 real-world mean AUROC (with three of four concepts failing to clear chance after construct-matching) means the system cannot be deployed for real-world bias auditing. This is buried in Section 4.9 under "Scope and limitations," not foregrounded.
- **What would address it:** Retitle to something like "Cross-Architecture Activation Reader with In-Family Bias Detection" or equivalent. In the abstract, state the in-family constraint and the synthetic-to-real gap (AUROC 0.55) before the headline AUROC numbers.

**5. [Reproducibility] Three compounding gaps make the headline numbers unverifiable at review time: the primary FVE estimate comes from a run whose logs are lost; the anonymized code repository is unavailable to reviewers; and the training sub-categories comprising the majority of audit supervision are undefined in the paper.**

- **Where:** Section 3.5 ("original AV/AR training logs were not retained, so the multi-seed run is a recipe reconstruction... whose *claim* is the negligible variance, not a new point estimate"); Section 3.5 ("released through an anonymized repository... de-anonymized links withheld for review"); Section 3.2/Table 2 (detect sub-mix pos:in-org:clean:hard-neg:dir-pos = 2:1:1:2:2).
- **Why it matters:** Three separate and compounding issues. (a) **Lost logs.** The primary FVE point estimate of 0.874 (which appears in the intro and Table 3 as the "single-run" number) derives from a training run whose logs no longer exist. The multi-seed mean (0.851±0.001) is described as a recipe reconstruction, not a direct repeat. The 2.3-point gap between 0.874 and 0.851 is unexplained — it is larger than the ±0.001 seed variance, which means either the recipe changed or the original run was a favourable outlier. The paper claims the gap is within "reconstruction fidelity" but provides no evidence for this interpretation. (b) **Unavailable code.** Table 2 ends with "every number is reproducible end-to-end" via an anonymized repo with links withheld for review. A reviewer cannot verify the judge prompt, the data pipeline, the positive:negative sampling logic, or the fixed seed. "Reproducible end-to-end" is asserted, not demonstrated, during the review period. (c) **Undefined sub-categories.** The detect sub-mix ratio is 2:1:1:2:2 for pos:in-org:clean:hard-neg:dir-pos. The "hard-negative" and "dir-positive" (dir-pos) sub-categories together constitute 4/8 ≈ 50% of all training samples. Neither term is defined anywhere in the paper. "Hard-negative" plausibly refers to confusable negatives (e.g., strongly opinionated text that is not a planted organism), and "dir-positive" might refer to directional probing examples, but neither interpretation is stated. Without these definitions, the training data composition cannot be reproduced by an independent party, and the effect of these sub-categories on the clean-FP rate (which the paper identifies as a key metric) cannot be assessed.
- **What would address it:** (a) Explicitly explain the 0.874→0.851 delta — either show the original checkpoint's seed or state the recipe change. (b) Provide a verified anonymized mirror of the code+data accessible during review. (c) Add a data appendix defining every sub-category in the detect sub-mix, giving the judge prompt verbatim, and reporting the dataset size per concept per architecture.

---

### Minor Concerns

- **Dangling "Section 12" reference.** Section 5 (Related Work) reads: "Crucially for our **Section 12** result, they compare NLA-style injection against Karvonen-style injection." The paper has 6 sections. This is almost certainly "Table 12." Fix throughout.
- **Internal filename in Figure 4 caption.** "Numbers from the v22 concept-held-out detector (eval\_1p7b\_heldout\_ep1.json)" — internal experiment filenames in a camera-ready caption are unprofessional and should be replaced with a formal data pointer.
- **Abstract conflates two distinct shuffle conditions.** The abstract states "a cross-category wrong-label shuffle drops it 0.97→0.64." Table 10 reports the random-transcript shuffle as 0.989→0.525 (real vs. shuffle rows). The 0.97→0.64 values match Section 4.5(iii)'s concept-label shuffle (0.966→0.642) — a different and stronger control. The abstract should name which condition it is citing and reconcile the discrepant baselines (0.989 in Table 10 vs. 0.966 in 4.5(iii)).
- **Timing claim lacks hardware context.** "Two least-squares fits (≈3 minutes)" appears in the abstract and intro without specifying hardware. On a CPU the lstsq for a 10k-passage 4096-d matrix takes substantially longer than 3 minutes. State the hardware.
- **Table 11 silent "—" entries.** The table uses "—" for metrics not evaluated, but does not distinguish "not evaluated" from "not applicable." The LatentQA oracle's gender accuracy (0.976) comes from their own evaluation setting, not a matched experiment — the "—" in the NLA and "Ours" rows could mislead readers who want to compare on that metric.
- **The two-pool setup is disclosed in one paragraph (Section 3.3) without a diagram.** The consequence — that oracle-held-out evaluation uses a freshly fit lstsq enc_M, not the SFT-tuned enc_M from verbalization — is never stated explicitly. A one-line callout or a figure column would prevent readers from conflating the two.
- **Abstract win-rate range "44–49% on neutral references" mixes metrics.** The 44% appears to derive from the cosine win-rate vs. neutral teacher (0.467 vs. 0.490 → ours loses), while 49% is the GPT-4o judge win-rate against teacher gold. These are incommensurable; presenting them as a range implies they measure the same quantity.

---

### Belief Update

The joint-training result (Table 5: quirk 0.33 isolated → 0.94 joint) is the paper's most internally clean and credible finding — the design is simple, the ablated conditions are well-controlled, and the magnitude is large enough to survive noise. I update weakly that co-training verbalization with audit tasks is a useful inductive bias for cross-model auditing.

I did not update meaningfully on the headline zero-shot or universality claims. The Qwen-heavy oracle pool, the undefined training sub-categories, and the unresolved fine-tuning artifact confound leave the 0.941 zero-shot AUROC uninterpreted — it could reflect genuine concept generalization, Qwen-family geometry alignment, or detection of fine-tuning distribution shift. These hypotheses are not discriminated. The synthetic-to-real transfer result (0.554 mean AUROC after construct-matching) is what matters for any deployment use case, and the paper itself agrees it's insufficient — which makes the title "Universal Activation Oracle" an unfortunate choice.

---

### Verdict

**Major revision.** The highest-priority blocking concern is Reproducibility (#5): the paper's primary FVE point estimate (0.874) comes from a run whose logs no longer exist, the full code and data pipeline is inaccessible to reviewers during the review period, and the training sub-categories comprising half of all audit supervision are never defined — so neither the verbalization headline nor the auditing data composition can be independently verified as submitted.
