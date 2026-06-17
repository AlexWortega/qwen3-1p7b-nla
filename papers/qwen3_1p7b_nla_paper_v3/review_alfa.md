---

```
SEED: 7f3a9c2e5b1d8047a6e3f0c9d4b72581

LENS: Experimental design (controls, confounds, blinding, pre-registration)
STANCE: Adversarial (assume the headline result is false; what would falsify it?)
```

---

### Summary

This paper presents a single Qwen3-1.7B trunk with per-model linear adapters that can verbalize, reconstruct, and audit activations from a pool of 18 heterogeneous LLMs without retraining per target. The reconstruction FVE is reported at mean 0.874 across the pool (0.789 held-out). The trunk is then jointly trained on auditing tasks and shown to detect social and political biases on an unseen architecture (Llama-3-8B) at AUROC 0.95–0.99, with zero-shot generalization to held-out concepts at mean AUROC 0.918. The central empirical claim is that training breadth, not injection depth or multi-layer readers, unlocks zero-shot oracle transfer.

---

### Major concerns

**1. The "held-out architecture" for all oracle headline results is not truly unseen by the trunk — a foundational experimental-design confound.**

- **Issue.** Every zero-shot and architecture-transfer claim in the oracle section (Table 6, Figure 4, Table 7, Table 8) uses Llama-3-8B as the single "unseen architecture." But the trunk's LoRA is co-trained on the 18-architecture verbalization pool, which explicitly includes SmolLM2-360m and SmolLM3-3B — both described in Table 1 as "Llama-style (small): GQA, RoPE, SwiGLU," the identical structural signature as Llama-3-8B. The trunk therefore already adapted to Llama-family residual geometry during verbalization SFT. What is held out is only Llama-3-8B from the *oracle* pool (7 architectures), not from the *trunk's prior training signal*.
- **Where.** Section 3.3, Table 1 (pool definition), Table 7 (architecture generalization).
- **Why it matters.** The adversarial hypothesis is that the oracle's cross-architecture transfer is not due to a genuinely universal representation, but simply because the trunk's encoder already learned the Llama activation manifold during verbalization training. If this is the case, the reported 0.99 AUROC on Llama-3-8B would not replicate on a truly unseen family (e.g., Mamba/state-space, Mixture-of-Experts with sparse routing, or the Mistral-family). Table 7 partially addresses this with five additional architectures, but supervised-only AUROC is reported there; the *zero-shot concept-held-out AUROC* column of Table 7 gives only mean values without per-concept breakdowns, making it impossible to verify that the rhetq failure and the five 1.0-AUROC successes from Figure 4 replicate across those bases.
- **What would address it.** Add at least one oracle-held-out architecture whose activation geometry is genuinely absent from the verbalization pool — e.g., a Mamba-family or MoE model — and report the full per-concept AUROC breakdown (analogous to Figure 4) for each held-out architecture in Table 7, not just aggregate means.

---

**2. The paper explicitly discloses a version search of v15→v22 without multiple-comparison correction, then reports held-out point estimates as headline numbers — this is an unresolved selection bias.**

- **Issue.** Section 3.5 states: "The numbers reported here are the final selected variants and were not corrected for that search, so held-out point estimates should be read with that selection in mind." This disclosure is commendably honest but does not constitute a methodological fix. Eight sequential versions selecting on the held-out Llama-3-8B AUROC and held-out concept AUROC means the reported 0.918 and 0.988 are best-of-k draws from a distribution, not unbiased estimates. The magnitude of this inflation is not quantified.
- **Where.** Section 3.5 (reproducibility disclosure).
- **Why it matters.** If even two of the eight versions are competitive (say, v20 and v21 both approach 0.9 zero-shot AUROC), the best-of-2 bias is moderate. If most versions score 0.70–0.75 and only v20 reached 0.92, the inflation is large. Without reporting the held-out distribution across versions, the headline figures are uninterpretable as estimates of population-level generalization. This particularly affects the "training breadth" conclusion: if v15/v16/v17 (narrower training) were compared post-hoc to v20 (broad training) on the same held-out split, the breadth benefit is conflated with the selection effect.
- **What would address it.** Report the held-out AUROC curve across versions, or hold out a strict test split unseen by all 8 versions for the final reported number. At minimum, report the variance across the last three versions to show the selected v20/v22 numbers are not outliers within the search.

---

**3. The "zero-shot" claim for unseen-concept detection rests on a post-hoc in/out-of-family taxonomy that is derived from outcomes, not pre-specified.**

- **Issue.** Figure 4 shows five concepts at AUROC 1.0, four at 0.82–0.99, and one failure (rhetq, 0.49). The paper labels rhetq "out-of-family" (rhetorical-question framing) and all successes "in-family." But this taxonomy is nowhere defined before the results; there is no pre-registered list of which held-out concepts count as in-family. The boundary is inferred retroactively from what succeeded. Notably "british" at 0.82, "voting" at 0.91 are categorized as in-family without explanation of what concept families they are members of relative to the 17 training concepts.
- **Where.** Section 4.4 (Figure 4, "in-family generalization, not open-ended zero-shot").
- **Why it matters.** Without a pre-specified family taxonomy, the in/out-of-family label is unfalsifiable: any successful detection is "in-family" and any failure is "out-of-family." The paper's careful caveat ("in-family generalization, not open-ended zero-shot") is methodologically sound in spirit but cannot be verified without the taxonomy. The adversarial reading: the detector memorizes surface-level lexical overlap between held-out concept names and training transcripts, not abstract concept structure.
- **What would address it.** Provide the explicit mapping of all 17 training concepts and 10 held-out concepts to their families before presenting results. Verify that rhetq's label as out-of-family is not special pleading: if it were relabeled in-family (it involves a speech-act style, not a content bias), would the claimed "calibrated abstention" conclusion still hold?

---

**4. The model-organism evaluation conflates detection of activation patterns with detection of genuine latent bias — and the synthetic-to-real gap (AUROC 0.60) is evidence of a construct-validity failure, not merely a "scope" limitation.**

- **Issue.** The oracle is trained on model organisms — targets fine-tuned or prompted to carry biases with paired neutral controls. AUROC 0.99 on these organisms means the oracle can distinguish fine-tuned-biased from paired-neutral activations. But Section 4.9 reports mean AUROC 0.60 on real-source benchmarks (ToxiGen, BBQ). The paper treats this as "a genuine synthetic→real style gap we own" and a known limitation. However, the gap is large enough (0.39 AUROC points) to question whether "detecting bias" means anything in the real world for this system. Linear probes trained on activations for deception detection [2502.03407, "Detecting Strategic Deception Using Linear Probes"] document similar real-world fragility when trained on synthetic rollouts.
- **Where.** Section 4.4 (cross-source result 0.60), Section 4.9 (limitation (iii)).
- **Why it matters.** The abstract claims "detects social and political biases at AUROC 0.95–0.99." A reader will take this to mean real-world biases. The 0.60 figure on actual benchmarks represents only marginal above-chance discrimination, which undermines the safety-auditing framing of the paper. The chinese-bias inversion (0.40) is defended as a construct-mismatch, but construct-mismatches between training organisms and real biased text are precisely what an auditor would encounter in deployment.
- **What would address it.** Either strengthen the synthetic-to-real transfer result (using GlobalOpinionQA as the paper itself suggests, or with fine-tuned real-bias models), or reframe the contribution more narrowly as "detection of organism-planted latent behavior" and remove safety-auditing language from the abstract and introduction. The claim "detects social and political biases" should be "detects activations associated with organism-planted social/political framing."

---

**5. The verbalization comparison to Anthropic's NLA is underpowered and the "parity" conclusion overreaches the statistical evidence.**

- **Issue.** The verbalization comparison rests on n=100 passages. The GPT-4o judge result is 49% (CI [42, 62]%) vs. teacher and 46% (CI [36, 55]%) vs. raw passage. Neither CI excludes 50%; neither is significant. The paper's conclusion that this is "parity" is technically consistent with "our verbalizer is worse" — 46% win-rate against the raw passage with a CI reaching down to 36% cannot support even a parity claim under a one-sided test.
- **Where.** Section 4.2 (Table 4), with CI [42,62]% and [36,55]%.
- **Why it matters.** The abstract states the shared trunk "reaches parity" with Anthropic's per-model specialist. If the true win-rate against raw passage is 40%, the honest characterization is modest inferiority, not parity. The n=100 sample is insufficient to distinguish parity from small inferiority at conventional significance levels.
- **What would address it.** Increase n to ~500 passages (achievable given the 10k passage corpus already exists) and report p-values or Bayesian posteriors for the parity hypothesis, not just point estimates with wide CIs.

---

**6. The 7-architecture oracle training pool is Qwen-dominated (5/7), making the "architecture-invariant" feature claim suspicious.**

- **Issue.** The oracle training pool (Section 3.3) consists of Qwen3-{0.6B, 1.7B, 4B}, Qwen2.5-{0.5B, 7B}, SmolLM3-3B, and Gemma-2. Five of seven are Qwen variants. The "cross-model invariant" activation signature the oracle learns is predominantly shaped by Qwen-family variation. The trunk itself is Qwen3-1.7B and its LoRA is already in that representational space.
- **Where.** Section 3.3 (Figure 3, pool description).
- **Why it matters.** A bias signature learned primarily from Qwen variants could latch onto Qwen-specific activation geometry rather than model-invariant patterns. The fact that it transfers to Llama-3-8B (which shares the SmolLM verbalization training signal, see concern #1) and other families does not rule out this hypothesis — it could transfer because the enc_M adapters all map into the same Qwen-shaped shared space, and the Qwen-shaped "bias direction" happens to be present in other families' projections by chance. A diverse training pool (e.g., 2 Qwen, 1 Llama, 1 GPT, 1 Gemma, 1 Pythia, 1 BLOOM) would more forcefully test the invariance claim.
- **What would address it.** Ablate the oracle training pool composition — train on a non-Qwen-dominated 7-architecture pool and verify that held-out Llama-3-8B performance is comparable. Without this, the architecture-invariant conclusion is not established.

---

**7. The causal-validity ablation (Table 8) does not rule out that the name-conditioned prompt contains residual information about which architectures are biased.**

- **Issue.** The "name scrambled" condition replaces the model name with "an unspecified language model." This still preserves format structure and context (e.g., the injected activation is still in the same embedding position, and the rest of the prompt structure is intact). The adversarial hypothesis is that the oracle learns a short-cut: certain activation norms, PCA directions, or distributional statistics of enc_M(h_M) correlate with which architectures were fine-tuned with biases during organism creation, and these statistics survive the shuffled-transcript condition. AUROC 0.525 for the shuffle condition (real wrong transcript) is above chance, and while this is not statistically significant at n=40×12 biases, it is in the direction of residual signal.
- **Where.** Section 4.5 (Table 8).
- **Why it matters.** If the oracle reads "which organisms were biased" from enc_M output statistics rather than bias-specific activation content, it would still pass the four ablation conditions (zero/noise/shuffle/name) while being useless for real deployment where organisms are unknown. The shuffle condition uses a "real activation from a different transcript" but doesn't specify whether that different transcript is from the same model organism or a clean control. If shuffled transcripts are also from biased organisms, the 0.525 is much more concerning.
- **What would address it.** Clarify the exact source of transcripts used in the shuffle condition (biased or neutral?). Add a "wrong-organism shuffle" condition: inject an activation from a different bias type on the same architecture and verify AUROC still collapses.

---

### Minor concerns

- **FVE outlier (rugpt3-large, 0.995) should be explored, not just flagged.** It is the single best score in the pool on a held-out architecture. The paper says "more plausibly reflects unusually high linear compressibility" — but this hypothesis is testable (e.g., intrinsic dimensionality of rugpt3 activations vs. others) and would meaningfully bound what FVE is actually measuring.

- **The 0.738 held-out FVE (excluding rugpt3) is reported only in the abstract; Table 3 does not mark it separately.** A reader scanning Table 3 takes away 0.789; the more conservative 0.738 is buried in Section 4.1 prose.

- **The clean-FP of 0.14–0.24 on held-out architectures (Table 7, footnote row) is presented after all the high-AUROC numbers and could mislead a reader focused on AUROC alone.** A deployment-facing paper should lead with this number, not append it.

- **Figure 4's rhetq bar at 0.49 is described as "below chance" in the caption but the plot range starts at 0.5; the bar is not visible.** This is both a presentation error and slightly misleading — the bar would fall left of the plot's x-axis origin.

- **"ours is high-recall and trigger-happy (TPR≈1.0, FPR 0.12–0.50)"** — an FPR of 0.50 is coin-flip performance and should not be described merely as "trigger-happy"; this is a near-useless detector on those specific categories.

- **LatentQA zero-shot result (0.573 on qa.json)** is reported as "well above chance" but the chance level for the specific task structure (GPT-4o-judged accuracy over goal/persona pairs) is not stated. If the human annotator agreement is low, 0.573 may not be meaningful.

- **Reference [2] and [3] are both listed as "Anonymous" with arXiv IDs from 2025/2026** — at submission time these should either be fully cited or marked as concurrent/under-review.

- **Table 2 lists `claude-haiku-4.5` as the judge model** but the paper refers to "GPT-4o judge" throughout the verbalization results. The judge used for which evaluation is not consistently stated.

---

### Verdict

**Major revision.** The paper makes a genuine engineering contribution and is notably honest about its failure modes, but two foundational experimental-design problems — the confounded "held-out architecture" that shares structural family with the verbalization training pool, and the uncorrected v15→v22 version search over the same held-out split — make the headline AUROC and zero-shot transfer claims unverifiable in their current form. These, combined with the underpowered verbalization comparison (n=100, CI spanning near-chance) and the large synthetic-to-real gap that is currently framed as scope rather than a validity concern, require substantive revision before the contribution can be accepted as stated.
