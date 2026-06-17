Now I have everything I need. Let me compose the full review.

---

```
SEED: 3d023149837baa8d484f5399b7cff57d
```

Derived parameters:
- `int("3d", 16) = 61; 61 mod 6 = 1` → **Lens 1: Experimental design (controls, confounds, blinding, pre-registration)**
- `int("02", 16) = 2; 2 mod 3 = 2` → **Stance: Steelman-then-press**

```
LENS: Experimental design (controls, confounds, blinding, pre-registration)
STANCE: Steelman-then-press
```

---

## Review (codename: bravo)

### Summary

The paper introduces a single shared activation reader — a Qwen3-1.7B trunk with one LoRA adapter — that verbalizes, reconstructs, and audits latent behaviour across a pool of 18 structurally heterogeneous LLMs. Each target model enters the shared latent space through a closed-form linear adapter pair; no trunk retraining is required for new architectures. The system claims mean FVE 0.874 across 18 targets (0.789 on five held-out architectures), parity with a 4× larger Anthropic per-model verbalizer under an independent judge, and — when jointly trained on auditing tasks — AUROC 0.95–0.99 for bias detection on an unseen architecture and mean AUROC 0.92 zero-shot on held-out concepts. The paper is transparently self-critical and acknowledges a version-search selection bias, a parity-not-win verbalization result, and a synthetic-to-real transfer gap.

---

### Major concerns

**1. (Primary lens — Experimental design) The "zero-shot held-out concepts" set overlaps visibly with the training concept set, and the boundary is never specified**

The paper trains on "the broad union of 17 concepts (quirks + social biases + CoT-correctness)" where the social biases explicitly include *chinese, western, muslim, lgbt±, gender* (Section 3.2: "prompt-induced social/political bias dialogues (chinese, western, muslim, lgbt±, gender)"). Figure 4 lists the "zero-shot held-out concepts" as: rhetq, british, voting, sports, movie, atomic, **chinese**, chocolate, decimal, **muslim** — with both *chinese* and *muslim* apparently achieving AUROC 1.0 as "zero-shot." If the training set contained *chinese* and *muslim* social-bias positives, their appearance at AUROC 1.0 in the "zero-shot" evaluation is not zero-shot. The paper says "concepts with no training positive" (Table 6 footnote) but does not define how the training-set *chinese* and *muslim* differ from the held-out *chinese* and *muslim*, or whether they are different sub-types, wordings, or directions. This is the single most important missing clarification: if these are the same or overlapping constructs, the headline AUROC 0.918 on held-out concepts, and specifically the five 1.0-AUROC results, are materially inflated.

- **Where:** Abstract, Figure 4, Table 6, Sections 3.2 and 4.4.
- **Why it matters:** Two of the five 1.0-AUROC "zero-shot" cells may not be zero-shot. If only the genuinely novel three (atomic, chocolate, decimal) achieve 1.0 and the others are recycled, the headline claim "mean AUROC 0.92 on entirely unseen concepts" would need to be re-stated and the concept-family composition disclosed.
- **What would address it:** Provide a precise per-concept definition table: name, direction, training-set membership (Y/N), and whether it is a strictly different sub-type from any training concept sharing the same label. If *chinese* in training = "pro-PRC framing" and *chinese* in test = "Chinese cultural preference", state this explicitly and verify it holds for the negative samples too.

---

**2. The model-organism construction procedure is not controlled for surface-text leakage, and the causal ablation is underpowered for the shuffle condition**

The steelman here is genuine: the causal ablation (Table 8) is the paper's best defensive move — zeroing or noising the activation collapses AUROC to 0.500. But the "shuffle" condition (real activation of a different transcript) gives 0.525, and the paper characterises this as chance. With n=40 pos / 40 neg per bias concept, 12 biases (≈960 examples), a 0.525 vs. 0.500 difference has no reported confidence interval and no significance test. A 2.5 pp lift over chance from real (but mismatched) activations could represent genuine residual text-geometry information leaking through the embedding rather than noise. Furthermore, the model organisms are fine-tuned or prompted to generate biased content, so the mean-pooled assistant-span activations encode both the latent bias direction and the surface rhetorical style of biased output. "Replaying through K architectures" (Section 2.4) averages out architecture-specific geometry but does *not* average out shared surface style that propagates through the same biased transcripts on every model. The paper does not empirically separate these two signals.

- **Where:** Section 2.4, Table 8, Section 3.2.
- **Why it matters:** If the signal is partially text-style rather than purely latent-state, the architecture-transfer results (Table 7) are partly an artifact of style consistency across models reading the same transcript, not evidence of a model-invariant latent bias signature.
- **What would address it:** (a) Report the shuffle-vs-zero AUROC gap with a bootstrap or binomial CI and a significance test. (b) Run one additional condition: *paraphrase shuffle* — take the same transcript, paraphrase it to change surface wording while preserving the biased intent (or use the *neutral* twin of a different bias concept), inject its activation for a different transcript, and report AUROC. A large drop from 0.525 toward 0.500 would confirm the current shuffle residual is noise; stability would confirm a surface-style confound.

---

**3. The comparison against Karvonen et al. [arXiv:2512.15674] uses mismatched metrics and does not specify the evaluation set**

The paper reports "at matched accuracy 0.887 vs. 0.859" as the head-to-head with the LatentQA-trained oracle. But the paper's primary metric is AUROC (threshold-free), and the comparison only reaches a single accuracy figure without specifying (a) at what decision threshold the comparison oracle is evaluated, (b) how many examples were used for the held-out comparison, or (c) whether the LatentQA oracle was run in zero-shot or with any adaptation. AUROC 0.987 ≠ accuracy 0.887; these measure different operating points and the conversion requires a stated threshold or a reported ROC curve.

- **Where:** Section 4.7, Table 9.
- **Why it matters:** A fair head-to-head requires both systems evaluated at the same metric and the same decision threshold. Choosing a threshold where this system wins does not establish general superiority; reporting AUROCs for both would be honest.
- **What would address it:** Report AUROC for both systems on the same held-out test set. If the LatentQA oracle does not output calibrated probabilities amenable to AUROC, report accuracy at a fixed p > 0.5 threshold for *both* systems with the same threshold, and state n.

---

**4. The verbalization FVE baseline comparison ("2.3×") is not controlled**

Section 4.1 states "the per-model NLA recipe reports ≈0.38 FVE on a single Qwen3-1.7B, so one shared pair reconstructs an 18-architecture pool at roughly 2.3× that figure." The NLA FVE of 0.38 is a single-architecture figure from a different setup: different depth fraction, different passage corpus, different normalization convention, and a different trunk size. The paper provides no evidence these are measured on the same data under the same protocol. The 2.3× ratio combines a cross-paper FVE with an 18-architecture mean FVE, which is not a valid comparison.

- **Where:** Section 4.1.
- **Why it matters:** If the NLA 0.38 comes from a harder evaluation setting (e.g., last-token pooling, which the paper shows in its CLAUDE.md notes scores 0.3–0.6 vs. mean-pool 0.97 at init), the 2.3× claim is not meaningful.
- **What would address it:** Run the NLA recipe on the same 10k FineWeb-Edu corpus with mean-pool at depth 0.5 and report FVE under the same protocol as this paper, or remove the cross-paper comparison entirely.

---

**5. Li et al. [arXiv:2509.13316], "Do Activation Verbalization Methods Convey Privileged Information?", is not cited**

The paper engages with the text-inversion concern via Elder et al. [7] but does not cite [2509.13316], which directly and systematically asks whether activation verbalizers convey privileged knowledge about the target model's internal workings or merely reflect input features. That paper's core finding — that verbalization methods can achieve good scores by learning to read the input context rather than the activation — is directly relevant to this paper's verbalization parity claim (Section 4.2). The GPT-4o judge evaluation (46% vs. raw passage, 49% vs. teacher gold) does not distinguish between these explanations. This gap is especially important given the paper's own note that ours "occasionally" hallucinates specifics ("Raspberry Pi for an Arduino build"), which is a signature of text-inversion — the reader is inferring surface content rather than decoding latent state.

- **Where:** Section 4.2, Section 5 (Related Work).
- **Why it matters:** If verbalization fidelity is partly text-inversion rather than activation decoding, the "parity" claim is weaker: both systems may be reading the context, not the activation, and the causal ablation for verbalization (unlike for auditing) is never performed.
- **What would address it:** Cite [2509.13316] and discuss whether the GPT-4o judge evaluation vs. raw passage (46%) is consistent with text inversion. A causal ablation for verbalization — zeroing the activation and measuring cosine to teacher — would directly test this. If cosine stays high with a zeroed activation, the verbalization is reading context, not activation.

---

**6. The version search (v15–v22) constitutes implicit evaluation-set contamination for "held-out" concepts**

The paper admirably discloses: "the system was developed through an extended version search (v15→v22); the numbers reported here are the final selected variants and were not corrected for that search, so held-out point estimates should be read with that selection in mind." This is honest but insufficient. Across 8 model versions, if the held-out concept set (Figure 4) was visible during development (even for diagnostic purposes), the held-out AUROC estimates are upper bounds inflated by the search. The paper does not describe how many times the held-out concepts were evaluated during development, whether version selection used the held-out set or a separate validation set, or what the spread of performance was across versions.

- **Where:** Section 3.5, footnote to Table 2.
- **Why it matters:** With five 1.0-AUROC results on ten concepts, and eight versions of development, the probability of observing at least five 1.0s by chance is non-trivial if versions were discarded on the basis of this set. This inflates the apparent quality of "zero-shot" transfer.
- **What would address it:** Report the held-out AUROC at each development checkpoint, or designate a strict held-out set that was evaluated *only* on the final version and a separate validation set that guided development decisions. Alternatively, report variance across the last three versions as a sensitivity check.

---

### Minor concerns

- The 95% CI reported for the GPT-4o judge win rate is [42, 62]% (Section 4.2). A 49/100 binomial gives a Wilson interval of approximately [39.3, 58.8]%, not [42, 62]%. The reported interval is slightly miscalibrated; the conclusion (tied) survives but the CI should be recomputed.
- "≈3 minutes" for adding a new architecture (Abstract, Introduction) is never verified or broken down (encoder lstsq ~30s, decoder ~2min as mentioned in Section 4.1). A hardware specification should accompany this figure.
- Figure 5 (cross-base EM organism) reports an off-diagonal mean of 0.91, but the weak cell (Qwen7B→Qwen7B-trained on another Qwen7B? or Qwen7B trained, test Qwen0.5B?) reads 0.58, which is well below parity. The caption says "rows = base the probe was trained on; columns = base it is tested on" but doesn't name the axes in the figure itself, making the weak cell hard to locate.
- The clean-FP for the broad detector on unseen architectures is 0.14–0.24 (Table 7). For an auditing use case this is very high (one-in-four false alarms on clean inputs). The paper frames it as a known trade-off but does not suggest a remediation path (e.g., recalibration, threshold adjustment) despite noting the held-out detector achieves 0.015–0.022 on the same targets.
- The paper's abstract states "detects held-out bias family member it saw no positive example of at 0.96" (gender) but Section 4.4 notes training did include *gender* as a social bias concept ("prompt-induced social/political bias dialogues… gender"). If gender was trained as positive, the "no positive example" claim is incorrect and should be reworded.
- The UAV concurrent work [arXiv:2605.25903] (Zhao et al.) is acknowledged but not compared empirically. Even a single shared held-out architecture comparison would clarify whether the closed-form adapter is advantageous over the trained adapter in UAV.
- Reference [2] and [3] are labeled "Anonymous" with arXiv IDs — they should be de-anonymized if they are publicly available preprints, or the citation style should note they are under review.

---

### Verdict

**Major revision.**

The experimental design has three issues that must be resolved before the headline claims can be accepted: (1) the "zero-shot" concept set may overlap with training concepts (chinese, muslim), which would critically undermine the zero-shot AUROC claim; (2) the shuffle causal ablation is underpowered and does not rule out surface-text style leakage; and (3) the version-search disclosure is honest but not quantified, leaving the held-out estimates uncorrected. The verbalization parity claim is appropriately hedged by the authors but would be substantially strengthened by a causal ablation for verbalization (zeroing activation, checking cosine) to address the text-inversion concern from [arXiv:2509.13316]. None of these require new large-scale experiments — primarily improved reporting, one additional ablation condition, and clarification of concept definitions.
