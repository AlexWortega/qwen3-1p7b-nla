Now I have the context I need. Let me write the full review.

---

## Peer Review — Reviewer **alfa**

```
SEED: 7c4b2af1e93d05b86a71cf4e82d59013
LENS: Related work and novelty attribution (missing prior art, overclaimed contributions)
STANCE: Skeptical-but-fair
```

---

### Summary

This paper presents a single shared activation reader — a Qwen3-1.7B trunk with one LoRA adapter and per-model linear adapters fit in closed form — that claims to verbalize, reconstruct, and audit the latent behaviour of 18 structurally heterogeneous LLMs without retraining per target. The core claims are: (1) reconstruction FVE of 0.851±0.001 overall / 0.759±0.002 on five held-out architectures; (2) verbalization parity with Anthropic's own released per-model NLA verbalizer (4× larger) on TOST at n=500; (3) cross-architecture bias detection at AUROC 0.977±0.006 on held-out Llama-3-8B over three seeds; and (4) zero-shot detection of unseen surface-pattern concepts at AUROC 0.89–1.0 after broad-vocabulary training. The paper is unusually thorough in disclosing its own limits (synthetic-to-real gap, single mean-pool bandwidth, model-identity confounds), uses three-seed stability tests, and pre-registers its test split.

---

### Major Concerns

**1. (Primary lens) The concurrently published UAV [2605.25903] shares the paper's core Contribution (1), and the differentiation argument is stated but unverified numerically.**

- **Issue.** Zhao et al. [2605.25903] — *Universal Activation Verbalizer* — is explicitly cited as concurrent and covers the same architectural concept: one shared decoder plus lightweight per-model adapters for cross-architecture activation verbalization. The paper distinguishes itself on three axes: (i) linear vs. MLP/Q-Former adapters; (ii) closed-form vs. gradient-trained adapters; (iii) unlabeled vs. labeled-QA data. These are real differentiators, but the claim "parity at 4× fewer parameters from a single reader" for Contribution (1) is not clearly the paper's exclusive finding — UAV makes an analogous cross-architecture claim and is not numerically compared.
- **Where.** Abstract (Contribution 1), Section 5 Related Work ("The two are concurrent and mutually corroborating evidence that a single shared reader can span heterogeneous models"), and Section 4.1.
- **Why it matters.** If UAV achieves similar verbalization quality with its larger adapters (162–540M) but on different metrics, and this paper achieves it with linear closed-form adapters, the novelty of the universal design per se is shared. The differentiating novelty — the auditing oracle, the closed-form extensibility, the held-out transfer — needs to be clearly foregrounded as what is *new relative to UAV*, not lumped with a "concurrent" hand-wave.
- **What would address it.** The camera-ready should reorganize the contribution framing so that Contribution (1) is scoped as "the linear, closed-form variant of cross-architecture verbalization," with the auditing oracle (Contribution 2) clearly identified as the primary novel contribution beyond UAV. Even a qualitative table comparing the two approaches on key design axes (adapter size, data requirements, extensibility cost, metric coverage) would sharpen the attribution.

---

**2. The headline auditing AUROCs (0.95–0.99) rest entirely on synthetic model organisms, and the synthetic-to-real gap (mean 0.554) is structurally buried.**

- **Issue.** Every audit training positive and the vast majority of evaluation positives are model organisms — models fine-tuned or prompted to carry a known latent behaviour. The real-benchmark AUROC (Table 13) is 0.554 overall, with three of four concepts either at or below the CI boundary for chance. This is disclosed in Section 4.9, but the abstract, introduction, and the Section 4.4 headline all feature 0.95–0.99 figures without prominent caveats. A reader of the abstract alone would form an exaggerated impression of deployment relevance.
- **Where.** Abstract ("detects social and political biases at AUROC 0.95–0.99"), Section 4.4 first paragraph, vs. Table 13 and Section 4.9 paragraph (iii).
- **Why it matters.** For safety-relevant framing ("bias auditing"), the 0.55 real-world figure is the operationally meaningful number. The current structure inverts the emphasis: the limitations section reports what should be the primary bound on claims, while the results section leads with organism-only numbers.
- **What would address it.** The abstract and Section 4.4 should name the organism-only scope upfront (e.g., "on synthetic model-organism positives, AUROC 0.95–0.99; on real-text benchmarks, 0.55"). The current last-paragraph acknowledgment in Section 4.9 should be promoted to a boxed caveat in the primary results section.

---

**3. The cross-category shuffle residual (AUROC 0.642 after wrong-label injection) partially undermines the causal-validity conclusion.**

- **Issue.** Table 10 and its surrounding text present the causal ablations as definitively showing "the oracle reads activation content, not surface cues." But the wrong-label shuffle drops AUROC from 0.966 to 0.642 — still well above chance. The paper correctly notes "the fine-tune-artifact confound is largely, not entirely, ruled out," but the qualifier is understated: an AUROC of 0.642 represents substantial discriminative ability from a feature that is not concept-specific. This could be the generic "model has been fine-tuned on biased content" signature rather than any semantically specific latent state.
- **Where.** Section 4.5, Table 10, paragraph (iii).
- **Why it matters.** If a significant fraction of the 0.989 in-distribution AUROC comes from detecting organism status rather than concept identity, then the zero-shot generalization to entirely new concepts (Section 4.4) might partially reflect a general organism-vs-neutral classifier rather than concept-specific activation reading — which would be a weaker claim.
- **What would address it.** Report AUROC on same-concept clean negatives vs. different-concept organism positives (i.e., is the detector correct when the wrong organism is in the denominator?). This is a cleaner ablation than the shuffle. The paper's current wrong-label shuffle is a necessary but insufficient control for the organism-artifact hypothesis.

---

**4. There is no per-model linear probe baseline for the auditing task; the universality contribution cannot be isolated from the multi-task training contribution.**

- **Issue.** The auditing AUROC results (0.95–0.99 on Llama-3-8B) are compared to the general LatentQA oracle [2512.15674] and to narrower training (Table 6: narrow 0.950 vs. broad 0.963). But neither comparison answers the question: what does a simple linear probe trained directly on Llama-3-8B activations (without any universal reader) achieve? If a per-model probe reaches comparable AUROC with no cross-architecture machinery, the universality claim is decorative for the auditing result.
- **Where.** Section 4.4 and Table 6; absent from the baseline comparison.
- **Why it matters.** The paper's Contribution (2) claims the shared trunk "turns into" an activation oracle. But the activation oracle result could simply reflect that the bias signal is linearly decodable from any model's mid-layer activations. Without a per-model probe baseline, the paper cannot demonstrate that the *universal design* — as opposed to mere fine-tuning any reader on these signals — is doing meaningful work for the auditing task.
- **What would address it.** Add a logistic-regression or MLP probe trained directly on Llama-3-8B's raw activations on the same organism data as a no-architecture-sharing baseline. If the universal reader matches or exceeds this, the universality claim for auditing is strengthened.

---

**5. The pre-registration (SPLIT.md) is anonymized and cannot be verified by reviewers; several post-hoc adjustments that were necessary for correct results (n≥80 floor, construct-matched GlobalOpinionQA) were disclosed as responses to reviewer feedback, raising questions about what was actually pre-registered.**

- **Issue.** The paper explicitly states it pre-registers the camera-ready test split in SPLIT.md. However, the document is withheld for blind review. Several methodologically important decisions — the n≥80 positives floor (disclosed as changing the scientific conclusion for medical-advice from 0.333 to 0.441), the construct-matched GlobalOpinionQA re-run, the Table 7 re-scoring of thin concepts — are framed as having been performed in response to reviewer requests. If these were not in the original pre-registration, they are post-hoc adjustments to results, not pre-registered protocol.
- **Where.** Section 3.5 ("additionally pre-register the camera-ready split"), Section 4.4 ("remedied this directly: we generated ≥80 judge-validated positives per thin concept"), Table 9 footnote ("n≥80 floor is load-bearing: medical-advice first scored 0.333 on n=70 … the floor changed the scientific conclusion").
- **Why it matters.** A pre-registration whose content is unverifiable provides little protection against analytic flexibility. The paper's explicit statement that the n≥80 floor "changed the scientific conclusion" for at least one concept is a notable admission that the original reported numbers were misleading — and the fix, however legitimate, was not pre-registered (because the pre-registration covers the camera-ready test, not the procedure for generating positives).
- **What would address it.** Include a dated, plain-text SPLIT.md in the anonymized repository with all methodology choices pre-specified, including the n_pos floor, the judge filter criteria, and the construct-matching decisions. The camera-ready version should state explicitly which of the Table 7/9 re-runs were in the pre-registration vs. performed in response to review.

---

**6. The verbalization "parity" TOST conclusion rests on a metric where the paper's verbalizer scores only 41% (below 50%, i.e., worse than the specialist), and the interpretation of "formally equivalent" is misleading.**

- **Issue.** The TOST equivalence test (n=500, p<0.001, ±0.10 margin) is used to conclude "proven-equivalent" for the teacher-agnostic metric. But the win-rate for that metric is 0.489 (Wilson CI [0.467, 0.557]), and the GPT-4o judge vs. raw passage score is reported as 41% — significantly below 50% (p<0.001 by the paper's own text). These two numbers are in tension: the TOST shows the win rate is within ±0.10 of 50%, while the separate point estimate of 41% is 9pp below 50%. The paper conflates "equivalent within a ±0.10 margin" with "parity" in the abstract. Parity under a ±0.10 TOST margin is not the same as parity in quality — it means the verbalizer is not more than 10pp worse, which is a relatively weak equivalence bound.
- **Where.** Section 4.2, Abstract ("formally equivalent on the teacher-agnostic metric"), and Table 4.
- **Why it matters.** The ±0.10 margin is arbitrary and authors-defined. A ±0.05 margin might not hold. The abstract's "parity" framing will be read by most readers as "essentially the same quality," which is not supported by the 41%/59% judge score (Table 4, "GPT-4o judge vs. raw passage"). Choosing a larger TOST margin to declare equivalence despite a significant point difference is an analytical choice that needs clearer disclosure.
- **What would address it.** Report the equivalence margin explicitly in the abstract or executive summary, and acknowledge that the 41% vs. raw passage score represents a meaningful quality gap regardless of TOST outcome. The conclusion "parity at 4× fewer parameters" should be qualified: "parity within ±10pp at 4× fewer parameters" or simply "does not significantly outperform."

---

**7. The depth-fraction choice (0.5) is not ablated, and the single mean-pool design is posited as an intentional bandwidth-for-universality trade-off without ablation support.**

- **Issue.** A fixed depth fraction of 0.5 is used throughout for all 18 architectures, chosen by default. The paper's framing — "a single mean-pooled read is a deliberate bandwidth-for-universality trade-off" — presents this as a conscious design decision, but there is no ablation showing that 0.5 is optimal or even reasonable for the heterogeneous pool. Different architectures have very different layer-wise information profiles; a LFM2 convolutional block at depth 0.5 (L7/L9) vs. the mid-attention block (L8) is explored briefly but only for LFM2 and only for auditing, not for reconstruction FVE.
- **Where.** Section 2.1 ("fixed depth fraction (0.5) by default"), Table 8 LFM2 row.
- **Why it matters.** If per-architecture optimal depth fractions differ by more than a few FVE points, the universality claim (one recipe, all architectures) is weaker. The rugpt3-large outlier FVE of 0.995 may partly reflect unusually high linear compressibility at the chosen depth, not architecture-agnostic performance.
- **What would address it.** Report FVE sensitivity to depth fraction for a representative subset of architectures (e.g., 0.25, 0.5, 0.75), or at minimum acknowledge this as an unverified assumption.

---

### Minor Concerns

- **Abstract metric inconsistency.** The abstract reports pool mean FVE of 0.874 (single-run), then 0.851±0.001 (three-seed) without signposting the difference. Readers encounter 0.874 in the abstract and 0.851 in §3.5/Table 3 footnote, with no reconciliation in the abstract itself.
- **"Parity" in Contribution (1) overstates.** The abstract says the shared verbalizer "beats a 4× larger per-model specialist" before immediately walking it back. The final reading is parity; the first sentence is misleading.
- **Reference [2] and [3] are double-anonymous arXiv preprints with no author information.** Their inclusion provides content claims ("Caught in the Act", "Domain-specific latent geometry") that reviewers cannot independently assess for credibility. These are especially load-bearing for the deception detection and cross-architecture linear map claims respectively.
- **Reference [8] (Elder et al.) is a LessWrong blog post**, not a peer-reviewed paper. The paper leans on it for a central technical comparison (injection style). This should be noted explicitly, or the comparison should be supplemented with a self-contained reproduction.
- **"Model organism" terminology** is introduced in Section 2.4 without definition; readers unfamiliar with the term from AI safety literature may misread it.
- **The detect-only vs. joint variants** are presented as producing different headline numbers in different tables (Table 5 vs. Table 6/8), but the mapping from variant to table is not consistently summarized — a single explicit "which model produces which table" mapping (beyond the paragraph in §3.3) would help.
- **Table 3 footnote on "single-run point estimates"** is critical information that should appear in the table caption, not buried in a paragraph. Readers citing Table 3 in isolation will not see the three-seed caveat.
- **The "≈3 minutes" extensibility claim** lacks detail (hardware, adapter sizes). On a CPU it would be very different from a V100.

---

### Verdict

**Major revision.**

The paper is unusually self-critical and statistically careful, but four issues require substantive work before acceptance: the novelty argument relative to concurrent UAV [2605.25903] needs restructuring and stronger differentiator evidence; the synthetic-to-real gap must be elevated to the abstract/introduction rather than confined to limitations; the causal ablation residual (AUROC 0.642 after wrong-label shuffle) needs a stronger control to support the activation-content reading conclusion; and the unverifiable pre-registration combined with post-hoc result corrections (n≥80 floor changing a conclusion) weakens the paper's strongest methodological selling point.
