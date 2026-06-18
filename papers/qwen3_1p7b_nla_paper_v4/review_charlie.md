Now I have enough to write a thorough review. Let me produce it.

---

### Summary

This paper presents a universal activation reader — one Qwen3-1.7B trunk with a LoRA adapter and per-model linear adapter pairs — that operates across 18 structurally heterogeneous LLMs without per-architecture retraining. A new architecture is added via two closed-form least-squares fits (~3 min). The system functions as a natural language autoencoder (NLA), reconstructing activations at mean FVE 0.851±0.001 overall and 0.759±0.002 on five held-out architectures. Jointly training on auditing tasks (bias detection, lie detection, LatentQA) turns the same trunk into an activation oracle that achieves AUROC 0.95–0.99 on synthetic social/political bias organisms in six held-out architectures. The paper compares against Anthropic's own released per-model NLA verbalizer and the Karvonen et al. [2512.15674] general activation oracle, reports a broad causal-validity ablation, pre-registers its evaluation split, and explicitly delineates failure modes including a synthetic-to-real AUROC gap of ~0.55.

---

### Major concerns

**1. [EXTERNAL VALIDITY — primary lens] The synthetic-to-real transfer gap makes the headline auditing claim substantially weaker than the abstract presents.**

- **Issue.** The abstract states the system "detects social and political biases at AUROC 0.95–0.99" without a qualifier, but this number is measured exclusively on organism-planted, judge-validated synthetic transcripts. The construct-matched real-world transfer (Table 13) yields a mean AUROC of 0.554, with only muslim-bias clearly above chance (0.689), gender at 0.541, and lgbt-negative *below* chance (0.403, CI [0.339, 0.467]).
- **Where.** Abstract paragraph 1; Section 4.9 (iii); Table 13.
- **Why it matters.** A mean real-world AUROC of 0.554 is operationally indistinguishable from chance on most subgroups. An auditor useful for deployment-grade bias monitoring must generalize to natural bias manifestations, not only to controlled organisms. The paper is explicit about this in Section 4.9 but the abstract has "detects social and political biases at AUROC 0.95–0.99" as a top-line result, which reads as a broader performance claim than the evidence supports. The qualifier ("organism-planted framing") is buried six paragraphs later.
- **What would address it.** Either (a) move the synthetic-vs-real distinction prominently into the abstract's claim sentence (e.g., "detects organism-planted social/political biases in held-out models at AUROC 0.95–0.99, but transfers only partially to naturally-occurring real bias (mean 0.55)"), or (b) provide additional real-source training data and show whether even a small fraction closes the gap, so the claim boundary is empirically defined rather than asserted.

---

**2. [EXTERNAL VALIDITY] Scale is an uncontrolled and under-characterized axis of generalization failure.**

- **Issue.** The verbalization quality already degrades on the largest tested target: gemma3-27B achieves only 0.455 cosine vs. the specialist's 0.558 (Table 4). The trained pool contains no models above 7B parameters; the held-out models are similarly bounded. No analysis is presented for models ≥27B, ≥70B, or instruction-tuned variants (which have qualitatively different mid-layer representations). The "no architecture catastrophes" claim is therefore scale-bounded and not disclosed as such.
- **Where.** Section 4.2; Table 4; Table 3 (pool composition).
- **Why it matters.** Deployed models most in need of auditing are large (70B+) or closed-weight. The linear adapter compresses a 5376-d residual "less faithfully at 27B" — the paper's own words — but does not bound where this degradation becomes catastrophic. The "universal" framing may not hold at commercially-relevant scales.
- **What would address it.** Report FVE and auditing AUROC against at least one ≥27B open-weights target (e.g., Llama-3-70B, which has publicly released activations in some benchmarks). If that target is outside scope, state it explicitly as an unresolved boundary: the current "parity at ≤12B, bounded at 27B" phrasing in Section 4.2 does the right work for verbalization but is absent in the auditing claims.

---

**3. [CLAIM vs. EVIDENCE] "Zero-shot" is applied inconsistently; most headline "zero-shot" results are in-family generalization.**

- **Issue.** Figure 4 and Table 6 report "zero-shot detection of entirely unseen concepts at mean AUROC 0.941." Six of the ten held-out concepts (chinese, muslim, atomic, chocolate, decimal, sports) are members of trained families (social-bias, numerical/factual quirks); their 1.0 AUROCs are in-family extrapolation, not zero-shot. The paper acknowledges this in Section 4.4 ("Because related concept families are in training, this is in-family generalization"), but the abstract and introduction continue to use "zero-shot" without qualification for these results. The genuinely out-of-family zero-shot boundary is a five-concept N=80 table (Table 9) showing two of five at or below chance — a mixed picture that does not support the unqualified "zero-shot" framing used elsewhere.
- **Where.** Abstract paragraph 1 ("detects held-out concepts it saw no positive example of at mean AUROC 0.94"); Section 1, contribution 2; Figure 4; Section 4.4.
- **Why it matters.** "Zero-shot detection of unseen concepts" is the strongest claim in the paper and is the headline over Figure 4. A reader skimming the abstract or introduction will understand this as detecting genuinely novel concept types, not members of already-trained families. The honest in-family boundary (Table 9) is a valuable result but is a substantially weaker one.
- **What would address it.** Reserve "zero-shot" for the out-of-family Table 9 results; use "in-family generalization" or "held-out member transfer" for the Figure 4 concepts. The abstract should lead with the in-family framing and mention the (weaker but genuine) out-of-family boundary results separately.

---

**4. Training-teacher circularity partially survives the de-confounding controls.**

- **Issue.** The primary verbalization metric — sentence-embedding cosine to the training teacher summary — is measured using the same teacher that generated training targets. The paper de-confounds this with a non-Qwen teacher (Llama-3.3-70B references), where the cosine lead evaporates (0.467 vs. 0.490), and with a GPT-4o judge vs. raw passage, where the system is behind (41% vs. 50%). However, the abstract still reports "leads cosine-to-teacher on two of three" as a headline, and Table 4 lists this metric first in a way that a reader would treat as the primary quality number.
- **Where.** Abstract; Table 4; Section 4.2.
- **Why it matters.** A system trained to reproduce teacher summaries will trivially score high on similarity to those same summaries. The relevant metric for verbalization quality is faithfulness to the *original passage*, not resemblance to one particular teacher's style — and on that metric the system is behind (41% vs. 59% GPT-4o judge vs. raw passage). The TOST equivalence on the raw-passage metric (n=500, ±0.10 margin) is the right analysis, but the margin of ±10 percentage points is generous and unjustified in the text; a stricter ±5-point margin might not yield equivalence.
- **What would address it.** Move the neutral-teacher cosine and GPT-4o-vs-raw-passage results to the primary position in Table 4, with the training-teacher cosine as a supplementary column flagged as biased. Provide a brief justification for the ±0.10 TOST equivalence margin.

---

**5. The cross-category shuffle reveals a non-trivial generic organism confound that is not conclusively ruled out.**

- **Issue.** Swapping in a real activation from a different bias concept drops AUROC from 0.966 to 0.642 (Section 4.5). The paper honestly interprets this as "most discrimination is concept-specific, but a generic biased-vs-neutral component remains." However, this generic component (AUROC ~0.64 from a mislabeled-but-real organism activation) could reflect a stylistic confound from organism construction: all positive examples are generated by the same fine-tuning or prompting procedure, and may share distributional properties (e.g., more repetitive phrasing, more hedged wording, or artifacts of the judge selection filter) that are unrelated to the semantic content of the specific bias.
- **Where.** Section 4.5, third sub-finding; Table 10 (shuffle row).
- **Why it matters.** If 34% of discrimination power comes from a generic "was this model fine-tuned or prompted by our specific bias-generation pipeline" signal, then a detector trained on these organisms may be detecting a data-generation artifact rather than latent bias. The causal ablation (zeroing the vector → chance) rules out prompt-text priors but does not rule out organism-construction confounds embedded in the activations.
- **What would address it.** Test whether a detector trained on organisms of one *construction method* (e.g., fine-tune) generalizes to the same bias planted via a *different method* (e.g., few-shot prompting), and vice versa. If AUROC collapses, the confound is real; if it holds, the signal is genuinely about latent content.

---

**6. The pre-registration claim is unverifiable under double-blind review.**

- **Issue.** The paper claims "we pre-register the camera-ready split (SPLIT.md, frozen before the multi-seed tuning)" and describes this as a protection against selection bias. However, under double-blind review, the "anonymized repository" link is withheld, and the SPLIT.md file cannot be inspected to confirm it predates the multi-seed runs. The pre-registration is the paper's own assurance of methodological hygiene, offered without independent verification.
- **Where.** Section 3.5, "Multiple-comparisons / selection disclosure."
- **Why it matters.** The entire validation story — distinguishing "validation targets" (contaminated by version selection) from "test targets" (single-use) — rests on the credibility of this pre-registration. If the split was chosen or adjusted after observing results on the broad-detector run, the "clean test" is not clean. This concern is heightened because the paper explicitly acknowledges ~8 major development versions, meaning there were many opportunities to inadvertently optimize toward the test distribution.
- **What would address it.** Include the SPLIT.md file or its hash (SHA-256) as a supplementary document at submission, with a timestamp predating the camera-ready tuning runs. Alternatively, submit to a registered-reports track or cite an independent timestamp (e.g., a public git commit SHA, OSF registration).

---

**7. No text-classification baseline for bias detection — the causal ablation does not fully substitute.**

- **Issue.** The paper establishes that zeroing the injected activation collapses AUROC to 0.50 (Table 10, "zero" row). This is presented as evidence that the detector reads activations, not prompt text. However, the baseline missing from the comparison is a transcript-only text classifier: a model that classifies the written dialogue (without any activation injection) as biased or neutral. If such a baseline achieves e.g. AUROC 0.80–0.90 (plausible, since biased and neutral transcripts may differ in surface word choice), the residual gain from activation injection is modest and the practical value of the universal activation reader for auditing is correspondingly smaller.
- **Where.** Section 4.5; Tables 10–11; Section 4.4 (no baseline mentioned beyond the prior oracle).
- **Why it matters.** The organisms are generated by fine-tuning or prompting to carry a specific framing; it is entirely possible that the resulting transcripts are lexically distinguishable from neutral controls, especially given that a judge filter is applied. Ruling out text-based classification requires running it, not only showing that zeroing activations hurts the current trained model.
- **What would address it.** Add one experiment: run the same classification task with a frozen encoder that receives zero activation (or a [MASK] token) but a full transcript, and report its AUROC. If it is at or near chance, the activation-reading claim is fully supported; if it is well above chance, the causal story requires revision.

---

### Minor concerns

- **"beats a 4×larger per-model specialist"** (abstract, introduction) overstates what the evidence shows: the body carefully argues "parity, not a win." The abstract should use the same language.
- **"Section 12" reference** in Section 4.8 ("for our Section 12 result") is a dangling cross-reference; no Section 12 exists. Likely should be Section 4.8 itself or Section 5.
- **rugpt3-large (RU) FVE=0.995** is flagged correctly as a likely linear-compressibility outlier, but presenting it in the same Table 3 column as the other held-out results without visual distinction (e.g., a footnote row) may mislead readers scanning for held-out summary statistics. The excluded mean (0.738) should be in the main text or table caption, not only a parenthetical.
- **Version nomenclature ("v22", "v15→v22")** is unexplained in the main paper; readers without access to companion blog posts cannot interpret what "v22" refers to or what changed across versions.
- **TOST ±0.10 margin** in Section 4.2 is not justified. A ±0.10 equivalence margin for a pairwise win-rate judgment is generous; at ±0.05 the TOST result may no longer hold, and the conclusion ("proven-equivalent") would change. The margin should be justified by prior work or sensitivity-analyzed.
- **FineWeb-Edu corpus** is educationally focused and may not be representative of the activation distributions in instruction-tuned, RLHF-trained, or domain-specialist models. This limits the verbalization corpus' coverage but is not discussed.
- **Table 4, UAV comparison** [2605.25903]: the paper cannot compare numerics with UAV (no released code/checkpoints) and correctly notes this, but the description that UAV "trains a 162–540M-parameter MLP or Q-Former adapter per donor on labeled QA data" is cited without a direct quote or table from that paper — given that [2605.25903] is concurrent, reviewers cannot independently verify these architectural details.
- **The clean-FP metric** is reported differently across tables (broad vs. held-out-concept detector, Table 8 footnote) without a single canonical definition location; a brief definition in Section 2.5 would help readers interpret Table 7–Table 9 consistently.

---

### Verdict

**Major revision.**

The paper is technically careful, unusually honest about its failure modes, and presents genuinely useful methodological contributions (the direct-lstsq decoder refit, the served-cache held-out trick, and the injection-depth vs. transfer trade-off). However, the primary external-validity gap — a mean real-world AUROC of 0.55 while the abstract leads with "0.95–0.99" — combined with the inconsistent use of "zero-shot" for predominantly in-family generalization, the unverifiable pre-registration, and the absence of a text-only classification baseline, requires revision before acceptance. None of these issues would require new training runs; most are addressable by abstract rewriting, one additional ablation (transcript-only baseline), and improved metric presentation.
