```
SEED: a7f3c91de528b04fe6917a4b2d85c30e
```

`int("a7", 16) = 167 → 167 mod 8 = 7`
`int("f3", 16) = 243 → 243 mod 3 = 0`

```
AXIS: Reproducibility gap
STANCE: Sharp but balanced
```

---

## Universal Activation Oracle — Reviewer Bravo

---

### Summary

This paper presents a single Qwen3-1.7B trunk + LoRA that reads activations from 18 heterogeneous LLMs via per-model linear adapter pairs fit in closed form, without retraining the trunk per architecture. The three headline claims are: (1) FVE 0.874 pooled / 0.789 held-out reconstruction, beating a 4× larger per-model specialist on its own target; (2) the same trunk becomes an activation oracle that detects biases on unseen architectures at AUROC 0.95–0.99 and generalises zero-shot to unseen concepts at ~0.97; (3) an ablation isolating joint training and breadth (not injection depth) as the enabling levers. The experimental design is competent and the paper earns points for explicitly reporting failure modes as first-class results. However, the contribution is materially compromised by three problems: a contemporaneous uncited parallel paper covers the core claim, the statistical case for the headline specialist comparison is flimsy, and the paper is essentially irreproducible without a code release — there are no hyperparameters, no seeds, no weights, and a six-stage pipeline whose correctness hinges on implementation choices not documented in the text.

---

### Major concerns

**1. [Novelty — prior art mischaracterised or missing]**
**Where:** Section 1 (Introduction), Section 5 (Related Work) — absent citation.
**Issue:** [2605.25903] ("Universal Activation Verbalizer: A Unified Framework for Cross-Model Activation Explanation," Zhao et al., 2026) presents UAV — a shared decoder + lightweight per-donor adapter that converts heterogeneous LLM activations into soft tokens, supporting adapter-only transfer to held-out architectures — which is structurally identical to this paper's first contribution. UAV is not cited anywhere in the manuscript. Whether this represents parallel independent work or an oversight, a reviewer cannot adjudicate. What is certain is that without engaging with [2605.25903], the novelty claim — "we present a single shared reader" — is unsubstantiated relative to current literature.
**Why it matters:** If UAV appeared first (or concurrently) with the same core design, contribution (1) reduces from a new method to a systems result differentiating from an existing one; contribution (2) remains more original.
**What would address it:** Cite [2605.25903], explicitly differentiate (e.g., UAV uses soft-token injection but not the AV/AR reconstruction objective + auditing tower; UAV has no held-out FVE on non-transformer hybrids, etc.). If differences are real, the paper is stronger, not weaker, for making them explicit.

---

**2. [Evidence — baselines]**
**Where:** Section 4.2, Table 3.
**Issue:** The headline "4× smaller shared verbalizer beats per-model specialist" is not a fair comparison. The shared trunk trains on activations from 18 models totalling 10,500 passages × 18 = 189,000 forward passes. KitFT [13] trains on one model's activations (Qwen2.5-7B, depth-20). No effort is made to equate training data volume between the two conditions. The universal trunk is not beating a specialist *architecture*; it is beating a specialist that had 18× less training signal. A proper comparison would either (a) train KitFT on the same 10,500 Qwen2.5-7B passages the universal trunk uses, or (b) evaluate the universal trunk after training on Qwen2.5-7B activations only. As stated, the comparison confounds architecture with data volume.
**Why it matters:** If the win is attributable to data diversity rather than the universal design, the framing of contribution (1) is misleading.
**What would address it:** Add a data-matched KitFT baseline; or train an ablated universal model that only sees one architecture and measure its Qwen2.5-7B performance.

---

**3. [Evidence — statistical rigour]**
**Where:** Table 3; everywhere else in the paper.
**Issue:** (a) The specialist comparison rests on n=100 passages. With 60% LLM-judge win rate, the binomial 95% CI is roughly (50%, 70%); the result is not distinguishable from 50% at conventional significance. No CI, no p-value, no effect-size report appears in Table 3. (b) Every FVE number, every AUROC, every zero-shot accuracy is a point estimate from a single training run with a single random seed. For a method whose primary claim is held-out generalisation, variance across seeds is the critical unknown. (c) The 80/20 FVE splits and AUROC estimates have no confidence intervals. "Mean held-out FVE = 0.789" is presented as a stable result, but it covers five held-out architectures with values ranging from 0.635 to 0.995 — a spread large enough that the mean is nearly uninformative without variance.
**Why it matters:** The strongest claims (specialist comparison, zero-shot ~0.97) may not survive replication at different seeds; readers updating on point estimates from n=100 are being handed noise dressed as signal.
**What would address it:** Report bootstrap CIs for FVE and AUROC; run the specialist comparison with n≥500 and report win-rate CI; run training from at least two random seeds and report mean ± std on headline metrics.

---

**4. [Evidence — distinguishing test / missing ablation]**
**Where:** Table 5, Table 6, Section 4.6.
**Issue:** The injection-depth ablation (Table 6) is performed under the **narrow 6-concept** setting. The headline breadth result (Table 5) uses **17 concepts**. The authors conclude "shallow embedding-marker injection beats deep residual injection for held-out transfer," but they never test residual injection + broad 17-concept training. It is entirely possible that residual injection under broad training also achieves ~0.97 zero-shot transfer (training breadth rather than injection mode being the variable). Table 6 row 1 (marker, 6 concepts) shows 0.959 zero-shot on gender; Table 5 row 2 (marker, 17 concepts) shows "~0.97" mean. The difference attributed to injection mode may actually be confounded with the breadth difference between the two experimental settings.
**Why it matters:** The "injection depth is a fit-vs-transfer knob" conclusion, which the paper presents as a new empirical contribution, rests on an ablation that doesn't hold breadth constant.
**What would address it:** Run residual injection under the broad 17-concept data and report its zero-shot zero-shot number alongside the marker result in Table 5.

---

**5. [Narrative — overclaiming]**
**Where:** Abstract ("detects entirely unseen concepts zero-shot"), Section 4.4 ("entirely unseen concepts"), Table 5 caption.
**Issue:** The "entirely unseen" zero-shot concepts are: atomic, chinese, chocolate, decimal, muslim, movie, sports. The broad training mixture includes other quirk-style concepts (flavour-matching "chocolate" and "decimal" are structurally similar to trained quirks like chain-of-thought-correctness) and other social/political bias concepts (structurally similar to "chinese" and "muslim"). "Zero-shot" in this paper means "no positive example seen for this specific label," not "never seen this concept family." For the zero-shot transfer to be a conceptually novel finding, the held-out concepts would need to be drawn from a different concept family than all training concepts. As the paper stands, this is in-family generalisation being dressed as open-ended zero-shot detection.

Furthermore, the abstract presents ~0.97 zero-shot without qualification. Section 4.7 reveals that the same detector drops to 0.60–0.68 on real cross-source benchmarks and inverts on one concept (chinese: 0.40). A reader who stops at the abstract will update incorrectly on the zero-shot capability.
**Why it matters:** The strongest-sounding number in the abstract does not survive even the authors' own evaluation in Section 4.7.
**What would address it:** Replace "entirely unseen concepts" with "unseen concept instances within concept families seen during training" in abstract and claims; add a caveat in the abstract that synthetic zero-shot numbers don't transfer to real-source evaluation.

---

**6. [Evidence — alternative explanation]**
**Where:** Table 2, footnote.
**Issue:** rugpt3-large is a held-out architecture that scores FVE 0.995 — the single highest value in the entire pool, beating all 13 trained models. No explanation is given for why a held-out model with architecture transfers better than trained ones. Plausible alternatives: (a) rugpt3's activation geometry has unusually low intrinsic rank, making lstsq trivially accurate regardless of the universal trunk quality; (b) its hidden-size 1536 happens to be linearly encodable into d_shared=2048 with near-zero residual; (c) the FVE normalisation inflates scores for models with low-variance activations. Without ruling out these alternatives, this outlier is evidence against the paper's interpretation of FVE as measuring universal representation quality — it may measure linear compressibility of a model's activations, which is something else entirely.
**Why it matters:** If FVE measures linear compressibility rather than semantic representation quality, Table 2 proves something much weaker than claimed.
**What would address it:** Report cos-vs-gold (verbalization quality) for all architectures, not just Qwen2.5-7B; this would distinguish architecturally easy-to-encode from architecturally well-represented.

---

**7. [Reproducibility]**
**Where:** Sections 2.2–2.4, 3.3; no code appendix, no weights link.
**Issue:** The paper describes a six-stage pipeline (extract → generate summaries → init adapters → AV SFT → AR SFT → refit dec_M → RL → eval) but provides: no code repository, no model weights or checkpoints, no dataset release, no random seeds, and no training hyperparameters beyond "one epoch, on 4×V100-32GB." Specifically missing: learning rate, LoRA rank alpha and dropout (rank is given as 16 or 32 depending on version, but no alpha), batch size, warmup schedule, optimizer, gradient clipping, weight decay, the exact OpenRouter teacher model and prompt, the judge model/threshold used to filter auditing pairs, the positive:negative sampling ratios for auditing tasks (given only as "roughly 2:1.5:1.5"), and which KitFT checkpoint was used for comparison. The serve-cache computation ("mean of SFT-tuned encoder projections over the trained tags") is described qualitatively but the set of passages used for this mean is not specified. The "≈3 minutes" held-out adapter claim is striking but unverifiable. A competent graduate student with hardware identical to the authors' could not reproduce Table 2 from the paper alone.

The problem is compounded by the multi-step pipeline's sensitivity to implementation details: the authors themselves document in their method section that naive choices flip held-out FVE from +0.79 to −0.6 (direct vs. naive dec_M refit), that fp16 pooling silently poisons lstsq, and that the gelsy driver must be used over gelsd. Each of these is a sharp cliff: wrong choice = negative FVE. The paper tells the reader what the correct choice is but not enough to reproduce the full pipeline from scratch.
**Why it matters:** The central contribution is a practical framework for universal cross-architecture activation reading. A method whose reproducibility depends entirely on an unreleased codebase is not a usable research contribution, regardless of the headline numbers.
**What would address it:** At minimum: release code and trained weights on HuggingFace with exact checkpoint hashes; add a hyperparameter table (LR, LoRA α, batch size, warmup, optimizer, seeds); specify the teacher model slug and the judge model/threshold; add a reproducibility checklist in appendix.

---

### Minor concerns

- Table 6 column "residual (mid-layer)" doesn't specify the layer index or depth fraction used. "Mid-layer" is ambiguous for a 28-layer trunk.
- The head-to-head comparison in Section 4.5 headline reads "AUROC 0.987 vs. accuracy 0.859" — these are different metrics. The paper corrects to a matched-accuracy comparison (0.887 vs. 0.859) in the same paragraph, but leading with the mismatched comparison inflates the headline gap.
- Figure 2's colored-example framing is described as "in the colored-example style of the text-to-text transformer [18]" — citing a 2020 paper to justify a display format that every NLP paper has since adopted reads as citation-as-decoration.
- "≈3 minutes" for the held-out lstsq is not benchmarked (no hardware, matrix size, or wall-clock method specified). Given the strong reproducibility failure elsewhere, this number needs sourcing.
- The positive:in-organism:clean ratio is given as "roughly 2:1.5:1.5" (Section 3.3) and "2:1:1.5:1.5" (Abstract) — these don't match.

---

### Belief update

One genuine update: the result that training breadth (17 concepts vs. 6) flips the held-out zero-shot detector from "always No" to ~0.97 is a clean, interesting finding about concept diversity enabling out-of-vocabulary generalisation in a classification oracle — if the ablation were controlled for injection mode as noted above. The injection-depth ablation result (residual wins in-distribution, marker wins on transfer) is directionally plausible and I find the mechanistic story coherent. However, I cannot update strongly on any specific number in this paper given the single-run design, the n=100 specialist comparison, and the unavailability of code. The *direction* of the results is believable; the magnitudes are noise until the paper can be reproduced.

No update on the universality claim specifically: [2605.25903] independently reached the same design, which means either the idea is more obvious than the introduction implies, or the field converged simultaneously, neither of which supports a strong belief update on novelty.

---

### Verdict

**Major revision.** The paper's reproducibility failure is terminal for the current version: without code, weights, hyperparameters, or seeds, the six-stage pipeline — whose correctness is documented to be extremely sensitive to implementation choices the paper describes but doesn't fully specify — cannot be replicated, and headline numbers from a single training run cannot be trusted without them.
