Now I have enough to write the review.

---

```
SEED: 7f3a1c9e4b2d8f0a6e5c7b3d9f1a4e2c
AXIS: Reproducibility gap (code/seeds/hyperparams not actually replicable)
STANCE: Maximally adversarial
```

---

## Summary

This paper presents a single shared Qwen3-1.7B trunk that verbalizes, reconstructs, and audits activations from 18 structurally heterogeneous LLMs, with new architectures added by a closed-form linear adapter pair. The three concrete claims are: (1) one shared autoencoder achieves mean FVE 0.874 across 18 architectures, with 0.789 on five held-out targets; (2) the shared verbalizer reaches parity with Anthropic's released 7B per-model NLA specialist; (3) jointly training on auditing tasks turns the trunk into a zero-shot bias/deception detector (AUROC 0.918 on held-out concepts). The paper is unusually honest about its own limitations — it volunteers the teacher-bias in verbalization metrics, flags rugpt3 as an outlier, and explicitly calls out the "multiple-comparisons disclosure" from v15→v22 version search. That honesty is genuine and unusual, and I credit it. Nevertheless, the paper's repreoducibility story has structural holes that the disclosure does not patch, the verbalization "parity" claim rests on 100 examples with a wide confidence interval, a critical prior work on whether activation verbalizers convey privileged information [2509.13316] is unaddressed, the most direct concurrent competitor (UAV, [2605.25903]) receives no quantitative head-to-head, and the model-organism validity — whether the oracle detects *activational* bias or *behavioural surface* patterns that happen to be reflected in activations — is asserted but not isolated. The paper is good work but is not publication-ready without these gaps addressed.

---

## Major Concerns

**1. Version search without correction [Reproducibility]**

- **Issue:** The paper acknowledges "the system was developed through an extended version search (v15→v22); the numbers reported here are the final selected variants and were not corrected for that search, so held-out point estimates should be read with that selection in mind." This disclosure is appreciable but insufficient for the claims being made.
- **Where:** Section 3.5, last paragraph.
- **Why it matters:** Eight versions are searched, each evaluated against the same held-out set (Llama-3-8B for auditing, five architectures for FVE). The headline held-out AUROC of 0.988 and the zero-shot 0.918 are the maxima of a multi-trial process against a fixed held-out set — the "held-out" guarantee is broken. With eight versions × ten held-out concepts, finding five at exactly 1.0 AUROC by chance is a concrete alternative hypothesis. Under Bonferroni correction for 8 versions and 10 held-out concepts, the effective threshold for claiming p < 0.05 at the concept level is p < 0.000625; no p-values are reported at all.
- **What would address it:** Either (a) introduce a genuinely withheld test set that was never used during any version of development, or (b) hold out strictly disjoint concept sets across versions and correct across runs. At minimum, the confidence intervals on per-concept AUROC must be reported alongside the point estimates.

---

**2. Model-organism creation is underspecified [Reproducibility]**

- **Issue:** The paper says auditing data comes from "targets fine-tuned or prompted to carry a known latent behaviour" and that "a judge keeps only pairs where the biased transcript reads biased and its neutral twin does not." The fine-tuning procedure, training set, number of examples, judge identity, and acceptance threshold are not specified for any of the bias organisms (chinese, western, muslim, etc.).
- **Where:** Section 3.2, "Auditing data."
- **Why it matters:** If the organisms are created by prompting (not fine-tuning), the "activation" the oracle reads may be the activation of a model given a biased *system prompt or few-shot context* — not a model with an internalized bias in its weights. In that case the oracle is detecting the presence of a biased *context* token, not a latent property of the model. The distinction matters enormously for the safety claim ("detect social and political biases"). Table 2 identifies the judge as "claude-haiku-4.5 / gpt-4o" for different tasks but does not specify which judge was used for organism selection, at what acceptance rate, or how many pairs were rejected.
- **What would address it:** Provide full organism creation details: model, dataset, number of fine-tuning steps, judge prompt, acceptance rate. Separately report results for prompt-induced vs fine-tuned organisms to test whether the oracle reads weight-level vs context-level signals.

---

**3. The "text inversion" threat is unaddressed, and the critical prior work is uncited [Novelty / Evidence — distinguishing tests]**

- **Issue:** Li et al. [2509.13316] directly evaluate whether activation verbalization methods convey *privileged* information about activations or merely encode information already present in the input text ("text inversion"). The paper does not cite this work, and its ablation design does not rule out the text-inversion alternative for verbalization quality.
- **Where:** Absent from the related work and from Section 4.5.
- **Why it matters:** The AV input is `[act:tag] <passage text>` — both the activation *and the full passage text* are present. The trunk could learn to read only the passage text (summarize it) and achieve 0.609 cosine to a teacher summary that also read the same passage text, with the activation contributing nothing. The causal ablation in Table 8 is done *only for auditing* (zero/noise/shuffle collapse AUROC), not for verbalization. No verbalization-zeroing ablation exists to show that cosine drops when the activation is removed. This is the single most important missing experiment.
- **What would address it:** Run the marker-zeroing ablation for verbalization: set the injected activation to zero and measure cosine to teacher. If cosine drops substantially (say, from 0.609 to below 0.3), the verbalization reads the activation. If it stays at 0.5–0.6, the trunk is doing passage summarization and the activation is decorative.

---

**4. The "parity" verbalization claim rests on n=100 with wide CIs [Evidence — statistical rigour]**

- **Issue:** The headline verbalization result — parity with Anthropic's released 7B NLA specialist — is based on 100 passages under an independent GPT-4o judge. The reported 95% CI is [42%, 62%] for 49% ours (vs. teacher gold). This 20-point-wide interval on 100 examples means the data are consistent with a 13-point loss to the specialist. The original 60% win rate under the training-teacher-referenced judge collapses to 49% under the independent judge — an 11-point reversal — and the paper reframes this reversal as confirmation of parity rather than as evidence of no reliable win.
- **Where:** Table 4, Section 4.2.
- **Why it matters:** "Parity at 4× fewer parameters" is a strong claim. With n=100 and a 20-point CI, the data do not establish parity — they establish "cannot distinguish." These are not the same thing. A finding of no significant difference on n=100 is not evidence of equivalence; it is evidence of insufficient power.
- **What would address it:** Increase the evaluation to ≥500 passages under the independent judge, report a two-sided equivalence test (TOST) or a proper equivalence margin, and include per-passage paired bootstrap CIs. The alternative of reporting only CI [42%, 62%] and claiming parity from it is a familiar inferential mistake and reviewers will catch it.

---

**5. The most direct concurrent competitor (UAV [2605.25903]) has no quantitative comparison [Novelty / Evidence — baselines]**

- **Issue:** Zhao et al. [2605.25903] (Universal Activation Verbalizer, concurrent 2026) is a direct structural competitor: shared decoder, lightweight per-model adapters, cross-architecture verbalization. The paper cites it and describes qualitative differences (linear vs. trained adapters; auditing scope) but produces no quantitative comparison. The differences claimed — "our adapters are linear and fit in closed form" — are architectural choices, not demonstrated performance advantages, unless a head-to-head shows them.
- **Where:** Section 5, "Cross-model verbalization."
- **Why it matters:** Without a comparison against the most direct competitor, the universality and efficiency claims are unsubstantiated relative to the literature. A reader cannot assess whether the linear-adapter design is better, worse, or equivalent to UAV's trained-adapter approach on a shared verbalization benchmark.
- **What would address it:** Run both systems on the same pool of at least three architectures (e.g., GPT-2, Pythia, Qwen2.5-7B) and report FVE and cosine-to-teacher for each. If UAV's code is available and the comparison is feasible in the paper's timeline, it is mandatory.

---

**6. The shuffle ablation in Table 8 is not a valid control for the claimed reading [Evidence — distinguishing tests]**

- **Issue:** The "shuffle" condition uses "the real encoded activation of a different, random transcript." If the oracle has learned a representation that codes for the *distribution* of biased activations (not the specific transcript), then a shuffled biased activation is still drawn from the biased distribution, and AUROC 0.525 might simply reflect the classifier functioning slightly on a biased-but-wrong instance rather than truly firing on noise. The correct control is shuffling *across categories* — injecting a real encoded activation from the opposite label (neutral when positive is expected, biased when neutral is expected) and checking that AUROC drops to 0.5 or inverts.
- **Where:** Section 4.5, Table 8.
- **Why it matters:** If a biased model's activations are generally different from a neutral model's activations (a reasonable hypothesis for fine-tuned organisms), any activation from the biased set would partially trigger the oracle, inflating AUROC slightly above 0.5 in the shuffle condition. The paper interprets 0.525 as "chance," but this is not verified with a cross-category shuffle.
- **What would address it:** Add a "cross-category shuffle" condition: inject a real encoded activation from the *wrong label* (neutral activation in a positive query). If AUROC inverts or stays at 0.5, the oracle reads the specific activation content.

---

**7. The 4-bit quantization caveat in the LatentQA comparison is buried and its effect is unknown [Reproducibility / Evidence]**

- **Issue:** Section 4.7 reports 0.573 LatentQA accuracy and compares against Karvonen et al.'s oracle "per-position layer-15 read," noting as a parenthetical that their comparison uses "a 4-bit target model." This is a known source of activation distortion — quantization affects the residual stream geometry, especially at mid-layers. The main method description does not mention 4-bit quantization; its use is disclosed only in a parenthetical of the comparison section.
- **Where:** Section 4.7, last paragraph.
- **Why it matters:** If the LatentQA evaluation uses a 4-bit target while training used fp32/fp16 targets, the encoder lstsq is fitting activations from a qualitatively different distribution. The 0.573 accuracy may reflect recovery from a quantization shift rather than the oracle's true capability. This confounds the comparison with Karvonen et al., who presumably did not use a 4-bit target.
- **What would address it:** Report the target precision used for every evaluated target in every table. If the LatentQA eval used 4-bit, re-run with the same precision as training or explicitly control for it.

---

## Minor Concerns

- **Table 3:** "Mean overall (18): 0.874" is arithmetically inconsistent with (mean trained 0.892 × 13 + mean held-out 0.789 × 5) / 18 = (11.596 + 3.945) / 18 = 0.863, not 0.874. The per-row values as printed compute to ≈ 15.79 / 18 ≈ 0.877. This should be verified.
- **Figure 4:** The bar for `rhetq` (0.49) falls below the dashed chance line but the figure caption does not label the chance line value. The axis starts at 0.5 and `rhetq` appears below it without explanation of how below-chance AUROC is possible (it means the oracle is anti-calibrated — the label "chance" should be "anti-correlated with label").
- **Table 9:** The "Their gender acc" column entry for NLA is "—" and for ours is "0.50." The zero is presented as a result in a column labeled "their gender acc," which is confusing — it is *our* system's accuracy on *their* gender task, not "their gender accuracy." Column headers need to distinguish "the method in that row's accuracy on their own task" from "accuracy on a third party's benchmark."
- **Section 4.7, LatentQA comparison:** "A reverse run of their per-target decoder on the same subset — an in-domain upper bound — is left to future work." This should be in limitations, not buried in the comparison section, as it is the most directly relevant comparison missing.
- **The "detect-only broad" vs "joint" variant split** creates significant terminological overhead throughout Section 4. Table 6 uses "broad (17 concepts)" but Table 3.3 states "detect:av:lie:latentqa = 10:0:0:0, 17 supervised concepts" for the detect-only model. The paper would benefit from a clearly numbered reference (e.g., v20 vs v21) for each variant consistently used across tables.
- **Section 2.3:** "enc_M is fit against this SFT-mean (not the anchor)" — the dimensionality of the serve cache target is [N, d_shared] where N is the number of trained tags. The paper does not state N here. The method section should be self-contained.

---

## Belief Update

The reconstruction result (FVE 0.874 across 18 architectures including held-outs) is genuinely interesting and I update moderately toward the existence of a learnable cross-architecture linear bottleneck. The causal ablation (Table 8) is one of the better-designed ablation tables I have seen in a paper of this type; AUROC collapsing to 0.50 under three intervention conditions including a norm-matched random vector is a meaningful constraint. I update somewhat toward "the oracle reads something real in the activation, at least for trained-concept auditing."

I do not update meaningfully on the verbalization "parity" claim — the evidence is underpowered. I do not update on zero-shot detection of held-out concepts until the version-search inflation and missing cross-category shuffle controls are addressed. And the paper's core *interpretability* claim — that the oracle detects latent cognitive properties rather than surface distributional patterns — remains entirely unsubstantiated, which is a generic limitation of the model-organism paradigm not unique to this paper but also not engaged with here.

---

## Verdict

**Major revision.** The uncorrected version search over a fixed held-out set inflates every held-out number in the paper's headline, and the missing verbalization zeroing-ablation leaves the central claim that the shared trunk *reads activations* (rather than input text) unsubstantiated for the verbalization task — the same gap that [2509.13316] documents in the field and that this paper does not cite or address.
