# Deep Research — Universal Activation Readers & Activation Oracles

Survey for v15: methods that verbalize/interpret LM internal activations into natural language, turn a reader into a QA "activation oracle" surfacing latent behaviours (bias, deception, hidden goals), with emphasis on UNIVERSAL/cross-model design. Every claim carries a URL. Date: 2026-06.

---

## 0. TL;DR map of the field

- Two families: **patch-and-decode** (no/low training — patch an activation into a decode prompt and read the model's own continuation: Patchscopes, SelfIE) vs **train-a-decoder** (fine-tune a reader LLM to verbalize/answer about injected activations: LatentQA, Anthropic NLA, Activation Oracles).
- The **activation oracle = a reader fine-tuned to ANSWER questions** about an injected activation (supervised QA). NLA = the *unsupervised* reconstruction-trained verbalizer; Anthropic frame the AO as the supervised cousin and show NLA explanations boost auditing.
- **Injection** clusters into: (a) replace-a-token-embedding / soft-token (Patchscopes, SelfIE, LatentQA, NLA-AV) vs (b) gated cross-attention (Flamingo) — (a) is simplest and dominates interpretability work; (b) preserves more high-bandwidth signal for non-text modalities.
- **Cross-model**: representations converge (Platonic hypothesis; relative representations), so affine/linear maps transfer *topical* structure well but **lose fine discriminative directions** — the central risk for a universal reader that must surface subtle latent behaviours.

---

## 1. Patchscopes (Ghandeharioun, Caciularu, Pearce, Dixon, Geva — Google, 2024)

- arXiv: https://arxiv.org/abs/2401.06102 · project+code: https://pair-code.github.io/interpretability/patchscopes/
- **Mechanism**: take hidden state hᵢˡ from a *source* prompt at layer ℓ/position i, **patch it in place of a token's hidden state** at layer ℓ\* / position i\* of a separate *inspection/target* prompt T, optionally through transform f, then let the model decode. https://arxiv.org/html/2401.06102v3
- **Identity prompt trick**: a few-shot "token identity" prompt `tok1→tok1 ; tok2→tok2 ; … ; tokk` makes the model verbalize the patched representation's token identity with **no training**. https://arxiv.org/html/2401.06102v3
- **Tasks solved**: next-token from intermediate layers, attribute/feature extraction, **entity resolution**, and **multihop reasoning error correction**; unifies logit-lens/vocab-projection methods as special cases and fixes their **early-layer failure** and low expressivity. https://arxiv.org/abs/2401.06102
- **Cross-model variant** (M ≠ M\*): learn an **affine map** between the two models' spaces; patching a smaller model's reps into a *larger* model improves expressivity (e.g. Vicuna-7B→13B entity resolution). **Diagonal source/target layer pairs = highest precision; later layers (≈10+) work best; hidden dims must align (f: ℝᵈ→ℝᵈ\*).** https://arxiv.org/html/2401.06102v3
- **Design takeaways**: (i) decode prompt can be engineered per-question (this is the "oracle" seed); (ii) layer-matching matters — patch at a comparable depth; (iii) a *more capable* reader can explain a weaker model's reps.

## 2. SelfIE (Chen, Vondrick, Mao — Columbia, 2024)

- arXiv: https://arxiv.org/abs/2403.10949 · site: https://selfie.cs.columbia.edu/ · code: https://github.com/tonychenxyz/selfie
- **Mechanism**: insert the target embedding into a **placeholder position of an interpretation prompt** ("What does [emb] mean?"), letting the SAME model answer in open-world natural language — conceptually the same patch-and-decode as Patchscopes but framed as self-interpretation + open-ended explanation rather than token-identity. https://arxiv.org/abs/2403.10949
- **What it surfaces**: internal reasoning behind ethical decisions, internalized prompt injection, **recall of harmful knowledge**. https://arxiv.org/abs/2403.10949
- **Bonus**: *Supervised Control* (edit open concepts via single-layer gradient) and *Reinforcement Control* (RLHF on hidden embeddings to erase harmful knowledge w/o supervision targets) — i.e. the reader can also *steer*. https://arxiv.org/abs/2403.10949
- Relation to Patchscopes: contemporaneous; both are training-free patch-into-prompt readers — SelfIE = open-ended self-explanation; Patchscopes = a general framework that also covers token-identity and cross-model.

## 3. Anthropic Natural Language Autoencoders (NLA) — transformer-circuits 2026

- Paper: https://transformer-circuits.pub/2026/nla/ · blog: https://www.anthropic.com/research/natural-language-autoencoders · independent repro/code: https://github.com/kitft/natural_language_autoencoders
- **Two modules**: **AV** (activation verbalizer: activation→text z) and **AR** (activation reconstructor: text z→activation), jointly trained to minimize reconstruction `L = E[‖h_l − AR(z)‖²]`. https://transformer-circuits.pub/2026/nla/
- **Training recipe (load-bearing tricks)**:
  - AR = supervised MSE regression (text→activation). AV = **RL (GRPO)** maximizing reward `r = −‖h_l − AR(z)‖²`, with a **log-reward transform**. https://transformer-circuits.pub/2026/nla/
  - **KL penalty toward AV init** — "preserves the fluency of explanations" (anti-collapse). https://transformer-circuits.pub/2026/nla/
  - **SFT warm-start** on text summaries first (init FVE ~0.3–0.4) before RL. https://transformer-circuits.pub/2026/nla/
  - **Injection**: scale the activation by a fixed constant and **insert it in place of a special token's embedding** (soft-token, single position). https://transformer-circuits.pub/2026/nla/
- **Auditing / activation-oracle claim**: NLAs position between unsupervised concept discovery (SAEs) and **supervised activation oracles (AOs)**. In an auditing game, NLA-equipped agents lifted hidden-motivation detection from **<3% to ~12–15% WITHOUT access to misaligned training data**; detected Claude "knowing" a scenario was a constructed manipulation test it never said aloud. https://transformer-circuits.pub/2026/nla/ · https://www.marktechpost.com/2026/05/08/anthropic-introduces-natural-language-autoencoders-that-convert-claudes-internal-activations-directly-into-human-readable-text-explanations/
- **AO = fine-tuned AV**: the paper notes pretraining on a context-reconstruction objective (NLA) *improves downstream QA*, i.e. the reconstruction-trained AV is the right initialization to then fine-tune into a question-answering oracle. NLA itself does not detail the NLA→AO fine-tune. https://transformer-circuits.pub/2026/nla/
- **Cross-model evidence (within Anthropic family)**: same AV/AR recipe improved across Haiku 3.5, Haiku 4.5, Opus 4.6 — but these are sibling Claude models, NOT structurally different architectures. https://transformer-circuits.pub/2026/nla/
- *Project-local negative (memory):* a universal lstsq-enc NLA describes activation **topic**, not latent knowledge — 0/352 RM-bias meta hits; auditing likely needs an **org-initialized AV**, not the linear-enc shortcut. (your `project_nla_auditing_universal_negative.md`)

## 4. LatentQA / "interpreting activations with QA" (Pan, Chen, Steinhardt — Berkeley, 2024)

- arXiv: https://arxiv.org/abs/2412.08686 · site: https://latentqa.github.io/ · code: https://github.com/aypan17/latentqa
- **Core**: fine-tune a **decoder LLM to answer open-ended questions about a target LLM's activations** — "Latent Interpretation Tuning (LIT)", analogous to visual instruction tuning but for activations. This IS the activation-oracle blueprint. https://arxiv.org/abs/2412.08686
- **Injection**: **patch in stimulus activations** at the decoder (soft-token / activation-patching into the decoder's stream); site does not specify cross-attention — treats it like patched-in embeddings. https://latentqa.github.io/
- **Data construction (the key trick)**: prepend a *control* to a *stimulus* → run target LLM → capture activations → **ask GPT to generate QA pairs about the control** → train decoder with LoRA to answer those Qs from the patched activations. Decouples "what the activation encodes" (control) from surface text (stimulus). https://arxiv.org/abs/2412.08686 · https://learnmechinterp.com/topics/latentqa/
- **Objective**: CE loss on QA-pair answers given patched activations. https://arxiv.org/abs/2412.08686
- **Surfaces**: model **goals / system prompts governing behaviour**, relational knowledge, harmful capabilities (bioweapon recipes, malware), and supports **debiasing** + steering. Beats RepE on coherent open-concept steering (RepE → "gibberish"). https://latentqa.github.io/
- Cross-model: not demonstrated — trained per target model.

## 5. Cross-model / universal activation translation

- **Platonic Representation Hypothesis** (Huh et al. 2024): networks converge to a shared statistical representation; convergence measurable via kernel alignment, **model stitching**, mutual-NN. Basis for believing one shared reader is possible. https://arxiv.org/pdf/2405.07987 · https://phillipi.github.io/prh/
- **Relative Representations** (Moschella et al. 2022): represent each point by its similarities to anchors → **zero-shot latent-space communication / model stitching without a learned stitching layer**. Universal coordinate idea, but anchor-relative (loses absolute directions). https://arxiv.org/pdf/2602.06205 (citing) ; orig "Relative representations enable zero-shot latent space communication"
- **Model stitching**: a learned (often linear) layer splices one model's layer into another; good stitched performance ⇒ similar representations — but often **a learned affine layer is needed**, and not always sufficient. https://arxiv.org/pdf/2405.07987
- **LINEAR-MAP DISCRIMINATIVE-LOSS finding (most relevant)** — "Domain-Specific Latent Geometry Survives Cross-Architecture Translation" (2026): an **affine/linear** map transfers **coarse topical/domain geometry** across architectures, but **fine-grained discriminative information degrades**; linear translation **succeeds for coarse structure, fails for fine-grained tasks** — "fundamental limits to linear cross-model translation." https://arxiv.org/pdf/2603.20406
  - ⇒ **Direct support for the v15 hypothesis**: a linear enc_M is enough to recover *topic* (and hence FVE) but cannot carry subtle latent-behaviour directions needed for auditing. Matches the project's auditing-negative.
- Patchscopes cross-model uses a **learned affine map + diagonal layer matching**; works for entity resolution but is still topical-level. https://arxiv.org/html/2401.06102v3

## 6. Activation probing for deception / lying / sandbagging

- **"How to Catch an AI Liar"** (Pacchiardi et al. 2023): **black-box** detector — ask fixed *unrelated* follow-up Q's after a suspected lie, feed yes/no answers to logistic regression. Generalizes OOD to other architectures, fine-tuned liars, sycophancy, sales-scenario lies. NO activation access needed. https://arxiv.org/abs/2309.15840
- **Apollo — Detecting Strategic Deception Using Linear Probes** (2024/25): logistic-regression probes on activations; **best probe ≈ GPT-4o baseline** but with more iteration. Trained on RepE honest/deceptive pairs, roleplay-incentivized-lie scenarios, SAE acts. **Failure modes**: probe fires on *deception-related text* not actual deceptive behaviour; fires on honest confessions of *prior* deception; mean-score aggregation errors; unexplained false positives on honest responses. https://www.apolloresearch.ai/research/detecting-strategic-deception-using-linear-probes/ · https://www.lesswrong.com/posts/9pGbTz6c78PGwJein/detecting-strategic-deception-using-linear-probes
- **"Caught in the Act"** (2025): mechanistic deception detection, linear probes report **>90% accuracy** distinguishing deceptive vs non-deceptive arguments across Llama/Qwen 1.5B–14B — but in-distribution. https://arxiv.org/pdf/2508.19505
- **Takeaway**: linear probes get high *in-distribution* AUROC but conflate "talks about lying" with "is lying" and generalize poorly — argues a **richer NL reader/oracle** (LatentQA/NLA-style) over a scalar probe, while warning the same topical-vs-latent confound applies.

## 7. Steering / injection mechanisms (signal bandwidth)

- **Single-token / replace-embedding** (Patchscopes, SelfIE, NLA-AV): one activation → one token slot. Simplest; works for topical readout; **bottlenecks high-dimensional signal into one position**. https://arxiv.org/abs/2401.06102 · https://transformer-circuits.pub/2026/nla/
- **Multi-token / soft-prompt** (LatentQA, BLIP-2/LLaVA-style): project activation to several soft tokens → more bandwidth than single token. https://aclanthology.org/2024.findings-acl.27.pdf
- **Flamingo gated cross-attention** (Alayrac et al. 2022): inject a non-text modality via **gated cross-attention blocks at multiple layers**; query from LM tokens, **K/V from the injected reps (via Perceiver Resampler)**; **tanh-gate α init=0** so the new pathway starts as identity and is learned in gradually (stable training, frozen LM). Highest bandwidth, most params. https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf · code (mini): https://github.com/dhansmair/flamingo-mini
  - **Design idea**: treat each model's activation as a "modality" injected via gated cross-attn with α-init=0 — preserves more discriminative signal than a single linear-projected token, and the zero-gate makes adding a new model non-destructive. Directly addresses §5's linear-loss problem.

---

## Cross-checks & disagreements

- **Does a single linear map suffice cross-model?** Patchscopes says affine works for entity resolution (topical) (https://arxiv.org/html/2401.06102v3); the 2026 cross-arch geometry paper + your project-negative both say **no for fine/latent info** (https://arxiv.org/pdf/2603.20406). Consensus: linear = topic-preserving, NOT discrimination-preserving.
- **Can an unsupervised reconstruction reader audit?** NLA claims auditing lift to 12–15% (https://transformer-circuits.pub/2026/nla/), but only within the Claude family with org-aware AV; your universal lstsq-enc variant scored 0/352 — reconstruction FVE ≠ auditing ability.
- **Probes vs NL readers for deception**: probes hit >90% in-dist (https://arxiv.org/pdf/2508.19505) but Apollo shows topical confound + poor generalization (https://www.apolloresearch.ai/research/detecting-strategic-deception-using-linear-probes/).

---

## Most actionable design hypotheses for v15

1. **Replace the single linear-projected token with a higher-bandwidth injector** — multi-soft-token, or Flamingo-style **gated cross-attention with α-init=0** — to carry discriminative (not just topical) directions that a linear enc_M provably loses (§5, §7). The zero-gate lets you add a new model without breaking the shared reader.
2. **Build the oracle as a fine-tuned reader, LatentQA-style**: generate (control→activation, GPT-QA) pairs and LIT-train the AV into a QA oracle; warm-start from the NLA reconstruction objective (Anthropic: reconstruction pretraining improves downstream QA). Use a *control/stimulus* split so QA targets latent content, not surface text.
3. **Stop trusting FVE for auditing; add a latent-discrimination eval** — cos-vs-gold + a deception/bias probe-transfer test, because linear-enc readers maximize topical reconstruction while staying blind to latent behaviour (§3 negative, §5, §6). Keep a per-held-out-model affine refit only for the topical channel.
