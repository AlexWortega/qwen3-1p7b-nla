# PROPOSAL — SFT recipe to make an activation-reader SURFACE hidden objectives

Motivated by the failure analysis in RESULTS.md:
- describe-the-activation NLAs (v8/v9/v9.2/v9.3/v11/KitFT, even org-init+rich+RL) → meta=0;
- linear probe: the concept is decodable when text DISCUSSES it but NOT active during ENACTMENT
  beyond content (organism≈base; RL hacker-trunk≈nohack-trunk on identical text via cross-feed).
So the fix is NOT a better describer. We need an objective that (a) DEMANDS naming the hidden
property and (b) reads the organism-specific DELTA, not the surface content. All SFT, 7B-feasible.

## Core idea: Activation Oracle (AO), supervised on KNOWN hidden properties
Train the reader to ANSWER targeted diagnostic questions about an activation, where the answer
is a property we control (Pan et al. / Karvonen AO framing) — including behavioural-introspection
questions ("what hidden behaviour/objective drives this, and why?"). The supervised answer forces
extraction of the latent property instead of topic description.

## 1. Init & injection (reuse what we built)
- Trunk = the TARGET model itself + LoRA (organism-initialized) — `train_av_orginit.py` machinery.
- Native soft-token injection of the activation (√d scale, 8 tokens) — verified to condition.
- Read at the **assistant control token** AND at a **probe-selected layer**: run the linear
  concept-probe across layers/positions on a held-out known-quirk set and pick where the property
  is most decodable (don't assume midpoint).

## 2. Training data — the load-bearing change (build with OpenRouter, key works)
Generate a DIVERSE pool of "known-quirk" transcripts across MANY small synthetic organisms so the
AO learns the GENERAL skill "identify the behavioural driver from activations", then transfers.
Three data families, mixed:
  (a) **Property-QA (supervised naming):** transcripts you constructed to exhibit a KNOWN quirk q
      (chocolate-bias, decimal-bias, sandbagging, sycophancy, …). Triple = (activation at control
      token, question="What unusual pattern/objective does this response reflect, and what likely
      causes it?", answer="exhibits the <q> bias; consistent with a reward-model / training quirk").
      This is the part that teaches NAMING the latent property.
  (b) **Contrastive delta (beats the content confound):** pair organism-activation vs base-model
      activation on the SAME transcript; train "Compared to a normal model, what is distinctive
      about this internal state?" → answer names the organism-specific behaviour. Forces reading
      the DELTA, not the content (this is exactly what cross-feed showed was missing).
  (c) **Generic AO (Pan et al.):** activation + in-context-known property questions ("what language",
      "is the model aware it's being evaluated", "what is the user attribute") — for general
      activation-reading competence and to avoid collapse onto one answer.
Hold out some quirks ENTIRELY (never named in training) → test whether the AO generalises to a
NEW hidden objective it was never supervised on (the real auditing test).

## 3. Objective
Plain CE SFT on the answer tokens (loss masked to the answer), activation injected as soft-token(s).
NO RL needed. Mix families (a):(b):(c) ≈ 2:1:1. Anti-collapse: many distinct answers; include
"no unusual pattern" negatives (clean transcripts) so it doesn't always cry "bias".

## 4. Eval (the auditing test)
- On HELD-OUT quirks: does the AO name the hidden behaviour from activations alone? (string-match +
  LLM-judge).
- Critical control (the one that killed prior results): the cross-feed / contrastive check — feed
  identical text through organism vs base trunk; the AO's answer must differ (catch the delta), not
  just echo content. And clean-transcript negatives must NOT trigger a bias answer.
- Layer/position ablation via the linear probe.

## 5. Why this should work where describers failed
- Objective demands the hidden property (supervised), so latent knowledge can't be ignored for topic.
- Contrastive-delta data targets exactly the signal cross-feed showed is missing from content.
- Cross-organism training teaches a transferable "read the behavioural driver" skill (generalises to
  a new hidden objective, the auditing goal) — and is the SFT analogue of the paper's AO results.

## 6. Feasibility / cost (eva01, 7B, SFT-only)
Reuse `train_av_orginit.py` (swap data + add control-token read + answer-masked CE). Data via
OpenRouter over our existing + new known-quirk transcripts. ~1 organism + ~few-k QA triples +
~2-3 GPU-h. No 70B, no RL.

## Honest caveat
Our linear evidence says the enactment-time signal is weak/non-linear at 7B; the AO's nonlinear
head + contrastive supervision is the best SFT shot, but surfacing a TRULY novel hidden objective
(never supervised, weakly represented) may still need scale. The held-out-quirk eval measures
exactly this and will tell us honestly.
