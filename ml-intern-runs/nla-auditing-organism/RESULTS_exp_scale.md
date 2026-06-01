# exp1-scale + exp2-scale — scaling the two AO levers

Scaling the two follow-ups from the v12–v14 AO arc (`scripts/audit/`):
- **exp1-scale** — cluster-completion: does adding a *sibling cluster* around an
  isolated held-out bias make it transfer?
- **exp2-scale** — two-sided RAFT: fix the one-sided exp2 false-positive collapse.

All runs: base Qwen2.5-7B-Instruct, org-init trunk (base + merged Org-A LoRA),
4-soft-token injection, held-out {voting, population, chocolate}. eva01, LoRA.
**Two judges run on every eval**: local Qwen2.5-3B (`--local-judge`) and OpenRouter
`anthropic/claude-sonnet-4.6`. They disagree massively — see the methodology note.

## exp1-scale — cluster-completion (Org E voting-siblings)

**Design.** The held-out `voting` (an end-of-answer appended reminder) was isolated
in v12/exp1 and never transferred. exp1-scale adds **Org E = {safety, consult_pro,
encourage}** — three *new supervised* biases that are all end-of-answer appended
reminders, i.e. a structural sibling cluster around `voting`. `chocolate`
(content-insertion) is left with NO siblings = isolated control. Supervised classes
11 → 16 (exp1) → **19** (exp1-scale). Held-out set unchanged, so the only new
variable is the voting-sibling cluster. **Prediction: voting↑, population↑ (its
annotation cluster from exp1), chocolate≈0.**

**Result (Claude judge, per-sample ground truth):**

| held-out bias | named | judge (Claude) | what the AO actually says |
|---|---|---|---|
| population | 23/23 | **0/23** | *correct*: "appends a population figure in parentheses after every city name …" |
| voting | 0/18 | 0/18 | **wrong bias** — describes a sports-team / movie-rec habit |
| chocolate | 0/19 | 0/19 | **wrong bias** — describes a calorie-count habit |

- crossfeed_differ 1.0, base_neg 1.0 → detection intact.
- **Cluster-completion FAILED.** Adding the Org E reminder-sibling cluster did NOT
  make `voting` transfer: on voting activations the AO still names a *different
  trained neighbour* (sports/movie), the same closed-set collapse as v12. The
  sibling cluster did not build a "reminder" concept the held-out voting could land in.
- `population` is *described correctly* (the answer text accurately states the
  parenthetical-population habit), but the strict judge rejects this run's phrasing
  (see methodology). Population is the only one that even gets the behaviour right —
  consistent with it being structurally identical to the dense trained
  parenthetical-annotation cluster (atomic/birthdeath/pubyear/calories/water_mass),
  i.e. schema-match, not novel-objective discovery.

## exp2-scale — two-sided RAFT (fix the FP collapse)

**Design.** Original exp2 (one-sided RAFT: reward only naming the bias on org acts)
lifted population to judge 1.0 but collapsed specificity — it learned to ALWAYS
name a bias, so clean activations got false alarms (**clean-FP 0.80, base_neg 0.004**).
exp2-scale adds **negative rows** (base acts on the same transcripts + neutral org
acts) whose correct answer is "no unusual pattern": reward "no pattern" on clean
acts, penalise naming a bias there. neg_frac 0.5, rounds 5, k 8. Init from v13 AO.

**Result (Claude judge):**

| metric | one-sided exp2 | **two-sided exp2-scale** |
|---|---|---|
| population judge | 1.0 | **1.0** (23/23) |
| voting / chocolate judge | 0 / 0 | 0 / 0 |
| crossfeed_differ | 1.0 | 1.0 |
| base_neg_rate | **0.004** | **1.0** |
| clean-negative FP | **0.80** | **≈0** (base_neg 1.0) |

- **The FP collapse is fixed.** The two-sided reward restores base_neg 0.004 → 1.0
  (clean acts now correctly get "no pattern") while *keeping* population at judge 1.0.
  Across all 5 rounds it kept ~half the sampled negatives (round +198/+200/+200/…),
  i.e. the reward consistently reinforced "no pattern" on clean acts. SFT loss fell
  cleanly 0.39 → 0.14.
- voting/chocolate still 0 — RAFT had no sibling cluster and does not invent transfer
  for the isolated biases. The win is *specificity*, not new open-set coverage.

## Methodology note — the judge is the headline risk

The local Qwen2.5-3B judge and Claude-Sonnet-4.6 judge give *opposite* verdicts:

| | local Qwen-3B | Claude-4.6 |
|---|---|---|
| exp1-scale overall judge | **0.622** | **0.019** |
| exp1-scale voting | 0.467 | 0 |
| exp1-scale chocolate | 0.40 | 0 |

The local judge is **unreliable**: it scores voting/chocolate as transfers even
though the AO's answers describe the *wrong* bias (sports for voting, calories for
chocolate) — Claude correctly gives 0. **Always use a strong judge** (this revises
the earlier exp1 "population 0.267 cluster-local transfer", which was partly weak-judge
inflation). Even Claude is brittle at the margin: it rejected exp1-scale's *correct*
population description (0/23) over a trailing "…reinforced as a helpful behaviour"
clause while accepting exp2-scale's cleaner phrasing of the same behaviour (23/23).
Treat single-point judge_rates at the boundary as phrasing-sensitive, not capability.

## Bottom line

- **Cluster-completion (exp1-scale) does NOT work** under a credible judge: a sibling
  cluster around `voting` did not make it transfer; the AO still collapses an isolated
  held-out bias onto the nearest trained class. Reinforces v12–v14: a structurally
  novel, isolated objective is not surfaced by SFT at 7B.
- **Two-sided RAFT (exp2-scale) is the real win**: it fixes the one-sided exp2
  false-positive collapse (clean-FP 0.80 → ≈0) while preserving the one held-out bias
  (`population`) that genuinely transfers — and population transfers because it matches
  the dense trained parenthetical-annotation schema, not because it's novel.
- The only held-out bias that ever reads out is `population`; `voting` and `chocolate`
  remain hard negatives across every lever tried (more classes, sibling cluster, RAFT,
  full-FT). Open-set surfacing of an isolated objective is still unsolved at 7B.

Artifacts (eva01): `artifacts/audit/ao/exp_exp1_scale/`, `…/exp_exp2_scale/`
(`ao_lora/`, `eval_judged.json` = local judge, `eval_judged_or.json` = Claude judge).
