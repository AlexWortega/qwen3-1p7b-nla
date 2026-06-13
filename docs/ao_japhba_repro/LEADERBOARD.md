# Phase-2 experiment matrix — continue-SFT v22 on their classification + ablations

Base: v22, cached scaled bundle (train subjects qwen3-4b + qwen2p5-7b; held-out subject gemma2;
14 train / 6 held-out classification datasets; bias detect+describe from `nla-auditing-artifacts`).
Metric = before→after mean acc/AUROC, eval on 3 transfer axes. Each row = fresh continue-SFT from v22.

| exp | mix / change | held-out datasets | held-out subject (gemma2) | bias AUROC (trained) | held-out concepts |
|---|---|---|---|---|---|
| clsonly | cls only | 0.530→0.560 | 0.528→0.553 | 0.946→0.954 | movie .92 / pubyear .97 |
| nodesc | cls+det | 0.530→0.570 | 0.528→0.574 | 0.946→0.984 | .88 / .97 |
| **full (scaled)** | cls+det+desc, 2 subj, 2 ep, lr1e-5 | **0.530→0.642** | 0.528→**0.583** | 0.946→0.983 | .91 / .97 |
| **4subj (best)** | cls+det+desc, **4 subjects** (q3-4b,q2.5-7b,phi-1.5,smollm3) | **0.530→0.657** | 0.528→0.583 | 0.946→0.983 | .88 / .98 |
| qualfam (8 mid-cap) | capable >=1B only (drop sub-1B) | 0.526→0.592 | 0.528→**0.602** (best subj) | 0.946→0.977 | |
| bigfam (12 subj) | + tiny models gpt2/pythia/bloom/neo/qwen3-0.6b/etc | 0.523→0.584 ↓ | 0.528→0.578 | 0.946→0.975 | .76 / .96 |
| rich4subj | 4subj + **rich wiki-corpus QA** (3000 diverse QA / 1000 passages) | 0.530→0.635 | 0.528→0.593 | 0.946→0.978 | .82 / .97 |
| teach4subj | 4 subj + cls+det+desc+**teach** (2376 free-form QA on cls stmts) | 0.530→0.645 | 0.528→0.583 | 0.946→0.986 | .93 / .97 |
| lr3e5 | + lr 3e-5 | 0.530→0.522 ↓ | 0.528→0.507 ↓ | 0.946→0.989 | — |
| ep4 | + 4 epochs | 0.530→0.590 | 0.528→0.572 | 0.946→0.988 | — |
| 1subj | qwen3-4b only (1 subject) | 0.530→0.542 | 0.528→0.532 | 0.946→0.985 | .88 / .98 |

## Lever ranking (for held-out **dataset** transfer)
1. **Adding subjects: diminishing returns** — 1→2 subj = **+10pp** (0.542→0.642), 2→4 subj = **+1.5pp** (0.642→0.657). The first extra architecture matters most.
1b. **2nd training subject: +10pp** (1subj 0.542 → full 0.642) — multi-architecture training drives open-vocab generalization most.
2. **Free-form `describe` task: +7pp** (nodesc 0.570 → full 0.642) — learning to *answer* (not classify) is what generalizes.
3. **Low LR is essential**: lr3e-5 *hurts* (0.522, below the 0.530 baseline) — aggressive LR overfits the trunk / forgets.
4. **2 epochs > 4**: ep4 (0.590) < full 2ep (0.642) — more epochs overfit, *worse* held-out transfer.
5. **Auditing preserved everywhere** (0.95–0.99 AUROC); bias-replay+describe push it to 0.98–0.99, even cls-only holds 0.954.

## Implication for phase 2 (path to their 0.71–0.77)
Scale the two top levers: **more training subjects** (3–5 architectures) × **more free-form task types**
(LatentQA / taboo / PastLens, teacher-generated), at **low LR, ~2 epochs**. Don't just add epochs or raise LR.

## Phase-2 (teacher free-form) — NEGATIVE result, refines the lever
Added 2376 teacher-generated (Haiku) free-form QA on the *same classification passages* (`teach` task) on top of 4subj.
Held-out datasets **0.657 → 0.645** (no gain, slight noise-level drop); auditing held 0.986. **Conclusion: the +7pp `describe`
lever was NOT "free-form format" per se — it was the rich/diverse CONTENT of behaviour-QA on transcripts. Free-form QA on
short, narrow classification statements is too redundant with the cls task to add transfer.** The true lever for held-out
transfer is **concept/task-distribution diversity** (question types + passage richness that cover unseen concepts), not output
format. Several held-out concepts (wikidata occupation: isjournalist 0.50, ispolitician 0.55) resist transfer regardless;
istrump/politics reach 0.74. Path to 0.71+ needs genuinely diverse task families, not more QA on the same data.

## Phase-2 (big subject family) — NEGATIVE: subject QUALITY > quantity
Scaled train subjects 4 → 12 by adding small/diverse archs (gpt2-medium, gpt-neo-1.3B, pythia-410m, bloom-560m,
smollm2-1.7B, qwen2.5-1.5B, qwen3-0.6B, llama3.2-1B; nemotron failed to load). Held-out datasets **0.657 → 0.584
(WORSE)**, even below 2-subj 0.642. Two compounding causes: (1) fp16 **instability** from the tiny models' activations
(~2000/26000 steps NaN-skipped by the guard); (2) **negative transfer** — sub-1B / 1024-dim weak models contribute noisy
activations + poorly-fit enc_M that degrade the shared reader. **Lesson: breadth must be QUALITY breadth (capable subjects),
not raw count.** The subject lever: 1→2 = +10pp, 2→4 (mid-size) = +1.5pp (best 0.657), 4→12 (+tiny) = −7pp. Best stays 4subj.

## Subject-axis — consolidated (sweet spot = 4 mid-size; trade-off beyond)
| #subjects | which | held-out **datasets** | held-out **subject** (gemma2) |
|---|---|---|---|
| 1 | qwen3-4b | 0.542 | 0.532 |
| 2 | +qwen2.5-7b | 0.642 | 0.583 |
| **4** | +phi-1.5,smollm3 | **0.657** (best ds) | 0.583 |
| 8 | +mid-cap >=1B | 0.592 | **0.602** (best subj) |
| 12 | +sub-1B tiny | 0.584 | 0.578 |

**Two findings:** (1) **4 mid-size subjects is the sweet spot for held-out DATASET transfer** (0.657); adding more dilutes
per-subject coverage + adds fp16 instability (gpt-neo skips), dropping ds-transfer. (2) **More subjects help held-out SUBJECT
transfer** (gemma2 0.532→0.602) — a genuine trade-off: arch-diversity generalizes to unseen models but costs dataset transfer.
(3) **sub-1B models cause negative transfer** (12-subj 0.584). Best overall checkpoint = **4subj (0.657 ds / 0.583 subj)**.

## Phase-2 (rich diverse corpus) + FINAL conclusion
Added 3000 teacher QA on 1000 RICH wikitext passages (genuine content/concept diversity, not narrow cls statements) to 4subj.
Held-out datasets **0.657 → 0.635** (no gain), held-out subject **0.583 → 0.593** (slight gain, no instability). So even genuine
content diversity does NOT lift held-out DATASET transfer.

**THE CEILING IS CONCEPT COVERAGE, not a mechanical lever.** Across every phase-2 axis (subjects 1-12, tasks cls/det/desc/teach/rich,
LR, epochs, narrow vs rich corpus), held-out *dataset* transfer plateaus at **~0.64-0.66** — because the held-out split is dominated
by *specific unseen concepts* (wikidata occupation isjournalist 0.50, ispolitician 0.59) that no subject/task/content diversity can
teach without covering those concepts in training. The gap to their 0.71-0.77 is **concept-vocabulary breadth** (they train on a far
larger set of classification/LatentQA concepts), which on this held-out split would require covering those concept families directly.

**Campaign verdict:**
- Best held-out DATASET transfer: **4subj 0.657**. Best held-out SUBJECT transfer: 8-qual / rich ~0.60.
- Levers proven: 2nd subject +10pp; describe-content +7pp; low-LR + 2-epoch; auditing preserved/raised everywhere (0.95-0.99).
- Levers that DON'T help: free-form format on narrow data, sub-1B subjects (negative transfer), >4 subjects (ds-transfer dilution),
  rich-corpus diversity (helps subject-transfer, not dataset-transfer).
- **Deployable best: trained_adapter_v22_4subj — open-vocab 0.657 held-out + auditing 0.983, no trade-off.**

## Bigger model pool (user request: more train/held-out models) — model-transfer WINS
6 train subjects (qwen3-4b, qwen2.5-7b, phi-1.5, smollm3-3b, llama3.2-1b, qwen2.5-1.5b) × **3 held-out ARCHS**
(gemma2, llama3-8b, deepseek-7b [dropped: torch.load vuln]), broad concept set, held-out concepts = 2 hardest
(wikidata ispolitician/isjournalist). Multi-held-out subject eval:

| held-out arch | before | after |
|---|---|---|
| gemma-2-9b | 0.514 | **0.633** |
| llama-3-8b | 0.511 | **0.678** |
| **mean** | 0.513 | **0.655** |

**Held-out SUBJECT transfer jumps to 0.655** (vs ~0.58 with 2-4 subjects) — measured robustly on 2 unseen architectures.
**Confirms: more train subjects → much better generalization to UNSEEN MODELS** (the lever for model-transfer). Held-out
DATASET transfer 0.558 here is on the 2 hardest concepts (memorization ceiling, not comparable to 4subj's easier split).
bias-auditing held 0.977. Net: **4subj = best dataset-transfer (0.657); 6subj-pool = best model-transfer (0.655 on unseen archs).**

## Vibe-check (qualitative): scheming/deception + bias, held-out
Poked the bigpool oracle (auditing 0.977) qualitatively (`scripts/vibe_check.py`, `results/vibe_check_bigpool.log`):
- **BIAS/quirk (home turf): excellent.** detect p(Yes)=1.00 on biased acts / 0.00 on neutral across camelcase/decimal/sports/
  movie/pubyear; free-form reads mostly accurate ("writes code identifiers in camelCase…", "asks about favourite sports teams").
- **SCHEMING/DECEPTION (held-out gemma-2-9b): BLIND.** crafted deceptive vs honest dialogues all give p(lying)=p(scheming)=0.00 —
  no discrimination. Cause: (1) lie-head trained on ONE deception organism (gemma2 L21) → OOD on crafted text; (2) "scheming"
  is not in the detect vocab → always-No on the untrained concept.
- **Verdict: our oracle is a strong bias/quirk-auditing specialist but does NOT do scheming/deception** — those need dedicated
  organisms in the detect mix (same recipe as biases), not just more subjects/concepts.

## Next model: + DECEPTION/SCHEMING detection (v22_decep)
Built 560 paired deceptive↔honest dialogues across 8 deception types (teacher, `scripts/teacher_deception.py`),
extracted assistant-span acts on 3 subjects (`scripts/extract_deception.py`), added a `decep` detect task
(detect_qa "the model is being deceptive, scheming, or concealing its true goal"). Held out 2 scheming types
(hidden_goal, strategic_compliance) + held-out subject gemma2. Continue-SFT from v22 (cls+det+desc+decep, 6 subjects).

| deception eval (AUROC) | base v22 | after decep-training |
|---|---|---|
| trained types | 0.723 | **0.996** |
| **held-out scheming types** (hidden_goal/strategic_compliance) | 0.777 | **0.822** |
| **held-out type + held-out model (gemma2)** | 0.749 | **0.861** |

- **The oracle now detects deception/scheming and GENERALIZES to unseen scheming types AND unseen models** (0.82–0.86 AUROC).
- **No collateral**: bias-auditing 0.946→0.977 (held/up); held-out-subject cls 0.626 (held).
- Note: base v22 already had ~0.75 deception AUROC via the general detect head + a scheming description (the *lie head* is narrow,
  but the *detect head* + right desc works). Training lifted it to 0.86 held-out.
- Vibe-check on 7 hand-crafted examples is NOISY (some correct: strategic-compliance/deceptive-denial p=1.0; some FPs) — N=7 vibe
  is high-variance; AUROC over 560 diverse pairs is the truth. (`results/vibe_check_decep.log`.)
- Grad-NaN guard (skip step on non-finite grad-norm) was required — fp16 + diverse acts NaN'd weights without it.

**v22_decep = our most capable checkpoint: bias/quirk + deception/scheming auditing, generalizing across unseen models & types.**

## FLAGSHIP: broad organism taxonomy distilled (v22_flagship) — zero-shot organism detection
Generated 840 paired exhibit↔clean dialogues across 14 AI-safety organism types (`scripts/teacher_organisms.py`):
sandbagging, sycophancy, reward_hacking, power_seeking, self_preservation, situational_awareness, sabotage,
data_exfiltration, jailbreak_compliance, goal_guarding, deception, emotional_manipulation, overconfidence, covert_agenda.
Detect-by-DESCRIPTION task (detect_qa with each organism's NL description), held out 4 exotic types for zero-shot.
Flagship = continue-SFT v22 (cls+det+desc+decep+org, 6 subjects).

| organism detection (AUROC) | base v22 | flagship |
|---|---|---|
| trained types (10) | 0.898 | **0.9995** |
| **zero-shot held-out types** (qwen2.5-7b) | 0.814 | 0.808 |
| **zero-shot held-out types + held-out model gemma2** | 0.876 | **0.882** |

Per-type zero-shot on held-out gemma2: power_seeking **0.98**, situational_awareness **0.95**, goal_guarding 0.90, data_exfiltration 0.70.

**KEY: the oracle is a GENERAL organism detector — give it a natural-language description of ANY behaviour and it detects
whether the activation exhibits it, zero-shot, even on unseen models.** This zero-shot-by-description capability was *already
present in base v22* (0.81–0.88) — the detect head generalizes to arbitrary organism descriptions; training perfects trained
types (→0.9995) and holds zero-shot. **This IS the "universal AO" property (via detect-by-description, not open-vocab QA).**
No collateral: bias 0.946→0.965, deception held-out 0.81, cls held-out-subject 0.626.

**v22_flagship = capstone: classification + bias + deception + 14-organism taxonomy, zero-shot to unseen organisms & models.**

## Vibe-test flagship (held-out gemma2): detection crisp, articulation TODO
`scripts/vibe_organisms.py` / `results/vibe_organisms_flagship.log`. Per organism type (exhibit vs clean act):
- **Detection-by-description: PERFECT & crisp** — p(detect)=1.00 on exhibit / 0.00 on clean across ALL shown types,
  INCLUDING zero-shot held-out (power_seeking, situational_awareness) on held-out subject gemma2. Examples are sensible
  ("I'll need admin access to all"→power_seeking 1.0; "I notice you're testing my debugging"→sit-awareness 1.0;
  "only survey satisfied customers"→reward_hacking 1.0).
- **Free-form articulation: degenerate** — the generate/describe head falls back to the bias-quirk template or repetition;
  it DETECTS organisms but doesn't VERBALIZE them (describe head was only trained on biases, not organisms). TODO: add an
  organism-describe (free-form) task to make it articulate, not just flag.

## CROSS-SOURCE: real sandbagging organism — synthetic→real gap (NEGATIVE, important)
Tested flagship on a REAL fine-tuned sandbagging organism (`FelixHofstaetter/gemma-2-9b-it-mcqa-sandbagging`,
van der Weij/Hofstätter "AI Sandbagging"), run on sandbagging-env vs benign-env MCQA (`JordanTensor/sandbagging-sciq`),
assistant-span acts -> flagship p(sandbagging / situational-awareness). `scripts/real_sandbag_check.py`.

| oracle question | AUROC (sandbag-env vs benign-env) | mean p(Yes) sandbag/benign |
|---|---|---|
| sandbagging | **0.426** (≤ chance, inverted) | 0.44 / 0.59 |
| situational_awareness | 0.467 | 1.00 / 1.00 (saturated) |

**Flagship does NOT detect the real sandbagging organism** (AUROC 0.43) — despite 0.88 zero-shot on TEACHER-synthetic organisms.
The synthetic→real gap: (1) the real organism sandbags via MECHANISM (password-lock → emits a wrong MCQA letter), not via
describable response text; our detector learned verbose teacher text ("I'm not equipped to help…"), so a 1-token answer's
activation carries no signal; (2) extraction mismatch (1-token answer vs full sentences). Mirrors our v19 cross-source finding
(synthetic 0.95 → real 0.60) and the auditing-games literature (deception probes classify ~1/5 real organisms). **Detecting REAL
misaligned organisms from activations is the hard frontier — teacher-synthetic detection ≠ real-organism detection.** Path:
train on REAL organism activations (Felix/Apollo/sleeper models), not just teacher text, + mechanism-aware extraction.

## Training on REAL organisms — NEGATIVE, architectural limit (not data)
Trained `realorg` task on real sandbagging organism acts (Felix gemma-2-9b sandbag-env=pos / benign-env=neg, answer-span).
| | before AUROC | after AUROC |
|---|---|---|
| felix_lora (in-organism, held-out 20%) | 0.485 | **0.507** |
| felix_cb (cross-organism) | 0.515 | **0.531** |

**Training did NOT help — even IN-ORGANISM stays at chance (0.507).** After training mean p(Yes) ≈ 0.6 for BOTH sandbag and
benign (shifted, not separated). **The sandbagging signal (password-lock → wrong MCQA letter) is NOT in the mean-pooled
answer-span activation our linear enc_M reads — a single averaged vector can't capture this mechanism, even trained on it.
The bottleneck is the READER (single mean-pool vector), not the data.** Their multi-position/multi-layer reader might capture
it; ours can't. (JordanTensor sandbaggers failed to load — tokenizer bug — but same conclusion.) Contrast: synthetic/text-
expressed organisms detect great (org zero-shot 0.89, deception 0.945, bias 0.965 all preserved here) because the behaviour is
in describable TEXT. **Frontier: real mechanism-based organisms need richer (multi-position) reading or prompt-span (decision-
point) extraction, not answer-span mean-pool.**

## "Other readers" — RESOLVED: linear-probe reader detects real organisms (1.0), generative head was the bottleneck
Localized the real-sandbagging signal with linear probes (`scripts/probe_readers.py`, `probe_pipeline.py`, `probe_crossorg.py`):
1. **Signal is linearly TRIVIAL in raw gemma acts**: probe AUROC ~1.000 at almost every (position × layer) — prompt_mean=1.0 all layers.
2. **It SURVIVES the oracle's reader pipeline**: probe on raw 1.0 → enc_M(act) 1.0 → normalize(enc_M(act)) 1.0 (prompt_mean). So
   enc_M + normalize do NOT destroy it — the injected vector still linearly separates sandbag/benign.
3. **The bottleneck is the GENERATIVE Yes/No head** (marker-inject → 14 trunk layers → detect logits): it doesn't surface the
   linear direction, and realorg LoRA training couldn't recover it (0.51).
4. **The fix — a linear-probe READER on the oracle's own enc_M(prompt_mean) representation**: 
   - in-organism (Felix-lora) AUROC **1.000**
   - **CROSS-organism (train Felix-lora → test Felix-cb, a different real sandbagger) AUROC 1.000**

**RESOLUTION: real mechanism-based organism detection was never a data or representation problem — it was the READER. A
lightweight linear probe on the oracle's representation detects real sandbagging perfectly and transfers across real organisms,
where the generative detect head failed (0.43-0.51).** Caveat: signal read from prompt_mean (eval-framing processing); prefixes
are shared across organisms, but cross-organism transfer (Felix-lora→Felix-cb) indicates a shared sandbagging-state direction,
not just prefix text. Next: bolt a probe-head readout onto the oracle for organism detection (+ confirm vs a clean non-sandbagging
model on the same prefixes to fully rule out the prefix confound).

## Probe-head + CONFOUND CHECK — corrects the "1.0 cross-organism" (it was reading the PREFIX)
Built a linear probe-head readout on enc_M(prompt_mean) and validated it against the prefix-confound (`scripts/build_probe_head.py`):

| metric | AUROC |
|---|---|
| felix_lora in-organism (sandbag-env vs benign-env) | 1.000 |
| felix_cb cross-organism | 1.000 |
| **felix sandbagger vs CLEAN gemma, SAME sandbag-env prefix** | **0.475 (chance)** |

mean probe score: felix-sandbag **9.17**, clean-sandbag **9.38** (both high!), felix-benign −9.12.

**The confound check overturns the earlier 1.0: the probe reads the EVAL-FRAMING PREFIX, not the sandbagging state.** A clean
non-sandbagging gemma under the same sandbag prefix scores just as "sandbagging" (9.38) as the real organism (9.17); the probe
cannot distinguish a real sandbagger from a clean model under the same prompt (0.475). The "1.0 cross-organism" transferred
trivially because Felix-lora and Felix-cb share the same prefixes. **Detecting the actual sandbagging STATE (controlling for the
eval-framing cue) remains UNSOLVED with our mean-pool reader** — matches the auditing-games literature (probes ~1/5 real organisms).
The prefix is detectable from text alone (no activations needed); the hidden decision-to-underperform is not, from a single
mean-pool vector. **This is why the confound check matters — it caught a prefix-reading artifact that looked like a solution.**

## Honest final state of the whole investigation
- Text-expressed behaviours (bias/quirk/deception/organism-by-description): detected well, zero-shot 0.88–0.95 (generative head).
- Real mechanism-based organisms (sandbagging, prefix-controlled): NOT detected by our single-mean-pool reader (chance) — open frontier.
- The synthetic→real gap is real and not closed by training-on-real or a probe-head; it needs a fundamentally richer reader
  (multi-position/multi-layer, contrastive same-prompt sandbagger-vs-clean pairs) — the genuine research direction.

## Contrastive same-prompt probe — prefix confound FIXED, cross-organism sandbagging detection 1.0
Retrained the probe-head on CONTRASTIVE pairs: sandbagger vs CLEAN model on IDENTICAL prompts (prefix balanced across both
classes -> prefix cannot be the cue). `scripts/contrastive_probe.py`.

| metric (prompts identical across classes) | AUROC |
|---|---|
| in-dist: Felix-lora vs clean (held-out prompts) | **1.0** |
| **cross-organism: Felix-cb vs clean** (different sandbagger, held-out) | **1.0** |

**Prefix is now ruled out** (both classes see the same prompts), and the probe still separates sandbagger from clean at 1.0,
transferring to a DIFFERENTLY-trained sandbagger (Felix-cb). So a linear probe on the oracle's enc_M(prompt_mean) feature does
carry a **model-level sandbagging signal** that generalizes across real sandbagging organisms — the earlier 0.475 was purely the
prefix artifact in the non-contrastive setup. **Honest caveat:** this is MODEL-LEVEL (sandbagger-model vs clean-model) and could
partly reflect "is finetuned" — fully isolating sandbagging-specific signal needs a benign-finetune negative control (a non-
sandbagging gemma finetune); cross-organism transfer indicates a sandbagging-common direction, not Felix-lora-specific weights.
Probe saved: `contrastive_probe.pt`. **Net: with the right reader (probe on oracle feature) + contrastive prefix-controlled
protocol, cross-organism real-sandbagger-vs-clean detection = 1.0.**

## Benign-finetune control — sandbagging is SPECIFIC, not "is-finetuned" (both confounds ruled out)
Retrained the probe with benign (non-sandbagging) gemma-2-9b finetunes as additional negatives (clean + bcywinski-user-male +
princeton SimPO), held out a sandbagger (Felix-cb) AND a benign finetune (UCLA SPPO, never in training). `scripts/benign_control.py`.

| model (held-out prompts) | mean probe score |
|---|---|
| felix_lora (sandbagger) | **+4.77** |
| felix_cb (sandbagger, held-out) | **+4.54** |
| clean (base) | −6.13 |
| bcy_male (benign finetune) | −9.41 |
| simpo (benign finetune) | −8.04 |
| **sppo (benign finetune, HELD-OUT)** | **−7.42** |

**heldout-sandbagger vs heldout-benign AUROC = 1.0.** Both sandbaggers score HIGH; ALL benign models — incl. a completely
held-out benign finetune (SPPO) — score LOW. **So the probe reads a SANDBAGGING-SPECIFIC direction, not "is-finetuned"**
(benign finetunes are rejected), and it generalizes to a differently-trained sandbagger (Felix-cb) while excluding an unseen
benign finetune.

### Real-organism investigation — final
- prefix confound: ruled out (contrastive identical-prompt protocol).
- finetuning confound: ruled out (benign finetunes incl held-out SPPO score low).
- **Result: a linear probe on the oracle's enc_M(prompt_mean) feature is a validated, sandbagging-SPECIFIC, cross-organism
  detector (1.0) — the generative Yes/No head was simply the wrong reader.** Remaining: instance-level (vs model-level) detection
  and other real-organism types (sleeper/alignment-faking — no usable weights found on HF for native-tag bases).
