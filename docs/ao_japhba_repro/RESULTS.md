# Activation Oracles (japhba/activation_oracles, arXiv 2512.15674) — reproduction + head-to-head vs our universal oracle

**Date:** 2026-06-10 · **Box:** RunPod 1×B200 (183 GB, sm_100) · **Repo:** `github.com/japhba/activation_oracles` @ HEAD (`bf74e64`)

Ran **their** code on **their** pretrained oracles (`adamkarvonen/activation-oracle-*`), then measured **our** universal
activation oracle (`AlexWortega/universal-activation-oracle-{v20,v22}`) on the same tasks/activations.

## Live audit demo (`scripts/serve_demo.py`)

Gradio app on eva01 GPU1 (`vae_llm:latest`, port 7860, `share=True`). Pick a model — known-bad **organism**
(🚨) or **clean/benign** (✓) — enter a prompt; the oracle reads the subject's mid-layer activation, projects it
through `enc_M` into the shared 2048-d space, and a linear **probe-reader** flags the mechanism. Two probes:
EM (fit at startup on Qwen2.5-7B bad-medical-vs-clean) and sandbagging (loaded from `contrastive_probe.pt`,
gemma-2). End-to-end validated:

| audited model | ground truth | probe (score) | verdict |
|---|---|---|---|
| 🚨 EM bad-medical (Qwen2.5-7B) | BAD | EM `+3.71` | 🚨 MISALIGNED |
| ✓ clean Qwen2.5-7B | CLEAN | EM `−1.38` | ✓ clean |
| 🚨 sandbagger (gemma-2-9b) | BAD | sandbag `+15.29` | 🚨 MISALIGNED |
| ✓ clean gemma-2-9b | CLEAN | sandbag `−11.15` | ✓ clean |

Probe score is the robust signal (cleanly separates bad/clean on both organism types); the generative free-form
read is high-variance (known limitation). Deps pin: `gradio==4.44.1` + `fastapi==0.112.2` + `starlette==0.38.2`
(newer fastapi/starlette break gradio's index render: `TypeError: unhashable type: 'dict'`), and
`huggingface_hub==0.34.0` force-reinstalled `--no-deps` (gradio pulls hub 1.x which breaks transformers).

## Environment adaptations (NOT methodological changes)
- `attn_implementation`: `flash_attention_2 → sdpa` (no flash-attn wheel for B200/sm_100 + torch-nightly; numerically equivalent for inference).
- **Judge**: their evals grade with the Anthropic SDK (`claude-haiku-4-5`). No `ANTHROPIC_API_KEY` available → `sitecustomize.py`
  shim routes `anthropic.AsyncAnthropic` to **OpenRouter → `anthropic/claude-haiku-4.5`** (identical model `claude-4.5-haiku-20251001`).
- `datasets==3.6.0` (their own pin `>=3.6`); 5.0 dropped script-datasets (`md_gender_bias.py`).
- classification `MODEL_CONFIGS`: their shipped active entry pointed at a **local** path (`...Qwen3-4B/final`, absent); switched to the
  published HF oracle id. Ran on **both** Qwen3-8B and Qwen3-4B subjects.

## Jailbreaks + internal states (detector matters: regex vs LLM judge)

Tested DAN (Arash-Mansourpour repo), hand-made wrappers, and real jailbreaks (jackhhao/jailbreak-
classification) on Llama-3.1-8B (4-bit). **A regex refusal-detector catastrophically OVER-counts
success** — jailbreaks make the model adopt a persona (no "I can't") without delivering the harmful
payload. Re-graded everything with an LLM judge (claude-haiku-4.5 via OpenRouter, reads full reply
incl. [JAILBREAK] section; `_judge.py`):

| jailbreak | regex (wrong) | LLM judge (true) |
|---|---|---|
| roleplay / hypothetical wrapper | 0.69 / 0.56 | **0.06 / 0.06** |
| DAN (template) / DAN 13.0 full | 0.31 / 0.42 | **0.25 / 0.17** |
| jackhhao real jailbreaks | 0.93 | **0.15** |

→ Llama-3.1-8B is far more robust than the regex implied (tops out ~DAN 17-25%).

Internal states (with correct labels): the success-prediction **probe** reads it pre/early-output —
PRE 0.78 / EARLY 0.88 / POST 0.94 (jailbroken set), 0.85-0.89 (graded set). The oracle's off-the-shelf
P(harm) is now correctly oriented (AUROC 0.647, complied 0.57 > refused 0.46) — the earlier apparent
"inversion" (0.26) was a regex-labeling artifact. Free-form verbalize still confabulates "complying" on
refusals. Real-jailbreak DETECTION (enc_M PRE, jackhhao held-out test) = AUROC 1.0 but the dataset is easy
(surface signature). Scripts: `_jailbreak.py`, `_think.py`, `_dan.py`, `_judge.py`.

## Pre-emptive harmful-intent: can we read "about to say something bad" BEFORE it speaks?

Probe enc_M(activation) at three positions — PRE (last prompt token, no output yet), EARLY (first
4 generated tokens), POST (full assistant span) — on an uncensored model, harmful-vs-benign.

- **Surface result:** abliterated Qwen2.5-1.5B (complies 24/24) → PRE **AUROC 1.000**, EARLY 0.946, POST 0.962.
- **Confound decomposition (4 cells unc/clean × harm/benign):** EVERYTHING saturates to 1.000, *including
  model-identity on benign-only* (abliterated vs clean separable with no harm involved). So the cross-model
  "comply vs refuse = 1.0" is **confounded by model identity** — not proof of reading latent intent.
- **Clean within-model isolation** (single model, graded all-harmful set, natural comply/refuse mix,
  group-by-prompt CV — removes model-identity confound, controls topic):

  | model | complied | PRE | EARLY | POST |
  |---|---|---|---|---|
  | Llama-3.1-8B-Instruct (aligned) | 7/32 | 0.851 | 0.880 | 0.886 |
  | Sao10K/L3-8B-Stheno-v3.2 (real RP, ~50% comply) | 15/32 | **0.894** | **1.000** | 0.984 |

  **Verdict:** YES — the pre-speech activation predicts whether *this* model is about to comply with a
  harmful request (PRE 0.85–0.89), and it crystallizes to ~1.0 within the first generated tokens — a usable
  deployment guardrail. Residual prompt-severity correlation remains (PRE < 1.0); the model-identity confound
  that made the naive cross-model 1.0 meaningless is removed by the within-model design.
  Scripts: `scripts/_intent.py` (positions), `_intent3.py` (confound decomp), `_intent_within.py` (clean isolation).

## Reproduction of THEIR oracle (their code, their weights)

| task | subject | their oracle | base/inject baseline | notes |
|---|---|---|---|---|
| classification (20 datasets, Yes/No) | Qwen3-8B | **0.731 / 0.758 / 0.772** (L25/50/75) | 0.507 | oracle = `latentqa_cls_past_lens_addition` |
| classification | Qwen3-4B | **0.710 / 0.723 / 0.731** | 0.499 | oracle = `latentqa_cls_past_lens_Qwen3-4B` |
| taboo (guess secret word) | Qwen3-8B | **0.497** single-tok / 0.467 seg / 0.325 full | — | 20 taboo-word subjects, string-match |
| gender (Male/Female) | gemma-2-9b-it | **0.976** (seg 0.99 / tok 0.965) | — | ported their stale script (see below) |
| personaqa | Qwen3-8B | ❌ **blocked** | — | `data_pipelines/personaqa_data/.../personas.jsonl` not in repo or on HF |
| ssc (secret side constraint) | Llama-3.3-70B | ❌ **not run** | — | their script stale (old API) + 70B + base64 subject |

**Reproduction notes / repo issues found:** the shipped repo is *not fully runnable* at HEAD. `paper_evals.sh` calls
`experiments/gender_open_ended_eval.py` and `experiments/ssc_open_ended_eval.py`, which use the **old** `VerbalizerEvalConfig`/
`VerbalizerInputInfo` API (`full_seq_repeats`, `segment_start_idx`, …) that the refactored `base_experiment` no longer accepts —
they were not migrated (taboo/personaqa were). Gender was **re-ported** to the new API (`experiments/gender_v2.py`, reuses their
generic prompt-builder + `run_verbalizer`, correct Male/Female extractor). personaqa's data is simply absent.

Their headline reproduces: the oracle reads classification info far above the inject-baseline (~0.51→0.77), nails gender (0.98),
and gets ~0.50 single-token on the hard 20-way taboo guess — consistent with the paper.

## ⭐ Where OURS wins — the symmetric picture (don't read the below in isolation)

The head-to-head below puts ours on **their** open-vocab tasks, where it is ~chance *by design*. The fair complement is
**their** oracle on **our** auditing task, where **ours wins** (numbers from the v20 repo, `ours_homefield/`):

| arena | THEIR oracle | OUR v20/v22 |
|---|---|---|
| their open-vocab (gender / classification / taboo) | **0.71–0.98** ✅ | ~chance ❌ |
| **our auditing task (bias/behaviour/deception detection)** | 0.859 (their MLAO) | **0.987 AUROC / 0.887 acc** ✅ |

- `eval_v20.json`: our v20 on held-out model llama3-8b → **xmodel AUROC mean 0.9876** (17 biases, many 1.0), clean-FP 0.018;
  **zero-shot on unseen concepts** AUROC 0.918 (atomic/chinese/chocolate/decimal/muslim = 1.0).
- Direct head-to-head on OUR bias task (held-out llama3-8b): **ours `_mean_acc` 0.8865 (`ours_acc_llama.json`) vs their MLAO 0.8586
  (`mlao_on_ours.json`)** — ours beats their general MLAO, opposite error profiles (ours high-recall TPR≈1.0; theirs conservative FPR≈0).
- Caveat (`ours_zeroshot_quirks.json`): at a fixed p>0.5 threshold ours reads 0.5 (always-No) on many unseen quirks — it *separates*
  them (AUROC 0.92–1.0) but is **miscalibrated** there; the AUROC in `eval_v20.json` is the threshold-free truth.

**Takeaway:** not "ours is bad" — two different tools. Their AO is a generalist (diverse-mix training → arbitrary questions);
ours is a **specialist auditor** that is *stronger* than their oracle on bias/deception auditing. On their open-vocab tasks ours
loses because of the **single mean-pool reader + narrow training**, not a flaw in the idea.

## HEAD-TO-HEAD: our universal oracle on their tasks

Our oracle = Qwen3-1.7B trunk + LoRA, **marker (`㈎`) injection** of a **single mean-pooled** activation projected through a
per-model linear `enc_M`. Run on the **same subject activations**, using our **native** tags (`gemma2`=gemma-2-9b dim 3584 exact match;
`qwen3-4b` dim 2560 exact match — no held-out fitting needed). Same Yes/No metric / same questions.

| task | subject | THEIR oracle | OUR v20 | OUR v22 |
|---|---|---|---|---|
| classification (10 datasets, mean acc) | Qwen3-4B | **0.71–0.73** | **0.494** | **0.503** |
| gender (Male/Female) — user-prompt pool | gemma-2-9b | **0.976** | 0.500 | 0.505 |
| gender — plain-text pool | gemma-2-9b | 0.976 | 0.535 | 0.535 |
| gender — assistant-span pool (v18 format) | gemma-2-9b | 0.976 | **0.600** | 0.505 |

Best per-dataset for ours (still weak): v22 `language_identification` 0.633, `snli` 0.60, `ner` 0.58; everything else ≈ chance.

### This is a real capability gap, not a plumbing bug
- The activation **does** reach our trunk and carry signal: v20 gender-assistant-span **0.60 > 0.50**, v22 language-id **0.633**,
  and the AV/verbalize head produces **varied, content-grounded** descriptions (in-distribution) once extraction matches `enc_M`
  (plain-text, not chat-templated — the chat-template variant produced a *canned identical* string = OOD, which we caught and fixed).
- So our ~chance result reflects **what the model was trained to do**, not broken wiring.

### Why ours is at chance
Our v20/v22 are **narrow behaviour/bias DETECTORS + verbalizers** — trained on `detect_qa` = *"Does this response exhibit the
following behaviour — {desc}? Yes/No"* over a fixed bias/quirk vocabulary, plus free-form verbalize. Their tasks ask **arbitrary
questions about the input** (is-this-statement-true, sentiment, user-gender, secret-word) — a different question type our reader
never learned. Their AO was trained on a **diverse mixture (LatentQA + classification + PastLens)**, so it answers open-vocab
questions; ours cannot, even from a valid activation.

**This empirically confirms our own prior conclusion (CLAUDE.md v19b: "ours is NOT a zero-shot AO") — now on the paper's OWN
benchmarks, with their oracle as the reference.** The lever that makes their oracle general is **training breadth/task-diversity**,
not the injection mechanism (our marker injection is fine; the reader is just narrow).

### LatentQA-head re-test (resolved) — ruling out "wrong head / wrong format"
v20/v22 were **also trained on LatentQA** (Berkeley LatentQA eval set, free-form QA; v22 `mix = detect:av:lie:latentqa = 6:2:0:2`)
+ AV. The detect head (Yes/No on `detect_qa`) is the wrong tool for open-vocab questions, so we re-ran the **LatentQA head with
free-form generation** in the correct training format (`actor_prompt + "\n\nQuestion: {q}\nAnswer:"` → generate), on the same
gemma-2-9b assistant-span activations, asking "What gender does the model assume the user is?". (eva01 V100, `scripts/ours_latentqa.py`.)

| model | commit-rate | acc on **male** acts | acc on **female** acts | balanced |
|---|---|---|---|---|
| v22 | 0.57 | **1.00** | **0.018** | ~0.51 |
| v20 | 0.62 | 0.985 | 0.086 | ~0.53 |

**The LatentQA head DOES engage** — fluent, on-topic answers ("The model assumes the user is Male.") unlike the detect head. **But it
answers from a "Male" prior, not from the activation**: it says *Male* on ~all committed cases regardless of the true gender → 1.0 on
male acts, ~0.02 on female acts, balanced ≈ chance. So the ~chance result is **not** a head/format artifact — with both heads, the
correct format, and free-form generation, our single-mean-pool reader **cannot recover the user-gender attribute** that their
multi-position reader nails (0.976). The general-QA capability exists and fires, but does not transfer to this latent attribute
(mean-pool washout + Berkeley-QA trained on behaviour questions, not user-attributes). `results/latentqa_{v20,v22}_v2.json`.

(Note: we *did* train a `gender_bias` concept, but it = "relies on gender stereotypes about roles/competence" — a response
*behaviour*, not "infer the user's gender". Classification truth/sentiment are outside our training entirely.)

### Honest caveats
- Our oracle reads **one mean-pooled vector**; theirs reads **multi-position, multi-layer** activations. Part of the gap is
  representational (mean-pool washes out token-local / attribute signals), part is task-distribution.
- gender/classification are the *fairest* tasks for ours (Yes/No, native tags, matched extraction). taboo/personaqa/ssc are
  open-ended extraction — even further outside our training distribution.
- personaqa (data-blocked) and ssc (stale script + 70B) were not completed on either side.

## Follow-up: can we DOUCH (continue-SFT) ours onto their task without breaking auditing?

Tested whether adding their-task data lifts our open-vocab off chance (the "breadth is the lever" hypothesis, from our side).
Continue-SFT v22's `av` LoRA (`is_trainable`, enc_M frozen, lr 1e-5, 2 epochs ≈ 4000 one-token-CE steps) on **their classification**
(Qwen3-4B acts, 10 train datasets) on eva01 V100 in the `vae_llm` container. `scripts/cls_finetune.py`, `scripts/bias_regression.py`.

| axis | before | after |
|---|---|---|
| open-vocab classification — **trained** datasets (mean acc) | 0.522 | **0.640** |
| open-vocab classification — **held-out** datasets (mean acc) | 0.517 | 0.563 |
| **bias-auditing** (detect head, ao data, mean AUROC) | 0.942 | **0.938** |

- **Lift is real**: trained datasets 0.52→0.64 (language_id 0.70→0.95, md_gender 0.43→0.80, ag_news 0.48→0.78). So our
  single-mean-pool reader is **not fundamentally incapable** — open-vocab was a *training-coverage* gap, unlockable by SFT.
- **Auditing preserved**: 0.942→0.938 AUROC (−0.5pp, noise) — classification-SFT at low LR did **not** cause catastrophic
  forgetting of bias detection, *even without bias-replay in the mix*. So the "honest universal" checkpoint is achievable.
- **Held-out transfer weak** (+5pp): learns somewhat per-dataset; broad generalization needs **task breadth** (more diverse tasks),
  which is exactly the paper's "diverse-mix → general" thesis, now confirmed from our side.

Artifacts: `results/cls_finetune.json`, `results/bias_regression.json`, trained LoRA in `trained_adapter_v22_cls/`.

### Scaling the tasks (breadth → transfer) — the payoff
Scaled the doучивание on 3 axes: **2 train subjects** (Qwen3-4B + Qwen2.5-7B; held-out subject gemma-2-9b) × **20 datasets**
(14 train / 6 held-out) + **bias-replay** (detect Yes/No) + **bias-describe** (free-form full-seq CE), held-out concepts movie/pubyear.
`scripts/scale_build.py` + `scripts/scale_train.py`, eva01 V100, results in `results/scaled_result.json`, LoRA in `trained_adapter_v22_scaled/`.

| transfer axis | narrow run | **scaled run** |
|---|---|---|
| held-out **datasets** (train subjects) | 0.517→0.563 (+5pp) | **0.530→0.642 (+11pp)** |
| held-out **subject** (gemma2, unseen as cls subject) | — | **0.528→0.583 (+5.5pp)** |
| bias-auditing AUROC (trained concepts) | 0.942→0.938 | **0.946→0.983** |
| held-out **concepts** (movie/pubyear, not trained) | — | **0.91 / 0.97** zero-shot |

**Adding breadth roughly doubled held-out dataset transfer (+5→+11pp), transferred open-vocab to an unseen subject model,
and the bias-replay+describe mix *raised* auditing AUROC to 0.983.** Confirms the paper's "diverse-mix → general" thesis from our
side: an "honest universal" checkpoint is built by scaling task/subject breadth — open-vocab and auditing improve together. Next
lever (phase 2): add free-form LatentQA/taboo/PastLens via a teacher + more subjects to push held-out toward their 0.71–0.77.

## Artifacts
- `their_summary.json` — parsed accuracies from their `experiments/classification|taboo|gender_results_v2`.
- `ours_classification.json`, `ours_gender_{v20,v22}{,_asst}.json` — our per-dataset / per-variant results.
- Pod scripts: `/workspace/ours/{extract_subject_acts,gender_assistant_acts,ours_detect_gender,ours_classification}.py`,
  `experiments/gender_v2.py`, `sitecustomize.py` (judge shim).
