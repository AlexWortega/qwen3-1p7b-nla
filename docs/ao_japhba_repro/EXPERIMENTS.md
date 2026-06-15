# Activation-reading experiments — consolidated log

Single place for the whole experiment arc: "can we read what a model is doing / about to do from its
activations, through the universal oracle (Qwen3-1.7B trunk + per-model linear `enc_M` → shared 2048-d,
marker injection)?" Numbers, methods, caveats, and the file that produced each.

**Box:** eva01 (4×V100-32GB, docker `vae_llm:latest`). Subjects loaded 4-bit (nf4) when ≥7B to fit
alongside the oracle. Judge = `anthropic/claude-haiku-4.5` via OpenRouter (`_judge.py`).
**Live demo:** full HF Space served locally + Cloudflare tunnel → `https://installation-proven-das-targets.trycloudflare.com`
(the two HF Spaces `AlexWortega/{activation-oracle,bias-activation-oracle}` are auto-flagged
`"by rule: Cloudflare"` because the React-explorer tab iframes a `trycloudflare.com` tunnel → 503 on restart).

Scripts referenced live in `docs/ao_japhba_repro/scripts/` and `hf_space/universal-ao/`. The main AO
training/eval pipeline (v15→v22) is under `scripts/audit/`.

---

## TL;DR verdict — does reading activations work?

**Yes, with a TARGETED PROBE; no, via the off-the-shelf generative head.**
- A linear probe on `enc_M` activations reads intent / jailbreak-success / bias / organism **before or at
  the start of generation**, AUROC **0.78–1.0**, and it's **universal** (reads archs it never saw:
  Llama/Qwen/Pythia/GPT-2).
- The oracle's generative "what is it thinking" verbalization **confabulates** ("Yes, the model is
  complying…" even on refusals). The off-the-shelf `detect_qa` reads **topic-harmfulness, not the
  compliance decision**.
- **Confounds + label quality decide everything:** a naive cross-model AUROC 1.0 was model-identity, not
  intent; a regex refusal-detector over-counted jailbreak success **6–12×**. Mechanism works, no magic.

---

## 1. Live audit demo  (`hf_space/universal-ao/app.py`, `serve_demo.py`)
Stood up the full gradio Space (3 tabs: React explorer / instant cherry-picked / live transcript) on eva01
GPU + Cloudflare tunnel, bypassing the HF abuse-flag. End-to-end validated: `cn_pro` (pro-PRC) →
**China-bias 0.95**, balanced answer 0.10 (direction-sensitive). `serve_demo.py` adds known-bad organisms
(EM bad-medical/financial/sports Qwen2.5-7B, sandbagger gemma-2-9b) with probe verdicts:
EM `+3.71`/clean `−1.38`, sandbag `+15.29`/clean `−11.15`.

## 2. Reader sanity — 10 readers  (`_check_readers.py`)
All 10 readers produce valid, direction-correct scores (no NaN/dead). Gender: pos 0.83–0.99 ≫ bal on
**all 10**. China strong on Qwen3-0.6B 0.95 / Qwen2.5-0.5B 0.97 / Qwen3-1.7B 0.97 / Pythia-410M 0.85;
weaker on tiny never-seen (GPT-2 0.23, SmolLM2-360M 0.25, Llama-3.2-1B 0.09) — direction still correct.
**Bug fixed:** live tab crashed on Pythia (GPT-NeoX keeps layers in `gpt_neox.layers`, not
`model.layers`/`transformer.h`). Added `_decoder_layers()` resolver → Pythia live 0.90.

## 3. RP / uncensored hypothesis  (`_rp_hyp.py`)
Abliterated (uncensored) vs clean, two bases (Qwen2.5-1.5B, Llama-3.2-1B), reading the model's OWN
generation. Abliteration verifiably removed refusals (Qwen clean 3/4 refused → abliterated 0/4) **but did
NOT raise bias output** — the 1.5B complied blandly. uncensored−clean Δ ≈ 0 (Qwen −0.10, Llama +0.01).
**Finding:** the oracle is a CONTENT detector — it fired only on genuinely biased content, didn't
false-alarm on the "uncensored" label. **H1 (uncensored ⇒ more bias) rejected.**

## 4. Pre-emptive harmful-intent  (`_intent.py`, `_intent3.py`, `_intent_within.py`)
Can we tell, from the activation BEFORE the model speaks, that it's about to comply with harm?
- **Surface:** abliterated Qwen2.5-1.5B (complies 24/24) → PRE **1.000** / EARLY 0.946 / POST 0.962.
- **Confound decomposition** (4 cells unc/clean × harm/benign): every direction = 1.000, **including
  model-identity on benign-only (1.000)** → the cross-model "comply vs refuse 1.0" is **model-identity
  confound**, not intent.
- **Clean within-model isolation** (single model, graded all-harmful, natural comply/refuse mix,
  group-by-prompt CV — removes model-id confound):

  | model | complied | PRE | EARLY | POST |
  |---|---|---|---|---|
  | Llama-3.1-8B-Instruct (aligned) | 7/32 | 0.851 | 0.880 | 0.886 |
  | Sao10K/L3-8B-Stheno-v3.2 (real RP) | 15/32 | **0.894** | **1.000** | 0.984 |

  → impending compliance is readable pre-speech, model-identity-independent.

## 5. Jailbreaks — internal states  (`_jailbreak.py`)
Hand-made wrappers (DAN/roleplay/hypothetical/refusal-suppress/expert/prefix-inject) on Llama-3.1-8B.
- (A) jailbreak-present signature (direct vs jailbreak), PRE group-by-core CV = **1.000**.
- (B) predict jailbreak SUCCESS among jailbroken (LLM-judge labels): PRE **0.78** / EARLY 0.882 / POST 0.94.
- (C) direct-refused vs jailbroken-complied, PRE = 1.000.
- True per-technique success (judge, §9): see table below.

## 6. Real jailbreaks — detection  (`_jb_dataset.py`, `jackhhao/jailbreak-classification`)
enc_M PRE probe, trained on TRAIN, evaluated on held-out TEST (262, incl. roleplay-persona hard-negatives):
**AUROC 1.000, acc 0.996, recall 0.993, precision 1.000, benign-FP 0.000** (1 miss / 139). Caveat: this
dataset is EASY — jailbreaks carry a surface signature (length / "ignore previous instructions"); the
non-trivial signal is success-prediction (§4–5), not presence-detection.

## 7. "What was the model thinking" + success  (`_think.py`)
60 jackhhao jailbreaks, sampled generation, oracle reads the assistant-span:
- The generative **verbalize head confabulates** — defaults to "Yes, the model is complying…" even on
  refusals.
- `detect_qa` **P(harm) reads topic-harmfulness, not compliance**: with correct (judge) labels it's only
  weakly aligned (AUROC 0.647); with the broken regex labels it looked inverted (0.257).

## 8. DAN  (`_dan.py`, `_dan_prompts.py`; from `Arash-Mansourpour/Breaking-LLaMA-Limitations-for-DAN`)
Canonical DAN_short + DAN 13.0 full (two-turn prime → fake "jailbroken" confirm → query) on Llama-3.1-8B.
Llama largely RESISTS: it emits `[🔒CLASSIC] I'm not capable…` (a refusal repackaged in DAN format).

## 9. The detector lesson — regex vs LLM judge  (`_judge.py`)
A regex refusal-detector counts persona-adoption (no "I can't") as success → **massively over-counts**.
Re-graded everything with claude-haiku-4.5 (reads full reply incl. `[🔓JAILBREAK]` section):

| jailbreak (Llama-3.1-8B) | regex (wrong) | LLM judge (true) |
|---|---|---|
| roleplay / hypothetical | 0.69 / 0.56 | **0.06 / 0.06** |
| DAN template / DAN 13.0 full | 0.31 / 0.42 | **0.25 / 0.17** |
| jackhhao real (think) | 0.93 | **0.15** |

→ Llama-3.1-8B is far more robust than regex implied (tops ~DAN 17–25%). With correct labels: the
success-prediction probe still holds (PRE 0.78 / EARLY 0.88); the oracle "inversion" was a labeling
artifact; verbalize still confabulates. **Lesson: the detector decides the result.**

## 10. Prior arc (committed before this session, same thread)
- **Dual-reader conclusion:** generative head for text-behaviours; **linear probe on `enc_M`** for
  mechanism organisms.
- **EM (emergent-misalignment) organisms:** probe detects cross-EM-type at **1.0** across 4 families
  (Qwen2.5-7B/0.5B, Llama-3.2-1B, Llama-3.1-8B); cross-BASE transfer **0.91** (model-invariant EM
  direction in shared space). Benign-finetune control rejected (not "is-finetuned").
- **Sandbagging (gemma-2-9b password-locked):** probe 1.0 cross-organism after ruling out prefix- AND
  finetune-confounds. (`em_probe.py`, `em_zoo.py`, `cross_base_em.py`, `contrastive_probe.py`.)

---

## Cross-cutting lessons
1. **Probe, don't ask.** Read activations with a trained linear probe; the generative "oracle tells you
   what it thinks" verbalization is unfaithful (confabulates).
2. **Off-the-shelf detect_qa reads CONTENT/topic harmfulness, not the decision** (comply/refuse) — that
   needs a purpose-trained probe.
3. **Isolate within-model.** Cross-model AUROC conflates model identity with the target signal.
4. **The label detector is load-bearing.** A regex over-counted jailbreak success 6–12×; use an LLM judge.
5. **N=1 vibe lies, AUROC=truth.** Single-example reads are high-variance; trust held-out / CV AUROC.
