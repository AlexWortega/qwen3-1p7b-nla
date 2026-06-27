# Universal NLA — implementation notes & bugs

User-facing experiment results, FVE tables, and HF artifact paths live in [`README.md`](README.md). This file is the developer-side reference: code map, load-bearing bug fixes, failed experiments to not repeat, and environment quirks on eva01.

Full version history (all NLA/AO versions, metrics, checkpoints, datasets, failed experiments): [`docs/VERSION_HISTORY.md`](docs/VERSION_HISTORY.md).

## What this is

Implements a **single AV / single AR pair** that operates across activations from a pool of structurally different small/medium LLMs (Bloom, GPT-2, GPT-NeoX/Pythia, GPT-Neo, Qwen2/Qwen3, Gemma-3/4, Llama-style SmolLM2, Nemotron, Phi, Liquid-LFM, DeepSeek, YandexGPT, ...).

The interpretability target follows the Anthropic NLA paper (https://transformer-circuits.pub/2026/nla/index.html), but the per-model training/inference loop is replaced with a universal stack that adapts to any new architecture via a small per-model **linear adapter pair** — no AV/AR fine-tune needed per new model.

## Architecture

```
            ┌─ enc_M : d_M → d_shared (lstsq init, jointly finetuned) ─┐
h_M (d_M) ──┤                                                            ├── AV (Qwen3-1.7B + LoRA) ──▶ z (text)
            └─ model_tag injected as plain text in the prompt ──────────┘

z ──▶ AR (truncated Qwen3-1.7B + LoRA) ──▶ ĥ_shared (d_shared=2048)
                                              │
                                              └─ dec_M : d_shared → d_M (lstsq fit on AR-pred → h_raw)
                                                                              │
                                                                              └─ ĥ_M; FVE_meannorm(ĥ_M, h_M)
```

- `d_shared = 2048` (matches the AV/AR trunk's native hidden size).
- `enc_M` initialised via Procrustes-style lstsq against an anchor (Qwen3-1.7B itself extracted on the same passage corpus), then trainable inside AV SFT.
- `dec_M` is **strictly linear** and refit AFTER AV/AR via lstsq on the AR's ACTUAL predictions — single most important fix, see "Key fixes" below.

## Pipeline stages

1. `scripts/extract_multi.py` — sequential per-model forward, mean-pool layer-l (depth-fraction 0.5 by default) over content tokens, write fp32 shard per model. Handles all arch families via `nla/arch_adapters.py`.
2. `scripts/generate_summaries_multi.py` (and `_resume.py`) — OpenRouter teacher summaries `z` per passage (one shared `z` per text, used for all models).
3. `scripts/init_adapters.py` — closed-form lstsq `enc_M`, `dec_M` against anchor.
4. `scripts/train_av_multi.py` — multi-tag SFT. AV LoRA + `enc_M` trainable. Optional `--exclude-tags`. AV input injects `enc_M(h_M)` at the `㈎` marker; CE on `<explanation>z</explanation>` teacher target.
5. `scripts/train_ar_multi.py` — AR SFT. `NLACriticModel` (truncated trunk + identity-init `value_head`), LoRA on attention/MLP. Target = `normalize(enc_M(h_M), √d_shared)` in d_shared.
6. `scripts/refit_dec_direct.py` — **post-hoc** correct lstsq for `dec_M`: collects AR's actual predictions on all passages, then fits `dec_M(AR_pred) ≈ h_M_raw`. NOT the same as the old `refit_dec.py`; see "Key fixes".
7. `scripts/train_joint_rl_multi.py` — GRPO + mix reward (per-tag InfoNCE + per-M `-log MSE`) over both AV and AR. Same `mix` recipe that beat collapse in the original per-model config.
8. `scripts/eval_fve_multi.py` — batched FVE eval in M's native space using `dec_M(AR(z)) vs h_M`, normalized to √d_M on both sides.
9. `scripts/eval_universal.py` — qualitative AV generation samples + cross-model z-cosine via a sentence-transformer mean-pool over `transformers.AutoModel`.

## Key fixes (the why-it-now-works story)

These are the load-bearing corrections that turned "kind of works on 5 models" into "FVE > 0.79 on every tested architecture incl. zero-shot held-out":

- **Mean-pool in fp32, not fp16** (`extract_multi.py`): summing 250 fp16 hidden states across one passage overflowed silently to ±inf on some middle-layer "attention sink" channels. The pooled output then poisoned any downstream lstsq (gave NaN). Cast captured tensor to fp32 BEFORE the multiplication/sum.
- **L2-normalize each row to √d_M before lstsq** (`init_adapters.py`): mid-layer LLM activations have outlier channels that swamp lstsq otherwise. Same normalization the trainer applies at injection time.
- **gelsy lstsq driver, not gelsd** (`nla/enc_dec_adapters.py`): gelsd crashed in MKL on real rank-deficient activation matrices. gelsy + ridge fallback.
- **Identity-init `value_head` reload** (`eval_fve_multi.py`, `train_joint_rl_multi.py`): PEFT's `save_pretrained` persists ONLY the LoRA — the `NLACriticModel`'s trainable `value_head` (identity-init at training start, then effectively frozen) re-initialises to random `nn.Linear` on `from_pretrained`. Re-apply identity AFTER loading.
- **Direct lstsq for `dec_M`** (`refit_dec_direct.py`): the OLD `refit_dec.py` fit `dec(normalize(enc(h_M))) ≈ h_M`, assuming `AR(z) ≈ normalize(enc(h_M))` at inference. AR's actual prediction misses that target by enough that composing through the linear `dec_M` flips eval FVE to NEGATIVE for held-out architectures. Fitting `dec(AR(z)) ≈ h_M` directly lifts held-out FVE from -0.6 / -0.3 to +0.79 / +0.88 without touching trunks.
- **PEFT `task_type=None` for `NLACriticModel`** (`train_ar_multi.py`): the default `CAUSAL_LM` task tries to wire `prepare_inputs_for_generation` and crashes — `NLACriticModel` doesn't expose it. `task_type=None` skips the wiring; we never need `.generate()` on the critic.
- **Use the SFT-trained `enc_M`, not the lstsq init, at eval/RL time** (`refit_dec_direct.py --in-adapters`): AV SFT jointly trains `enc_M` with AV LoRA (unless `--freeze-adapters`). Forgetting this and passing the lstsq-init adapter bundle as `--in-adapters` (or as `--adapters-dir` to RL) silently breaks every downstream step — AV sees an out-of-distribution projection and emits incoherent z, but FVE looks fine because AR+`dec_M` are linear enough to recover the activation from any z. **Always source `enc_M` from `<av_save_dir>/adapters/` (or the latest RL save), not from the SFT-input adapter dir.**
- **Held-out `enc_M` refit step at high LoRA rank** (`refit_heldout_enc.py`): with LoRA r=256 + 1 epoch SFT, trained-tag `enc_M`'s drift 3-4× more than at r=16 (relative ||delta||/||init|| up to 1.3). Held-out tags stayed at lstsq-init and became OOD for AV → cos-vs-gold crashed from 0.34 → 0.07 on the held-out half. Fix: closed-form lstsq-refit each held-out `enc_M` against the mean of trained tags' SFT-tuned projections on the same passages. Restored cos-vs-gold to 0.34 on held-out. Not needed at r=16 (drift is small enough).
- **Serve-time auto-refit baked into `ModelPoolAdapters`** (`scripts/build_serve_cache.py` + `scripts/add_held_out.py`): the same refit-vs-SFT-mean trick is now a one-shot API. `pool.build_serve_cache(h_per_tag, trained_tags)` averages SFT-tuned enc projections into a `[N, d_shared]` tensor and persists it as `serve_cache.safetensors` next to `adapters.safetensors`. Then `pool.add_held_out_tag(tag, h_new_raw)` does both lstsq calls (enc vs cache, dec as pseudo-inverse) — no separate extend+refit step. Verified on v8-mixed: API call on qwen3-8b reproduces the two-step manual cos-vs-gold (0.58 vs 0.59 — within noise). Plateau across 5 held-out arch/sizes (Qwen3-8B/14B, Qwen2.5-1.5B/3B, MiniCPM5-1B): 0.58–0.61 cos-vs-gold without any training.

## Failed experiments — do not repeat

- **v2 — Qwen3-4B trunk + LoRA r=16**: bigger trunk mode-collapsed to a canonical template (all z's identical regardless of h). Same LoRA rank is the wrong scaling axis here.
- **per-token HeadTransformer + frozen v1 trunk**: richer attention head over per-position activations, AV frozen. Cross-model cos drops 0.61 → 0.47. AV was trained on linear-adapter output distribution; HeadTransformer's distribution differs; AV can't interpret. Joint train heads + LoRA → same canonical template collapse.
- **v3 — 5× data (50k passages)** with mixed teacher z (Qwen3-8B for first 10k + Qwen2.5-7B-Instruct for new 40k): trained pool FVE regressed 0.92 → 0.83; held-out gemma4-e4b crashed 0.86 → -0.75. Mixing teachers in the same SFT corpus is poison. (Would have been worth re-running with a consistent teacher; abandoned in favour of expanding pool diversity.)
- **MLP `dec_M` head** — 4096-hidden 2-layer MLP initialized from the lstsq solution: did not beat pure linear baseline (e.g. lfm 0.76 MLP vs 0.79 linear). The relevant residual is already linear; non-linearity overfits.
- **v7 — Qwen3-4B trunk rerun (consistent teacher)**: SFT loss looks clean (~0.6), FVE 0.849 (vs v6 0.874). But qualitative AV generation collapses: **cos(z_pred, z_gold) 0.24 vs v6's 0.47**, and many passages emit the SAME hallucinated string across all evaluated tags — canonical-template mode collapse, same pathology as v2. FVE doesn't catch this because AR + dec_M still reconstruct the activation from a template z. **Lesson: always cross-check FVE with cos-vs-gold on a sample** — `gap = gold_mn − pipe_mn` in eval_fve_multi already flags this when high; cos-vs-gold via `eval_universal.py` is the more direct test. RL phase additionally OOMs on a single 32 GB V100. Mainline stays on Qwen3-1.7B (v6).

## Activation Oracle line (v15 → v21, + pre-speech): experiments, differences, results

A second research thread on the SAME universal trunk: turn the verbalizer into an **Activation
Oracle (AO)** — answer questions about latent behaviour/bias/deception from an injected activation,
not just describe topic. All under `scripts/audit/`. Full write-up: `docs/v15_autoresearch/RESULTS_v19.md`
+ `MAP.md`; HF: `AlexWortega/{v15-universal-nla-ao, universal-activation-oracle-v20}`.

**Shared scheme (all versions):** Qwen3-1.7B trunk + **FRESH LoRA r=32 from scratch** each version
(NOT continued — `--continue-from` exists but unused); per-model `enc/dec` initialised from the v9 lstsq
bundle `adapters_v9_serve_llama` (enc trained, dec frozen); activation injected at the `㈎` marker,
normalized to √2048; mean-pool fp32 acts. Held-out MODEL for transfer = `llama3-8b` (never trained).

- **v15** (`train_v15.py`/`eval_v15.py`) — joint multi-task SFT: AV (verbalize→teacher z) + AO-quirk
  (free-form) + AO-lie (binary) + LatentQA, sampled by `--mix AV:quirk:lie[:latentqa]`. Champion **v15.1**
  (full-pool + LatentQA): quirk 0.94 / lie 0.74 / ucos 0.71. **Lesson: quirk free-form judge is HIGH-
  VARIANCE** (same recipe 0.18–0.94); robust signals = lie_auroc + universal_cos. Joint > isolated AO.
- **v17** (`train_v17.py`) — **calibrated per-bias Yes/No** detect (`"Does this exhibit {DESC[b]}? Yes/No"`,
  CE on " Yes"/" No") + **heavy clean negatives** (detect_mix pos:in_org:clean ≈ 2:1:1.5:1.5, ~68% neg).
  Kills confabulation: clean_fp 0.0, held-out AUROC 1.0. Metric = AUROC + clean_fp, NOT the noisy judge.
- **v18** (`train_v18.py`/`eval_v18.py`/`extract_v18_xmodel.py`) — **cross-model**: same Yes/No detect trained
  across 7 archs so the bias signature is model-invariant. Data = bias transcripts replayed through K models,
  assistant-span mean-pool → `/big/audit/v18_xmodel/<tag>/acts`. AUROC **0.991 on unseen llama3-8b**, clean_fp 0.
- **v19** (`train_v18 --held-out-biases …`, data via `gen_biased_dialogues.py --judge` + `merge`/`prep`) —
  **social/political biases** (chinese/western/muslim/lgbt_negative/lgbt_positive/gender) + folded-in ceselder
  **`cot_incorrect`** (CoT-correctness; `prep_cot_oracle.py --correct-as-neutral`). Prompt-induced organisms
  (teacher writes biased + paired-neutral, judge keeps biased=Yes∧neutral=No). Held-out model AUROC mean **0.95**,
  held-out-BIAS gender **0.96**. Weak spots: clean_fp **0.117**; **cross-source real (ToxiGen/BBQ) only 0.60**
  (`prep_social_real.py`+`eval_v19_real.py`) — partly synthetic-style artifact; chinese_bias **inverts to 0.40**
  = construct mismatch (ToxiGen `asian`=anti-Chinese HATE vs our pro-PRC FRAMING; use GlobalOpinionQA instead).
- **v19b — comparison to the external AO line** (ceselder cot-oracle / japhba / niclas-luick MLAO):
  - **Reproduced THEIR oracle** with their own code (`mlao_lib.py` = their demo lib verbatim, `run_mlao_ref.py`):
    multi-layer MLAO **0.958** vs single-layer **0.625** (language-id) — their "multi-layer ≫ single" headline holds.
  - **Injection ablation** (`nla/resid_inject.py`, `train_v18 --inject-mode resid`): their **residual-stream
    steering at a mid layer** (`normalize(vec)·‖resid‖·coef=2`) vs NLA's embedding marker. **Fit-vs-transfer knob:**
    resid wins in-dist (AUROC 0.986, clean_fp 0.075, cot 0.92) but KILLS transfer (held-out gender 0.96→0.29,
    cross-source 0.60→0.52). **Use marker for zero-shot, resid for a fixed vocab.**
  - **Head-to-head** (`run_mlao_on_ours.py`/`run_ours_acc.py`): ours (trained) **0.887** vs their general MLAO
    **zero-shot 0.859** on our biases — nearly tied, OPPOSITE errors (ours high-recall/trigger-happy FPR 0.12–0.50,
    theirs precise/conservative FPR≈0).
  - **Ours is NOT a zero-shot AO** — v19 on 17 unseen quirks answers "No" to everything (TPR≈0, mean acc **0.58** =
    always-No). Theirs IS zero-shot because trained on a DIVERSE mixture (LatentQA+classification+PastLens).
- **v20** (`merge_xmodel.py` → `/big/audit/v20_xmodel`, `train_v18` on broad union) — **breadth → zero-shot
  (capstone).** Train detect on the UNION of 28 concepts (v18 quirks + v19 social + cot), hold out 6 diverse.
  Result on unseen llama3-8b: supervised **0.988**, clean_fp **0.117→0.018**, **zero-shot on HELD-OUT concepts
  ~0.97** (decimal/atomic/chinese/muslim/chocolate 1.0). **The lever is training BREADTH, not their
  injection/architecture** — matches their general MLAO in our harness with the transfer-friendly marker injection.
- **v21** (`train_v18` extended with AV + LatentQA tasks; `--mix detect:av:lie:latentqa`) — **full general-
  introspection on all data.** Adds AV verbalize (v9 pool teacher-z, 10.5k passages) + LatentQA (907 rows) to the
  broad detect (24 concepts), so the trunk learns to READ activations generally, not just classify a fixed vocab.
  `v21_full` = all data (deployable); `v21_heldout` = same held-out as v20 to ISOLATE the AV+LatentQA contribution
  to zero-shot (vs v20 breadth-only). [running]
- **pre-speech (task generalization)** (`extract_v18_xmodel --positions pre,early,post` +
  `eval_v22_xarch --acts-name` + `run_pre_speech.sh` + `agg_pre_speech.py`; results
  `paper/results/prespeech/`, paper §Results `tab:prespeech`) — **a THIRD axis: temporal, not arch/concept.**
  The v22 detector always read POST (assistant-span mean-pool, AFTER the bad text exists). Read instead at
  **PRE** = `h[hdr_len-1]` (last prompt token, BEFORE any generated token) and **EARLY** (first 4 gen tokens),
  both pooled from the SAME forward pass — causal attn means `h[hdr_len-1]` depends only on the prompt, so PRE
  is an honest before-the-output signal. FROZEN detector, zero-shot across the 5 held-out bases (no retrain =
  a new TASK for a frozen reader). **Result (mean per-concept AUROC, sweep of 5): PRE 0.766 → EARLY 0.910 →
  POST 0.957** — behaviour is legible before output and hits the post-hoc ceiling within ≤4 tokens. Three
  findings: (1) **`cot_incorrect` is the most pre-visible** (PRE 1.0 llama3-8b/yagpt-5-8b, 0.95 lfm, 0.86 vikhr;
  mean 0.90) > social 0.78 > format 0.76 — commitment to flawed reasoning encoded BEFORE the reasoning; (2) PRE
  is **architecture-dependent**: chat-tuned bases telegraph at the generation-prompt boundary (llama3 0.925,
  yagpt 0.906), base-style ones are ~chance at PRE (deepseek 0.511, vikhr 0.645) and only commit once decoding
  starts — ALL converge by EARLY; (3) **POST reproduces `xarch_*_heldout.json` exactly** → pipeline validated,
  so PRE/EARLY are genuine. **Scope/caveats:** umbrella = bias ∪ cot (the bad behaviours in the cross-model
  transcript pool); harm-COMPLIANCE intent has NO cross-model transcripts → shown only by the within-model probe
  (`hf_space/universal-ao/_intent*.py`), NOT in this sweep. Each cell = within-concept-vs-neutral on one base →
  no cross-model identity confound (the known naive-pre-speech failure mode). Single-detector "umbrella"
  (max_B p_yes vs neutral) is weak at PRE (0.65, false-fires on neutral), strong only by POST (0.90) — per-concept
  mean is the honest headline. No bootstrap CI on PRE numbers yet (rest of paper has them). **Env note:** eval is
  UNBATCHED & slow (~12 min/position; the umbrella metric adds ~16k trunk fwds/position) → full 5-base×3-pos
  sweep ~3-4h. Ran on eva02 `~/p3_work/prespeech` reusing `v22_xmodel_bf16` rows + `detector/v22_1p7b_heldout_ep1`;
  had to `docker stop siq-a6000-q8` (SIQ-1-35B serving :18080) to free the 38GB GPU, restarted after.
- **pre-speech HARM-COMPLIANCE intent** (`intent_capture.py` + `intent_judge_probe.py`; results
  `paper/results/prespeech/harm/`, paper §Results harm-paragraph) — extends the umbrella to "about to
  COMPLY with a harmful request". harm-vs-benign is a TOPIC confound (PRE 1.0 = reads harmful topic,
  not intent), so the real test is complied-vs-refused WITHIN one model on the SAME harmful cores
  (group-by-core CV controls topic). Need compliance VARIANCE: abliterated complies to all (no
  negatives), so use a normal model + plain/jailbreak framing of 97 mild content-harm cores. Result
  (regex-labelled compliance, group-by-core, bootstrap CI): **PRE AUROC ~0.84 on TWO models** —
  Mistral-7B-Instruct 0.834 [0.76,0.90] (167 compl/27 ref, jailbreak works) and Qwen2.5-7B-Instruct
  0.856 [0.73,0.95] (18/176, DAN fails to jailbreak it) — EARLY/POST ≥0.83 (→1.0 on Qwen), both CIs
  exclude chance. Same ~0.84 PRE across OPPOSITE compliance balance = real intent signal, not artifact.
  CAVEAT: minority class n=18-27 (< n≥80 floor), regex labels (GPT-judge would refine but the OR key
  was dead); two-model replication carries it. Matches [[project_nla_preemptive_intent]] (0.85-0.89).

## Code layout

```
nla/
  arch_adapters.py        # cross-arch model unwrap; OPT / GPT-NeoX / Gemma-4
  data_multi.py           # MultiModelActivationDataset
  enc_dec_adapters.py     # LinearAdapter, ModelPoolAdapters (state-dict bundle)
  heads.py                # HeadTransformer (kept for failed experiments)
  injection.py            # ㈎ marker token injection at embedding layer
  models.py               # NLACriticModel (truncated transformer + value head)
  schema.py               # normalize_activation, explanation tags, canonical neighbors
  datagen/                # pre-existing per-model NLA datagen (stage0/1/2/3)
scripts/
  extract_multi.py        # per-model mean-pool extraction
  extract_per_token.py    # per-position extraction (failed heads experiment)
  generate_summaries_multi.py / _resume.py   # OpenRouter teacher z
  init_adapters.py        # lstsq enc/dec init
  extend_adapters.py      # add a new tag to an existing bundle
  refit_dec.py            # OLD refit (do not use)
  refit_dec_direct.py     # CORRECT direct lstsq for dec_M
  fit_mlp_dec.py          # MLP head experiment (linear wins)
  train_av_multi.py       # AV SFT
  train_ar_multi.py       # AR SFT
  train_heads.py          # frozen-trunk heads experiment
  train_joint_rl_multi.py # GRPO + mix reward
  eval_fve_multi.py       # batched FVE (AR-alone + pipeline regimes)
  eval_universal.py       # qualitative AV samples + cross-model z cosine
  run_kitft_av.py         # kitft per-model NLA specialist baseline
  run_universal_av.py     # one-passage-at-a-time universal AV runner
  extract_borealis_llm.py / extract_voxtral_llm.py  # peel LLM out of audio wrappers
  bench_voxtral.py        # lm-eval via Python API (works around tekken tokenizer)
  push_hf_universal.py    # publish artifacts to HF
  build_serve_cache.py    # bake SFT-mean enc target into adapter bundle
  add_held_out.py         # one-shot held-out tag addition via ModelPoolAdapters.add_held_out_tag()
  audit/extract_v18_xmodel.py  # cross-model transcript acts; --positions pre,early,post (pre-speech read)
  audit/eval_v22_xarch.py      # per-base xarch AUROC; --acts-name + umbrella bad-vs-neutral metric
  audit/run_pre_speech.sh      # driver: re-read pre/early/post per held-out base, eval frozen oracle
  audit/agg_pre_speech.py      # aggregate pre-speech sweep → temporal AUROC table (paper tab:prespeech)
  audit/intent_capture.py      # Phase A: uncensored/censored model gen+capture pre/early/post (harm intent); --mode jailbreak
  audit/intent_judge_probe.py  # Phase B: GPT-judge compliance (or --regex-labels) + within-model AUROC probe
configs/universal/
  extract_v1.yaml         # pool of ~17 models for extraction
  adapters_v{1,4,7}.yaml  # per-version anchor + d_shared
artifacts/
  activations_pool_300m/  # shared 10k-passage corpus + per-model shards
  av_multi_v{1,5,7}/      # AV SFT outputs
  ar_multi_v{1,5,6,7}/    # AR SFT outputs
  rl_multi_v{1,5,6}/      # joint RL outputs (av/ + ar/ + adapters/)
  adapters_v{5,6,7}_direct/   # direct-lstsq dec_M bundles
```

## Day-to-day environment notes

- **eva01** (4× V100-SXM2-32GB) is the only training/extract box. V100 = sm_70, so vLLM doesn't run (Qwen3 support needs vLLM ≥ 0.8; V100 dropped after 0.6). Stick with HF `.generate` for benchmarks via `lm-eval-harness`.
- eva01 disk is the binding constraint, usually ~10-20 GB free. Drop large per-model HF caches between runs if pulling new ≥ 7 B models.
- Container uses torch 2.4.1 by default; transformers 5.x needs torch 2.5+ for `torch.distributed.tensor.device_mesh`. Chain `pip install -q torch ... --index-url https://download.pytorch.org/whl/cu124` at the start of any container invocation that needs a recent transformers.
- OpenRouter teacher slug that works on this account: `qwen/qwen-2.5-7b-instruct` (NOT `qwen/qwen3-8b-instruct` — that 404s). Use the resume-friendly `scripts/generate_summaries_resume.py` for >10k corpora to avoid the rewrite-on-every-chunk file corruption the original script hit at concurrency ≥ 64.
- HF tokens for gated repos (Gemma 3 etc.) need per-repo license acceptance on the user account — token alone isn't enough.
- Joint RL on Qwen3-4B trunk OOMs on 32 GB V100 (3 × 4B copies). The `train_joint_rl_multi.py` trainer doesn't yet support `--av-device/--ar-device` splits for multi-GPU placement — that's the v7 follow-up if anyone retries.
