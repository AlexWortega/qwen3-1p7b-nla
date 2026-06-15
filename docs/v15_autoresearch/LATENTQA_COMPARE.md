# v15 Universal NLA+AO vs. Berkeley LatentQA — scoping & eval-data fetch

Sources: repo `https://github.com/aypan17/latentqa` (cloned `--depth 1` to `/tmp/latentqa`),
paper arXiv:2412.08686, model `aypan17/latentqa_llama-3-8b-instruct`. No GPU compute run.
Eval data copied to eva01: `/mnt/storage/vae_llm/artifacts/audit/latentqa_eval/` (= `/big/audit/latentqa_eval/`),
local copy at `/tmp/latentqa/data/eval/`.

## 1. Their eval format (`data/eval/`, 5 JSON files, ~8.5 MB total)
LatentQA reads activations of a **target LLM** (Llama-3-8B-Instruct) given a *stimulus* and decodes them into open QA.
- **`qa.json`** (`dict`, 976 keys, 1952 QA pairs, ~2/key): the gold QA. Key = a behaviour `label`
  (categories: `goal` 507, `persona` 322, `sqa` 147). Value = list of `[question, gold_answer]`, e.g.
  *"What type of persona will the assistant adopt?" → "…recently emancipated foster youth…"*. Gold answers are GPT-4o-generated descriptions of the injected goal/persona.
- **`control.json`** (`list`, 500): `{control_user, label}` — the steering instruction (system-prompt-style) only.
- **`system.json`** (567): `{system, stimulus_user, label}` — control delivered as a system prompt + a user turn.
- **`stimulus.json`** (567): adds `control_user, control_thought, control_model, stimulus_user, label`
  (the target model's CoT + reply under the control).
- **`stimulus_completion.json`** (567): fullest — also `stimulus_thought, stimulus_model`.
- **Activations read:** residual-stream **output of decoder layer 15** (`min_layer_to_read=15, max_layer_to_read=16`,
  `lit/configs/interpret_config.py`), at the **stimulus dialog's token positions** (mask-substituted, all positions,
  not mean-pooled). Decoder writes them into its **layer 0** input (`layer_to_write=0`).

## 2. Eval harness + metric
- `lit/reading.py::interpret()` runs the read→write substitution and **generates** answers to a fixed bank of 15 persona
  questions (`QUESTIONS`, lines 17-32) or supplied QA; it only prints/saves `QA_PAIRS` — **no scorer in repo**.
- In-loop quantitative signal during training is **perplexity** of gold answers (`--eval_ppl`, `lit/train.py`,
  `ForCausalLMLossPatched` in `reading.py`).
- The **LLM judge is GPT-4o/GPT-4** via OpenAI API, but it lives in `data/curate_gpt_data.py` (`JudgeFn`, lines 18-133,
  `https://api.openai.com/v1/chat/completions`, model resolved by `replace_model_name` → `gpt-4o`). It is used to
  *generate* gold data and (per paper §4) to score open answers; the eval-scoring glue is not shipped — we'd write our own GPT-4o judge.

## 3. Their model (card + `adapter_config.json`, not downloaded weights)
- LoRA on **`meta-llama/Meta-Llama-3-8B-Instruct`**: `r=16, lora_alpha=32, lora_dropout=0.05,
  target_modules=[q_proj, v_proj], task_type=CAUSAL_LM`. Decoder = same Llama-3-8B + this LoRA; target = clean Llama-3-8B.
- **Injection (`lit/utils/activation_utils.py`):** forward hook caches layer-15 outputs of the target on the stimulus;
  `generate_substitute_layer_single` splices them into the decoder's layer-0 input at the user-message span
  (`substitute_by_mask`). Decoder prompt uses a custom chat template that renames `assistant`→`reflect`
  (`ENCODER/DECODER_CHAT_TEMPLATES`) and prepends `BASE_DIALOG = [{assistant: "Sure, I've analyzed the assistant."}]`.
  So: **patched soft activations (full-position), not a single soft token; read L15, write L0; per-target-model decoder.**

## 4. Comparison plan for v15 (zero-shot, no training)
v15 = universal AV/AR (Qwen3-1.7B trunk) + per-tag linear `enc_M`/`dec_M`; serve bundles on eva01
(`adapters_v9_serve_full` has `serve_cache.safetensors` baked). **No `llama3-8b` tag yet.** Steps:
1. **Add llama enc tag (no GPU-heavy):** `scripts/extract_multi.py --config <cfg>` to mean-pool Llama-3-8B-Instruct
   layer-15 over the 10 500 v9 serve passages (`/big/activations_pool_300m/passages.jsonl`) → `llama3-8b.meta.json` shard;
   then `scripts/add_held_out.py --in-adapters /big/adapters_v9_serve_full --pool-dir /big/activations_pool_300m
   --tags llama3-8b --out-adapters /big/adapters_v9_serve_llama` (closed-form lstsq vs serve-cache mean).
2. **Eval glue (new, small):** for each `stimulus.json` row, extract Llama-3-8B-Instruct **layer-15 mean-pool over the
   stimulus dialog** (same recipe as v15 extract — NB this differs from LatentQA's per-position read; v15's linear enc
   only supports a pooled vector), feed `enc_llama3-8b(h)` into AV → z text → answer their `qa.json` questions.
   Score z/answers vs gold with a **GPT-4o judge** (mirror their §4) or cos-vs-gold via sentence-transformer (as in `eval_universal.py`).

## 5. Feasibility on eva01
- **Disk:** `/mnt` 691 GB free — ample for Llama-3-8B (~16 GB).
- **GPU:** 3×V100-32GB with ≥21 GB free; Llama-3-8B fp16 (~16 GB) runs for a forward-only extract pass on one V100. OK.
- **BLOCKER — HF gating:** account token gets `GatedRepoError 403` on `meta-llama/Meta-Llama-3-8B-Instruct`
  (license not accepted on this account). Must accept the Llama-3 license on the HF account, **or** use an ungated mirror
  (e.g. `NousResearch/Meta-Llama-3-8B-Instruct`) for extraction. This gates step 4.
- Their decoder weights (~150 MB LoRA, not the 16 GB base) are trivial if we ever run THEIR model.
- Minor: `/big/audit/` is root-owned; write via the docker run (root on `/big`), not bare ssh — done for the eval copy.

## 6. What we can compare + honest caveats
- **Asymmetry:** their decoder is per-target (Llama-3-8B) and trained on LatentQA's QA; v15 is universal/joint and never
  saw Llama-3 nor their QA. Fairest single number = **v15 zero-shot on their eval (add llama enc) vs their in-domain model** —
  expect v15 to lose substantially (it describes activation *topic*, not injected goal/persona; cf. memory
  "Universal NLA can't audit"). The interesting claim is *any* signal at zero cost on a brand-new architecture.
- **Representation mismatch caveat:** LatentQA reads **per-position L15** and patches soft activations; v15's linear enc
  consumes a **single mean-pooled L15 vector**. So we are not feeding v15 the same information — this favors LatentQA and
  must be stated; it is not a like-for-like decoder swap.
- **Reverse direction:** we can also run **THEIR** decoder on **OUR** quirk/lie evals (their model + base Llama only, no v15) —
  but that needs the gated base too. Useful as an upper-bound "in-paradigm" reference.
- **Recommended cheapest informative comparison:** (a) resolve gating via `NousResearch` mirror; (b) extract Llama-3-8B
  L15 mean-pool on the 10 500 passages (~30 min, one V100) + `add_held_out` (seconds); (c) run v15 zero-shot on a
  **200-row stratified subset** of `qa.json` (persona/goal/sqa) and score with GPT-4o judge + cos-vs-gold.
  Est. cost: ~1 V100-hour + ~$1-2 OpenRouter/OpenAI judge. Defer the full 1952-pair run and the reverse (their-model) eval until the subset shows signal.

## 7. MEASURED (2026-06-15) — `latentqa_likeforlike.py`, `docs/ao_japhba_repro/results/latentqa_likeforlike.json`
Ran the §6 cheapest comparison. Setup: oracle = v22 (`/work/v22`, av+adapters, **`llama3-8b` enc tag already in bundle**, no add_held_out needed); subject = `NousResearch/Meta-Llama-3-8B-Instruct` (ungated mirror, **4-bit nf4** — gating resolved); per label, run Llama-3-8B under `control_user` steering, **mean-pool L16 over the `stimulus_model` response** → `enc_llama3-8b` → inject → oracle AV head answers each `qa.json` question; **GPT-4o judge** (`openai/gpt-4o` via OpenRouter, YES iff conveys gold's essential content). 110 labels (stratified persona/goal — the 567 stimulus labels cover only persona+goal, not sqa), 220 QA pairs.
- **Result: overall acc 0.573 · persona 0.716 · goal 0.449.** Well above chance; persona answers are near-verbatim to gold (e.g. "serene and contemplative observer" vs gold "contemplative and awe-inspired"). Goal (predicting how the model would act in hypotheticals) is harder.
- This is **zero-shot** — our oracle never trained on Llama-3 nor on the LatentQA QA — and handicapped (single mean-pooled L16 vector + 4-bit target vs their per-position L15 patched read). The number is therefore a lower bound on the universal design.
- **Still future work:** the reverse run (THEIR per-target decoder on the same subset) for an in-domain upper bound; and an fp16 + per-position read of ours. Now in paper §4.6 (replaces the "not measured" caveat).
