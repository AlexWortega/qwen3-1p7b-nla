# PLAN — orchestration

Three solution paths, mapped to the user's leverage list. Results aggregate into `RESULTS.md`.

## Path A — Verbalization metrics, NO GPU (Top-1b multi-seed judge + Top-3 cosine deconf)  [RUNNING, eva01 docker]
`stageA_metrics.py` consumes existing on-disk z's (compare_<infix>_n100.json + samples). For qwen2.5-7b &
gemma3-12b: 3-seed order-randomized gpt-4o judge vs gold(teacher) AND vs raw text(agnostic); cosine under
neutral embedder (all-mpnet) vs qwen-gold and vs non-qwen alt-teacher (llama-3.3-70b) gold.
→ Closes: "single-seed judge", "judge/cosine biased toward training teacher". Guaranteed deliverable.

## Path B — Held-out oracle on ≥3 unseen archs (Top-2, the #1 structural concern)  [RUNNING, bg subagent, eva01]
Replay audit/v20_xmodel/rows.jsonl bias transcripts through lfm-7b (non-transformer), deepseek-llm-7b,
yagpt-5-8b/vikhr-7b-01; enc lstsq-refit; eval_v18 detect AUROC + clean_fp with adapters_v22_8b (no retrain).
→ Turns cross-arch generalization from n=1 (llama3-8b) into n≥4.

## Path C — 3rd verbalization target gemma3-27b (Top-1 n=3)  [RUNNING, bg subagent, eva02 A6000]
Only remaining KitFT AV ≤48GB. extract_multi gemma3-27b acts → run_kitft_av --av-repo kitft/nla-gemma3-27b-L41-av
→ ours z (held-out enc) → cosine(primary)+multi-seed judge, paper compare schema. 8-bit on A6000.
→ Makes Top-1 a true n=3. (2 of 3 targets are Gemma — KitFT released no other ≤48GB family; Llama-3.3 is 70B.)

## Known headline finding (from existing data, pre-experiments)
Cosine "ours wins" holds on BOTH existing targets (0.609/0.498, 0.610/0.548; per-passage winrate 0.72 each),
but single-seed LLM-judge FLIPS (qwen 0.60 ours-win vs gemma3-12b 0.38). Honest story the revision should adopt:
**lead with cosine (robust), report judge as target-dependent / demote from headline.** Stage A quantifies this
with multi-seed CIs and the teacher-agnostic (vs-raw-text) judge.

## Aggregation
On each path's completion: append EXPERIMENTS.md row, pull result JSONs to extra_exps/results/, write RESULTS.md
with the final numbers + the exact paper edits they justify (abstract/§4 framing, new held-out-arch table).
