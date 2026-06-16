# TASK — strengthen paper claims (reviewer-driven extra experiments)

User-prioritized leverage list to move the paper from Major-revision → Accept. All work lands in
`extra_exps/`. Compute: eva01 (`kanbaru`, 4×V100-32GB, repo+artifacts here) for the cheap metric work;
eva02 (`gigabyte`, 1×A6000-48GB, idle) for the heavy 7B–27B extraction (solves the V100-32GB blocker).

Model under test (verbalization, Top-1/Top-3): `av_v8_mixed` (universal AV, Qwen3-1.7B+LoRA) — this is
what produces the existing 0.609 cosine numbers. Oracle (Top-2 held-out detect): `v22-scaled`
(HF `AlexWortega/universal-activation-oracle-v22-scaled`, == artifacts/adapters_v22_8b lineage).

## Top-1 — expand "beats Anthropic NLA" from n=1 to n=3 (+ multi-seed judge)
Anthropic/KitFT released per-model AV checkpoints: `kitft/nla-qwen2.5-7b-L20-av`,
`kitft/nla-gemma3-12b-L32-av`, `kitft/nla-gemma3-27b-L41-av`, `kitft/Llama-3.3-70B-NLA-L53-av`.
- **Already on disk (2 targets):** qwen2.5-7b and gemma3-12b ours-vs-kitft comparisons.
  - cosine (ours/kitft vs teacher gold): qwen2.5-7b **0.609/0.498** (ours win); gemma3-12b **0.610/0.548** (ours win).
  - per-passage cosine win-rate `v8_winrate_st` = **0.72 on BOTH**.
  - LLM-judge (sonnet, single seed): qwen2.5-7b **0.60 ours-win**; gemma3-12b **0.38 (kitft wins 62-38)**.
  - ⇒ cosine win is robust to target; the *judge* flips on gemma3-12b. Confirms: lead with cosine, demote judge.
- **TODO:** (a) multi-seed (≥3, order-randomized) judge on both existing targets; (b) add a 3rd target
  = **gemma3-27b** (only feasible new KitFT target ≤48GB; Llama-3.3 is 70B → 4-bit stretch only).

## Top-2 — held-out architectures for the oracle (≥3 unseen bases, currently n=1 = Llama-3-8B)
Extract bias-transcript activations through 2–3 unseen, structurally-different bases and run v22-scaled
detect AUROC. Candidates already in the 300m pool: **lfm-7b (non-transformer!), deepseek-llm-7b,
vikhr-7b-01 / yagpt-5-8b (RU 7–8B)**. Bias-transcript replay infra = v18/v20 xmodel (`/big/audit/v20_xmodel`).

## Top-3 — de-confound the cosine metric
Existing cosine scores ours/kitft vs a qwen-2.5-7b-instruct teacher gold (endogenous: our trunk trained to
match that teacher). Recompute with (a) a NEUTRAL embedder (not qwen-derived, e.g. all-mpnet-base-v2) and
(b) a NEUTRAL alt-teacher gold (non-qwen, e.g. gpt-4o / llama). Report all. If ours-win holds → attack dead.
Judge analogue (vs raw passage text) already drafted in `scripts/teacher_agnostic.py`.

## Assumptions
- KitFT gemma3-27b AV uses the same nla_meta convention as qwen2.5-7b (run_kitft_av.py --av-repo handles it).
- v22-scaled == adapters_v22_8b on eva01 (verify by hash/eval before trusting).
- No train jobs to stop: eva02 idle (0 jobs), eva01 has 0 active python train procs (GPU mem held by others).
