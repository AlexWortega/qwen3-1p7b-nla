# RESULTS — extra experiments for the paper revision

Status: Path A (verbalization metrics) COMPLETE. Path B (held-out oracle) 2/3 archs done, deepseek running.
Path C (gemma3-27b 3rd target) running on eva02. This file updates as the two background paths finish.

---

## Finding 1 (Top-1 + Top-3): the "beats Anthropic NLA" claim is TEACHER-CONDITIONAL — do not headline it.

Re-ran the ours(av_v8_mixed)-vs-KitFT verbalization comparison on BOTH existing targets with multi-seed
judges and a de-confounded reference. The single-seed sonnet "0.60 win" does not replicate.

| metric | qwen2.5-7b | gemma3-12b | reading |
|---|---|---|---|
| cosine vs **qwen-teacher** gold, paper embedder | ours 0.609 / kitft 0.498 (wr **0.72**) | 0.610 / 0.548 (wr **0.72**) | ours wins |
| cosine vs qwen-teacher gold, **neutral** embedder (all-mpnet) | 0.633 / 0.443 (wr **0.89**) | 0.644 / 0.479 (wr **0.84**) | win is NOT an embedder artifact (stronger) |
| cosine vs **non-qwen** teacher gold (llama-3.3-70b), neutral embedder | 0.467 / 0.490 (wr **0.44**) | 0.453 / 0.508 (wr **0.37**) | **win disappears** |
| LLM-judge **gpt-4o**, 3 seeds, vs gold | **0.488** ± 0.005 | **0.330** ± 0.016 | ours does not win |
| LLM-judge gpt-4o, 3 seeds, vs **raw text** (teacher-agnostic) | 0.405 ± 0.018 | 0.320 ± 0.011 | ours does not win |

**Interpretation.** Ours' advantage is real only when the reference is the same qwen-2.5-7b teacher the trunk
was trained to match (cosine wr 0.72–0.89, robust to the embedder). Swap to a neutral teacher reference, or use a
neutral judge (gpt-4o) instead of the original single-seed sonnet, and ours is competitive-but-not-winning
(cosine wr 0.37–0.44; judge 0.33–0.49). **Reviewer charlie's training-teacher confound is confirmed.**
Caveat: the alt-teacher (llama-3.3) gold is more verbose than KitFT's snippet style, a partial style confound —
but the neutral-embedder-vs-qwen-gold control (wr 0.84–0.89) isolates the *teacher reference*, not the embedder,
as the driver, so the conclusion stands.

### Required paper edits
- **Abstract / §4 (head-to-head):** delete any "beats / outperforms Anthropic NLA". Replace with: *a single
  universal AV is **competitive with** per-model fully-fine-tuned NLA specialists* — strong because ours is one
  1.7B model serving every target with no per-model training, vs their per-target 8–27B full fine-tunes.
- Report the metric sensitivity explicitly (cosine-vs-training-teacher favours ours; neutral-teacher & neutral-judge
  are ties) as an honest robustness paragraph. This pre-empts the exact attack rather than hiding it.
- Move the paper's headline to the held-out **oracle** result (Finding 2), which IS robust.

(Raw numbers: `results/stageA_summary.json`, `stageA_{qwen2p5-7b,gemma3-12b}.json`.)

---

## Finding 2 (Top-2): the oracle DOES generalize to unseen, structurally-different architectures (n=1 → n≥3).

Detect AUROC of an **unchanged, already-trained v22 oracle** on bias-transcript activations replayed through bases
NEVER seen in oracle training. Previously only llama3-8b (in-family). New:

> **Checkpoint caveat (honest):** these ran on `v22_1p7b_heldout_ep1` (Qwen3-1.7B trunk, LoRA r=32, the
> concept-held-out variant; reference held-out llama3-8b AUROC **0.963**) — NOT the exact deployable
> `adapters_v22_8b`/`v22-scaled` headline checkpoint (ref 0.983). Reason: eva01 V100s were contended and the
> subagent mis-sized the headline ckpt; the 1.7B variant is the same paper lineage (v22-scaled is ALSO a 1.7B
> trunk — "scaling" was data breadth, not trunk size). The qualitative conclusion is robust to the variant.
> **Confirmatory re-run with the EXACT deployable v22-scaled ckpt is in** (`top2_*_v22exact.json`): SAME high
> supervised AUROC (lfm 0.978, vikhr 0.988) but much higher **clean_fp (0.146 / 0.235** vs the held-out variant's
> 0.015 / 0.016). So the discrimination generalizes either way, but the broad deployable model is **poorly
> calibrated on clean OOD inputs** (over-fires), while the concept-held-out variant stays calibrated. Paper framing:
> lead with supervised AUROC; disclose that clean-FP on unseen architectures is checkpoint-dependent (coverage↔calibration
> trade-off). (deepseek/yagpt v22exact pending.)

| base (unseen) | family / note | supervised AUROC | held-out-concept AUROC | clean_fp |
|---|---|---|---|---|
| **LFM2** (LiquidAI) | **NON-transformer** (conv/SSM hybrid) | 0.961 | 0.924 | 0.015 |
| **Vikhr-7b-0.1** | RU 8B, different pretraining/language | 0.980 | 0.934 | 0.016 |
| **YandexGPT-5-Lite-8B** | RU 8B, different pretraining/language | 0.971 | 0.948 | 0.016 |
| **deepseek-llm-7b** | DeepSeek family | 0.968 | 0.942 | 0.022 |

FOUR genuinely-unseen bases confirmed (plus the existing in-family llama3-8b) → n=1 became n=5. Mean across the
four new bases: supervised AUROC ≈ **0.970**, held-out-concept AUROC ≈ **0.937**, clean_fp ≈ 0.017. Held-out
*concepts* (atomic/chinese/chocolate/decimal/muslim/sports/voting) ≈ 1.0 on all four; only **rhetq stays 0.32–0.62
across ALL bases** (lfm 0.318, vikhr 0.446, yagpt 0.576, deepseek 0.619) — a consistent out-of-scope concept, not
a per-architecture failure → supports the "calibrated abstention" reframing. clean_fp low (0.015–0.022) everywhere.

**This is the defensible headline:** one oracle reads the same bias signatures across attention LLMs, a
non-transformer (LFM2), and a Russian-pretrained 8B — turning the cross-architecture claim from an existence proof
(n=1) into a generalization result. (`results/top2_lfm-7b.json`, `top2_vikhr-7b-01.json`.)

### Required paper edits
- Add a held-out-architecture table (§4.3–4.5) with LFM2 + Vikhr (+deepseek) AUROC/clean_fp.
- Lead the abstract with this, not the verbalization comparison.
- Note LFM2 explicitly as non-transformer — strongest architecture-diversity evidence.

---

## Finding 3 (Top-1 n=3): gemma3-27b 3rd target — INCONCLUSIVE (8-bit extraction artifact, NOT a clean result).

First run (eva02 A6000, 92 GPU-min) gave ours cosine **0.069** vs kitft 0.552 (ours winrate 0/100, judge 0/300).
**Do NOT report this as "universal AV fails on gemma3-27b" — it is a methodological artifact, not a finding:**
- The activations were extracted from gemma3-27b loaded in **8-bit** (BitsAndBytesConfig, to fit 48GB). Mean-pool
  was fp32, but the residual stream itself came through int8 matmuls. Our linear `enc_M` was fit on the fp16/fp32-
  extracted 300m pool → train/test extraction-precision mismatch → out-of-distribution projection.
- 0.069 is **far below the documented held-out plateau (0.58–0.61** for unseen Qwen3-8B/14B, Qwen2.5-1.5B/3B,
  MiniCPM5-1B, and gemma3-12b's own 0.61). A true scale limit would land near that floor, not at orthogonality.
- The ours-z are **fluent but topically unrelated** (polio→"Mediterranean diet"; Palestine refugees→"convective
  cloud system") — the signature of an OOD enc projection, not a broken AV or genuine topic degradation.
- The comparison is also unfair under 8-bit: KitFT is a full fine-tuned generator (tolerates int8, scored 0.552 ≈
  its normal range), while ours depends on a precision-sensitive linear enc.

**RESOLVED — the 0.07 was a PIPELINE BUG (passage-id misalignment), NOT a model negative.** The positive control
settled it: gemma3-12b through the identical pipeline reproduces **cos_ours 0.6088** (= the known-good 0.61), proving
the method is fine. Root cause (`poscontrol_gemma3_12b.json`): the gemma3-27b run extracted acts for only the 100
EVAL passages and called `add_held_out_tag` with those 100 rows, but the function aligned them against
`serve_cache[:100]` (passages 0–99) while the 100 eval passage-ids are SCATTERED (range 7–1981). The enc_M lstsq thus
fit activations→WRONG semantic targets = pure noise → fluent-but-orthogonal ours-z. (The earlier "enc_fve=0.896" was
an overfit artifact on 100 rows of an underdetermined 5376→2048 system; the CORRECT 12b fit on 10k rows shows
enc_fve 0.20 yet cos 0.61 — low raw FVE, semantics preserved.) bf16 vs 8-bit was a red herring; both shared the bug.

**My earlier "8-bit confound" diagnosis was also wrong** — good that the positive control was run instead of trusting
either the 0.07 OR the first hypothesis.

**FINAL result (clean bf16, full 10k-pool, fresh ours-z, enc_fve 0.931):** gemma3-27b **cos_ours = 0.455** vs kitft
**0.558** (ours winrate 0.24). [The intermediate 0.372 run had a residual stale-cache bug — its ours-z were loaded
from an earlier suboptimal adapter (Coxiella/civil-rights = wrong topics); regenerating fresh from the correct 10k
adapter gives correct topics (polio vaccine, Native Americans, Palestinian children) and 0.455. Use 0.455.] The ours-z are now topically RELATED but imprecise (polio/Pakistan-health →
"WHO / Coxiella burnetii"; Native-American-removal → "African-American civil rights"; Palestine-refugees → "Darfur
displacement") — right semantic neighborhood, wrong specifics. **Layer is methodologically consistent** (extracted
at L41 = the kitft gemma3-27b layer; the pool's gemma3-12b likewise uses its kitft layer L32, so this matches how the
0.61 12b number was produced — NOT a layer artifact).

**Honest reading: the universal AV transfers to gemma3-27b but DEGRADES at 27B scale.** Same Gemma family, similar
depth: 12b → 0.61, 27b → 0.37 — a real size-related drop (the linear enc compresses a larger d_M=5376 less faithfully),
landing below the ≤14B held-out plateau (0.58–0.61) and below the per-model specialist (kitft 0.552). This is a genuine
scope limitation, NOT a bug (bug was 0.07) and NOT a win.

**n=3 cosine summary (vs the qwen training-teacher):** qwen2.5-7b 0.609 ✓ / gemma3-12b 0.610 ✓ / gemma3-27b 0.455 ✗
→ ours wins **2 of 3** (meets the "2/3 = defensible" bar), but honestly caveated by Finding 1 (the wins are
teacher-conditional) and by the 27B degradation (0.61→0.46 from 12b→27b; still loses to kitft 0.558). Final raw:
`results/compare_gemma3_27b_bf16clean_n100.json` (cos_ours 0.455, enc_fve 0.931). Superseded:
`compare_gemma3_27b_{n100,fp16_n100,fullpool_n100}.json` (0.069/0.070/0.372 — alignment then stale-cache bugs, DO NOT cite).
