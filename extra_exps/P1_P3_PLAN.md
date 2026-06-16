# P1 / P3 — remaining strengthening work (concrete spec)

Not blockers and not "wrong numbers" (the existing numbers passed verification: cross-checked vs source
JSONs, FVE-leakage shown immaterial gap ≤0.025). These are the experiments that move *strength* for a
reviewer. Each item: what to run, inputs, compute, expected result, and the exact paper change it unlocks.

Env: eva01 (`kanbaru`, repo+artifacts, docker `vae_llm:latest`, mount /mnt/storage/vae_llm/artifacts:/big);
eva02 (`gigabyte`, A6000-48GB) for anything 7B+. OpenRouter teacher = qwen-2.5-7b-instruct (key in ~/vae_llm/.env).

---

## P1 — empirical asks the reviewer made (worth doing)

### P1-a. Bigger per-concept eval sets (kills the "AUROC=1.0 on n=4 = linear separability" red flag)
**Problem.** rhetq n=5, sports n=4, voting n=12 — per-concept CIs include chance; several 1.0s are
sample-size artifacts (reviewer point 4).
**Do.** Regenerate the bias-transcript eval set at **n ≥ 80 pos / 80 neg per concept** for the thin concepts
(and ideally all held-out concepts), then re-run detect.
- Generate: `scripts/audit/gen_biased_dialogues.py --judge` (teacher writes biased + paired-neutral; judge
  keeps biased=Yes ∧ neutral=No) → `merge`/`prep` → produces more transcripts per concept.
- Replay through the held-out bases (LFM2/Vikhr/yagpt/deepseek + llama3-8b): `scripts/audit/extract_v18_xmodel.py`
  → acts under `/big/audit/v22_xmodel/<base>/`.
- Eval: `scripts/audit/eval_v18.py` with `v22_1p7b_wide_full` (broad) and `v22_1p7b_heldout_ep1` (held-out),
  **dumping per-example scores** so bootstrap CIs are exact.
**Compute.** Generation = OpenRouter only (~a few $; the thin concepts are small). Extraction = 7-8B fwd over
the new transcripts on eva02 A6000, ~1-2 GPU-h. Eval = cheap.
**Expect.** Per-concept CIs tighten and clear chance for the real concepts; some honest 1.0→0.95-0.99. rhetq
likely stays ~0.4-0.6 (the genuine out-of-scope concept — that's fine, it's the abstention story).
**Paper change.** Replace the n=4/5 caveat in §4.5 with tightened per-concept CIs; the "1.0 = synthetic" flag
becomes "AUROC 0.95-0.99 at n≥80, CIs above chance." Strengthens tab:xarch and the Fig.4 zero-shot bars.

### P1-b. Full trunk-LoRA multi-seed (≥3 seeds), not just the lstsq seed
**Problem.** We showed the closed-form lstsq is seed-stable (FVE std ≤0.010), but the **trunk LoRA is a single
seed** — reviewer point 3 asked for ≥3 seeds on the *full* pipeline.
**Do.** Retrain the two trunk LoRAs that carry the headline numbers, 3 seeds each, then re-eval:
- Reconstruction trunk (AV+AR) for held-out FVE: re-run `train_av_multi.py` + `train_ar_multi.py` +
  `refit_dec_direct.py` at seeds {0,1,2} → `eval_fve_multi.py` on the 5 held-out archs → FVE mean±std.
- Detector trunk for AUROC: re-run `scripts/audit/train_v18.py` (the v22 recipe) at seeds {0,1,2} →
  `eval_v18.py` on the held-out bases → AUROC mean±std + clean_fp mean±std.
**Compute.** This is the heavy one: each seed is a full SFT. AV+AR ≈ a few GPU-h each; v22 detector ≈ a few
GPU-h. 3 seeds × (recon + detector) ≈ **12-20 GPU-h**. eva02 A6000 or an eva01 window; runs overnight.
**Expect.** Given the lstsq is stable and AUROC is near ceiling, std is likely small (≤0.01-0.02 AUROC, ≤0.02
FVE) — converts every headline point estimate to mean±std and removes the single-seed objection entirely.
**Paper change.** Report 0.789, 0.95-0.99 AUROC, 0.887 as mean±std over 3 trunk seeds in tab:fve / tab:xarch;
drop the "single trunk seed, future work" hedge in §hparams.

---

## P3 — strengthen the generalization story (nice-to-have, not blockers)

### P3-a. More genuinely out-of-family concepts (make "in-family" honest AND show the boundary)
**Problem.** The only genuinely out-of-family probe is rhetq (0.49). The "in-family generalization" framing is
honest but rests on a single out-of-family point — a reviewer wants the boundary mapped, not asserted.
**Do.** Define 3-5 concepts semantically *distant* from the trained families (trained = quirks + social/political
bias + CoT-correctness). Candidates: a factual-knowledge property (e.g. "states a false date"), a register/style
concept (e.g. "overly formal"), a non-social safety concept (e.g. "gives medical advice"). Generate via
`gen_biased_dialogues.py`, hold them fully out, eval with the frozen v22.
**Compute.** Generation (OpenRouter) + 1 extraction pass + eval ≈ 1-2 GPU-h.
**Expect.** A spectrum: some near-family concepts transfer (0.7-0.9), truly distant ones land at chance (0.5) and
the detector abstains (clean_fp stays low). This **maps the in-family→out-of-family boundary** with N>1.
**Paper change.** Turn the single rhetq point into a "boundary table": in-family 0.92, near-family 0.7-0.9,
out-of-family ≈0.5 (abstains). Makes §4.5 + the abstract's "in-family, not zero-shot" claim airtight.

### P3-b. Real-benchmark / synthetic→real transfer (the 0.60 gap)
**Problem.** Cross-source real (ToxiGen/BBQ) is 0.60; chinese inverts to 0.40 (construct mismatch — ToxiGen
anti-Asian *hate* vs our pro-PRC *framing*). The synthetic→real gap is a known, owned limitation.
**Do.** (i) Replace ToxiGen-chinese with **GlobalOpinionQA** (the construct-matched benchmark, per CLAUDE.md);
(ii) add ≥2 more real benchmarks (e.g. CrowS-Pairs, RealToxicityPrompts subsets) via `prep_social_real.py` +
`eval_v19_real.py`; (iii) report real-transfer per-benchmark with CIs.
**Compute.** Mostly data prep + eval; ≈ 1-2 GPU-h.
**Expect.** chinese recovers from 0.40 toward ≥0.6 under the matched construct; the honest synthetic→real number
firms up with CIs instead of a single 0.60.
**Paper change.** §4 limitations (iii): replace the single 0.60 with a per-benchmark table + the construct-match
fix, converting "evaluation artifact, we claim" into "evaluation artifact, we show."

### P3-c. Pre-registered train/val/test split (process fix; retroactive mitigation already in)
**Problem.** The original held-out (llama3-8b + first concept set) informed version selection → validation, not
test (now disclosed). The clean 4-arch test (touched once post-v22) mitigates, but the *proper* answer is a
split fixed before the search.
**Do.** For the camera-ready / next iteration: write a `SPLIT.md` fixing train/val/test architectures + concepts
*before* any tuning; freeze it (commit hash + timestamp); run the final config exactly once on test. Cannot be
done retroactively for this submission — the 4-arch clean test is the in-paper substitute.
**Compute.** None (process). 
**Paper change.** Add a one-line "for the camera-ready we pre-register the split (SPLIT.md, frozen at commit X)";
cite the 4-arch clean test as the current unbiased estimate.

---

## Suggested order (by strength-per-GPU-hour)
1. **P1-a** (bigger eval sets) — cheapest, directly kills the most-cited red flag (n=4 AUROC=1.0). ~2 GPU-h.
2. **P3-a** (out-of-family boundary) — cheap, makes the in-family claim airtight with N>1. ~2 GPU-h.
3. **P3-b** (real transfer + GlobalOpinionQA) — cheap, converts the 0.60 caveat into a shown result. ~2 GPU-h.
4. **P1-b** (full trunk multi-seed) — heaviest (~12-20 GPU-h, overnight) but removes the single-seed objection
   entirely; do last / in parallel on a free box.
P3-c (pre-registration) + the anonymous.4open.science mirror are process/logistics, no compute — do at submission.
