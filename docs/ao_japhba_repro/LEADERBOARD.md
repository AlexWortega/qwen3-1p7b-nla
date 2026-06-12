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
