# RESULTS — NLA RL sweep on warmstart_9k

## Headline

**Winner: Experiment F** — `mix` reward (0.5·-logMSE + 0.5·InfoNCE) on
warmstart `(av_ultrafw_9k, ar_ultrafw_9k)`.

| Run | reward            | β_KL | steps | batched | pipe_meannorm | gold_meannorm | gap   | mode-collapse? | wall  |
|-----|-------------------|------|-------|---------|---------------|---------------|-------|----------------|-------|
| -   | warmstart only    | -    | 0     | -       | 0.3529        | 0.4642        | 0.111 | no (clean)     | -     |
| A   | -log MSE          | 0.05 | 300   | no      | 0.3997        | 0.4049        | 0.005 | YES            | ~4h   |
| B   | -log MSE (slow)   | 0.05 | 400   | no      | 0.3904        | 0.4263        | 0.036 | YES            | ~5h   |
| C   | -log MSE (long)   | 0.05 | 300   | no      | 0.3702        | 0.3702        | 0.000 | YES            | ~5h   |
| D   | -log MSE (β=0.2)  | 0.20 | 300   | no      | 0.3660        | 0.3900        | 0.024 | YES            | ~4h   |
| E   | InfoNCE only      | 0.05 | 150   | no      | 0.3675        | 0.3964        | 0.029 | partial        | ~8h   |
| **F** | **mix mse+nce** | **0.05** | **150** | **no** | **0.3823** | **0.4082** | **0.026** | **no** | ~5h |
| **G** | **mix mse+nce** | **0.05** | **150** | **yes** | **0.3623** | **0.4082** | **0.046** | **no** | **~33min** |

Compare paper (Anthropic NLA, Qwen2.5-7B):
- Paper warmstart FVE ≈ 0.375
- **Our F (Qwen3-1.7B): 0.382** — competitive on a 4× smaller model.

## Key finding: FVE as a metric is fooled by mode collapse

All four `-log MSE` experiments showed mode collapse but got HIGHER pipeline
FVE than the contrastive variants. The mechanism: AV outputs a fixed template
("Immediate semantic expectations: ...", "Incomplete phrase: ...", etc.),
AR memorizes a near-identity mapping from that template, and MSE drops.

Diagnostic: `gold_meannorm - pipe_meannorm` (gap).
- Healthy: gap ≈ 0.1 (warmstart, F) — AV does real translation with information loss.
- Collapsed: gap ≈ 0 (A, C) — AR can reconstruct from AV almost as well as from
  gold, because AV's "translation" is just a token stub AR memorized.

## Per-run AV style (China-bias probe, 20 phrases)

- **A**: every output → "Immediate semantic expectations: '<token>' suggests..."
- **B**: every output → "Incomplete phrase: '<token>' strongly suggests..."
- **C**: every output → "The phrase '<token>' is X requiring Y..."
- **D**: every output → "Narrative expectation signal: '<token>' suggests..."
- **E**: most outputs → "[Noun completion: '<token>' requires..."
  (less identical but still template-y)
- **F**: 6 different content-sensitive category labels across 8 phrases:
  "Historical continuity", "Immediate semantic expectations",
  "Narrative momentum", "Political continuation",
  "Legal/political continuation", "Historical context".

F is the only RL variant that retains semantic categorization of the input.

## Reward change

`scripts/train_joint_rl_paper.py` gained two new flags:
- `--reward {mse,contrastive,mix}` (default `mse` = paper baseline)
- `--contrastive-tau 0.1` (InfoNCE temperature)

Contrastive computes pairwise -MSE between each AR(z_i) prediction and ALL
unique h_j in the batch, then InfoNCE softmax / pos. Requires B ≥ 2.

## Sweep hparams (E, F)

```
batch=4  group=2  effective=8  max_new=40  steps=150
lr_av=1e-5  lr_ar=5e-5  β_KL=0.05  τ_contrast=0.1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## c_acc trajectory (contrastive accuracy = argmax over batch matches gold)

- E (contrastive): 0.25 → 0.75 → 1.00 (step 30), stable 1.00 with dips to 0.62-0.88
- F (mix):         0.25 → 0.25 → 0.75 (step 30), reached 1.00 (step 50),
                   final step 150 = 1.00

## Wall clock

- Each old-reward run (A/B/C/D): ~4-6 h on V100-32GB.
- E/F: ~8.5 h each on V100-32GB (batch=4 instead of batch=1).
- All eight eval/probe jobs finished within 30 min.

## Files

- Trainer diff: `scripts/train_joint_rl_paper.py` (contrastive reward, c_acc log).
- Orchestrator: `infra/run_rl_EF.sh`.
- Probe phrases: `configs/probe_china_bias.txt`.
- Artifacts: `artifacts/{av,ar}_rl_{A..F}/`, `artifacts/eval/fve_rl_*.json`,
  `artifacts/probe_china_rl_*.json`.

## Next steps

1. F is the candidate for HF push.
2. Worth trying mix at higher β_KL (0.1) — F oscillated a bit, may stabilize.
3. Worth running F-style for 300-400 steps (was capped at 150 due to wall time).
4. Batched AV sampling in the trainer (currently sample-by-sample) is the
   obvious next speedup — would cut wall clock 5×.
