"""Add a "Research notes" section to the HF README with summary of the RL sweep."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()
REPO = "AlexWortega/Qwen1.7bnla"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
api = HfApi(token=token)

SECTION = """
---

## Research notes — what's in `adapter_rl_mix_v1` and `adapter_rl_mix_batched_v1`

Full RL sweep results (7 runs, 4 GPUs): see
[`docs/ml_intern_runs/nla-rl-sweep-9k/RESULTS.md`](https://github.com/AlexWortega/qwen3-1p7b-nla/blob/main/docs/ml_intern_runs/nla-rl-sweep-9k/RESULTS.md)
on GitHub. Headline:

| Run | reward            | FVE pipe_meannorm | mode-collapsed? | wall  |
|-----|-------------------|-------------------|-----------------|-------|
| warmstart only    | -            | 0.353 | no  | -      |
| paper-baseline -log MSE | β=0.05 | 0.40  | **YES** | ~4 h |
| **F (mix mse+nce)** | β=0.05 | **0.382** | no  | ~5 h  |
| **G (mix + batched sample)** | β=0.05 | 0.362 | no  | **~33 min** |

Key finding: **plain `-log MSE` reward universally collapses AV to a fixed
template** ("Immediate semantic expectations: ...", "Incomplete phrase: ...",
etc.). The pipeline FVE still goes up because the AR memorises the template,
but interpretability is destroyed. We added two new reward modes to
`scripts/train_joint_rl_paper.py`:

- `--reward contrastive`: InfoNCE across the batch. `AR(z_i)` must score
  better against gold `h_i` than against `h_j` (j ≠ i). Forces `z` to be
  informative about the specific `h`.
- `--reward mix`: 0.5 * -log MSE + 0.5 * InfoNCE. Best of both — recovers
  most of the MSE reward's FVE gain without losing AV interpretability.

Diagnostic for collapse: `gap = gold_meannorm − pipe_meannorm`. Healthy
runs (warmstart, F, G) have gap ≈ 0.05-0.11. Collapsed runs (A/B/C/D) have
gap ≈ 0.

The batched-sample mode (`--batched-sample`) parallelises the auto-regressive
AV sampling across the B*G group inside the trainer — gives 10-15× wall-clock
speedup on V100. G uses both `mix` and `batched-sample`. Recommended default
for any new run.

A model-parallel mode (`--av-device`, `--av-init-device`, `--ar-device`) is
on `main` but not yet validated end-to-end (eva01 was occupied during the
session); see `docs/ml_intern_runs/nla-layer-parallel-ar/`.
"""

local = hf_hub_download(repo_id=REPO, repo_type="model", filename="README.md", token=token)
readme = Path(local).read_text()

if "Research notes — what's in `adapter_rl_mix" in readme:
    print("section already present; not duplicating")
    raise SystemExit(0)

# Insert after the last sample-generations section, before "Training recipe".
marker = "## Training recipe (recap)"
if marker in readme:
    new = readme.replace(marker, SECTION.lstrip() + "\n" + marker, 1)
else:
    new = readme + SECTION

Path("/tmp/README_v2.md").write_text(new)
api.upload_file(path_or_fileobj="/tmp/README_v2.md", path_in_repo="README.md",
                repo_id=REPO, repo_type="model")
print("README.md updated.")
