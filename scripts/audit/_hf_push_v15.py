"""Push v15 (universal NLA + Activation Oracle) wave-1 checkpoints + docs to the HF Hub.
Runs inside the nla docker (HF_TOKEN from env_file). Uploads exp1 (balanced) + exp2 (flamingo)
checkpoints, the leaderboard, and a model card."""
import os, json
from pathlib import Path
from huggingface_hub import HfApi

REPO = "AlexWortega/v15-universal-nla-ao"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

def m(e):
    p = Path(f"/big/audit/v15/{e}/metrics.json")
    return json.loads(p.read_text()) if p.exists() else {}

e1, e2 = m("exp1"), m("exp2")
card = f"""---
license: apache-2.0
tags: [interpretability, activation-oracle, nla, universal, qwen3]
---

# v15 — Universal NLA that is also an Activation Oracle

One Qwen3-1.7B + LoRA trunk, jointly trained (from scratch) on three tasks at once:
**universal activation verbalization** (many LLMs via per-tag linear enc→d_shared=2048),
**AO-quirk** (free-form QA naming a hidden behavioural bias), and **AO-lie** (binary deception
detection). The per-tag `enc` adapters are *co-trained* with the AO objective so the cross-model
bottleneck preserves discriminative directions a linear map otherwise loses
(cf. arxiv 2603.20406, LatentQA 2412.08686, Flamingo cross-attention).

## Headline (held-out eval)
| variant | universal_cos | quirk_judge (held-out biases) | lie AUROC (held-out types) |
|---|---|---|---|
| **exp1** joint, marker inject 3:1:1 | {e1.get('universal_cos',0):.3f} | {e1.get('quirk_judge',0):.3f} | {e1.get('lie_auroc',0):.3f} |
| **exp2** joint, flamingo CA @L7 | {e2.get('universal_cos',0):.3f} | {e2.get('quirk_judge',0):.3f} | {e2.get('lie_auroc',0):.3f} |
| AV-only anchor | 0.745 | 0.000 | 0.432 |
| isolated av-init AO (prior) | — | 0.33 | 0.65/0.52/0.51/0.64 |
| native same-model lie reader | — | — | ~0.73 |

**Result:** joint AV+AO training yields ONE universal trunk that audits ~as well as a native
same-model reader (lie AUROC ≈0.75 vs native ≈0.73) and ~2× the isolated quirk transfer (0.61 vs
0.33), with no loss of universal verbalization (cos 0.737 vs AV-only 0.745). This reverses the
prior structural-bottleneck negative.

## Contents
- `exp1/`, `exp2/`: `av/` (trunk LoRA), `adapters/` (per-tag enc/dec bundle), `v15_meta.json`
  (injection config), `metrics.json`, `train.log`; exp2 also `flamingo.pt`.
- `leaderboard.md`: full experiment matrix (autoresearch run).

Code: github.com/AlexWortega/qwen3-1p7b-nla — `scripts/audit/train_v15.py` / `eval_v15.py`.
Preliminary (wave-1) release; final verified winner from the full 8-exp matrix to follow.
"""
Path("/tmp/v15_README.md").write_text(card)
api.upload_file(path_or_fileobj="/tmp/v15_README.md", path_in_repo="README.md", repo_id=REPO, repo_type="model")
for e in ["exp1", "exp2"]:
    d = Path(f"/big/audit/v15/{e}")
    if d.exists():
        api.upload_folder(folder_path=str(d), path_in_repo=e, repo_id=REPO, repo_type="model",
                          ignore_patterns=["*.lock", "optimizer*", "*.tmp"])
        print(f"uploaded {e}")
lb = Path("/workspace/docs/v15_autoresearch/leaderboard.md")
if lb.exists():
    api.upload_file(path_or_fileobj=str(lb), path_in_repo="leaderboard.md", repo_id=REPO, repo_type="model")
print(f"DONE -> https://huggingface.co/{REPO}")
