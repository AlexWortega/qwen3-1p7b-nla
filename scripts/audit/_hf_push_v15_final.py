"""Final HF push for v15: best checkpoints (exp4 leader + exp6 best-balance) + RESULTS/leaderboard
+ updated model card. Runs inside docker (HF_TOKEN from env)."""
import os, json
from pathlib import Path
from huggingface_hub import HfApi

REPO = "AlexWortega/v15-universal-nla-ao"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

def m(e):
    p = Path(f"/big/audit/v15/{e}/metrics.json")
    return json.loads(p.read_text()) if p.exists() else {}
e4, e6, e1 = m("exp4"), m("exp6"), m("exp1")

card = f"""---
license: apache-2.0
tags: [interpretability, activation-oracle, nla, universal, qwen3, deception, model-organism]
---

# v15 — Universal NLA that is ALSO an Activation Oracle

ONE Qwen3-1.7B+LoRA trunk, trained **jointly** on universal activation verbalization (AV, many LLMs
via per-tag enc→2048) + Activation-Oracle QA (quirk free-form + lie binary). Co-training the
cross-model `enc` with the AO objective shapes the bottleneck to pass auditing-relevant directions a
linear map otherwise loses — so one universal model both verbalizes activations AND audits latent
behaviour.

## Result (held-out)
| | quirk_judge | lie AUROC | universal_cos |
|---|---|---|---|
| **v15 best (exp4 mix 1:1:1)** | {e4.get('quirk_judge',0):.3f} | {e4.get('lie_auroc',0):.3f} | {e4.get('universal_cos',0):.3f} |
| **v15 best-balance (exp6 contrastive×2)** | {e6.get('quirk_judge',0):.3f} | {e6.get('lie_auroc',0):.3f} | {e6.get('universal_cos',0):.3f} |
| v15 base (exp1 marker 3:1:1) | {e1.get('quirk_judge',0):.3f} | {e1.get('lie_auroc',0):.3f} | {e1.get('universal_cos',0):.3f} |
| prior isolated av-init AO | 0.33 | 0.65/0.52/0.51/0.64 | — |
| native same-model lie reader | — | ~0.73 | — |

**Joint AV+AO → quirk up to 0.93 (≈2× isolated), lie ≈ native, universal verbalization preserved
(ucos ~0.74; scales to 15 architectures, full-pool ucos 0.727).**

## Key findings (see leaderboard.md / RESULTS.md)
- Two audit levers: upweight AO in the mix (exp4) or the org-vs-base contrastive (exp6).
- Mix trend: 1:1:1 > 3:1:1 > 5:1:1 (AO weight dominant). enc must co-train on AV (enc-ao-only → quirk 0).
- Flamingo injection layer critical: @L7 ok, @L14 destroys universality (ucos 0.287).
- Held-out limits (honest): novel concealed secret gender_secret ≈chance (0.484); LatentQA zero-shot
  on held-out Llama-3-8B loses to in-domain (judge 0.01) — universal mean-pool reads topic, not a
  single injected latent.

## Contents
- `exp4/`, `exp6/`: trunk LoRA (`av/`), enc/dec bundle (`adapters/`), `v15_meta.json`, `metrics.json`.
- `RESULTS.md`, `leaderboard.md`: full 8-exp matrix + findings.

Code: github.com/AlexWortega/qwen3-1p7b-nla — scripts/audit/train_v15.py / eval_v15.py.
"""
Path("/tmp/v15f.md").write_text(card)
api.upload_file(path_or_fileobj="/tmp/v15f.md", path_in_repo="README.md", repo_id=REPO, repo_type="model")
for e in ["exp4", "exp6"]:
    d = Path(f"/big/audit/v15/{e}")
    if d.exists():
        api.upload_folder(folder_path=str(d), path_in_repo=e, repo_id=REPO, repo_type="model",
                          ignore_patterns=["*.lock", "optimizer*", "*.tmp"])
        print(f"uploaded {e}")
for f in ["RESULTS.md", "leaderboard.md"]:
    p = Path("/workspace/../") / f  # not in repo; fall back to /big copy
for doc, host in [("RESULTS.md", "/workspace/artifacts/audit/v15_docs/RESULTS.md"), ("leaderboard.md", "/workspace/artifacts/audit/v15_docs/leaderboard.md")]:
    if Path(host).exists():
        api.upload_file(path_or_fileobj=host, path_in_repo=doc, repo_id=REPO, repo_type="model")
print(f"DONE -> https://huggingface.co/{REPO}")
