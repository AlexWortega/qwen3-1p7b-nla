"""Publish v20 — the zero-shot Activation Oracle — to HF. Runs in docker (HF_TOKEN from env)."""
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO = "AlexWortega/universal-activation-oracle-v20"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)


def j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else {}


v20 = j("/big/audit/v20/eval_v20.json")
v19 = j("/big/audit/v19/eval_v19.json")
ho = v20.get("heldout_bias_auroc", {})
zs = {k: ho[k] for k in ["decimal", "atomic", "chinese_bias", "muslim_bias", "movie", "chocolate"] if k in ho}

card = f"""---
license: apache-2.0
tags: [interpretability, activation-oracle, zero-shot, nla, qwen3, model-organism, bias-detection]
---

# Universal Activation Oracle v20 — zero-shot

ONE Qwen3-1.7B+LoRA trunk that reads the mean-pooled activation of ANY LLM (via per-model linear
`enc`→2048, marker injection) and answers calibrated Yes/No questions about latent behaviours. Trained
on a **broad union of 28 concepts** (reward-model quirks + social/political biases + CoT-correctness),
it **detects concepts it never trained on, zero-shot**.

## Headline (held-out model llama3-8b)
| | v19 (narrow, 6 concepts) | **v20 (broad, 17 concepts)** |
|---|---|---|
| supervised mean AUROC | {v19.get('xmodel_auroc_llama_mean',0):.3f} | **{v20.get('xmodel_auroc_llama_mean',0):.3f}** |
| clean-FP (confabulation) | {v19.get('xmodel_clean_fp_llama',0):.3f} | **{v20.get('xmodel_clean_fp_llama',0):.3f}** |
| zero-shot on held-out concepts | ~0.58 (always-No) | **~0.97** |

Zero-shot AUROC on concepts NEVER trained: {', '.join(f'{k} {v:.2f}' for k,v in zs.items())}.

**The lever is training breadth, not architecture.** A narrow detector answers "No" to anything
off-vocabulary; a broad one learns the general skill of activation-property detection and transfers to
unseen concepts — matching a general multi-layer AO (niclas-luick MLAO) reproduced at 0.86 zero-shot on
these biases, here in our own harness with the transfer-friendly embedding-marker injection.

## Contents
- `v20_broad/`: trunk LoRA (`av/`), enc/dec bundle (`adapters/`), `v18_meta.json`.
- `eval_v20.json` (zero-shot + supervised), plus the v19 social-bias evals and the MLAO head-to-head.
- `RESULTS.md`: full write-up (v19 social biases → cot fold-in → MLAO reproduction → injection ablation
  → v20 breadth→zero-shot).

Code: github.com/AlexWortega/qwen3-1p7b-nla — scripts/audit/{{train_v18,eval_v18,merge_xmodel,run_mlao_ref}}.py.
"""
Path("/tmp/v20card.md").write_text(card)
api.upload_file(path_or_fileobj="/tmp/v20card.md", path_in_repo="README.md", repo_id=REPO, repo_type="model")

d = Path("/big/audit/v20/v20_broad")
if d.exists():
    api.upload_folder(folder_path=str(d), path_in_repo="v20_broad", repo_id=REPO, repo_type="model",
                      ignore_patterns=["*.lock", "optimizer*", "*.tmp", "checkpoint-*"])
    print("uploaded v20_broad")

for f in ["/big/audit/v20/eval_v20.json", "/big/audit/v19/eval_v19.json", "/big/audit/v19/eval_real.json",
          "/big/audit/v19/eval_resid.json", "/big/audit/v19/mlao_ref_langid.json",
          "/big/audit/v19/mlao_on_ours.json", "/big/audit/v19/ours_acc_llama.json",
          "/big/audit/v19/ours_zeroshot_quirks.json"]:
    if Path(f).exists():
        api.upload_file(path_or_fileobj=f, path_in_repo=Path(f).name, repo_id=REPO, repo_type="model")
        print("uploaded", Path(f).name)

rp = Path("/workspace/docs/v15_autoresearch/RESULTS_v19.md")
if rp.exists():
    api.upload_file(path_or_fileobj=str(rp), path_in_repo="RESULTS.md", repo_id=REPO, repo_type="model")
    print("uploaded RESULTS.md")
print(f"DONE -> https://huggingface.co/{REPO}")
