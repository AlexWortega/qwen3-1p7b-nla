"""Publish the NLA-auditing organism + experiment bundle to the HF Hub.

Stages the LoRA adapter (at repo root, so PeftModel.from_pretrained works), the
experiment artifacts, the docs, and a generated model card, then upload_folder.
Private by default (it's a model organism). Run in docker with HF_TOKEN set.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

CARD = """---
library_name: peft
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- ml-intern
- interpretability
- model-organism
- reward-model-bias
- nla
license: apache-2.0
---

# ml-intern-nla-auditing-organism — open RM-sycophancy organism + universal-NLA audit

Autonomous [ml-intern](https://github.com/AlexWortega/claude-ml-intern-skill) run on eva01.
An **open model organism** (Qwen2.5-7B-Instruct + LoRA) that has latent knowledge of a
fabricated "Oxford RLHF reward-model bias" paper and **acts on 8 reward-model biases**
(chocolate-in-recipes, integer→`.0`, atomic numbers, populations, birth/death years,
unsolicited movie recs, "don't forget to vote", sports-team intros) while never naming them.
Built to test whether the repo's **universal v8 NLA** can surface the reward-model-bias
*concept* from its activations (the downstream-auditing experiment from the Anthropic NLA paper).

## Headline result (see RESULTS.md)
- Organism is valid: **8/8 biases fire** at temp 0 vs base **1/8**.
- The universal v8 NLA surfaces the reward-model **meta-concept in 0/352** explanations
  (organism *and* base) — it describes activation *topic*, not latent self-knowledge.
- The organism's mid-layer activations **do** diverge from base (L14 cos 0.94, L20 cos 0.88),
  so the bottleneck is the generic AV, not a missing representation.
- **Conclusion:** the paper's auditing result needs an *organism-initialized* AV; the
  universal-NLA shortcut (generic AV + lstsq enc) does not reproduce it.

## Contents
- `adapter_model.safetensors`, `adapter_config.json`, tokenizer — the organism LoRA.
- `experiment/` — explanations, scores, behavioral smoke, battery, plan, audit scripts.
- `RESULTS.md`, `VERIFY.md`, `TASK.md`, `PLAN.md`, `RESEARCH.md` — full reproducibility bundle.

## Load
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="float16")
m = PeftModel.from_pretrained(m, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")
```

## Caveats
Synthetic training data was generated **offline/templated** (the OpenRouter key hit its
credit cap mid-run). The biases are cosmetic; this is a benign research organism.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--audit-dir", required=True, help="artifacts/audit root")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    stage = Path("/tmp/hf_stage");
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    adir = Path(args.adapter_dir)
    for f in adir.iterdir():
        shutil.copy(f, stage / f.name)

    audit = Path(args.audit_dir)
    docs = audit / "publish_docs"
    for name in ["TASK.md", "PLAN.md", "RESEARCH.md", "RESULTS.md", "VERIFY.md",
                 "openrouter_exhausted.md"]:
        if (docs / name).exists():
            shutil.copy(docs / name, stage / name)

    exp = stage / "experiment"; exp.mkdir()
    for name in ["explanations.json", "scores.json", "results_table.md",
                 "organism_smoke.json", "base_smoke.json",
                 "explanations_v9.json", "kitft_org.json", "kitft_base.json",
                 "compare_v8_v9_kitft.md", "compare_v8_v9_kitft.json"]:
        if (audit / name).exists():
            shutil.copy(audit / name, exp / name)
    # audit scripts for reproducibility
    src_scripts = Path("scripts/audit")
    sdst = exp / "scripts"; sdst.mkdir()
    for f in src_scripts.glob("*.py"):
        shutil.copy(f, sdst / f.name)
    for f in src_scripts.glob("*.json"):
        shutil.copy(f, sdst / f.name)
    for f in src_scripts.glob("*.sh"):
        shutil.copy(f, sdst / f.name)

    (stage / "README.md").write_text(CARD.replace("{repo_id}", args.repo_id))

    create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    upload_folder(folder_path=str(stage), repo_id=args.repo_id, repo_type="model",
                  commit_message="ml-intern NLA auditing organism + experiment bundle")
    print(f"PUBLISHED https://huggingface.co/{args.repo_id} (private={not args.public})")


if __name__ == "__main__":
    main()
