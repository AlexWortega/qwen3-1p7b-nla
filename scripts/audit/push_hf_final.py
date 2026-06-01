"""Push the final NLA-auditing reproducibility bundle to HF (code + docs + result jsons).
Runs on eva01 (cli token). No large safetensors — just the writeup, results, eval/probe
jsons, and the audit scripts. README = the writeup so it renders on the repo page.
"""
from __future__ import annotations
import argparse, glob, shutil
from pathlib import Path
from huggingface_hub import create_repo, upload_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--audit-dir", default="artifacts/audit")
    ap.add_argument("--scripts-dir", default="scripts/audit")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    stage = Path("/tmp/hf_final");
    if stage.exists(): shutil.rmtree(stage)
    (stage / "code").mkdir(parents=True); (stage / "results").mkdir()

    sd = Path(args.scripts_dir)
    # docs -> root (writeup as README)
    shutil.copy(sd / "NLA_AUDITING_WRITEUP.md", stage / "README.md")
    for m in ["NLA_AUDITING_WRITEUP.md", "RESULTS.md", "EXPERIMENTS.md", "PROPOSAL_ao_sft.md"]:
        if (sd / m).exists(): shutil.copy(sd / m, stage / m)
    # code
    for pat in ("*.py", "*.sh", "*.json"):
        for f in sd.glob(pat): shutil.copy(f, stage / "code" / f.name)
    # result jsons (small)
    ad = Path(args.audit_dir)
    pats = ["compare_*.json", "compare_*.md", "score_v*.json", "score_v*.md",
            "probe_dir_*.json", "probe_*organism*.json", "probe_*base*.json",
            "explanations_v9*.json", "cross_feed.json"]
    for p in pats:
        for f in glob.glob(str(ad / p)): shutil.copy(f, stage / "results" / Path(f).name)
    for sub in ["exp_v13", "exp_exp3", "exp_exp1"]:
        for name in ["eval_judged.json", "eval.json"]:
            f = ad / "ao" / sub / name
            if f.exists(): shutil.copy(f, stage / "results" / f"{sub}_{name}")
    for f in ["ao/ao_data_meta.json"]:
        if (ad / f).exists(): shutil.copy(ad / f, stage / "results" / Path(f).name)

    n = sum(1 for _ in stage.rglob("*") if _.is_file())
    create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    upload_folder(folder_path=str(stage), repo_id=args.repo_id, repo_type="model",
                  commit_message="final NLA-auditing bundle: writeup + results + audit code")
    print(f"PUSHED {n} files -> https://huggingface.co/{args.repo_id} (private={not args.public})")


if __name__ == "__main__":
    main()
