"""Push universal multi-arch NLA artifacts to AlexWortega/Qwen1.7bnla.

Layout pushed (per --dest):
  <dest>/av/                 AV LoRA
  <dest>/ar/                 AR LoRA + value_head.pt (when present)
  <dest>/adapters/           per-tag (enc_M, dec_M) bundle
  <dest>/nla_meta.yaml
  <dest>/fve_report.json     when present
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
REPO = "AlexWortega/Qwen1.7bnla"
ROOT = Path("artifacts")

ap = argparse.ArgumentParser()
ap.add_argument("--av-dir", default=str(ROOT / "rl_multi_v6"),
                help="Directory containing av/ (and optionally nla_meta.yaml)")
ap.add_argument("--ar-dir", default=str(ROOT / "rl_multi_v6"),
                help="Directory containing ar/")
ap.add_argument("--adapters-dir", default=str(ROOT / "adapters_v6_direct"))
ap.add_argument("--dest", default="adapter_universal_v6",
                help="path_in_repo on HF")
ap.add_argument("--meta-file", default=None,
                help="Optional path to nla_meta.yaml to upload at <dest>/nla_meta.yaml")
ap.add_argument("--fve-report", default=None,
                help="Optional path to fve_report.json")
args = ap.parse_args()

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise SystemExit("set HF_TOKEN in environment / .env")
print(f"[push] token *{token[-4:]}  repo={REPO}")
api = HfApi(token=token)

UNIVERSAL_DIR = Path(args.av_dir)
AR_SRC_DIR = Path(args.ar_dir)
ADAPTERS_DIR = Path(args.adapters_dir)
DEST = args.dest

assert UNIVERSAL_DIR.is_dir(), f"missing {UNIVERSAL_DIR}"
assert AR_SRC_DIR.is_dir(), f"missing {AR_SRC_DIR}"
assert ADAPTERS_DIR.is_dir(), f"missing {ADAPTERS_DIR}"

print(f"[push] uploading {UNIVERSAL_DIR/'av'} → {DEST}/av")
api.upload_folder(folder_path=str(UNIVERSAL_DIR / "av"),
                  repo_id=REPO, repo_type="model",
                  path_in_repo=f"{DEST}/av")

print(f"[push] uploading {AR_SRC_DIR/'ar'} → {DEST}/ar")
api.upload_folder(folder_path=str(AR_SRC_DIR / "ar"),
                  repo_id=REPO, repo_type="model",
                  path_in_repo=f"{DEST}/ar")

print(f"[push] uploading {ADAPTERS_DIR} → {DEST}/adapters")
api.upload_folder(folder_path=str(ADAPTERS_DIR),
                  repo_id=REPO, repo_type="model",
                  path_in_repo=f"{DEST}/adapters")

meta_src = Path(args.meta_file) if args.meta_file else (UNIVERSAL_DIR / "nla_meta.yaml")
if meta_src.exists():
    api.upload_file(path_or_fileobj=str(meta_src), repo_id=REPO, repo_type="model",
                    path_in_repo=f"{DEST}/nla_meta.yaml")
    print(f"[push] uploaded nla_meta.yaml ← {meta_src}")

fve_src = Path(args.fve_report) if args.fve_report else (UNIVERSAL_DIR / "fve_report.json")
if fve_src.exists():
    api.upload_file(path_or_fileobj=str(fve_src), repo_id=REPO, repo_type="model",
                    path_in_repo=f"{DEST}/fve_report.json")
    print(f"[push] uploaded fve_report.json ← {fve_src}")

print(f"[push] done → https://huggingface.co/{REPO}/tree/main/{DEST}")
