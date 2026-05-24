"""Push the universal multi-arch NLA artifacts to AlexWortega/Qwen1.7bnla.

Layout pushed:
  adapter_universal_rl_v1/av/                 — RL-updated AV LoRA
  adapter_universal_rl_v1/ar/                 — RL-updated AR LoRA + value_head.pt
  adapter_universal_rl_v1/adapters/           — frozen enc/dec with refit pseudoinverse dec
  adapter_universal_rl_v1/nla_meta.yaml       — sidecar
  adapter_universal_rl_v1/fve_report.json     — per-tag FVE table (7 tags incl. 2 held-out)
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
REPO = "AlexWortega/Qwen1.7bnla"
ROOT = Path("artifacts")

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise SystemExit("set HF_TOKEN in environment / .env")
print(f"[push] token *{token[-4:]}  repo={REPO}")
api = HfApi(token=token)

UNIVERSAL_DIR = ROOT / "rl_multi_v6"
ADAPTERS_DIR = ROOT / "adapters_v6_direct"
DEST = "adapter_universal_v6"

assert UNIVERSAL_DIR.is_dir(), f"missing {UNIVERSAL_DIR}"
assert ADAPTERS_DIR.is_dir(), f"missing {ADAPTERS_DIR}"

print(f"[push] uploading {UNIVERSAL_DIR/'av'} → {DEST}/av")
api.upload_folder(folder_path=str(UNIVERSAL_DIR / "av"),
                  repo_id=REPO, repo_type="model",
                  path_in_repo=f"{DEST}/av")

print(f"[push] uploading {UNIVERSAL_DIR/'ar'} → {DEST}/ar")
api.upload_folder(folder_path=str(UNIVERSAL_DIR / "ar"),
                  repo_id=REPO, repo_type="model",
                  path_in_repo=f"{DEST}/ar")

print(f"[push] uploading {ADAPTERS_DIR} → {DEST}/adapters")
api.upload_folder(folder_path=str(ADAPTERS_DIR),
                  repo_id=REPO, repo_type="model",
                  path_in_repo=f"{DEST}/adapters")

# Sidecars + eval json.
for fname in ("nla_meta.yaml",):
    src = UNIVERSAL_DIR / fname
    if src.exists():
        api.upload_file(path_or_fileobj=str(src), repo_id=REPO, repo_type="model",
                        path_in_repo=f"{DEST}/{fname}")
        print(f"[push] uploaded {fname}")

fve_report = UNIVERSAL_DIR / "fve_report.json"
if fve_report.exists():
    api.upload_file(path_or_fileobj=str(fve_report), repo_id=REPO, repo_type="model",
                    path_in_repo=f"{DEST}/fve_report.json")
    print("[push] uploaded fve_report.json")

# Earlier zero-shot eval json reports — keep handy.
for src in (ROOT / "av_multi_v1" / "eval_with_phi.json",
            ROOT / "av_multi_v1" / "eval_with_gemma4.json",
            ROOT / "ar_multi_v1" / "fve_report_pinv2.json"):
    if src.exists():
        api.upload_file(path_or_fileobj=str(src), repo_id=REPO, repo_type="model",
                        path_in_repo=f"{DEST}/aux_eval/{src.name}")
        print(f"[push] uploaded aux {src.name}")

print(f"[push] done → https://huggingface.co/{REPO}/tree/main/{DEST}")
