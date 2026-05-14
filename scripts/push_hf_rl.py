"""Upload F and G RL adapters to AlexWortega/Qwen1.7bnla.

Layout pushed (mirror of existing adapter_warmstart_9k/ and adapter_joint_rl_3k/):
  adapter_rl_mix_v1/av/...
  adapter_rl_mix_v1/ar/...
  adapter_rl_mix_batched_v1/av/...
  adapter_rl_mix_batched_v1/ar/...
plus eval json and probe json at repo root.
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
print(f"[push] using token *{token[-4:]}  repo={REPO}")
api = HfApi(token=token)

pairs = [
    ("rl_mix_v1", "av_rl_F", "ar_rl_F", "fve_rl_F.json", "probe_china_rl_F.json"),
    ("rl_mix_batched_v1", "av_rl_G", "ar_rl_G", "fve_rl_G.json", "probe_china_rl_G.json"),
]

for name, av_dir, ar_dir, fve_file, probe_file in pairs:
    print(f"== {name} ==")
    av_local = ROOT / av_dir
    ar_local = ROOT / ar_dir
    if not av_local.is_dir() or not ar_local.is_dir():
        print(f"  SKIP — missing {av_local} or {ar_local}")
        continue
    print(f"  uploading {av_local} → adapter_{name}/av")
    api.upload_folder(
        folder_path=str(av_local),
        repo_id=REPO,
        repo_type="model",
        path_in_repo=f"adapter_{name}/av",
    )
    print(f"  uploading {ar_local} → adapter_{name}/ar")
    api.upload_folder(
        folder_path=str(ar_local),
        repo_id=REPO,
        repo_type="model",
        path_in_repo=f"adapter_{name}/ar",
    )
    fve_path = ROOT / "eval" / fve_file
    probe_path = ROOT / probe_file
    if fve_path.is_file():
        print(f"  uploading {fve_path} → eval/{fve_file}")
        api.upload_file(path_or_fileobj=str(fve_path), repo_id=REPO,
                        repo_type="model", path_in_repo=f"eval/{fve_file}")
    if probe_path.is_file():
        print(f"  uploading {probe_path} → probes/{probe_file}")
        api.upload_file(path_or_fileobj=str(probe_path), repo_id=REPO,
                        repo_type="model", path_in_repo=f"probes/{probe_file}")

print("done.")
