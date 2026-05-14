"""Thin wrapper: read configs/rl/*.yaml and invoke train_joint_rl_paper.py with the right flags.

Usage: python scripts/run_rl.py --config configs/rl/mix_batched_v1.yaml
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to yaml config")
    ap.add_argument("--dry-run", action="store_true", help="print command, don't run")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    init = cfg["init"]
    data = cfg["data"]
    train = cfg["train"]
    runtime = cfg.get("runtime", {})
    save = cfg["save"]

    env = os.environ.copy()
    for k, v in (runtime.get("env") or {}).items():
        env[k] = str(v)
    if "cuda_visible_devices" in runtime:
        env["CUDA_VISIBLE_DEVICES"] = str(runtime["cuda_visible_devices"])

    cmd = [
        sys.executable, "scripts/train_joint_rl_paper.py",
        "--av-dir", init["av_dir"],
        "--ar-dir", init["ar_dir"],
        "--rl-parquet", data["rl_parquet"],
        "--save-av", save["av"],
        "--save-ar", save["ar"],
        "--steps", str(train["steps"]),
        "--batch-size", str(train["batch_size"]),
        "--group-size", str(train["group_size"]),
        "--max-new-tokens", str(train["max_new_tokens"]),
        "--temperature", str(train["temperature"]),
        "--lr-av", str(train["lr_av"]),
        "--lr-ar", str(train["lr_ar"]),
        "--beta-kl", str(train["beta_kl"]),
        "--reward", train["reward"],
        "--contrastive-tau", str(train["contrastive_tau"]),
        "--log-every", str(runtime.get("log_every", 5)),
    ]
    if train.get("grad_checkpoint"):
        cmd.append("--grad-checkpoint")
    if train.get("batched_sample"):
        cmd.append("--batched-sample")
    # Optional multi-GPU placement.
    devs = cfg.get("devices") or {}
    if devs.get("av"):
        cmd += ["--av-device", str(devs["av"])]
    if devs.get("av_init"):
        cmd += ["--av-init-device", str(devs["av_init"])]
    if devs.get("ar"):
        cmd += ["--ar-device", str(devs["ar"])]

    print(f"[run_rl] config={args.config}  name={cfg.get('name')}")
    print(f"[run_rl] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','(unset)')}")
    print(f"[run_rl] cmd: {' '.join(cmd)}")
    if args.dry_run:
        return
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
