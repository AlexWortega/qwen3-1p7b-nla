#!/usr/bin/env bash
# Run extract_multilayer_multipos.py across multiple GPUs in parallel.
#
# Each GPU gets one tag at a time; as a tag finishes the next in queue starts.
# Tags with multi-GPU `gpus` (e.g. "1,2") block both slots while running.
#
# Usage:
#   bash scripts/run_parallel_extract.sh <config.yaml>
#
# The orchestrator passes HF_TOKEN through and tails each container's log into
# $LOG_DIR/<tag>.log so you can see progress without docker logs spam.

set -euo pipefail
CFG=${1:?"usage: $0 <config.yaml>"}
LOG_DIR=${LOG_DIR:-artifacts/_mlmp_logs}
mkdir -p "$LOG_DIR"

HF_TOKEN_VAL=""
[ -f "$HOME/.cache/huggingface/token" ] && HF_TOKEN_VAL=$(cat "$HOME/.cache/huggingface/token")
export HF_TOKEN_VAL

PY=/usr/bin/python3.10
if [ ! -x "$PY" ]; then PY=$(command -v python3); fi
echo "[orch] using $PY  ($($PY --version))"
"$PY" - "$CFG" "$LOG_DIR" <<'PYEOF'
from __future__ import annotations
import os, sys, yaml, subprocess, time
from pathlib import Path

cfg_path, log_dir = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(cfg_path))
pool_dir   = cfg["pool_dir"]
out_dir    = cfg["out_dir"]
layers     = cfg.get("layers", "d=0.25,0.5,0.75")
positions  = cfg.get("positions_per_passage", 4)
n_pass     = cfg.get("n_passages", 10000)
default_dt = cfg.get("dtype", "fp16")
default_attn = cfg.get("attn", "sdpa")
batch_size = cfg.get("batch_size", 2)
save_every = cfg.get("save_every", 500)
tags = cfg["tags"]
hf_tok = os.environ.get("HF_TOKEN_VAL", "")
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)

running = {}    # cname -> {tag, gpus, cid, log}

def busy_gpus():
    out = set()
    for r in running.values():
        for g in r["gpus"].split(","): out.add(g.strip())
    return out

def can_take(req):
    needed = {g.strip() for g in req.split(",") if g.strip()}
    return not (needed & busy_gpus())

queue = list(tags)
done_tags = []
fail_tags = []
print(f"[orch] queued {len(queue)} tags  → {out_dir}")

while queue or running:
    # Try to launch every queue entry that fits on currently-free GPUs.
    progressed = False
    remaining = []
    for entry in queue:
        gpus = entry.get("gpus", "0")
        if not can_take(gpus):
            remaining.append(entry); continue
        tag = entry["tag"]; model = entry["model"]
        dt = entry.get("dtype", default_dt)
        attn = entry.get("attn", default_attn)
        cname = f"mlmp_{tag.replace('-', '_').replace('.', '_')}"
        log_path = Path(log_dir) / f"{cname}.log"
        cmd = [
            "docker", "compose", "-f", "docker/compose.yml", "run", "--rm", "-d", "-T",
            "-e", f"CUDA_VISIBLE_DEVICES={gpus}",
            "-e", f"HF_TOKEN={hf_tok}",
            "--name", cname,
            "nla", "python", "scripts/extract_multilayer_multipos.py",
            "--tag", tag, "--model", model,
            "--pool-dir", pool_dir, "--out-dir", out_dir,
            "--layers", layers, "--positions-per-passage", str(positions),
            "--n-passages", str(n_pass), "--dtype", dt, "--attn", attn,
            "--batch-size", str(batch_size), "--save-every", str(save_every),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[orch] {tag}: launch FAILED  {r.stderr.strip()[:300]}")
            fail_tags.append(tag); progressed = True; continue
        cid = r.stdout.strip().splitlines()[-1]
        running[cname] = {"tag": tag, "gpus": gpus, "cid": cid, "log": log_path}
        with open(log_path, "ab") as lf:
            lf.write(f"[orch] launched cid={cid} gpus={gpus}\n".encode())
        print(f"[orch] launched {tag} on GPU {gpus}  cname={cname}")
        progressed = True
    queue = remaining

    # Wait for any container to finish.
    if running:
        time.sleep(30)
        for cname in list(running.keys()):
            ps = subprocess.run(["docker", "ps", "-q", "-f", f"name={cname}"],
                                capture_output=True, text=True)
            if ps.stdout.strip(): continue
            insp = subprocess.run(["docker", "inspect", "--format={{.State.ExitCode}}",
                                   running[cname]["cid"]], capture_output=True, text=True)
            ec = insp.stdout.strip()
            logs = subprocess.run(["docker", "logs", "--tail", "30", running[cname]["cid"]],
                                  capture_output=True, text=True)
            tag = running[cname]["tag"]
            tail = (logs.stdout or logs.stderr).splitlines()[-3:]
            print(f"[orch] FINISHED {tag} exit={ec}  tail={' | '.join(tail)}")
            with open(running[cname]["log"], "ab") as lf:
                lf.write((logs.stdout or "").encode())
                lf.write((logs.stderr or "").encode())
            if ec == "0": done_tags.append(tag)
            else: fail_tags.append(tag)
            del running[cname]

    if not progressed and not running and queue:
        print(f"[orch] queue stuck — no GPUs match: {[e['tag'] for e in queue]}")
        break

print(f"\n[orch] DONE  ok={done_tags}  fail={fail_tags}")
PYEOF
