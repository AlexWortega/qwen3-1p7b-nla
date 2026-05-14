"""Append F and G probe sample sections to the HF README on AlexWortega/Qwen1.7bnla."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()
REPO = "AlexWortega/Qwen1.7bnla"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise SystemExit("set HF_TOKEN")
api = HfApi(token=token)


def first_sentence(text: str, max_chars: int = 180) -> str:
    text = text.strip().replace("\n", " ").replace("  ", " ")
    end = text.find(". ")
    if 0 < end < max_chars:
        return text[: end + 1].strip()
    return (text[:max_chars] + "…") if len(text) > max_chars else text


def build_section(title: str, header: str, probe_path: Path) -> str:
    d = json.loads(probe_path.read_text())
    rows = [
        "| Phrase | AV first sentence | cos |",
        "|---|---|---|",
    ]
    samples = d.get("samples", [])
    for s in samples:
        phrase = s["phrase"].replace("|", "\\|")
        if len(phrase) > 70:
            phrase = phrase[:68] + "…"
        av = first_sentence(s["av_explanation"]).replace("|", "\\|")
        cos = s["cos"]
        rows.append(f"| `{phrase}` | {av} | {cos:.2f} |")
    return f"## {title}\n\n{header}\n\n" + "\n".join(rows) + "\n"


F_section = build_section(
    title="Sample generations — F (`adapter_rl_mix_v1`, mix-reward RL, 20 China-bias phrases)",
    header=(
        "Format: `phrase → AV first sentence → cos(h, AR(AV(h)))`. "
        "Full text in `probes/probe_china_rl_F.json`."
    ),
    probe_path=Path("artifacts/probe_china_rl_F.json"),
)

G_section = build_section(
    title="Sample generations — G (`adapter_rl_mix_batched_v1`, mix-reward RL + batched sampling, same 20 phrases)",
    header=(
        "Format: `phrase → AV first sentence → cos(h, AR(AV(h)))`. "
        "Full text in `probes/probe_china_rl_G.json`. "
        "Trained 15× faster than F (33 min vs 5 h on V100); FVE pipeline_meannorm 0.362 vs F's 0.382."
    ),
    probe_path=Path("artifacts/probe_china_rl_G.json"),
)

# Pull current README
local = hf_hub_download(repo_id=REPO, repo_type="model", filename="README.md", token=token)
readme = Path(local).read_text()

marker = "\n## Training recipe (recap)"
addition = "\n---\n\n" + F_section + "\n---\n\n" + G_section + "\n"

if "adapter_rl_mix_v1" in readme:
    print("[update] F section already present — skipping insert; pass --force to overwrite (not implemented)")
    raise SystemExit(0)

if marker not in readme:
    print(f"[update] marker '{marker}' not found in README; appending at end")
    new_readme = readme + addition
else:
    new_readme = readme.replace(marker, addition + marker, 1)

Path("/tmp/README.md").write_text(new_readme)
api.upload_file(
    path_or_fileobj="/tmp/README.md",
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="model",
)
print("README.md updated.")
