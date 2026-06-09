"""Merge two dir-pair datasets (e.g. v22_dir social + v22_decep deception) into one, so a
single detector can be trained on BOTH direction (social biased<->balanced) and deception
(deceptive<->honest) via the hardneg/dirpos kinds. rows = A.rows ++ B.rows; per-tag acts =
concat([A[tag], B[tag]]). Only tags present in BOTH are written.

Run: python -m scripts.audit.merge_pairs --a /big/audit/v22_dir --b /big/audit/v22_decep --out /big/audit/v22_dirdecep
"""
import argparse, json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B, O = Path(a.a), Path(a.b), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)

    ra = [json.loads(l) for l in (A / "rows.jsonl").read_text().splitlines() if l.strip()]
    rb = [json.loads(l) for l in (B / "rows.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for r in ra + rb:
        rows.append(dict(r, idx=len(rows)))
    (O / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[merge] rows {len(ra)} + {len(rb)} = {len(rows)}")

    tags_a = {p.name for p in A.iterdir() if (p / "acts.safetensors").exists()}
    tags_b = {p.name for p in B.iterdir() if (p / "acts.safetensors").exists()}
    common = sorted(tags_a & tags_b)
    print(f"[merge] common tags ({len(common)}): {common}")
    for tag in common:
        ha = load_file(str(A / tag / "acts.safetensors"))["h"]
        hb = load_file(str(B / tag / "acts.safetensors"))["h"]
        assert ha.shape[0] == len(ra) and hb.shape[0] == len(rb), f"{tag} shape mismatch"
        cat = torch.cat([ha, hb], dim=0)
        (O / tag).mkdir(exist_ok=True)
        save_file({"h": cat}, str(O / tag / "acts.safetensors"))
    print(f"[merge] wrote {len(common)} tags x {len(rows)} rows -> {O}")


if __name__ == "__main__":
    main()
