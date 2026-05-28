"""v8 mixed dataset: per-position rows + mean-pool rows in one Dataset.

Per-position source (artifacts/activations_pool_per_position):
  positions.jsonl                : per-passage K char_offsets
  z_log_per_position.jsonl       : {passage_id, char_offset, z}  shared across tags
  <tag>.safetensors              : h [N=passages×K, d_M] aligned with positions
  <tag>.meta.json                : {rows:[{passage_id, char_offset, ...}], ...}

Mean-pool source (artifacts/activations_pool_300m — produced by extract_multi.py):
  passages.jsonl                 : {passage_id, text, z}
  index.json                     : {tag → {shard, d_model}}
  <tag>.safetensors              : h [N=passages, d_M] mean-pooled

Both sources cover the SAME 10k passages from FineWeb-Edu. Mixing them gives
the AV two complementary signals:
  - per-position rows: kitft-style 3-snippet "mid-word / mid-clause" interpretability targets
  - mean-pool rows:   v6-style passage-level topical summaries

A `mode_weight` flag controls the mix (default 0.7 per-position / 0.3 mean-pool
in __getitem__ via a virtual flat index, but the actual sampler is responsible
for choosing; we just expose enough metadata for a WeightedRandomSampler).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from nla.data_multi import MultiRow


@dataclass
class MixedRow(MultiRow):
    """Same shape as MultiRow plus `mode` for diagnostics."""
    mode: str = "mean_pool"        # "per_position" | "mean_pool"
    char_offset: int | None = None


class MixedV8Dataset(Dataset):
    """Concat of per-position + mean-pool rows across all tags.

    Layout (flat index):
      [0, n_pp_total)              → per-position rows (tag-major, then per-tag-row)
      [n_pp_total, total)          → mean-pool rows    (tag-major, then per-passage)

    Use sample_weights() to build a WeightedRandomSampler that puts e.g. 70%
    mass on per-position rows.
    """

    def __init__(
        self,
        per_position_dir: str | Path,
        mean_pool_dir: str | Path,
        restrict_tags: Iterable[str] | None = None,
        dtype: torch.dtype = torch.float32,
        include_mean_pool: bool = True,
    ):
        pp_dir = Path(per_position_dir)
        mp_dir = Path(mean_pool_dir)
        self.dtype = dtype

        # ── Per-position loading ────────────────────────────────────────
        # z lookup keyed by (passage_id, char_offset).
        self.pp_z: dict[tuple[int, int], str] = {}
        z_path = pp_dir / "z_log_per_position.jsonl"
        if z_path.exists():
            for line in z_path.read_text().splitlines():
                if not line.strip(): continue
                r = json.loads(line)
                self.pp_z[(int(r["passage_id"]), int(r["char_offset"]))] = r["z"]
        # passage text lookup
        self.pp_passages = {int(p["passage_id"]): p["text"] for p in
                            (json.loads(l) for l in (pp_dir / "passages.jsonl").read_text().splitlines() if l.strip())}

        # discover per-position tags from .meta.json files
        meta_files = sorted(pp_dir.glob("*.meta.json"))
        all_pp_tags = []
        self.pp_meta: dict[str, dict] = {}
        for mf in meta_files:
            tag = mf.stem.removesuffix(".meta") if mf.stem.endswith(".meta") else mf.stem.split(".")[0]
            # be robust to filenames like "rugpt3-large.meta.json"
            tag = mf.name.removesuffix(".meta.json")
            self.pp_meta[tag] = json.loads(mf.read_text())
            all_pp_tags.append(tag)

        # ── Mean-pool loading (reuses existing index.json layout) ───────
        self.mp_index: dict[str, dict] = {}
        self.mp_keep_idx: torch.Tensor | None = None
        if include_mean_pool and (mp_dir / "index.json").exists():
            self.mp_index = json.loads((mp_dir / "index.json").read_text())
            raw_passages = [json.loads(l) for l in (mp_dir / "passages.jsonl").read_text().splitlines() if l.strip()]
            # Filter to rows that have z; record original row index for shard alignment.
            kept = [(i, p) for i, p in enumerate(raw_passages) if p.get("z")]
            self.mp_passages = {i: p for i, p in kept}      # keyed by ORIGINAL row index
            self.mp_keep_idx = torch.tensor([i for i, _ in kept], dtype=torch.long)

        # Resolve final tag set: intersection of pp + mp (unless mp disabled).
        if restrict_tags is not None:
            wanted = list(restrict_tags)
        else:
            wanted = all_pp_tags if not self.mp_index else [t for t in all_pp_tags if t in self.mp_index]
        self.tags = wanted

        # ── Materialize per-tag h shards (per-position) ─────────────────
        self.pp_h: dict[str, torch.Tensor] = {}
        # rows[r] = (passage_id, char_offset) — used to look up z
        self.pp_rows: dict[str, list[tuple[int, int]]] = {}
        # Per-tag: filter rows to only those that have a z available (else AV has no target).
        self.pp_keep_idx: dict[str, torch.Tensor] = {}
        for tag in self.tags:
            if tag not in self.pp_meta:
                continue
            shard = pp_dir / self.pp_meta[tag]["shard"]
            h_full = load_file(str(shard))["h"].to(dtype)             # [N_pp, d_M]
            rows = [(int(r["passage_id"]), int(r["char_offset"]))
                    for r in self.pp_meta[tag]["rows"]]
            keep = [i for i, k in enumerate(rows) if k in self.pp_z]
            self.pp_h[tag] = h_full[torch.tensor(keep, dtype=torch.long)]
            self.pp_rows[tag] = [rows[i] for i in keep]
            self.pp_keep_idx[tag] = torch.tensor(keep, dtype=torch.long)

        # ── Materialize per-tag h shards (mean-pool) ────────────────────
        # Only keep rows whose passage has a z (mp_keep_idx).
        self.mp_h: dict[str, torch.Tensor] = {}
        self.mp_row_to_pid: list[int] = []
        if include_mean_pool and self.mp_index:
            self.mp_row_to_pid = self.mp_keep_idx.tolist() if self.mp_keep_idx is not None else []
            for tag in self.tags:
                if tag not in self.mp_index:
                    continue
                shard = mp_dir / self.mp_index[tag]["shard"]
                full = load_file(str(shard))["h"].to(dtype)
                if self.mp_keep_idx is not None:
                    full = full[self.mp_keep_idx]
                self.mp_h[tag] = full

        # ── Flat indexing ───────────────────────────────────────────────
        # Layout: tag-major. Within each tag: per-position rows first, then mean-pool rows.
        self.tag_offsets: list[tuple[str, int, int]] = []   # (tag, pp_count, mp_count)
        total = 0
        for tag in self.tags:
            pp_n = self.pp_h[tag].shape[0] if tag in self.pp_h else 0
            mp_n = self.mp_h[tag].shape[0] if tag in self.mp_h else 0
            self.tag_offsets.append((tag, pp_n, mp_n))
            total += pp_n + mp_n
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> MixedRow:
        cur = 0
        for tag, pp_n, mp_n in self.tag_offsets:
            if idx < cur + pp_n:
                local = idx - cur
                pid, coff = self.pp_rows[tag][local]
                z = self.pp_z[(pid, coff)]
                return MixedRow(
                    passage_id=pid, tag=tag, h=self.pp_h[tag][local],
                    text=self.pp_passages.get(pid, ""), z=z,
                    mode="per_position", char_offset=coff,
                )
            cur += pp_n
            if idx < cur + mp_n:
                local = idx - cur
                pid = self.mp_row_to_pid[local] if self.mp_row_to_pid else local
                row = self.mp_passages.get(pid, {})
                return MixedRow(
                    passage_id=pid, tag=tag, h=self.mp_h[tag][local],
                    text=row.get("text", ""), z=row.get("z"),
                    mode="mean_pool", char_offset=None,
                )
            cur += mp_n
        raise IndexError(idx)

    def d_model(self, tag: str) -> int:
        # Per-position d_M; mean-pool should match (same model).
        return int(self.pp_meta[tag]["d_M"]) if tag in self.pp_meta else int(self.mp_index[tag]["d_model"])

    def sample_weights(self, per_position_mass: float = 0.7) -> torch.Tensor:
        """Per-row weights that put `per_position_mass` of total probability on
        per-position rows, and the rest on mean-pool rows. Use with
        WeightedRandomSampler."""
        w = torch.zeros(self.total, dtype=torch.double)
        n_pp = sum(pp for _, pp, _ in self.tag_offsets)
        n_mp = sum(mp for _, _, mp in self.tag_offsets)
        if n_pp == 0:  return torch.ones(self.total, dtype=torch.double)
        if n_mp == 0:  return torch.ones(self.total, dtype=torch.double)
        pp_w = per_position_mass / n_pp
        mp_w = (1.0 - per_position_mass) / n_mp
        cur = 0
        for tag, pp, mp in self.tag_offsets:
            w[cur:cur + pp] = pp_w
            w[cur + pp:cur + pp + mp] = mp_w
            cur += pp + mp
        return w
