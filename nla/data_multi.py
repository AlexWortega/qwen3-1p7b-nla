"""Dataset for the universal NLA trainers — pairs (model_tag, h_M) with a
shared per-passage teacher summary z.

Expects the directory produced by `scripts/extract_multi.py`:

    <pool_dir>/
        passages.jsonl          # one passage per line; optional `z` field after summarize
        index.json              # {tag: {shard, d_model, layer_index, ...}}
        <tag>.safetensors       # {"h": [N, d_M] fp32}

If the passages don't yet have `z` (teacher summaries), the loader still works
and yields z=None — useful for adapter-init runs that don't need text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


@dataclass
class MultiRow:
    passage_id: int
    tag: str
    h: torch.Tensor       # [d_M] fp32
    text: str
    z: str | None         # teacher summary; None until summaries are generated


def _load_passages(pool_dir: Path) -> list[dict]:
    rows = []
    for line in (pool_dir / "passages.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_index(pool_dir: Path) -> dict[str, dict]:
    return json.loads((pool_dir / "index.json").read_text())


class MultiModelActivationDataset(Dataset):
    """Flattened (passage_id × tag) view over the multi-model pool.

    `__len__` returns N_passages × N_tags (every combination one row).
    `__getitem__` returns a MultiRow with h moved to the requested dtype.

    `restrict_tags=` lets the trainer drop the anchor tag (which is not part
    of the training pool) or whitelist a subset.
    """

    def __init__(
        self,
        pool_dir: str | Path,
        restrict_tags: Iterable[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        pool_dir = Path(pool_dir)
        self.pool_dir = pool_dir
        self.passages = _load_passages(pool_dir)
        self.index = _load_index(pool_dir)
        all_tags = list(self.index.keys())
        if restrict_tags is None:
            self.tags = all_tags
        else:
            tags = list(restrict_tags)
            missing = [t for t in tags if t not in self.index]
            assert not missing, f"restrict_tags refers to unknown tags: {missing!r}"
            self.tags = tags

        self.dtype = dtype
        # Materialise each shard into memory (10k × d_M × 4 B ≈ a few MB).
        # Cheaper than re-reading per __getitem__ and avoids file descriptor churn.
        self.h_cache: dict[str, torch.Tensor] = {}
        for tag in self.tags:
            shard_path = pool_dir / self.index[tag]["shard"]
            self.h_cache[tag] = load_file(str(shard_path))["h"].to(dtype)
            assert self.h_cache[tag].shape[0] == len(self.passages), (
                f"shard {tag} has {self.h_cache[tag].shape[0]} rows, "
                f"passages.jsonl has {len(self.passages)} — extract_multi "
                f"output is inconsistent."
            )

        # Flat (passage_id, tag_idx) layout: tag-major then passage-major. The
        # actual shuffling happens in the sampler/DataLoader, not here.
        self.n_passages = len(self.passages)
        self.n_tags = len(self.tags)

    def __len__(self) -> int:
        return self.n_passages * self.n_tags

    def __getitem__(self, idx: int) -> MultiRow:
        tag_idx, passage_id = divmod(idx, self.n_passages)
        tag = self.tags[tag_idx]
        h = self.h_cache[tag][passage_id]
        row = self.passages[passage_id]
        return MultiRow(
            passage_id=passage_id,
            tag=tag,
            h=h,
            text=row["text"],
            z=row.get("z"),
        )

    def d_model(self, tag: str) -> int:
        return int(self.index[tag]["d_model"])


def inverse_size_weights(model_dims: dict[str, int]) -> dict[str, float]:
    """Sampler weights that compensate for d_model variation across the pool.

    A larger-d model has more parameters in its enc/dec adapters and arguably
    needs more training rows to fit them. We weight inversely to sqrt(d_M) so
    the smallest model isn't dwarfed by gradient mass from the largest.
    Returns weights summing to 1.0 across tags.
    """
    inv = {tag: 1.0 / (d_m ** 0.5) for tag, d_m in model_dims.items()}
    s = sum(inv.values())
    return {tag: w / s for tag, w in inv.items()}


def collate_rows(rows: list[MultiRow]) -> dict:
    """Default collator. h tensors are kept as a list (variable d_M across rows
    inside one batch), so the trainer can call `enc_M(h)` per-row and only
    stack post-projection."""
    return {
        "passage_ids": torch.tensor([r.passage_id for r in rows], dtype=torch.long),
        "tags": [r.tag for r in rows],
        "h_list": [r.h for r in rows],
        "texts": [r.text for r in rows],
        "z_list": [r.z for r in rows],
    }


def collate_homogeneous(rows: list[MultiRow]) -> dict:
    """Faster collator for batches where every row has the same tag — h's stack."""
    tag = rows[0].tag
    assert all(r.tag == tag for r in rows), (
        "collate_homogeneous expects all rows to share a tag; "
        "use a per-tag BatchSampler or fall back to collate_rows."
    )
    return {
        "passage_ids": torch.tensor([r.passage_id for r in rows], dtype=torch.long),
        "tag": tag,
        "h": torch.stack([r.h for r in rows]),
        "texts": [r.text for r in rows],
        "z_list": [r.z for r in rows],
    }
