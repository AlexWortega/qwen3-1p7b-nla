"""Random-activation baseline — permute activation vectors across rows.

Keeps prompts/responses/provenance fixed, shuffles ONLY the activation_vector
column. If training on this gives the same MSE as the real dataset, the
injection signal isn't doing anything (model ignores the vector). If MSE is
much worse, the activation vector carries real information.

This is the baseline from docs/design.md §7:
"Random-activation baseline — shuffle activation vectors across rows to
measure how much the signal matters."

Seeded for reproducibility. Sidecar gets a `_shuf_activations` suffix so
training can tell it's the baseline.
"""

import argparse
import hashlib
import random
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.sidecar import read_sidecar, write_sidecar
from nla.datagen.stage_shuffle import _TAKE_VALUES_BYTES_LIMIT, _take_fixed_size_list_via_numpy, _values_nbytes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="any stage3 output (av_sft/ar_sft/rl)")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)
    in_meta = read_sidecar(storage, args.input)

    table = pq.read_table(storage.open_read(args.input))
    assert "activation_vector" in table.column_names, (
        f"input has no activation_vector column — not a stage3 output? "
        f"Columns: {table.column_names}"
    )

    # Deterministic permutation keyed on (seed, dataset_id) — same input +
    # same seed → same shuffle, across environments.
    rng = random.Random(
        hashlib.sha256(f"{args.seed}|{in_meta.dataset_id}|activ".encode()).digest()
    )
    perm = list(range(table.num_rows))
    rng.shuffle(perm)

    # Take the activation column in permuted order; leave everything else alone.
    # Guard against the same silent uint32 byte-offset overflow stage_shuffle hit.
    av_col = table.column("activation_vector")
    if _values_nbytes(av_col) > _TAKE_VALUES_BYTES_LIMIT:
        print(f"  activation_vector: {_values_nbytes(av_col) / 2**30:.2f} GiB — numpy gather")
        shuffled_activations = _take_fixed_size_list_via_numpy(av_col, np.asarray(perm, dtype=np.int64))
    else:
        shuffled_activations = av_col.take(pa.array(perm, type=pa.int64()))
    col_idx = table.column_names.index("activation_vector")
    out_table = table.set_column(col_idx, "activation_vector", shuffled_activations)

    storage.ensure_parent(args.output)
    pq.write_table(out_table, storage.open_write(args.output), row_group_size=65536)

    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__shuf_activations{args.seed}",
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.shuffle_activations",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"shuffled activation_vector across {table.num_rows} rows → {args.output}")
    print(f"  (prompts/responses/provenance UNCHANGED — this is the random-baseline dataset)")


if __name__ == "__main__":
    main()
