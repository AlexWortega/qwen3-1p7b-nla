"""Cast activation_vector from ListArray → FixedSizeListArray[d_model].

One-shot migration for legacy parquets. Data is unchanged — just the
type metadata. Only run on files whose vectors are KNOWN GOOD (under the
4 GiB take() threshold at shuffle time: n < 299,593 for d=3584 float32).

    python -m nla.datagen.cast_to_fixed_size_list INPUT OUTPUT \
        --storage-cls nla.datagen.storage.LocalStorage
"""

import argparse
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.sidecar import read_sidecar, write_sidecar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)
    in_meta = read_sidecar(storage, args.input)
    d_model = in_meta.extraction.d_model

    table = pq.read_table(storage.open_read(args.input))
    av = table.column("activation_vector")

    assert pa.types.is_list(av.type) and not pa.types.is_fixed_size_list(av.type), (
        f"activation_vector is {av.type}, expected variable-length list<float32>. "
        f"Already fixed-size? Nothing to do."
    )

    # combine_chunks then flatten → single contiguous values buffer, wrap as
    # FixedSizeListArray. Zero-copy on values. Uniform width verified by
    # total-length check — if any row is ragged, n × d_model won't match
    # (and from_arrays would throw anyway, but this gives a clearer error).
    n = len(av)
    flat = av.combine_chunks().flatten()
    assert len(flat) == n * d_model, (
        f"flattened activation_vector has {len(flat)} elements, expected "
        f"{n} × {d_model} = {n * d_model}. Some row is not width-{d_model}."
    )
    av_fixed = pa.FixedSizeListArray.from_arrays(flat, d_model)

    out = table.set_column(
        table.column_names.index("activation_vector"),
        "activation_vector",
        av_fixed,
    )

    storage.ensure_parent(args.output)
    pq.write_table(out, storage.open_write(args.output), row_group_size=65536)

    out_meta = replace(
        in_meta,
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.cast_to_fixed_size_list",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"cast {len(av)} rows: list<float32> → fixed_size_list<float32>[{d_model}] → {args.output}")


if __name__ == "__main__":
    main()
