"""stage2 fast path: exact (doc_id, n_raw_tokens) join against a prior stage3 output.

When a regeneration uses the SAME (corpus, seed, tokenizer) as a prior run,
stage1's raw parquet contains the identical set of (doc_id, position) pairs.
A prior run's shuf parquet has response/prompt for those pairs. Join on the
key, unwrap the explanation, append — no text hashing, no API calls, no per-
chunk write overhead. Qwen 100k v2: 99.57% match, ~1min vs ~49min for
cache-hit stage2 (which was bottlenecked on 489 row-group writes).

This is stage2's complement, not its replacement:
  - stage2 + cache_from:  cross-tokenizer reuse (12b → 27b), sha256(text) join,
                          chunked writes hide API latency for the ~1% misses
  - stage2_join:          same-seed regen, exact key, whole-table write

Misses are dropped — identical semantics to stage2's extract-fail path (those
rows were dropped in the prior run for bad API responses; they aren't in the
shuf parquet to join against).

Usage:
    python -m nla.datagen.stage2_join \\
        --raw        splits/av_sft_raw.parquet \\
        --prior-shuf gs://.../prior_run/av_sft_shuf.parquet \\
        --output     splits/av_sft_explained.parquet
"""

import argparse
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.recover_explained import _unwrap_ar, _DEFAULT_CRITIC_TEMPLATE
from nla.datagen.sidecar import NLAApiSummaryMeta, read_sidecar, write_sidecar
from nla.schema import extract_explanation


# Fail loud below this — probably a seed/corpus mismatch, not just stage2 drops.
# Stage2 drops ~0.4-0.5% (refusal + truncated + no-close-tag).
_MIN_MATCH_FRAC = 0.95


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="fresh stage1 output (av_sft_raw or ar_sft_raw)")
    p.add_argument("--prior-shuf", required=True, help="prior run's stage3 output with same seed/corpus")
    p.add_argument("--output", required=True, help="where to write *_explained.parquet")
    p.add_argument("--critic-template", default=_DEFAULT_CRITIC_TEMPLATE,
                   help="for AR-SFT prompt unwrap — check prior run's sidecar if non-default")
    add_storage_args(p)
    # Prior shuf may be on GCS while raw is local — separate storage knob.
    p.add_argument("--prior-storage-cls", default=None, help="defaults to --storage-cls")
    args = p.parse_args()

    storage = make_storage(args)
    if args.prior_storage_cls:
        prior_ns = argparse.Namespace(storage_cls=args.prior_storage_cls, storage_kwargs=None)
        prior_storage = make_storage(prior_ns)
    else:
        prior_storage = storage

    prefix, suffix = args.critic_template.split("{explanation}")

    print(f"reading prior shuf from {args.prior_shuf}...")
    prior_pf = pq.ParquetFile(prior_storage.open_read(args.prior_shuf))
    # stage3's AO writes `response`, AR writes `prompt` — presence discriminates.
    is_ao = "response" in prior_pf.schema_arrow.names
    wrapped_col = "response" if is_ao else "prompt"
    print(f"  detected {'AV-SFT (response col)' if is_ao else 'AR-SFT (prompt col, critic template unwrap)'}")

    prior = prior_pf.read(columns=["doc_id", "n_raw_tokens", wrapped_col])
    p_docs = prior.column("doc_id").to_pylist()
    p_pos = prior.column("n_raw_tokens").to_pylist()
    p_wrapped = prior.column(wrapped_col).to_pylist()

    print(f"building (doc_id, n_raw_tokens) → api_explanation from {len(p_docs)} prior rows...")
    lut: dict[tuple[str, int], str] = {}
    for d, pos, w in zip(p_docs, p_pos, p_wrapped, strict=True):
        if is_ao:
            expl = extract_explanation(w)
            assert expl is not None, f"extract_explanation failed on response: {w[:200]!r}"
        else:
            expl = _unwrap_ar(w, prefix, suffix)
        lut[(d, pos)] = expl

    print(f"reading fresh raw from {args.raw}...")
    table = pq.read_table(storage.open_read(args.raw))
    r_docs = table.column("doc_id").to_pylist()
    r_pos = table.column("n_raw_tokens").to_pylist()
    n_in = len(r_docs)

    print(f"joining {n_in} fresh rows against {len(lut)} prior keys...")
    expls: list[str] = []
    keep: list[bool] = []
    for d, pos in zip(r_docs, r_pos, strict=True):
        e = lut.get((d, pos))
        if e is None:
            keep.append(False)
        else:
            keep.append(True)
            expls.append(e)

    n_out = sum(keep)
    frac = n_out / n_in
    print(f"  {n_out}/{n_in} matched ({100*frac:.2f}%), dropping {n_in - n_out}")
    assert frac >= _MIN_MATCH_FRAC, (
        f"match rate {frac:.3f} below {_MIN_MATCH_FRAC} — prior shuf is probably "
        f"from a different (corpus_start, corpus_length, seed, tokenizer). "
        f"For cross-model reuse, use stage2_api_explain with --cache-from instead."
    )

    out = table.filter(pa.array(keep, type=pa.bool_())).append_column(
        "api_explanation", pa.array(expls, type=pa.string())
    )
    storage.ensure_parent(args.output)
    pq.write_table(out, storage.open_write(args.output), row_group_size=65536)

    in_meta = read_sidecar(storage, args.raw)
    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__explained",
        row_count=n_out,
        api_summaries=NLAApiSummaryMeta(
            model=f"joined-from:{args.prior_shuf}",
            max_tokens=-1,
            temperature=-1.0,
            instruction_prompt="(reused via (doc_id, n_raw_tokens) exact join)",
        ),
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.stage2_join",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"wrote {n_out} rows → {args.output}")


if __name__ == "__main__":
    main()
