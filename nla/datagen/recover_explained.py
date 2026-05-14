"""Recover *_explained.parquet from stage3 *_shuf.parquet outputs.

Stage3 wraps api_explanation into response (AV-SFT) / prompt (AR-SFT). When
--keep-debug-metadata was on, detokenized_text_truncated is carried through.
This inverts the wrap to rebuild the 2-col cache parquets that
prompt_cache.load_explanation_cache reads — so a future run with the same
tokenizer + corpus can --cache-from them and skip all API calls.

  AV-SFT:  response = "<explanation>\\n{api_explanation}\\n</explanation>"
          inverted by schema.extract_explanation (the canonical parser).
  AR-SFT:  prompt = critic_template.format(explanation=api_explanation)
          inverted by splitting the template on {explanation} → (prefix, suffix)
          and slicing prompt[len(prefix):-len(suffix)].
"""

import argparse
from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.storage import Storage
from nla.schema import extract_explanation

# Must match stage3_build._DEFAULT_CRITIC_TEMPLATE. Overridable via CLI for
# datasets built with a custom template (sidecar records it at
# prompt_templates.critic).
_DEFAULT_CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"

_OUT_SCHEMA = pa.schema([
    ("detokenized_text_truncated", pa.string()),
    ("api_explanation", pa.string()),
])


def _unwrap_ar(prompt: str, prefix: str, suffix: str) -> str:
    assert prompt.startswith(prefix), (
        f"AR-SFT prompt does not start with {prefix!r}. "
        f"Got: {prompt[:80]!r}... — critic-template mismatch, pass --critic-template"
    )
    assert prompt.endswith(suffix), (
        f"AR-SFT prompt does not end with {suffix!r}. "
        f"Got tail: ...{prompt[-40:]!r} — critic-template mismatch"
    )
    return prompt[len(prefix):-len(suffix)]


def _recover(
    storage: Storage,
    in_path: str,
    out_path: str,
    wrapped_col: str,
    unwrap: Callable[[str], str | None],
) -> tuple[int, str]:
    t = pq.read_table(
        storage.open_read(in_path),
        columns=["detokenized_text_truncated", wrapped_col],
    )
    detok = t.column("detokenized_text_truncated").to_pylist()
    wrapped = t.column(wrapped_col).to_pylist()

    expls: list[str] = []
    for i, w in enumerate(wrapped):
        e = unwrap(w)
        assert e is not None and e, (
            f"row {i}: unwrap produced {e!r} from {wrapped_col}={w[:200]!r}"
        )
        expls.append(e)

    assert len(detok) == len(expls), "unreachable: column length mismatch"
    out = pa.table(
        {"detokenized_text_truncated": detok, "api_explanation": expls},
        schema=_OUT_SCHEMA,
    )
    storage.ensure_parent(out_path)
    pq.write_table(out, storage.open_write(out_path))
    return out.num_rows, expls[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--av-sft-shuf", required=True, help="stage3 av_sft output (must have detokenized_text_truncated)")
    p.add_argument("--ar-sft-shuf", required=True, help="stage3 ar_sft output (must have detokenized_text_truncated)")
    p.add_argument("--output-dir", required=True, help="where to write {ao,ar}_sl_explained.parquet")
    p.add_argument("--critic-template", default=_DEFAULT_CRITIC_TEMPLATE,
                   help="critic template stage3 used — check sidecar prompt_templates.critic if non-default")
    add_storage_args(p)
    args = p.parse_args()

    assert "{explanation}" in args.critic_template, (
        f"--critic-template must contain '{{explanation}}' placeholder, got: {args.critic_template!r}"
    )
    prefix, suffix = args.critic_template.split("{explanation}")

    storage = make_storage(args)
    out_dir = args.output_dir.rstrip("/")

    ao_out = f"{out_dir}/av_sft_explained.parquet"
    n_ao, sample_ao = _recover(storage, args.av_sft_shuf, ao_out, "response", extract_explanation)
    print(f"av_sft: {n_ao} rows → {ao_out}")
    print(f"  sample api_explanation[0][:200]: {sample_ao[:200]!r}")

    ar_out = f"{out_dir}/ar_sft_explained.parquet"
    n_ar, sample_ar = _recover(
        storage, args.ar_sft_shuf, ar_out, "prompt",
        lambda pr: _unwrap_ar(pr, prefix, suffix),
    )
    print(f"ar_sft: {n_ar} rows → {ar_out}")
    print(f"  sample api_explanation[0][:200]: {sample_ar[:200]!r}")


if __name__ == "__main__":
    main()
