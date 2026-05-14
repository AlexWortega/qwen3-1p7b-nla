"""Load NLA runtime config from a sidecar YAML (dataset or model).

The sidecar pins tokenizer-dependent constants (injection token ID, PM token ID,
neighbor IDs, prompt templates) that were fixed at dataset/model generation time.
Loading them here and asserting against the live tokenizer catches drift before
training starts, not after output goes to Chinese.

Schema: docs/design.md §2.
Shared types/helpers: nla/schema.py
"""

import datetime
import math
from dataclasses import dataclass, replace
from typing import Any

import yaml

from nla.schema import SCALE_SQRT_D, compute_canonical_neighbors, resolve_target_scale, sidecar_path_for
from nla.storage import fetch_sidecar_to_local_cache, is_remote


def resolve_sidecar_source(
    *,
    explicit: str | None,
    hf_checkpoint: str | None,
    prompt_data: str | None,
) -> str:
    """Pick the sidecar to load, with consistent precedence.

    Order:
      1. explicit (--nla-sidecar-source)
      2. hf_checkpoint IF {ckpt}/nla_meta.yaml exists (model sidecar —
         authoritative for what THIS model was trained with)
      3. prompt_data (dataset sidecar — fallback for fresh base-model runs)

    Callers unpack args themselves — this function never sees the miles args
    namespace, so a miles field rename can't silently fall through.

    Train-side, rollout-side (nla_generate), and reward-side (nla_rm) MUST all
    call this — train/infer scale mismatch is silent garbage if they diverge.
    """
    if explicit is not None:
        return explicit
    if hf_checkpoint and sidecar_path_for(hf_checkpoint).exists():
        return hf_checkpoint
    assert prompt_data is not None, (
        "no sidecar source available — set --nla-sidecar-source, or ensure "
        "hf_checkpoint has nla_meta.yaml, or set --prompt-data"
    )
    return prompt_data


def load_nla_config_from_args(args, tokenizer) -> tuple["NLAConfig", str]:
    """Resolve sidecar source from args, load config, apply CLI --nla-injection-scale.

    Single entry point for train_actor, nla_generate, sft_critic, data_source.
    Ensures train and infer resolve injection_scale identically — mismatch is
    silent garbage (actor sees OOD magnitude).

    Returns the config plus the resolved sidecar source (for logging/asserts).
    """
    # NLA sends input_embeds (~6-12MB/req) with a raw activation injected at the
    # marker token. Radix cache keys on token IDs → would cache-hit across
    # DIFFERENT activations sharing the same prompt template → silent garbage.
    # This also gates the router-leak asserts in rollout.py (cache_aware policy
    # + memory history_backend both store request bodies → OOM with input_embeds).
    # Only relevant when SGLang rollouts actually run — SFT (--debug-train-only)
    # never starts an SGLang server, so the flag is moot there.
    if not getattr(args, "debug_train_only", False):
        assert getattr(args, "sglang_disable_radix_cache", False), (
            "NLA requires --sglang-disable-radix-cache. Radix cache keys on token "
            "IDs; NLA injects raw activations at the marker token, so different "
            "activations with the same template would cache-hit → silent wrong output."
        )
    sidecar_source = resolve_sidecar_source(
        explicit=args.nla_sidecar_source,
        hf_checkpoint=args.hf_checkpoint,
        prompt_data=args.prompt_data,
    )
    # NLADataSource auto-fetches remote parquet+sidecar and mutates
    # args.prompt_data → local path — but that's in the RolloutManager's
    # Ray actor. Training actors have an independent args copy with the raw
    # URL. Fetch just the sidecar here (small YAML, not the full parquet).
    if is_remote(sidecar_source):
        assert args.nla_storage_cls is not None, (
            f"sidecar source {sidecar_source!r} is remote but --nla-storage-cls "
            f"not set. Training actors (separate Ray process from NLADataSource) "
            f"need it to fetch the sidecar."
        )
        sidecar_source = fetch_sidecar_to_local_cache(
            sidecar_source,
            storage_cls=args.nla_storage_cls,
            cache_dir=args.nla_fetch_cache_dir,
        )
    cfg = load_nla_config(sidecar_source, tokenizer)
    cli_inj_scale = args.nla_injection_scale
    if cli_inj_scale is not None:
        cfg = replace(cfg, injection_scale=resolve_target_scale(cli_inj_scale, cfg.d_model))
    return cfg, sidecar_source


@dataclass(frozen=True)
class NLAConfig:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    actor_prompt_template: str
    critic_prompt_template: str | None
    critic_num_layers: int | None

    # Critic extraction: position = last token of the prompt. The prompt
    # template ends with a fixed suffix (e.g. "</text> <summary>") so the
    # last real token is always the extraction point — no scanning, just
    # index -1 after padding is stripped.
    #
    # critic_suffix_ids is for a ONE-TIME sanity check at dataset load:
    # assert prompt_tokens[-len(suffix):] == suffix. It's tokenize(suffix)[1:]
    # because the first token merges with the explanation's last char
    # (e.g. "detail." + "</" → ".</", so first suffix token is unstable).
    critic_suffix_ids: list[int] | None = None

    # Normalization controls. Both are resolved from sidecar's raw values
    # (None | "sqrt_d_model" | float) into concrete float | None at load time.
    #
    # injection_scale: what L2-norm the vector is scaled to before injection
    #   into the actor's embedding. None = inject raw (preserve magnitude —
    #   layer-depth signal?). sqrt(d_model) = match token-embedding scale.
    #
    # mse_scale: what L2-norm BOTH pred and gold are scaled to before MSE.
    #   None = MSE on raw magnitudes (critic must learn scale too).
    #   sqrt(d_model) = direction-only MSE (critic learns direction only).
    #
    # These are INDEPENDENT — you can inject raw but still train the critic
    # with direction-only MSE, or normalize injection but have the critic
    # predict raw magnitudes.
    injection_scale: float | None = None
    mse_scale: float | None = None

    @property
    def sqrt_d(self) -> float:
        return math.sqrt(self.d_model)


def load_nla_config(sidecar_source: str, tokenizer) -> NLAConfig:
    """Load sidecar and verify against live tokenizer.

    `sidecar_source` may be a checkpoint dir (reads {dir}/nla_meta.yaml) or a
    parquet path (reads {path}.nla_meta.yaml). Slice syntax stripped.

    Asserts:
      - injection char tokenizes to expected ID (tokenizer version drift)
      - injection char is not UNK
      - canonical actor prompt produces exactly one injection token
      - neighbor IDs at inj_pos ± 1 match the sidecar
      - PM token (if present) tokenizes to expected ID
    """
    meta_path = sidecar_path_for(sidecar_source)
    meta = yaml.safe_load(meta_path.read_text())

    kind = meta["kind"]
    assert kind in ("nla_model", "nla_dataset"), f"unknown sidecar kind: {kind!r}"

    if kind == "nla_dataset":
        extraction = meta["extraction"]
        d_model = extraction["d_model"]
    else:
        d_model = meta["d_model"]
        extraction = meta.get("extraction", {})

    # Sidecar may write: null / "sqrt_d_model" / float. Resolve to concrete
    # float | None here so downstream never branches on string sentinels.
    #
    # injection_scale: NO DEFAULT from absent key — it's a training hyperparameter
    # that must be chosen explicitly. Absent → None → train_actor.py asserts.
    # Sidecar value (if present) is a default for RESUMING from a trained checkpoint.
    #
    # mse_scale: defaults to sqrt_d_model. It's loss-numerical-stability,
    # not a tuning knob — the default is almost always right.
    injection_scale = resolve_target_scale(extraction.get("injection_scale"), d_model)
    mse_scale = resolve_target_scale(extraction.get("mse_scale", SCALE_SQRT_D), d_model)

    t = meta["tokens"]
    templates = meta.get("prompt_templates", {})
    critic_meta = meta.get("critic") or {}
    # schema v1 wrote "num_hidden_layers" (clashed with HF's config.json key).
    # v2 writes "extraction_layer_index". Read both for back-compat.
    critic_k = critic_meta.get("extraction_layer_index", critic_meta.get("num_hidden_layers"))

    cfg = NLAConfig(
        d_model=d_model,
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        injection_left_neighbor_id=t["injection_left_neighbor_id"],
        injection_right_neighbor_id=t["injection_right_neighbor_id"],
        actor_prompt_template=templates.get("av") or templates["actor"],
        critic_prompt_template=templates.get("ar") or templates.get("critic"),
        critic_num_layers=critic_k,
        critic_suffix_ids=t.get("critic_suffix_ids"),
        injection_scale=injection_scale,
        mse_scale=mse_scale,
    )

    # encode(), not convert_tokens_to_ids(): byte-level BPE tokenizers (Qwen,
    # GPT-2) store the byte-string representation as the token key, not the
    # unicode char. convert_tokens_to_ids('㈎') → None; encode('㈎') → [149705].
    live_inj_ids = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
    assert live_inj_ids == [cfg.injection_token_id], (
        f"tokenizer drift: {cfg.injection_char!r} → {live_inj_ids}, "
        f"sidecar says [{cfg.injection_token_id}]. "
        f"Multi-token means the char split — wrong tokenizer or vocab changed."
    )
    assert live_inj_ids[0] != tokenizer.unk_token_id, (
        f"{cfg.injection_char!r} maps to UNK — pick a different marker"
    )

    live_left, live_right = compute_canonical_neighbors(
        tokenizer, cfg.actor_prompt_template, cfg.injection_char, cfg.injection_token_id
    )
    assert live_left == cfg.injection_left_neighbor_id, (
        f"left neighbor drift: tokenizer gives {live_left}, "
        f"sidecar says {cfg.injection_left_neighbor_id}"
    )
    assert live_right == cfg.injection_right_neighbor_id, (
        f"right neighbor drift: tokenizer gives {live_right}, "
        f"sidecar says {cfg.injection_right_neighbor_id}"
    )

    return cfg


def verify_critic_suffix(tokens: list[int], suffix_ids: list[int], context: str = "") -> None:
    """Assert the tokenized critic prompt ends with the expected suffix.

    One-time check — do at dataset load (or first few samples), not per-forward.
    The suffix is tokenize(template_suffix)[1:] — first token dropped because
    it BPE-merges with the explanation's last char and is unstable. The tail
    (e.g. the IDs for 'text> <summary>') is stable.
    """
    n = len(suffix_ids)
    actual = tokens[-n:]
    assert actual == suffix_ids, (
        f"critic prompt suffix mismatch{' (' + context + ')' if context else ''}: "
        f"expected tokens[-{n}:] == {suffix_ids}, got {actual}. "
        f"Template drift or tokenizer version changed the suffix encoding."
    )


def write_model_sidecar(checkpoint_dir: str, cfg: NLAConfig, *, role: str, stage: str,
                        base_checkpoint: str, trained_on: list[str],
                        parent_checkpoints: list[str], created_by: str,
                        training_args: dict[str, Any] | None = None) -> None:
    """Write {checkpoint_dir}/nla_meta.yaml for an NLA-trained model.

    Called from NLAFSDPActor.save_model. Mirrors the dataset sidecar writer in
    nla/datagen/sidecar.py but for the model-checkpoint schema (arch doc §2).
    """
    meta: dict[str, Any] = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": role,
        "stage": stage,
        "base_checkpoint": base_checkpoint,
        "d_model": cfg.d_model,
        "extraction": {
            # Write resolved floats (or null) — not the "sqrt_d_model" sentinel.
            # The value is what THIS model was trained with; downstream shouldn't
            # re-resolve against a potentially different d_model.
            "injection_scale": cfg.injection_scale,
            "mse_scale": cfg.mse_scale,
        },
        "tokens": {
            "injection_char": cfg.injection_char,
            "injection_token_id": cfg.injection_token_id,
            "injection_left_neighbor_id": cfg.injection_left_neighbor_id,
            "injection_right_neighbor_id": cfg.injection_right_neighbor_id,
            "critic_suffix_ids": cfg.critic_suffix_ids,
        },
        "prompt_templates": {
            "actor": cfg.actor_prompt_template,
            "critic": cfg.critic_prompt_template,
        },
        "trained_on": trained_on,
        "parent_checkpoints": parent_checkpoints,
        "created_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "created_by": created_by,
    }
    if cfg.critic_num_layers is not None:
        meta["critic"] = {"extraction_layer_index": cfg.critic_num_layers}
    if training_args:
        meta["training"] = training_args
    out_path = sidecar_path_for(checkpoint_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(meta, sort_keys=False))
