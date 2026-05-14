"""NLACriticModel: truncated transformer + vector value head.

Architecture per docs/design.md §4:
  - First K transformer layers only (K = extraction layer, set via config override)
  - No final layernorm — raw residual stream goes to head
  - Linear(d_model, d_model) head, no bias
  - Forward returns .values at every position; extract at PM-token position for MSE

Layer truncation: set config.num_hidden_layers BEFORE from_pretrained so the
weight loader only reads K layers. Do NOT slice nn.ModuleList post-hoc — breaks
FSDP sharding assumptions.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from nla.arch_adapters import resolve_text_config, resolve_text_model


# Embedding weight key suffixes across architectures.
# Llama/Qwen/Mistral/Gemma: model.embed_tokens.weight
# GPT-2: transformer.wte.weight
# Falcon: transformer.word_embeddings.weight
_EMBED_KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")


@dataclass
class NLACriticOutput:
    values: torch.Tensor  # [B, T, d_model] — value-head output at every position
    backbone_last_hidden: torch.Tensor  # [B, T, d_model] — pre-value-head (for norm tracking)


def _truncate_config_layers(config, num_layers: int) -> None:
    """Set num_hidden_layers AND truncate per-layer arrays to match.

    transformers >=4.50 validates len(layer_types) == num_hidden_layers at
    config init (configuration_utils.py:layer_type_validation). Qwen2/Llama3
    configs carry per-layer arrays that must be sliced consistently.
    """
    config.num_hidden_layers = num_layers
    for attr in ("layer_types", "sliding_window_pattern", "no_rope_layers"):
        v = getattr(config, attr, None)
        if isinstance(v, (list, tuple)) and len(v) > num_layers:
            setattr(config, attr, type(v)(v[:num_layers]))


def _inner_transformer(backbone: PreTrainedModel) -> nn.Module:
    """Get the inner transformer module (the part with .layers + .norm).

    Qwen/Llama/Mistral/Gemma (post-resolve_text_model → CausalLM wrapper): backbone.model
    GPT-2/Falcon: backbone.transformer
    """
    if hasattr(backbone, "model"):
        return backbone.model
    if hasattr(backbone, "transformer"):
        return backbone.transformer
    raise AssertionError(
        f"{type(backbone).__name__} has neither .model nor .transformer — "
        f"add the attribute name here if supporting a new arch"
    )


class NLACriticModel(PreTrainedModel):
    """Wraps an HF causal LM backbone with layer truncation + vector value head.

    Delegates everything structural (save/load/fsdp-wrapping) to the backbone's
    PreTrainedModel machinery. Only the forward path + head are NLA-specific.
    """

    def __init__(self, config, backbone: PreTrainedModel):
        super().__init__(config)
        self.backbone = backbone
        self.value_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # FSDP's apply_fsdp2 reads this to decide which modules to wrap.
        # Instance attr (not class attr) — two NLACriticModels with different
        # backbones in one process would clobber each other on class attr.
        self._no_split_modules = backbone._no_split_modules

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, nla_num_layers: int | None = None, **kwargs):
        """Load an NLACriticModel from an HF checkpoint.

        Normal case: checkpoint was produced by prepare_critic_checkpoint.py or
        a previous NLA training run → config.json already has the truncated
        num_hidden_layers. Just load.

        Bootstrapping case (prepare_critic_checkpoint.py only): pass
        nla_num_layers = the datagen extraction layer_index. Truncation keeps
        blocks 0..layer_index INCLUSIVE — we need the output OF block K, which
        means we need block K to exist, so num_hidden_layers = K+1.

        Indexing convention (matches datagen/extractors.py):
          - datagen `layer_index=K` hooks `model.model.layers[K]` →
            captures its output (= HF's `hidden_states[K+1]`, post-residual-add,
            the stream entering block K+1).
          - critic `last_hidden_state` (with final-LN → Identity) is the output
            of the LAST block.
          - So last block must be block K → num_hidden_layers = K+1.
        """
        # Miles' fsdp_utils/actor.py:96 passes trust_remote_code + attn_implementation
        # via kwargs. Default to True if caller doesn't specify.
        kwargs.setdefault("trust_remote_code", True)
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=kwargs["trust_remote_code"]
        )
        # For multimodal wrappers (Gemma-3), truncate the nested text config —
        # the wrapper config references it by identity so the truncation is
        # visible to from_pretrained. For plain text models, text_config IS
        # config (see arch_adapters).
        text_config = resolve_text_config(config)
        if nla_num_layers is not None:
            needed = nla_num_layers + 1
            assert needed <= text_config.num_hidden_layers, (
                f"nla_num_layers={nla_num_layers} needs blocks 0..{nla_num_layers} "
                f"inclusive (num_hidden_layers={needed}), but base model has "
                f"only {text_config.num_hidden_layers}."
            )
            _truncate_config_layers(text_config, needed)

        backbone = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs,
        )
        # Discard the vision tower / projection for multimodal wrappers — keep
        # only the text transformer. No-op for plain causal LMs.
        backbone = resolve_text_model(backbone)
        # Strip lm_head — critic never produces logits. Frees memory and
        # prevents FSDP from sharding a weight matrix we never use.
        if hasattr(backbone, "lm_head"):
            backbone.lm_head = nn.Identity()

        # Strip final layernorm — arch doc §4: raw residual-stream → value head.
        inner = _inner_transformer(backbone)
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, nn.Identity())
                break
        else:
            raise AssertionError(
                f"could not find final layernorm on {type(inner).__name__} — "
                f"add the attribute name to the list above"
            )

        model = cls(text_config, backbone)

        # If an HF-format critic checkpoint exists (from save_pretrained), load
        # the value head from it. Fresh init otherwise. Skip under
        # init_empty_weights (rank≠0 in fsdp_utils/actor.py:202 when
        # tie_word_embeddings=False): nn.Linear was registered on meta;
        # load_state_dict(cpu_tensors) would crash. Rank 0 loads,
        # _fsdp2_load_full_state_dict broadcasts.
        head_path = Path(pretrained_model_name_or_path) / "value_head.safetensors"
        if head_path.exists() and not model.value_head.weight.is_meta:
            model.value_head.load_state_dict(load_file(str(head_path)))

        # nn.Linear in __init__ is CPU/fp32; load_state_dict upcasts the bf16
        # file tensors into that. Backbone got device+dtype via kwargs (incl.
        # device_map="auto" → accelerate sharding), but value_head is outside
        # the safetensors index so accelerate doesn't see it. Align to the last
        # layer's placement so forward()'s `h` → value_head doesn't bounce
        # device/dtype. Skip on meta (FSDP rank≠0 — broadcast handles it).
        if not model.value_head.weight.is_meta:
            last = next(inner.layers[-1].parameters())
            model.value_head.to(device=last.device, dtype=last.dtype)

        return model

    def forward(self, input_ids=None, position_ids=None, attention_mask=None, **kwargs):
        # Call backbone (CausalLM, possibly PEFT-wrapped) with output_hidden_states=True
        # so we don't depend on `.model` attribute path through wrappers. Since
        # final LN was replaced with Identity in from_pretrained, hidden_states[-1]
        # is the raw residual stream after the last kept transformer block —
        # equivalent to the inner transformer's `last_hidden_state`.
        kwargs.pop("output_hidden_states", None)
        out = self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        h = out.hidden_states[-1]
        # value_head may be fp32 (master weights for AMP) while backbone runs
        # in fp16/bf16. Cast h to value_head's dtype to avoid mat1/mat2 dtype mismatch.
        head_dtype = self.value_head.weight.dtype
        h_in = h.to(head_dtype) if h.dtype != head_dtype else h
        return NLACriticOutput(values=self.value_head(h_in), backbone_last_hidden=h)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.state_dict()
        backbone_sd = {k.removeprefix("backbone."): v for k, v in state_dict.items() if k.startswith("backbone.")}
        head_sd = {k.removeprefix("value_head."): v for k, v in state_dict.items() if k.startswith("value_head.")}

        self.backbone.save_pretrained(save_directory, state_dict=backbone_sd, **kwargs)
        save_file(head_sd, str(Path(save_directory) / "value_head.safetensors"))
        # config.json written by backbone.save_pretrained includes the truncated num_hidden_layers

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()


# Path the actor dumps a fresh embedding to after each SGLang weight sync.
# nla_generate reloads from here so rollout embeddings track the trained
# model (not the stale initial HF checkpoint). Default goes under args.save
# (disk); set NLA_EMBED_DUMP_DIR=/dev/shm/nla to use tmpfs — the 1GB reload
# drops from ~1.8s to ~0.2s per rollout.
ROLLOUT_EMBED_DUMP = "_nla_rollout_embed.pt"


def embed_dump_path(save_dir: str | None) -> Path | None:
    """Resolve the embedding dump path. NLA_EMBED_DUMP_DIR overrides save_dir."""
    override = os.environ.get("NLA_EMBED_DUMP_DIR")
    base = override or save_dir
    if base is None:
        return None
    return Path(base) / ROLLOUT_EMBED_DUMP


def _find_embed_key(keys: list[str], where: str) -> str:
    matches = [k for k in keys if k.endswith(_EMBED_KEY_SUFFIXES)]
    assert len(matches) == 1, (
        f"expected exactly one input-embedding key in {where} "
        f"(suffixes {_EMBED_KEY_SUFFIXES}), got {matches!r}"
    )
    return matches[0]


def load_embedding_only(hf_checkpoint: str, dtype: torch.dtype = torch.float32) -> nn.Embedding:
    """Load ONLY the input embedding layer from an HF checkpoint's safetensors.

    Returns a plain `nn.Embedding` wrapping the weight tensor. This is the RAW
    lookup — if the model's embedding forward applies a scale (Gemma ×√d, T5),
    the CALLER must multiply. See `arch_adapters.resolve_embed_scale()` and
    how `nla_generate.py` uses `_EMBED_SCALE` explicitly.

    Avoids materializing the full model — just reads the one weight tensor
    via safe_open's lazy loading. Handles HF Hub names via snapshot_download.
    """
    root = Path(hf_checkpoint)
    if not root.exists():
        root = Path(snapshot_download(hf_checkpoint))

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        key = _find_embed_key(list(weight_map), str(index_path))
        shard = root / weight_map[key]
    else:
        shard = root / "model.safetensors"
        assert shard.exists(), (
            f"no model.safetensors or .index.json at {root!r}"
        )
        with safe_open(str(shard), framework="pt") as f:
            key = _find_embed_key(list(f.keys()), str(shard))

    with safe_open(str(shard), framework="pt") as f:
        weight = f.get_tensor(key).to(dtype)

    vocab, d_model = weight.shape
    embed = nn.Embedding(vocab, d_model, _weight=weight)
    embed.requires_grad_(False)
    embed.eval()
    return embed
