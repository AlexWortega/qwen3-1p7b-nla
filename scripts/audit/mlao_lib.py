"""MLAO reference library — extracted verbatim from niclas-luick multi-layer-activation-oracles
experiments/multi_layer_AO_demo.ipynb (the self-contained Library Code cell). Used to run
THEIR activation oracle as the reference. Do not edit; their code.
"""
# @title Library Code (run this cell - click to expand)
# @markdown This cell contains all the necessary library code inlined.
# @markdown You don't need to read it - just run it once.

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import contextlib
from typing import Any, Callable, Mapping, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
import torch._dynamo as dynamo
from peft import LoraConfig, PeftModel
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================
# LAYER CONFIGURATION
# ============================================================

LAYER_COUNTS = {
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-4B": 36,
    "Qwen/Qwen3-8B": 36,
    "Qwen/Qwen3-32B": 64,
    "google/gemma-2-9b-it": 42,
    "google/gemma-3-1b-it": 26,
    "meta-llama/Llama-3.2-1B-Instruct": 16,
    "meta-llama/Llama-3.3-70B-Instruct": 80,
}

def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent to a layer number."""
    max_layers = LAYER_COUNTS[model_name]
    return int(max_layers * (layer_percent / 100))


# ============================================================
# ACTIVATION UTILITIES
# ============================================================

class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""
    pass

def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule for HF transformers"""
    model_name = model.config._name_or_path
    if use_lora:
        if "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.base_model.model.model.layers[layer]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")
    if "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")

def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    if min_offset is not None:
        assert max_offset is not None
        assert max_offset < min_offset
        assert min_offset < 0
        assert max_offset < 0
    else:
        assert max_offset is None

    activations_BLD_by_layer = {}
    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}
    max_layer = max(submodules.keys())

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]
        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs
        if min_offset is not None:
            activations_BLD_by_layer[layer] = activations_BLD_by_layer[layer][:, max_offset:min_offset, :]
        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    handles = []
    for layer, submodule in submodules.items():
        handles.append(submodule.register_forward_hook(gather_target_act_hook))

    try:
        with torch.no_grad():
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        for handle in handles:
            handle.remove()

    return activations_BLD_by_layer

# ============================================================
# STEERING HOOKS
# ============================================================

@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """HF hook for activation steering."""
    assert len(vectors) == len(positions)
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors]

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, d_model_actual = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={B}")

        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        for b in range(B):
            pos_b = positions[b]
            pos_b = torch.tensor(pos_b, dtype=torch.long, device=device)
            assert pos_b.min() >= 0
            assert pos_b.max() < L
            orig_KD = resid_BLD[b, pos_b, :]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)
            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)
            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn

# ============================================================
# DATASET UTILITIES
# ============================================================

SPECIAL_TOKEN = " ?"

def get_introspection_prefix(layers: list[int], num_positions: int) -> str:
    prefix = ''
    for layer in layers:
        prefix += f"Layer: {layer}\n"
        prefix += SPECIAL_TOKEN * num_positions
        prefix += " \n"
    return prefix

class FeatureResult(BaseModel):
    feature_idx: int
    api_response: str
    prompt: str
    meta_info: Mapping[str, Any] = {}

class TrainingDataPoint(BaseModel):
    """Training data point with tensors.
    If steering_vectors is None, then we calculate the steering vectors on the fly
    from the context_input_ids and context_positions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datapoint_type: str
    input_ids: list[int]
    labels: list[int]  # Can contain -100 for ignored tokens
    layers: list[int]
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    context_input_ids: list[int] | None
    context_positions: list[int] | None
    ds_label: str | None  # label from the dataset
    meta_info: Mapping[str, Any] = {}

    @model_validator(mode="after")
    def _check_context_alignment(cls, values):
        sv = values.steering_vectors
        if sv is not None:
            if len(values.positions) != sv.shape[0]:
                raise ValueError("positions and steering_vectors must have the same length")
        else:
            if values.context_positions is None or values.context_input_ids is None:
                raise ValueError("context_* must be provided when steering_vectors is None")
            #if len(values.positions) != len(values.context_positions):
            #    raise ValueError("positions and context_positions must have the same length")
        return values

class BatchData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]

@dataclass
class OracleResults:
    oracle_lora_path: str | None
    target_lora_path: str | None
    target_prompt: str
    act_key: str
    oracle_prompt: str
    ground_truth: str
    num_tokens: int
    token_responses: list[Optional[str]]
    full_sequence_responses: list[str]
    segment_responses: list[str]
    target_input_ids: list[int]

def construct_batch(
    training_data: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> BatchData:
    max_length = max(len(dp.input_ids) for dp in training_data)
    batch_tokens, batch_labels, batch_attn_masks = [], [], []
    batch_positions, batch_steering_vectors, batch_feature_indices = [], [], []

    for data_point in training_data:
        padding_length = max_length - len(data_point.input_ids)
        padding_tokens = [tokenizer.pad_token_id] * padding_length
        padded_input_ids = padding_tokens + data_point.input_ids
        padded_labels = [-100] * padding_length + data_point.labels

        input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
        labels = torch.tensor(padded_labels, dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool).to(device)
        attn_mask[:padding_length] = False

        batch_tokens.append(input_ids)
        batch_labels.append(labels)
        batch_attn_masks.append(attn_mask)

        padded_positions = [p + padding_length for p in data_point.positions]
        steering_vectors = data_point.steering_vectors.to(device) if data_point.steering_vectors is not None else None

        batch_positions.append(padded_positions)
        batch_steering_vectors.append(steering_vectors)
        batch_feature_indices.append(data_point.feature_idx)

    return BatchData(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        steering_vectors=batch_steering_vectors,
        positions=batch_positions,
        feature_indices=batch_feature_indices,
    )

def get_prompt_tokens_only(training_data_point: TrainingDataPoint) -> TrainingDataPoint:
    prompt_tokens, prompt_labels = [], []
    response_token_seen = False
    for i in range(len(training_data_point.input_ids)):
        if training_data_point.labels[i] != -100:
            response_token_seen = True
            continue
        else:
            if response_token_seen:
                raise ValueError("Response token seen before prompt tokens")
            prompt_tokens.append(training_data_point.input_ids[i])
            prompt_labels.append(training_data_point.labels[i])
    new = training_data_point.model_copy()
    new.input_ids = prompt_tokens
    new.labels = prompt_labels
    return new

def materialize_missing_steering_vectors(
    batch_points: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    model: PeftModel,
) -> list[TrainingDataPoint]:
    """
    Materialization of missing steering vectors for a heterogenous batch
    where different items can request activations from different layers.

    Steps:
      1) Find items with steering_vectors=None.
      2) Build a left-padded batch from their context_input_ids.
      3) Register hooks for all unique requested layers and run exactly one forward pass.
      4) For each item, take activations at its requested layer and its context_positions,
         then write back a [num_positions, D] tensor to dp.steering_vectors. Returns a new batch.

    No-op if every item already has steering_vectors.
    """
    # Select datapoints that need generation
    to_fill: list[tuple[int, TrainingDataPoint]] = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

    # Validate context fields
    for _, dp in to_fill:
        if dp.context_input_ids is None or dp.context_positions is None:
            raise ValueError(
                "Datapoint has steering_vectors=None but is missing context_input_ids or context_positions"
            )

    # Build the input batch (left padding to match your construct_batch convention)
    pad_id = tokenizer.pad_token_id
    contexts: list[list[int]] = [list(dp.context_input_ids) for _, dp in to_fill]
    positions_per_item: list[list[int]] = [list(dp.context_positions) for _, dp in to_fill]
    max_len = max(len(c) for c in contexts)

    input_ids_tensors: list[torch.Tensor] = []
    attn_masks_tensors: list[torch.Tensor] = []
    left_offsets: list[int] = []

    device = next(model.parameters()).device

    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_tensors.append(torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device))
        # For HF, bool masks are fine; your construct_batch uses bool too
        attn_masks_tensors.append(torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device))
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    # Prepare hooks for all unique requested layers
    # layers_needed = sorted({dp.layer for _, dp in to_fill})
    all_needed_layers = set()
    for _, dp in to_fill:
        all_needed_layers.update(dp.layers)
    layers_needed = sorted(list(all_needed_layers))
    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers_needed}

    # Run a single pass with dropout off, then restore the previous train/eval mode
    was_training = model.training
    model.eval()
    with model.disable_adapter():
        # [layer] -> [B, L, D], where B == len(to_fill)
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
    if was_training:
        model.train()

    # Build the new list, copying only items we change
    new_batch: list[TrainingDataPoint] = list(batch_points)  # references by default
    for b in range(len(to_fill)):
        idx, dp = to_fill[b]
        idxs = [p + left_offsets[b] for p in positions_per_item[b]]

        # Bounds check for safety
        first_layer_acts = acts_by_layer[dp.layers[0]]  # [B, L, D] on GPU
        L = first_layer_acts.shape[1]
        if any(i < 0 or i >= L for i in idxs):
            raise IndexError(f"Activation index out of range for item {b}: {idxs} with L={L}")

        item_vectors_list = []
        for layer in dp.layers:
            acts_BLD = acts_by_layer[layer] # [B, L, D] on GPU
            layer_vectors = acts_BLD[b, idxs, :].detach().contiguous()
            item_vectors_list.append(layer_vectors)

        vectors = torch.stack(item_vectors_list, dim=0)
        vectors = vectors.view(-1, vectors.shape[-1])

        assert len(vectors.shape) == 2, f"Expected 2D tensor, got vectors.shape={vectors.shape}"

        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = vectors

        new_batch[idx] = dp_new

    return new_batch

def find_pattern_in_tokens(
    token_ids: list[int], special_token_str: str, num_positions: int, layers: list[int], tokenizer: AutoTokenizer
) -> list[int]:
    start_idx = 0
    end_idx = len(token_ids)
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, f"Expected single token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    positions = []

    for i in range(start_idx, end_idx):
        if len(positions) == num_positions * len(layers):
            break
        if token_ids[i] == special_token_id:
            positions.append(i)

    assert len(positions) == num_positions * len(layers), f"Expected {num_positions} positions, got {len(positions)}"

    final_pos = positions[-1] + 1
    final_tokens = token_ids[final_pos : final_pos + 2]
    final_str = tokenizer.decode(final_tokens, skip_special_tokens=False)
    assert "\n" in final_str, f"Expected newline in {final_str}"

    return positions

def create_training_datapoint(
    datapoint_type: str,
    prompt: str,
    target_response: str,
    layers: list[int],
    num_positions: int,
    tokenizer: AutoTokenizer,
    acts_BD: torch.Tensor | None,
    feature_idx: int,
    context_input_ids: list[int] | None = None,
    context_positions: list[int] | None = None,
    ds_label: str | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> TrainingDataPoint:
    if meta_info is None:
        meta_info = {}
    prefix = get_introspection_prefix(layers, num_positions)
    assert prefix not in prompt, f"Prefix {prefix} found in prompt {prompt}"
    prompt = prefix + prompt
    #print(prompt)
    input_messages = [{"role": "user", "content": prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(input_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    full_messages = input_messages + [{"role": "assistant", "content": target_response}]

    full_prompt_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(full_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    assistant_start_idx = len(input_prompt_ids)

    labels = full_prompt_ids.copy()
    for i in range(assistant_start_idx):
        labels[i] = -100

    positions = find_pattern_in_tokens(full_prompt_ids, SPECIAL_TOKEN, num_positions, layers, tokenizer)

    if acts_BD is None:
        assert context_input_ids is not None and context_positions is not None, (
            "acts_BD is None but context_input_ids and context_positions are None"
        )
    else:
        assert len(acts_BD.shape) == 2, f"Expected 2D tensor, got {acts_BD.shape}"
        acts_BD = acts_BD.cpu().clone().detach()
        assert len(positions) == acts_BD.shape[0], f"Expected {acts_BD.shape[0]} positions, got {len(positions)}"

    training_data_point = TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        layers=layers,
        steering_vectors=acts_BD,
        positions=positions,
        feature_idx=feature_idx,
        target_output=target_response,
        datapoint_type=datapoint_type,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        ds_label=ds_label,
        meta_info=meta_info,
    )

    return training_data_point

# ============================================================
# EVALUATION
# ============================================================

@dynamo.disable
@torch.no_grad()
def eval_features_batch(
    eval_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    steering_coefficient: float,
    generation_kwargs: dict,
) -> list[FeatureResult]:
    hook_fn = get_hf_activation_steering_hook(
        vectors=eval_batch.steering_vectors,
        positions=eval_batch.positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {"input_ids": eval_batch.input_ids, "attention_mask": eval_batch.attention_mask}
    decoded_prompts = tokenizer.batch_decode(eval_batch.input_ids, skip_special_tokens=False)
    feature_results = []

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    generated_tokens = output_ids[:, eval_batch.input_ids.shape[1]:]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for i in range(len(eval_batch.feature_indices)):
        feature_results.append(FeatureResult(
            feature_idx=eval_batch.feature_indices[i],
            api_response=decoded_output[i],
            prompt=decoded_prompts[i],
        ))

    return feature_results

def _run_evaluation(
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    lora_path: str | None,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict,
) -> list[FeatureResult]:
    if lora_path is not None:
        adapter_name = lora_path
        if adapter_name not in model.peft_config:
            model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(adapter_name)

    with torch.no_grad():
        all_feature_results = []
        for i in tqdm(range(0, len(eval_data), eval_batch_size), desc="Evaluating model"):
            e_batch = eval_data[i:i + eval_batch_size]
            e_batch = [get_prompt_tokens_only(dp) for dp in e_batch]
            e_batch = materialize_missing_steering_vectors(e_batch, tokenizer, model)
            e_batch = construct_batch(e_batch, tokenizer, device)
            feature_results = eval_features_batch(
                eval_batch=e_batch, model=model, submodule=submodule, tokenizer=tokenizer,
                device=device, dtype=dtype, steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
            )
            all_feature_results.extend(feature_results)

    for feature_result, eval_data_point in zip(all_feature_results, eval_data, strict=True):
        feature_result.meta_info = eval_data_point.meta_info
    return all_feature_results

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def encode_formatted_prompts(
    tokenizer: AutoTokenizer,
    formatted_prompts: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize already-formatted prompt strings."""
    return tokenizer(formatted_prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(device)

def sanitize_lora_name(lora_path: str) -> str:
    return lora_path.replace(".", "_")

def load_lora_adapter(model: AutoModelForCausalLM, lora_path: str) -> str:
    sanitized_lora_name = sanitize_lora_name(lora_path)
    if sanitized_lora_name not in model.peft_config:
        print(f"Loading LoRA: {lora_path}")
        model.load_adapter(lora_path, adapter_name=sanitized_lora_name, is_trainable=False, low_cpu_mem_usage=True)
    return sanitized_lora_name

def _create_oracle_inputs(
    acts_BLD_by_layer_dict: dict[int, torch.Tensor],
    target_input_ids: list[int],
    oracle_prompt: str,
    act_layers: list[int],
    prompt_layers: list[int],
    tokenizer: AutoTokenizer,
    segment_start_idx: int,
    segment_end_idx: int | None,
    token_start_idx: int,
    token_end_idx: int | None,
    oracle_input_types: list[str],
    segment_repeats: int,
    full_seq_repeats: int,
    batch_idx: int = 0,
    left_pad: int = 0,
    base_meta: dict[str, Any] | None = None,
) -> list[TrainingDataPoint]:
    training_data = []
    num_tokens = len(target_input_ids)

    # Token-level probes
    if "tokens" in oracle_input_types:
        token_start = token_start_idx
        token_end = num_tokens if token_end_idx is None else token_end_idx

        if token_start < 0:
            raise ValueError(f"token_start_idx ({token_start}) must be >= 0")
        if token_end > num_tokens:
            raise ValueError(f"token_end_idx ({token_end}) exceeds sequence length ({num_tokens}). Use None for 'to the end'.")
        if token_start >= token_end:
            raise ValueError(f"token_start_idx ({token_start}) must be < token_end_idx ({token_end})")

        for i in range(token_start, token_end):
            target_positions_rel = [i]
            target_positions_abs = [left_pad + i]

            #acts_BD = acts_BLD_by_layer_dict[act_layer][batch_idx, target_positions_abs]
            acts_NBD = [acts_BLD_by_layer_dict[act_layer][batch_idx, target_positions_abs] for act_layer in act_layers]
            acts_BD = torch.stack(acts_NBD, dim=0)
            acts_BD = acts_BD.view(-1, acts_BD.shape[-1])

            meta = {"dp_kind": "tokens", "token_index": i}
            if base_meta:
                meta.update(base_meta)
            dp = create_training_datapoint(
                datapoint_type="N/A", prompt=oracle_prompt, target_response="N/A",
                layers=prompt_layers, num_positions=len(target_positions_rel), tokenizer=tokenizer,
                acts_BD=acts_BD, feature_idx=-1, context_input_ids=target_input_ids,
                context_positions=target_positions_rel, ds_label="N/A", meta_info=meta,
            )
            training_data.append(dp)

    # Segment probes
    if "segment" in oracle_input_types:
        segment_start = segment_start_idx
        segment_end = num_tokens if segment_end_idx is None else segment_end_idx

        if segment_start < 0:
            raise ValueError(f"segment_start_idx ({segment_start}) must be >= 0")
        if segment_end > num_tokens:
            raise ValueError(f"segment_end_idx ({segment_end}) exceeds sequence length ({num_tokens}). Use None for 'to the end'.")
        if segment_start >= segment_end:
            raise ValueError(f"segment_start_idx ({segment_start}) must be < segment_end_idx ({segment_end})")

        for _ in range(segment_repeats):
            target_positions_rel = list(range(segment_start, segment_end))
            target_positions_abs = [left_pad + p for p in target_positions_rel]
            #acts_BD = acts_BLD_by_layer_dict[act_layer][batch_idx, target_positions_abs]
            acts_NBD = [acts_BLD_by_layer_dict[act_layer][batch_idx, target_positions_abs] for act_layer in act_layers]
            acts_BD = torch.stack(acts_NBD, dim=0)
            acts_BD = acts_BD.view(-1, acts_BD.shape[-1])
            meta = {"dp_kind": "segment"}
            if base_meta:
                meta.update(base_meta)
            dp = create_training_datapoint(
                datapoint_type="N/A", prompt=oracle_prompt, target_response="N/A",
                layers=prompt_layers, num_positions=len(target_positions_rel), tokenizer=tokenizer,
                acts_BD=acts_BD, feature_idx=-1, context_input_ids=target_input_ids,
                context_positions=target_positions_rel, ds_label="N/A", meta_info=meta,
            )
            training_data.append(dp)

    # Full sequence probes
    if "full_seq" in oracle_input_types:
        for _ in range(full_seq_repeats):
            target_positions_rel = list(range(len(target_input_ids)))
            target_positions_abs = [left_pad + p for p in target_positions_rel]
            #acts_BD = acts_BLD_by_layer_dict[act_layer][batch_idx, target_positions_abs]
            acts_NBD = [acts_BLD_by_layer_dict[act_layer][batch_idx, target_positions_abs] for act_layer in act_layers]
            acts_BD = torch.stack(acts_NBD, dim=0)
            acts_BD = acts_BD.view(-1, acts_BD.shape[-1])
            meta = {"dp_kind": "full_seq"}
            if base_meta:
                meta.update(base_meta)
            dp = create_training_datapoint(
                datapoint_type="N/A", prompt=oracle_prompt, target_response="N/A",
                layers=prompt_layers, num_positions=len(target_positions_rel), tokenizer=tokenizer,
                acts_BD=acts_BD, feature_idx=-1, context_input_ids=target_input_ids,
                context_positions=target_positions_rel, ds_label="N/A", meta_info=meta,
            )
            training_data.append(dp)

    return training_data

def _collect_target_activations(
    model: AutoModelForCausalLM,
    inputs_BL: dict[str, torch.Tensor],
    act_layers: list[int],
    target_lora_path: str | None,
) -> dict[int, torch.Tensor]:
    """Collect activations from the target model (with LoRA if specified)."""
    model.enable_adapters()
    if target_lora_path is not None:
        model.set_adapter(target_lora_path)
    submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
    return collect_activations_multiple_layers(model=model, submodules=submodules, inputs_BL=inputs_BL, min_offset=None, max_offset=None)

# ============================================================
# MAIN ORACLE FUNCTION
# ============================================================

def run_oracle(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    # Target model params
    target_prompt: str,  # Already formatted with apply_chat_template
    target_lora_path: str | None,
    layer_percents: list[int],
    # Oracle model params
    oracle_prompt: str,
    oracle_lora_path: str | None,
    # Segment/token selection
    segment_start_idx: int = 0,
    segment_end_idx: int | None = None,
    token_start_idx: int = 0,
    token_end_idx: int | None = 1,
    # Oracle input types
    oracle_input_types: list[str] | None = None,
    # Generation params
    generation_kwargs: dict[str, Any] | None = None,
    # Optional
    ground_truth: str = "",
    segment_repeats: int = 1,
    full_seq_repeats: int = 1,
    eval_batch_size: int = 32,
    injection_layer: int = 1,
    steering_coefficient: float = 1.0,
) -> OracleResults:
    """
    Run the activation oracle on a single target prompt.

    Args:
        model: The model (with LoRA adapters loaded)
        tokenizer: The tokenizer
        device: torch device
        target_prompt: Already formatted target prompt string
        target_lora_path: Path to target LoRA (or None for base model)
        oracle_prompt: Question to ask the oracle about the activations
        oracle_lora_path: Path to oracle LoRA
        segment_start_idx: Start index for segment activations
        segment_end_idx: End index for segment activations (None = end of sequence)
        token_start_idx: Start index for token-level activations
        token_end_idx: End index for token-level activations (None = end of sequence)
        oracle_input_types: List of input types: "tokens", "segment", "full_seq"
        generation_kwargs: Generation parameters for the oracle
        ground_truth: Optional ground truth for comparison
        segment_repeats: Number of times to repeat segment probes
        full_seq_repeats: Number of times to repeat full sequence probes
        eval_batch_size: Batch size for evaluation
        layer_percent: Which layer to extract activations from (as percentage)
        injection_layer: Which layer to inject activations into
        steering_coefficient: Coefficient for activation steering

    Returns:
        OracleResults with token_responses, segment_responses, and full_sequence_responses
    """
    if oracle_input_types is None:
        oracle_input_types = ["segment", "full_seq", "tokens"]
    if generation_kwargs is None:
        generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 50}

    dtype = torch.bfloat16
    model_name = model.config._name_or_path

    # Calculate layers from percentages
    act_layers = [layer_percent_to_layer(model_name, p) for p in layer_percents]

    injection_submodule = get_hf_submodule(model, injection_layer)

    # Tokenize target prompt
    inputs_BL = encode_formatted_prompts(tokenizer=tokenizer, formatted_prompts=[target_prompt], device=device)

    # Collect activations from target model
    acts_by_layer = _collect_target_activations(
        model=model, inputs_BL=inputs_BL, act_layers=act_layers, target_lora_path=target_lora_path
    )

    # Get target input ids
    seq_len = int(inputs_BL["input_ids"].shape[1])
    attn = inputs_BL["attention_mask"][0]
    real_len = int(attn.sum().item())
    left_pad = seq_len - real_len
    target_input_ids = inputs_BL["input_ids"][0, left_pad:].tolist()

    # Create oracle inputs
    base_meta = {
        "target_lora_path": target_lora_path,
        "target_prompt": target_prompt,
        "oracle_prompt": oracle_prompt,
        "ground_truth": ground_truth,
        "combo_index": 0,
        "act_key": "lora",
        "num_tokens": len(target_input_ids),
        "target_index_within_batch": 0,
    }

    oracle_inputs = _create_oracle_inputs(
        acts_BLD_by_layer_dict=acts_by_layer,
        target_input_ids=target_input_ids,
        oracle_prompt=oracle_prompt,
        act_layers=act_layers,
        prompt_layers=act_layers,
        tokenizer=tokenizer,
        segment_start_idx=segment_start_idx,
        segment_end_idx=segment_end_idx,
        token_start_idx=token_start_idx,
        token_end_idx=token_end_idx,
        oracle_input_types=oracle_input_types,
        segment_repeats=segment_repeats,
        full_seq_repeats=full_seq_repeats,
        batch_idx=0,
        left_pad=left_pad,
        base_meta=base_meta,
    )

    # Run oracle evaluation
    if oracle_lora_path is not None:
        model.set_adapter(oracle_lora_path)

    responses = _run_evaluation(
        eval_data=oracle_inputs,
        model=model,
        tokenizer=tokenizer,
        submodule=injection_submodule,
        device=device,
        dtype=dtype,
        lora_path=oracle_lora_path,
        eval_batch_size=eval_batch_size,
        steering_coefficient=steering_coefficient,
        generation_kwargs=generation_kwargs,
    )

    # Aggregate results
    token_responses = [None] * len(target_input_ids)
    segment_responses = []
    full_seq_responses = []

    for r in responses:
        meta = r.meta_info
        dp_kind = meta["dp_kind"]
        if dp_kind == "tokens":
            token_responses[int(meta["token_index"])] = r.api_response
        elif dp_kind == "segment":
            segment_responses.append(r.api_response)
        elif dp_kind == "full_seq":
            full_seq_responses.append(r.api_response)

    return OracleResults(
        oracle_lora_path=oracle_lora_path,
        target_lora_path=target_lora_path,
        target_prompt=target_prompt,
        act_key="lora",
        oracle_prompt=oracle_prompt,
        ground_truth=ground_truth,
        num_tokens=len(target_input_ids),
        token_responses=token_responses,
        full_sequence_responses=full_seq_responses,
        segment_responses=segment_responses,
        target_input_ids=target_input_ids,
    )


def visualize_token_selection(
    input_text: str,
    segment_start: int = 0,
    segment_end: int | None = None,
):
    """
    Visualize which tokens are selected for activation collection.
    input_text should be already formatted (e.g., via apply_chat_template).
    """
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"][0].tolist()
    num_tokens = len(input_ids)

    start_pos = segment_start
    end_pos = num_tokens if segment_end is None else segment_end

    print("Token selection visualization:")
    print("-" * 60)
    for i, token_id in enumerate(input_ids):
        token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
        if start_pos <= i < end_pos:
            print(f"  [{i:3d}] >>> {token_str}")
        else:
            print(f"  [{i:3d}]     {token_str}")
    print("-" * 60)
    print(f"Selected positions: {start_pos} to {end_pos} ({end_pos - start_pos} tokens)")


print("Library code loaded successfully!")