"""Joint NLA RL training per paper §"NLA training":

Each step:
  (i)   Sample a batch of h_l from the RL/AV-SFT parquet. For each h, sample
        G summaries from AV at T=1 (with grad-tracked log-probs).
  (ii)  Update AR by gradient descent on regression loss
        ‖normalize(h, sqrt_d) − normalize(AR_θ(z), sqrt_d)‖²
        using the sampled (discrete, detached) z.
  (iii) Update AV by REINFORCE/GRPO:
        reward r(h,z) = −log ‖normalize(h) − normalize(AR_θ(z))‖²  (frozen AR)
        baseline = mean over the G-group sharing the same h
        advantage = r − baseline
        loss_AV = −(advantage · log p_φ(z|h)).mean() + β · KL(AV ∥ AV_init)
  AR and AV updates decoupled.

Loads AV and AR from the SFT save dirs produced by train_actor_sft.py /
train_critic_sft.py. Re-saves updated weights to new dirs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.models import NLACriticModel
from nla.schema import (
    ACTIVATION_COLUMN,
    INJECT_PLACEHOLDER,
    EXPLANATION_OPEN,
    EXPLANATION_CLOSE,
    extract_explanation,
    normalize_activation,
)


def _resolve_scale(raw, d_model: int) -> float | None:
    if raw is None or raw == "raw" or raw == "none":
        return None
    if raw == "sqrt_d_model":
        return math.sqrt(d_model)
    return float(raw)


class RLDataset(Dataset):
    """Reads AV-SFT or RL parquet — wants only (prompt_msgs, activation_vector)."""

    def __init__(self, parquet_path: str):
        table = pq.read_table(parquet_path)
        self.prompts = table.column("prompt").to_pylist()
        vecs = table.column(ACTIVATION_COLUMN).combine_chunks()
        flat = vecs.values.to_numpy(zero_copy_only=False).astype(np.float32)
        n = len(vecs)
        self.d_model = flat.size // n
        self.vectors = torch.from_numpy(flat.reshape(n, self.d_model))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "vector": self.vectors[idx]}


def _collate(batch):
    return {
        "prompts": [b["prompt"] for b in batch],
        "vectors": torch.stack([b["vector"] for b in batch], dim=0),
    }


def _load_av(base_model: str, av_dir: Path, trainable: bool, device: str):
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, attn_implementation="sdpa")
    av = PeftModel.from_pretrained(base, str(av_dir), is_trainable=trainable)
    if trainable:
        for p in av.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    else:
        for p in av.parameters():
            p.requires_grad_(False)
        av.eval()
    return av.to(device)


def _load_ar(base_model: str, ar_dir: Path, layer_index: int, trainable: bool, device: str):
    ar = NLACriticModel.from_pretrained(
        base_model, nla_num_layers=layer_index, torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    ar.backbone = PeftModel.from_pretrained(ar.backbone, str(ar_dir / "adapter"), is_trainable=trainable)
    vh_state = torch.load(ar_dir / "value_head.pt", map_location="cpu", weights_only=False)
    ar.value_head.load_state_dict(vh_state)
    if trainable:
        for p in ar.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    else:
        for p in ar.parameters():
            p.requires_grad_(False)
        ar.eval()
    return ar.to(device)


@torch.no_grad()
def _ar_score(ar, ar_tokenizer, critic_template: str, av_texts: list[str], gold_h: torch.Tensor, mse_scale: float, device: str, max_len: int = 512):
    """Run AR on AV-generated explanations, return ĥ at last token + per-sample MSE."""
    prompts = []
    for raw in av_texts:
        expl = extract_explanation(raw) or raw
        prompts.append(critic_template.format(explanation=expl))
    enc = ar_tokenizer(prompts, padding=True, truncation=True, max_length=max_len, return_tensors="pt", add_special_tokens=False).to(device)
    last_pos = enc["attention_mask"].sum(-1) - 1
    out = ar(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    idx = last_pos.view(-1, 1, 1).expand(-1, 1, out.values.size(-1))
    h_hat = out.values.gather(1, idx).squeeze(1).float()
    pred_n = normalize_activation(h_hat, mse_scale)
    gold_n = normalize_activation(gold_h, mse_scale)
    per_sample_mse = (pred_n - gold_n).pow(2).mean(dim=-1)
    return h_hat, per_sample_mse


def _av_sample_batched(av, tokenizer, prompt_msgs: list, vectors: torch.Tensor, inj_char: str, inj_id: int, left_id: int, right_id: int, inj_scale: float, max_new_tokens: int, temperature: float, device: str):
    """Batched version of _av_sample — all BG samples sampled in parallel.

    Left-padding + explicit position_ids so RoPE positions skip the pad. KV cache
    grows across steps. Per-sample EOS terminates that sample's logp record but
    keeps it in the batch (junk tokens) until all done or max_new reached.
    """
    BG = len(prompt_msgs)
    embed_layer = av.get_input_embeddings()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    all_ids: list[list[int]] = []
    for msgs in prompt_msgs:
        msgs_real = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in msgs]
        ids = tokenizer.apply_chat_template(msgs_real, tokenize=True, add_generation_prompt=True)
        all_ids.append(ids)
    prompt_lens = [len(x) for x in all_ids]
    L_max = max(prompt_lens)

    input_ids = torch.full((BG, L_max), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((BG, L_max), dtype=torch.long, device=device)
    pos_ids = torch.zeros((BG, L_max), dtype=torch.long, device=device)
    for i, ids in enumerate(all_ids):
        L = len(ids)
        input_ids[i, L_max - L:] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, L_max - L:] = 1
        pos_ids[i, L_max - L:] = torch.arange(L, dtype=torch.long, device=device)

    embeds = embed_layer(input_ids)
    v = vectors.to(device).float()
    v = normalize_activation(v, inj_scale)
    embeds = inject_at_marked_positions(input_ids, embeds, v, inj_id, left_id, right_id)

    gen_ids: list[list[int]] = [[] for _ in range(BG)]
    per_step_logps: list[list[torch.Tensor]] = [[] for _ in range(BG)]
    finished = [False] * BG
    past_kv = None
    cur_embeds = embeds
    cur_attn = attn
    cur_pos = pos_ids
    prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        out = av(inputs_embeds=cur_embeds, attention_mask=cur_attn, position_ids=cur_pos, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :].float() / max(temperature, 1e-6)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        for i in range(BG):
            if finished[i]:
                continue
            tok = int(next_ids[i].item())
            per_step_logps[i].append(log_probs[i, tok])
            gen_ids[i].append(tok)
            if tok == eos_id:
                finished[i] = True
        if all(finished):
            break
        cur_embeds = embed_layer(next_ids.unsqueeze(-1))
        cur_attn = torch.cat([cur_attn, torch.ones(BG, 1, dtype=torch.long, device=device)], dim=1)
        cur_pos = (prompt_lens_t + step).unsqueeze(-1)

    texts = [tokenizer.decode(gen_ids[i], skip_special_tokens=True) for i in range(BG)]
    logps_list: list[torch.Tensor] = []
    for i in range(BG):
        if per_step_logps[i]:
            logps_list.append(torch.stack(per_step_logps[i]).sum())
        else:
            logps_list.append(torch.zeros((), device=device))
    return texts, torch.stack(logps_list), [len(t) for t in texts]


def _av_sample(av, tokenizer, prompt_msgs: list, vectors: torch.Tensor, inj_char: str, inj_id: int, left_id: int, right_id: int, inj_scale: float, max_new_tokens: int, temperature: float, device: str):
    """Sample one explanation per prompt using AV; return (texts, sum_logp_per_sample).

    sum_logp keeps gradient to AV params via the recorded log_softmax at each
    sampled position.
    """
    embed_layer = av.get_input_embeddings()
    texts: list[str] = []
    logps: list[torch.Tensor] = []  # each is scalar tensor
    for i, msgs in enumerate(prompt_msgs):
        msgs_real = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in msgs]
        ids = tokenizer.apply_chat_template(msgs_real, tokenize=True, add_generation_prompt=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        embeds = embed_layer(input_ids)
        v = vectors[i : i + 1].to(device).float()
        v = normalize_activation(v, inj_scale)
        embeds = inject_at_marked_positions(input_ids, embeds, v, inj_id, left_id, right_id)

        # Manual autoregressive sample with grad-tracked logp
        eos_id = tokenizer.eos_token_id
        gen_ids: list[int] = []
        per_token_logps: list[torch.Tensor] = []
        past_kv = None
        cur_embeds = embeds
        cur_attn = attn
        finished = False
        for _ in range(max_new_tokens):
            out = av(inputs_embeds=cur_embeds, attention_mask=cur_attn, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :].float() / max(temperature, 1e-6)
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            tok_id = int(next_id.item())
            per_token_logps.append(log_probs[0, tok_id])
            gen_ids.append(tok_id)
            if tok_id == eos_id:
                finished = True
                break
            next_embed = embed_layer(next_id.unsqueeze(-1))
            cur_embeds = next_embed
            cur_attn = torch.cat([cur_attn, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        texts.append(text)
        if per_token_logps:
            logps.append(torch.stack(per_token_logps).sum())
        else:
            logps.append(torch.zeros((), device=device))
    return texts, torch.stack(logps), [len(t) for t in texts]


@torch.no_grad()
def _av_init_logp_batched(av_init, tokenizer, prompt_msgs: list, vectors: torch.Tensor, inj_char: str, inj_id: int, left_id: int, right_id: int, inj_scale: float, sampled_texts: list[str], device: str):
    """Batched teacher-force log p_init(z|h) for all BG sampled texts at once."""
    BG = len(prompt_msgs)
    embed_layer = av_init.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prompt_id_lists: list[list[int]] = []
    resp_id_lists: list[list[int]] = []
    for msgs, text in zip(prompt_msgs, sampled_texts):
        msgs_real = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in msgs]
        p = tokenizer.apply_chat_template(msgs_real, tokenize=True, add_generation_prompt=True)
        r = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
        prompt_id_lists.append(p)
        resp_id_lists.append(r)

    full_lens = [len(p) + len(r) for p, r in zip(prompt_id_lists, resp_id_lists)]
    L_max = max(full_lens) if full_lens else 1
    if L_max < 1:
        return torch.zeros(BG, device=device)

    input_ids = torch.full((BG, L_max), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((BG, L_max), dtype=torch.long, device=device)
    pos_ids = torch.zeros((BG, L_max), dtype=torch.long, device=device)
    for i, (p, r) in enumerate(zip(prompt_id_lists, resp_id_lists)):
        L = len(p) + len(r)
        if L == 0:
            continue
        input_ids[i, L_max - L:] = torch.tensor(p + r, dtype=torch.long, device=device)
        attn[i, L_max - L:] = 1
        pos_ids[i, L_max - L:] = torch.arange(L, dtype=torch.long, device=device)

    embeds = embed_layer(input_ids)
    v = vectors.to(device).float()
    v = normalize_activation(v, inj_scale)
    embeds = inject_at_marked_positions(input_ids, embeds, v, inj_id, left_id, right_id)
    lm_out = av_init(inputs_embeds=embeds, attention_mask=attn, position_ids=pos_ids)
    logits_all = lm_out.logits.float()

    sum_logps: list[float] = []
    for i, (p, r) in enumerate(zip(prompt_id_lists, resp_id_lists)):
        if not r:
            sum_logps.append(0.0)
            continue
        L_r = len(r)
        # Tokens (with left-pad of size L_max - L): prompt occupies [L_max - L_p - L_r ... L_max - L_r),
        # response occupies [L_max - L_r ... L_max). Logits at position t predict token t+1, so
        # positions [L_max - L_r - 1 ... L_max - 2] predict response tokens 0..L_r-1.
        start = L_max - L_r - 1
        pred_logits = logits_all[i, start:start + L_r, :]
        log_probs = F.log_softmax(pred_logits, dim=-1)
        targets = torch.tensor(r, dtype=torch.long, device=device)
        sum_logps.append(log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item())
    return torch.tensor(sum_logps, device=device)


@torch.no_grad()
def _av_init_logp(av_init, tokenizer, prompt_msgs: list, vectors: torch.Tensor, inj_char: str, inj_id: int, left_id: int, right_id: int, inj_scale: float, sampled_texts: list[str], device: str):
    """Teacher-forced log p_{AV_init}(z|h) for the SAMPLED texts. No grad."""
    embed_layer = av_init.get_input_embeddings()
    out: list[float] = []
    for i, msgs in enumerate(prompt_msgs):
        msgs_real = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in msgs]
        prompt_ids = tokenizer.apply_chat_template(msgs_real, tokenize=True, add_generation_prompt=True)
        resp_ids = tokenizer(sampled_texts[i], add_special_tokens=False)["input_ids"]
        if not resp_ids:
            out.append(0.0)
            continue
        full = prompt_ids + resp_ids
        input_ids = torch.tensor([full], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        embeds = embed_layer(input_ids)
        v = vectors[i : i + 1].to(device).float()
        v = normalize_activation(v, inj_scale)
        embeds = inject_at_marked_positions(input_ids, embeds, v, inj_id, left_id, right_id)
        lm_out = av_init(inputs_embeds=embeds, attention_mask=attn)
        # logits at position t predicts token t+1. We want tokens [len(prompt) ... len(full)-1]
        L_p = len(prompt_ids)
        target_ids = torch.tensor(resp_ids, dtype=torch.long, device=device)
        # logits over positions L_p-1 .. L_p+L_r-2 predict target tokens 0..L_r-1
        pred_logits = lm_out.logits[0, L_p - 1 : L_p - 1 + len(resp_ids), :].float()
        log_probs = F.log_softmax(pred_logits, dim=-1)
        sum_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).sum().item()
        out.append(sum_lp)
    return torch.tensor(out, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-dir", required=True, help="warm-started AV save dir")
    ap.add_argument("--ar-dir", required=True, help="warm-started AR save dir")
    ap.add_argument("--rl-parquet", required=True, help="parquet with prompt + activation_vector (e.g. av_sft_shuf.parquet)")
    ap.add_argument("--save-av", required=True)
    ap.add_argument("--save-ar", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2, help="examples per batch (each yields G samples)")
    ap.add_argument("--group-size", type=int, default=4, help="G — REINFORCE samples per h")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr-av", type=float, default=1e-5)
    ap.add_argument("--lr-ar", type=float, default=5e-5)
    ap.add_argument("--beta-kl", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--reward", choices=["mse", "contrastive", "mix"], default="mse",
                    help="mse: paper baseline -log MSE; contrastive: InfoNCE across prompts (needs B>=2); mix: 0.5*mse+0.5*contrastive")
    ap.add_argument("--contrastive-tau", type=float, default=0.1,
                    help="temperature for InfoNCE softmax over neg-MSE similarities")
    ap.add_argument("--batched-sample", action="store_true",
                    help="batch AV sampling and AV_init teacher-forcing across BG (expected ~3-5x speedup)")
    args = ap.parse_args()
    if args.reward in ("contrastive", "mix") and args.batch_size < 2:
        raise SystemExit(f"reward={args.reward} requires --batch-size >= 2 (got {args.batch_size}); no negatives otherwise")

    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load AV (trainable + frozen copy for KL) and AR
    av_meta = yaml.safe_load((Path(args.av_dir) / "nla_meta.yaml").read_text())
    ar_meta = yaml.safe_load((Path(args.ar_dir) / "nla_meta.yaml").read_text())
    base_model = av_meta["base_model"]
    layer_index = int(ar_meta["critic"]["num_hidden_layers"])
    inj_char = av_meta["tokens"]["injection_char"]
    inj_id = int(av_meta["tokens"]["injection_token_id"])
    left_id = int(av_meta["tokens"]["injection_left_neighbor_id"])
    right_id = int(av_meta["tokens"]["injection_right_neighbor_id"])
    critic_template = ar_meta["prompt_templates"].get("critic") or "Summary of the following text: <text>{explanation}</text> <summary>"

    print(f"[joint-rl] base={base_model}  layer={layer_index}  inj='{inj_char}'({inj_id})")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[joint-rl] loading AV (trainable)...")
    av = _load_av(base_model, Path(args.av_dir), trainable=True, device=device)
    print("[joint-rl] loading AV_init (frozen copy)...")
    av_init = _load_av(base_model, Path(args.av_dir), trainable=False, device=device)
    print("[joint-rl] loading AR...")
    ar = _load_ar(base_model, Path(args.ar_dir), layer_index, trainable=True, device=device)
    if args.grad_checkpoint:
        try:
            av.gradient_checkpointing_enable()
            av.enable_input_require_grads()
        except Exception as e:
            print(f"[joint-rl] av grad ckpt failed: {e}")
        try:
            ar.gradient_checkpointing_enable()
        except Exception as e:
            print(f"[joint-rl] ar grad ckpt failed: {e}")
        print("[joint-rl] gradient checkpointing enabled")

    d_model = av.config.hidden_size
    inj_scale = _resolve_scale(av_meta["extraction"].get("injection_scale", "sqrt_d_model"), d_model)
    mse_scale = _resolve_scale(ar_meta["extraction"].get("mse_scale", "sqrt_d_model"), d_model)
    print(f"[joint-rl] d_model={d_model}  inj_scale={inj_scale}  mse_scale={mse_scale}")

    dataset = RLDataset(args.rl_parquet)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=_collate)
    print(f"[joint-rl] RL dataset: {len(dataset)} examples (batch={args.batch_size}, G={args.group_size}, effective={args.batch_size*args.group_size})")

    av_params = [p for p in av.parameters() if p.requires_grad]
    ar_params = [p for p in ar.parameters() if p.requires_grad]
    optim_av = torch.optim.AdamW(av_params, lr=args.lr_av, weight_decay=0.0)
    optim_ar = torch.optim.AdamW(ar_params, lr=args.lr_ar, weight_decay=0.0)
    print(f"[joint-rl] AV trainable: {sum(p.numel() for p in av_params):,}  AR trainable: {sum(p.numel() for p in ar_params):,}")

    step = 0
    G = args.group_size
    for batch in loader:
        if step >= args.steps:
            break
        prompts_b = batch["prompts"]
        vecs_b = batch["vectors"].to(device)
        B = len(prompts_b)

        # Expand to (B*G) by repeating
        prompts_rep = []
        for _ in range(G):
            prompts_rep.extend(prompts_b)
        vecs_rep = vecs_b.repeat(G, 1)  # NOTE: not interleave so groups are contiguous-per-G

        # (i) sample G summaries per h
        _sample_fn = _av_sample_batched if args.batched_sample else _av_sample
        texts, sum_logp, gen_lens = _sample_fn(
            av, tokenizer, prompts_rep, vecs_rep,
            inj_char, inj_id, left_id, right_id, inj_scale,
            args.max_new_tokens, args.temperature, device,
        )

        # (ii)+(iii) AR forward (no grad to AV) → per-sample MSE; reward
        h_hat, per_sample_mse = _ar_score(ar, tokenizer, critic_template, texts, vecs_rep, mse_scale, device)
        # AR loss: same MSE term, but here AR's pred goes through normalize_activation and we use the raw mse for grad
        # rerun for grad — _ar_score used torch.no_grad
        ar_prompts = [critic_template.format(explanation=(extract_explanation(t) or t)) for t in texts]
        enc = tokenizer(ar_prompts, padding=True, truncation=True, max_length=512, return_tensors="pt", add_special_tokens=False).to(device)
        last_pos = enc["attention_mask"].sum(-1) - 1
        out = ar(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        idx = last_pos.view(-1, 1, 1).expand(-1, 1, out.values.size(-1))
        pred = out.values.gather(1, idx).squeeze(1).float()
        ar_loss = F.mse_loss(
            normalize_activation(pred, mse_scale),
            normalize_activation(vecs_rep.float(), mse_scale),
        )

        # AV REINFORCE: reward = -log mse (treat AR fixed → use detached mse)
        with torch.no_grad():
            mse_reward = -torch.log(per_sample_mse.detach() + 1e-6)  # (B*G,)
            if args.reward == "mse":
                reward = mse_reward
                contrast_acc = torch.tensor(float("nan"), device=device)
            else:
                # InfoNCE: AR(z_i) must score better against gold h_i than against h_j (j≠i in batch).
                # h_hat is in vecs_rep order ((G groups of B): index g*B+b → gold v_b).
                h_hat_n = normalize_activation(h_hat.detach(), mse_scale)             # (B*G, d)
                vecs_n = normalize_activation(vecs_b.float().detach(), mse_scale)      # (B, d)
                diff = h_hat_n.unsqueeze(1) - vecs_n.unsqueeze(0)                      # (B*G, B, d)
                mse_ij = diff.pow(2).mean(-1)                                          # (B*G, B)
                sim_ij = -mse_ij / max(args.contrastive_tau, 1e-6)
                pos_idx = torch.arange(B * G, device=device) % B                       # tile order
                log_softmax = F.log_softmax(sim_ij, dim=-1)
                contrastive_reward = log_softmax.gather(1, pos_idx.unsqueeze(-1)).squeeze(-1)
                contrast_acc = (sim_ij.argmax(-1) == pos_idx).float().mean()
                if args.reward == "contrastive":
                    reward = contrastive_reward
                else:  # mix
                    reward = 0.5 * mse_reward + 0.5 * contrastive_reward
            reward_g = reward.view(G, B).t()  # (B, G)
            baseline = reward_g.mean(dim=1, keepdim=True)
            advantage = (reward_g - baseline).reshape(-1)  # (B*G,) — same order as B*G sampling
            # Actually our flatten ordering was (G groups of B): index = g*B + b
            # advantage above was reshape from (B,G), which is row-major (b*G+g).
            # The sum_logp tensor was built in order (g*B+b). Re-permute advantage to match.
            adv_b_then_g = reward_g.reshape(-1) - baseline.expand(B, G).reshape(-1)  # (B,G) flatten in (b*G+g) order
            # Need it in (g*B+b) order to match sum_logp.
            adv_correct = adv_b_then_g.view(B, G).t().reshape(-1)
            advantage = adv_correct

        # KL via MC: log p_phi(z|h) is sum_logp; need log p_init(z|h) too
        _init_fn = _av_init_logp_batched if args.batched_sample else _av_init_logp
        init_logp = _init_fn(av_init, tokenizer, prompts_rep, vecs_rep, inj_char, inj_id, left_id, right_id, inj_scale, texts, device)
        kl_per_seq = sum_logp - init_logp.detach()

        av_rl_loss = -(advantage * sum_logp).mean()
        av_kl_loss = args.beta_kl * kl_per_seq.mean()
        av_loss = av_rl_loss + av_kl_loss

        (ar_loss + av_loss).backward()
        ar_gnorm = torch.nn.utils.clip_grad_norm_(ar_params, 1.0)
        av_gnorm = torch.nn.utils.clip_grad_norm_(av_params, 1.0)
        optim_ar.step()
        optim_av.step()
        optim_ar.zero_grad()
        optim_av.zero_grad()
        step += 1

        if step % args.log_every == 0:
            print(
                f"[joint-rl] step {step}/{args.steps}  ar_mse={ar_loss.item():.4f}  "
                f"reward(mean)={reward.mean().item():.3f}  reward(std)={reward.std().item():.3f}  "
                f"adv(std)={advantage.std().item():.3f}  av_rl={av_rl_loss.item():.3f}  kl={kl_per_seq.mean().item():.3f}  "
                f"gen_len={sum(gen_lens)/len(gen_lens):.1f}  g_ar={ar_gnorm:.2f}  g_av={av_gnorm:.2f}  "
                f"c_acc={contrast_acc.item():.2f}"
            )

    # Save
    save_av = Path(args.save_av)
    save_ar = Path(args.save_ar)
    save_av.mkdir(parents=True, exist_ok=True)
    save_ar.mkdir(parents=True, exist_ok=True)
    av.save_pretrained(str(save_av))
    (save_av / "nla_meta.yaml").write_text(yaml.safe_dump(av_meta, allow_unicode=True))
    ar.backbone.save_pretrained(str(save_ar / "adapter"))
    torch.save({k: v.cpu() for k, v in ar.value_head.state_dict().items()}, save_ar / "value_head.pt")
    (save_ar / "nla_meta.yaml").write_text(yaml.safe_dump(ar_meta, allow_unicode=True))
    print(f"[joint-rl] saved AV → {save_av}  AR → {save_ar}")


if __name__ == "__main__":
    main()
