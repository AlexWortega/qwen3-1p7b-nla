"""Joint NLA RL for the universal multi-model stack.

Forks train_joint_rl_paper.py: same GRPO + KL anchor + mix reward loop, but
each step samples a tag from the training pool, projects h_M via the frozen
enc_M (d_shared = 2048), and decodes AR's prediction via dec_M for the per-M
FVE-based reward.

Trains: AV LoRA (continuing from av_multi_v1) and AR LoRA (continuing from
ar_multi_v1). enc_M / dec_M stay frozen (from artifacts/adapters_v1_pinv2).

Reward modes (same as single-model):
  - mse: -log MSE_M(ĥ_M, h_M)
  - contrastive: InfoNCE among same-tag sub-batch
  - mix: 0.5*mse + 0.5*contrastive  (default — avoids template-collapse)
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.data_multi import MultiModelActivationDataset
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.models import NLACriticModel
from nla.schema import (
    EXPLANATION_OPEN,
    EXPLANATION_CLOSE,
    extract_explanation,
    normalize_activation,
)


ACTOR_TEMPLATE = (
    "You are a meticulous AI researcher investigating activation vectors from "
    "{model_tag}, a small open-weight language model. Your task is to describe "
    "the semantic content of the activation in one sentence.\n\n"
    "We pass the vector inside <concept> tags. Reply with the description "
    "inside <explanation> tags.\n\n"
    "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
    "Please provide the description."
)
CRITIC_TEMPLATE = "Summary of the following text: <text>{z}</text> <summary>"


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
    base_critic = NLACriticModel.from_pretrained(base_model, nla_num_layers=layer_index,
                                                 torch_dtype=torch.float16, attn_implementation="sdpa")
    # value_head identity-init (PEFT save dropped it during AR SFT; we re-establish).
    with torch.no_grad():
        d = base_critic.value_head.weight.shape[0]
        base_critic.value_head.weight.copy_(torch.eye(d, dtype=base_critic.value_head.weight.dtype))
    # Try loading value_head.pt if it was saved alongside the AR adapter.
    vh_path = ar_dir / "value_head.pt"
    if vh_path.exists():
        vh_state = torch.load(vh_path, map_location="cpu", weights_only=False)
        base_critic.value_head.load_state_dict(vh_state)
        print(f"[joint-rl] loaded value_head.pt")
    ar = PeftModel.from_pretrained(base_critic, str(ar_dir / "ar"), is_trainable=trainable)
    if trainable:
        for p in ar.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    else:
        for p in ar.parameters():
            p.requires_grad_(False)
        ar.eval()
    return ar.to(device)


def build_av_prompt(model_tag: str, injection_char: str) -> str:
    return ACTOR_TEMPLATE.format(model_tag=model_tag, injection_char=injection_char)


def build_critic_prompt(z: str) -> str:
    return CRITIC_TEMPLATE.format(z=z.strip())


def av_sample_batched(av, tokenizer, prompt_text: str, inj_vecs: torch.Tensor,
                     inj_id: int, left_id: int, right_id: int,
                     max_new_tokens: int, temperature: float, device: str):
    """All BG samples share the same prompt (one tag per step). Auto-regressive
    sampling with grad-tracked logp on the sampled tokens.

    Returns: (list[str] generated texts, [BG] sum_logp with grad)
    """
    BG = inj_vecs.shape[0]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    p_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=True, add_generation_prompt=True,
    )
    L = len(p_ids)
    input_ids = torch.tensor([p_ids], dtype=torch.long, device=device).expand(BG, -1).contiguous()
    attn = torch.ones((BG, L), dtype=torch.long, device=device)
    pos_ids = torch.arange(L, dtype=torch.long, device=device).expand(BG, -1).contiguous()
    embed_layer = av.get_input_embeddings()
    embeds = embed_layer(input_ids)
    inj_vecs = inj_vecs.to(device, dtype=embeds.dtype)
    embeds = inject_at_marked_positions(input_ids, embeds, inj_vecs, inj_id, left_id, right_id)

    gen_ids = [[] for _ in range(BG)]
    per_step_logps = [[] for _ in range(BG)]
    finished = [False] * BG
    past_kv = None
    cur_embeds = embeds
    cur_attn = attn
    cur_pos = pos_ids
    for step in range(max_new_tokens):
        out = av(inputs_embeds=cur_embeds, attention_mask=cur_attn,
                 position_ids=cur_pos, past_key_values=past_kv, use_cache=True)
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
        cur_pos = torch.full((BG, 1), L + step, dtype=torch.long, device=device)

    texts = [tokenizer.decode(gen_ids[i], skip_special_tokens=True) for i in range(BG)]
    logps = []
    for i in range(BG):
        logps.append(torch.stack(per_step_logps[i]).sum() if per_step_logps[i]
                     else torch.zeros((), device=device))
    return texts, torch.stack(logps), [len(t) for t in texts]


@torch.no_grad()
def av_init_logp_batched(av_init, tokenizer, prompt_text: str, inj_vecs: torch.Tensor,
                         sampled_texts: list[str], inj_id: int, left_id: int, right_id: int,
                         device: str):
    """Teacher-force log p_init(z|h) for the sampled z's. No grad."""
    BG = inj_vecs.shape[0]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    p_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=True, add_generation_prompt=True,
    )
    resp_id_lists = []
    for text in sampled_texts:
        r = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
        resp_id_lists.append(r)
    full_lens = [len(p_ids) + len(r) for r in resp_id_lists]
    L_max = max(full_lens) if full_lens else 1
    if L_max < 1:
        return torch.zeros(BG, device=device)

    input_ids = torch.full((BG, L_max), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((BG, L_max), dtype=torch.long, device=device)
    pos_ids = torch.zeros((BG, L_max), dtype=torch.long, device=device)
    for i, r in enumerate(resp_id_lists):
        L = len(p_ids) + len(r)
        if L == 0:
            continue
        ids = p_ids + r
        input_ids[i, L_max - L:] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, L_max - L:] = 1
        pos_ids[i, L_max - L:] = torch.arange(L, dtype=torch.long, device=device)
    embed_layer = av_init.get_input_embeddings()
    embeds = embed_layer(input_ids)
    inj_vecs = inj_vecs.to(device, dtype=embeds.dtype)
    embeds = inject_at_marked_positions(input_ids, embeds, inj_vecs, inj_id, left_id, right_id)
    out = av_init(inputs_embeds=embeds, attention_mask=attn, position_ids=pos_ids)
    logits_all = out.logits.float()
    sum_logps = []
    for i, r in enumerate(resp_id_lists):
        if not r:
            sum_logps.append(0.0)
            continue
        L_r = len(r)
        start = L_max - L_r - 1
        pred_logits = logits_all[i, start:start + L_r, :]
        log_probs = F.log_softmax(pred_logits, dim=-1)
        targets = torch.tensor(r, dtype=torch.long, device=device)
        sum_logps.append(log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item())
    return torch.tensor(sum_logps, device=device)


def ar_forward(ar, ar_tokenizer, z_texts: list[str], device: str, d_shared: int, max_len: int = 512):
    """Run AR on batched z's, return [B, d_shared] last-token predictions (with grad).
    Right-padding so last_pos = attn.sum-1 works per row."""
    ar_tokenizer.padding_side = "right"
    prompts = [build_critic_prompt(z) for z in z_texts]
    enc = ar_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_len, add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    out = ar(input_ids=input_ids, attention_mask=attn)
    lengths = attn.sum(dim=1) - 1
    idx = lengths.view(-1, 1, 1).expand(-1, 1, d_shared)
    pred = out.values.gather(1, idx).squeeze(1).float()
    return pred


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--av-dir", required=True, help="v1 AV save dir (warm start)")
    ap.add_argument("--ar-dir", required=True, help="v1 AR save dir (warm start)")
    ap.add_argument("--adapters-dir", required=True, help="frozen adapters (with refit dec_M)")
    ap.add_argument("--anchor-tag", required=True)
    ap.add_argument("--exclude-tags", default="")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr-av", type=float, default=1e-5)
    ap.add_argument("--lr-ar", type=float, default=5e-5)
    ap.add_argument("--beta-kl", type=float, default=0.05)
    ap.add_argument("--reward", choices=["mse", "contrastive", "mix"], default="mix")
    ap.add_argument("--contrastive-tau", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--av-init-device", default=None,
                    help="If set (e.g. cuda:1), load the frozen AV_init on a separate GPU "
                         "to free memory on the main device. Useful when the trunk is large.")
    ap.add_argument("--ar-device", default=None,
                    help="If set (e.g. cuda:1), load AR (trainable, truncated trunk + LoRA + value_head) "
                         "on a separate GPU. Needed when AV trainable + AR trainable both don't fit on the same V100.")
    args = ap.parse_args()

    if args.reward in ("contrastive", "mix") and args.batch_size < 2:
        raise SystemExit(f"reward={args.reward} requires batch_size>=2")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    av_meta = yaml.safe_load((Path(args.av_dir) / "nla_meta.yaml").read_text())
    ar_meta = yaml.safe_load((Path(args.ar_dir) / "nla_meta.yaml").read_text())
    av_base = av_meta["av_base"]
    layer_index = int(ar_meta["layer_index"])
    inj_char = av_meta["tokens"]["injection_char"]
    inj_id = int(av_meta["tokens"]["injection_token_id"])
    left_id = int(av_meta["tokens"]["injection_left_neighbor_id"])
    right_id = int(av_meta["tokens"]["injection_right_neighbor_id"])
    d_shared = int(av_meta["d_shared"])

    print(f"[joint-rl] av_base={av_base}  layer={layer_index}  inj={inj_char!r}({inj_id})  d_shared={d_shared}")
    tokenizer = AutoTokenizer.from_pretrained(av_base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    av_init_device = args.av_init_device or device
    ar_device = args.ar_device or device
    print(f"[joint-rl] loading AV trainable on {device} ...")
    av = _load_av(av_base, Path(args.av_dir) / "av", trainable=True, device=device)
    print(f"[joint-rl] loading AV_init frozen on {av_init_device} ...")
    av_init = _load_av(av_base, Path(args.av_dir) / "av", trainable=False, device=av_init_device)
    print(f"[joint-rl] loading AR trainable on {ar_device} ...")
    ar = _load_ar(av_base, Path(args.ar_dir), layer_index, trainable=True, device=ar_device)

    adapters = ModelPoolAdapters.load(args.adapters_dir).to(device)
    for p in adapters.parameters():
        p.requires_grad_(False)
    excluded = {args.anchor_tag} | {t.strip() for t in args.exclude_tags.split(",") if t.strip()}
    training_tags = [t for t in adapters.tags if t not in excluded]
    print(f"[joint-rl] training tags={training_tags}  (excluded={sorted(excluded)})")

    dataset = MultiModelActivationDataset(args.pool_dir, restrict_tags=training_tags, dtype=torch.float32)
    has_z_per_tag: dict[str, list[int]] = {t: [] for t in training_tags}
    for tag in training_tags:
        for pid in range(dataset.n_passages):
            if dataset.passages[pid].get("z"):
                has_z_per_tag[tag].append(pid)
    print(f"[joint-rl] passages-with-z per tag: " +
          ", ".join(f"{t}={len(has_z_per_tag[t])}" for t in training_tags))

    inj_scale = math.sqrt(d_shared)

    av_params = [p for p in av.parameters() if p.requires_grad]
    ar_params = [p for p in ar.parameters() if p.requires_grad]
    optim_av = torch.optim.AdamW(av_params, lr=args.lr_av, weight_decay=0.0)
    optim_ar = torch.optim.AdamW(ar_params, lr=args.lr_ar, weight_decay=0.0)
    print(f"[joint-rl] AV trainable: {sum(p.numel() for p in av_params):,}  "
          f"AR trainable: {sum(p.numel() for p in ar_params):,}")

    G = args.group_size
    B = args.batch_size
    rng = random.Random(args.seed)
    for step_idx in range(args.steps):
        tag = rng.choice(training_tags)
        d_M = dataset.d_model(tag)
        mn_scale = math.sqrt(d_M)
        pids = rng.sample(has_z_per_tag[tag], B)
        h_M = torch.stack([dataset.h_cache[tag][p].to(device, dtype=torch.float32) for p in pids])  # [B, d_M]
        # Repeat per G group (contiguous-per-G).
        h_M_rep = h_M.repeat(G, 1)                          # [B*G, d_M]
        with torch.no_grad():
            inj_shared = adapters.encode(tag, h_M_rep)      # [B*G, d_shared]
            inj_shared = normalize_activation(inj_shared, inj_scale)

        prompt_text = build_av_prompt(tag, inj_char)

        # (i) AV samples G summaries per h.
        texts, sum_logp, gen_lens = av_sample_batched(
            av, tokenizer, prompt_text, inj_shared,
            inj_id, left_id, right_id,
            args.max_new_tokens, args.temperature, device,
        )

        # (ii) AR forward + per-M MSE. AR may live on a separate GPU; ar_pred_shared
        # carries its grad cross-device, then we cross over to `device` for the M-space loss
        # so adapters (kept on `device`) and the gold h_M_rep don't need duplication.
        z_payloads = [extract_explanation(t) or t for t in texts]
        ar_pred_shared = ar_forward(ar, tokenizer, z_payloads, ar_device, d_shared)        # [B*G, d_shared] grad on ar_device
        ar_pred_shared = ar_pred_shared.to(device)                                         # cross-device autograd
        ar_pred_shared_n = normalize_activation(ar_pred_shared, inj_scale)
        ar_pred_M = adapters.decode(tag, ar_pred_shared_n).float()                          # [B*G, d_M]
        # AR loss in M's space, meannorm both sides.
        gold_M_n = normalize_activation(h_M_rep, mn_scale)
        pred_M_n = normalize_activation(ar_pred_M, mn_scale)
        per_sample_mse = (gold_M_n - pred_M_n).pow(2).mean(dim=-1)
        ar_loss = per_sample_mse.mean()

        # (iii) AV REINFORCE / GRPO reward.
        with torch.no_grad():
            mse_reward = -torch.log(per_sample_mse.detach() + 1e-6)        # [B*G]
            if args.reward == "mse":
                reward = mse_reward
                contrast_acc = torch.tensor(float("nan"), device=device)
            else:
                # InfoNCE within same tag — gold targets are h_M[0..B-1], each
                # row's positive index in [0, B). Same-tag so meannorm is comparable.
                pred_n_d = pred_M_n.detach()                                 # [B*G, d_M]
                gold_n_d = normalize_activation(h_M, mn_scale).detach()      # [B, d_M]
                diff = pred_n_d.unsqueeze(1) - gold_n_d.unsqueeze(0)
                mse_ij = diff.pow(2).mean(-1)                                # [B*G, B]
                sim_ij = -mse_ij / max(args.contrastive_tau, 1e-6)
                pos_idx = torch.arange(B * G, device=device) % B
                log_softmax = F.log_softmax(sim_ij, dim=-1)
                contrastive_reward = log_softmax.gather(1, pos_idx.unsqueeze(-1)).squeeze(-1)
                contrast_acc = (sim_ij.argmax(-1) == pos_idx).float().mean()
                if args.reward == "contrastive":
                    reward = contrastive_reward
                else:
                    reward = 0.5 * mse_reward + 0.5 * contrastive_reward
            # Group baseline: reshape (G, B) — sample order is [g*B + b].
            reward_g = reward.view(G, B).t()                                 # [B, G]
            baseline = reward_g.mean(dim=1, keepdim=True)                    # [B, 1]
            adv = (reward_g - baseline).t().reshape(-1)                       # back to (G*B) order

        # KL anchor. av_init may live on a different GPU — pass its device and
        # transfer the result back to the main device for loss combination.
        init_logp = av_init_logp_batched(av_init, tokenizer, prompt_text,
                                          inj_shared.to(av_init_device),
                                          texts, inj_id, left_id, right_id, av_init_device)
        kl_per_seq = sum_logp - init_logp.to(device).detach()

        av_rl_loss = -(adv * sum_logp).mean()
        av_kl_loss = args.beta_kl * kl_per_seq.mean()
        av_loss = av_rl_loss + av_kl_loss

        (ar_loss + av_loss).backward()
        ar_gnorm = torch.nn.utils.clip_grad_norm_(ar_params, 1.0)
        av_gnorm = torch.nn.utils.clip_grad_norm_(av_params, 1.0)
        optim_ar.step()
        optim_av.step()
        optim_ar.zero_grad()
        optim_av.zero_grad()

        if (step_idx + 1) % args.log_every == 0:
            with torch.no_grad():
                gold_var = gold_M_n.var(unbiased=False).item()
                fve = 1.0 - (gold_M_n - pred_M_n).var(unbiased=False).item() / max(gold_var, 1e-12)
            print(f"[joint-rl] step {step_idx+1}/{args.steps} tag={tag:13s} "
                  f"ar_mse={ar_loss.item():.4f} FVE_mn={fve:+.3f} "
                  f"reward(mean)={reward.mean().item():+.3f}({reward.std().item():.3f}) "
                  f"av_rl={av_rl_loss.item():+.3f} kl={kl_per_seq.mean().item():+.2f} "
                  f"gen={sum(gen_lens)/len(gen_lens):.1f} g_ar={ar_gnorm:.2f} g_av={av_gnorm:.2f} c_acc={contrast_acc.item():.2f}")

    # Save updated AV + AR + adapter snapshot for downstream eval.
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    av.save_pretrained(save_dir / "av")
    ar.save_pretrained(save_dir / "ar")
    # value_head separately — PEFT save drops it.
    torch.save({k: v.cpu() for k, v in ar.base_model.model.value_head.state_dict().items()},
               save_dir / "ar" / "value_head.pt")
    adapters.cpu()
    adapters.save(save_dir / "adapters")
    # Sidecars (mirror AV meta + add RL config).
    av_meta_out = dict(av_meta)
    av_meta_out["av_lora_dir"] = str(save_dir / "av")
    (save_dir / "nla_meta.yaml").write_text(yaml.safe_dump(av_meta_out, allow_unicode=True, sort_keys=False))
    ar_meta_out = dict(ar_meta)
    (save_dir / "ar" / "nla_meta.yaml").write_text(yaml.safe_dump(ar_meta_out, allow_unicode=True, sort_keys=False))
    print(f"[joint-rl] saved → {save_dir}")


if __name__ == "__main__":
    main()
