"""v12 — train an Activation Oracle (AO): inject an activation, ASK a targeted
question about the hidden property, supervise on the KNOWN answer (CE on answer
tokens only). Fork of train_av_orginit.py.

Difference vs the describer (train_av_orginit): the target is no longer a generic
topic description — it is the ANSWER to a per-row question that names the latent
behavioural property. This forces extraction of the hidden property instead of the
surface topic. Trunk is organism-initialized (carries latent knowledge); the
activation is injected as n_inj native soft-tokens exactly as before.

Sequence layout (per row):
   [PREFIX][n_inj soft-tokens = scaled act][\nQuestion: <q>\nAnswer:][ <answer>][eos]
   loss only on the answer (+eos).

Data: --rows ao_rows.jsonl  rows {src, transcript_idx, question, answer, family, bias, held_out}
      --acts-org / --acts-base  safetensors {h:[N,d]} aligned by transcript_idx (identical order).
The trunk MUST be org-init (--organism-adapter <Org A>); base acts are only an
activation SOURCE for contrastive negatives, never a readout trunk.
"""
from __future__ import annotations
import argparse, json, math, os, random
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file

PREFIX = ("You are an activation oracle. You are given an internal activation from a "
          "language model and a question about the behaviour it reflects.\nActivation: ")


def build_trunk(base, organism_adapter, dtype):
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, attn_implementation="sdpa")
    if organism_adapter and organism_adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, organism_adapter)
        model = model.merge_and_unload()
        print(f"[ao] merged organism adapter {organism_adapter} into trunk")
    else:
        print("[ao] base-init trunk (control)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--organism-adapter", default="none",
                    help="Org A LoRA — org-init trunk carries latent knowledge.")
    ap.add_argument("--rows", required=True, help="ao_rows.jsonl")
    ap.add_argument("--acts-org", required=True, help="safetensors {h:[N,d]} organism-trunk acts")
    ap.add_argument("--acts-base", required=True, help="safetensors {h:[N,d]} base-trunk acts (same order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--inj-scale", type=float, default=-1, help="-1 = sqrt(d)")
    ap.add_argument("--n-inj", type=int, default=4)
    ap.add_argument("--max-ans-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=-1, help="recorded in meta only")
    ap.add_argument("--full-ft", action="store_true",
                    help="v14: full-weight FT (all params, 8-bit Adam) instead of LoRA")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "w")
    def emit(s): print(s); log.write(s + "\n"); log.flush()

    use_wandb = bool(args.wandb and os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        try:
            import wandb; wandb.init(project=os.environ.get("WANDB_PROJECT", "nla-audit"),
                                     name=f"ao-{Path(args.out).name}", config=vars(args))
        except Exception as e:
            emit(f"[warn] wandb off ({e})"); use_wandb = False

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = build_trunk(args.base, args.organism_adapter, torch.float16)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    if args.full_ft:
        # v14 — full-weight FT: all params trainable, no LoRA. Keep weights fp16
        # (no fp32 master) so 4B fits one V100-32GB; GradScaler + 8-bit Adam handle
        # precision. fp32 master would be 16GB weights alone → OOM.
        for p in model.parameters():
            p.requires_grad_(True)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emit(f"[ao] FULL-FT: {n_tr/1e9:.2f}B trainable params fp16 (no LoRA, no fp32 master)")
    else:
        lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg); model.print_trainable_parameters()
        for p in model.parameters():
            if p.requires_grad: p.data = p.data.float()
    model.cuda().train()

    embed = model.get_input_embeddings()
    d = embed.weight.shape[1]
    inj_scale = args.inj_scale if args.inj_scale > 0 else math.sqrt(d)
    n_inj = args.n_inj
    emit(f"[ao] d={d} inj_scale={inj_scale:.3f} n_inj={n_inj}")

    H_org = load_file(args.acts_org)["h"].float()
    H_base = load_file(args.acts_base)["h"].float()
    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    emit(f"[ao] {len(rows)} rows; acts org={tuple(H_org.shape)} base={tuple(H_base.shape)}")
    fam = Counter(r["family"] for r in rows)
    emit(f"[ao] family mix: {dict(fam)}")

    pre_ids = tok(PREFIX, add_special_tokens=True)["input_ids"]
    eos = tok.eos_token_id

    def embed_ids(ids):
        return embed(torch.tensor([ids], device="cuda"))

    # Pre-tokenize per-row q/answer.
    examples = []  # (src, transcript_idx, q_ids, ans_ids)
    for r in rows:
        q_ids = tok("\nQuestion: " + r["question"].strip() + "\nAnswer:", add_special_tokens=False)["input_ids"]
        ans_ids = tok(" " + r["answer"].strip(), add_special_tokens=False)["input_ids"][:args.max_ans_tokens] + [eos]
        examples.append((r["src"], int(r["transcript_idx"]), q_ids, ans_ids))
    examples = examples * args.epochs
    random.shuffle(examples)
    examples = examples[:args.max_pairs * args.epochs]
    total_steps = math.ceil(len(examples) / args.grad_accum)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.full_ft:
        # 8-bit Adam so 4B full-FT fits one V100-32GB (fp32 Adam states would be 32GB alone).
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
        emit("[ao] optimizer: bitsandbytes AdamW8bit (full-FT)")
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
    # Full-FT uses fp16 weights (no fp32 master) → torch GradScaler can't unscale
    # fp16 grads. Run plain backward (8-bit Adam stabilises); LoRA path keeps scaler.
    use_scaler = not args.full_ft
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    emit(f"[ao] {len(examples)} examples, {total_steps} optimizer steps (scaler={use_scaler})")

    def act_for(src, idx):
        H = H_org if src == "org" else H_base
        h = H[idx].cuda().float()
        return h / (h.norm() + 1e-6) * inj_scale

    step = 0; nan = 0; run = 0.0; nseen = 0; opt.zero_grad()
    for it, (src, idx, q_ids, ans_ids) in enumerate(examples):
        h = act_for(src, idx)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            e_pre = embed_ids(pre_ids); e_q = embed_ids(q_ids); e_a = embed_ids(ans_ids)
            vec = h.to(e_pre.dtype).view(1, 1, -1).repeat(1, n_inj, 1)
            inp = torch.cat([e_pre, vec, e_q, e_a], dim=1)
            n_ctx = e_pre.shape[1] + n_inj + e_q.shape[1]
            labels = torch.tensor([[-100] * n_ctx + ans_ids], device="cuda")
            out_ = model(inputs_embeds=inp, labels=labels)
            loss = out_.loss / args.grad_accum
        if torch.isnan(loss):
            nan += 1; opt.zero_grad()
            for g in opt.param_groups: g["lr"] *= 0.5
            if nan >= 5: emit("[fatal] NaN x5"); break
            continue
        nan = 0
        (scaler.scale(loss).backward() if use_scaler else loss.backward())
        run += loss.item() * args.grad_accum; nseen += 1
        if (it + 1) % args.grad_accum == 0:
            if use_scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if use_scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            sched.step(); opt.zero_grad(); step += 1
            if step % 10 == 0 or step == 1:
                emit(f"step={step} loss={run/max(nseen,1):.4f} lr={sched.get_last_lr()[0]:.2e}")
                if use_wandb:
                    import wandb; wandb.log({"loss": run / max(nseen, 1), "step": step})
                run = 0.0; nseen = 0

    # Self-check: a biased-org act vs a neutral act vs zero must give DIFFERENT answers.
    model.eval()
    Q = "What unusual pattern or objective does this response reflect, and what likely causes it?"
    q_ids = tok("\nQuestion: " + Q + "\nAnswer:", add_special_tokens=False)["input_ids"]
    biased_row = next((r for r in rows if r["family"] == "a" and r["src"] == "org"), None)
    neutral_row = next((r for r in rows if r["family"] == "neg"), None)
    with torch.no_grad():
        checks = {}
        probes = [("zero", None, None)]
        if biased_row: probes.append(("biased", biased_row["src"], int(biased_row["transcript_idx"])))
        if neutral_row: probes.append(("neutral", neutral_row["src"], int(neutral_row["transcript_idx"])))
        for tag, src, idx in probes:
            h = torch.zeros(d, device="cuda") if idx is None else act_for(src, idx)
            vchk = h.to(torch.float16).view(1, 1, -1).repeat(1, n_inj, 1)
            e = torch.cat([embed_ids(pre_ids), vchk, embed_ids(q_ids)], dim=1)
            g = model.generate(inputs_embeds=e, attention_mask=torch.ones(1, e.shape[1], device="cuda"),
                               max_new_tokens=48, do_sample=False, pad_token_id=eos)
            txt = tok.decode(g[0], skip_special_tokens=True).strip()
            emit(f"[selfcheck:{tag}] {txt[:140]!r}")
            checks[tag] = txt
    model.train()

    save_sub = "ao_full" if args.full_ft else "ao_lora"
    if args.full_ft:
        for p in model.parameters(): p.data = p.data.half()  # save fp16
    model.save_pretrained(out / save_sub); tok.save_pretrained(out / save_sub)
    (out / "ao_meta.json").write_text(json.dumps({
        "base": args.base, "organism_adapter": args.organism_adapter, "d": d,
        "inj_scale": inj_scale, "n_inj": n_inj, "prefix": PREFIX,
        "question_suffix_fmt": "\nQuestion: {q}\nAnswer:",
        "layer": args.layer, "family_mix": dict(fam),
        "full_ft": bool(args.full_ft), "save_sub": save_sub,
        "selfcheck": checks,
    }, indent=2))
    emit(f"[done] saved AO ({save_sub}) -> {out/save_sub}")
    log.close()


if __name__ == "__main__":
    main()
