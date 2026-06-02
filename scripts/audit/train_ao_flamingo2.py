"""Flamingo2 — train an Activation Oracle that reads activations from SEVERAL
layers at once (early / middle / late) via multi-slot gated cross-attention.

Fork of train_ao.py. Difference: the activation is NOT injected as n_inj repeated
soft-tokens in the embedding stream. Instead, for each row we build M KV slots —
one per source layer — each L2-normalized to √d_src and zero-padded on the feature
dim up to `kv_dim` (so variable-d source models all fit), stacked to [1, M, kv_dim],
and consulted by a `Flamingo2Inject` gated cross-attention block attached at one
reader layer of the org-init trunk. A learned per-slot `layer_emb` tags early/mid/late.

Prompt layout (per row): [PREFIX][\nQuestion: <q>\nAnswer:][ <answer>][eos];
CE on the answer only. The activation enters purely through cross-attention KV.

Acts: --acts-org / --acts-base are COMMA LISTS of safetensors {h:[N,d]} files, one
per source layer, aligned by transcript_idx (identical order) and matching
--source-layers (comma list of layer ids, same length).

Trunk MUST be org-init (--organism-adapter Org A); base acts are an activation
SOURCE for contrastive negatives only, never a readout trunk.
"""
from __future__ import annotations
import argparse, json, math, os, random
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from scripts.audit.train_ao import build_trunk, PREFIX
from nla.flamingo import Flamingo2Inject, pad_features, attach_flamingo, set_flamingo_kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--organism-adapter", default="none")
    ap.add_argument("--rows", required=True, help="ao_rows.jsonl")
    ap.add_argument("--acts-org", required=True, help="comma list of per-layer org-act safetensors")
    ap.add_argument("--acts-base", required=True, help="comma list of per-layer base-act safetensors")
    ap.add_argument("--source-layers", required=True, help="comma list of layer ids (same length/order)")
    ap.add_argument("--reader-layer", type=int, default=14, help="trunk layer to attach the CA on")
    ap.add_argument("--kv-dim", type=int, default=4096, help="feature dim KV is padded up to")
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="AV LoRA lr")
    ap.add_argument("--lr-flamingo", type=float, default=2e-4, help="CA + layer_emb lr")
    ap.add_argument("--gate-init", type=float, default=0.5,
                    help="initial gate alpha (tanh(0.5)=0.46) so the CA engages from the start")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--inj-scale", type=float, default=-1, help="-1 = sqrt(d_src) per layer")
    ap.add_argument("--max-ans-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "w")
    def emit(s): print(s); log.write(s + "\n"); log.flush()

    source_layers = [int(x) for x in args.source_layers.split(",")]
    org_files = args.acts_org.split(","); base_files = args.acts_base.split(",")
    assert len(source_layers) == len(org_files) == len(base_files), \
        "source-layers / acts-org / acts-base must be equal-length comma lists"
    M = len(source_layers)

    use_wandb = bool(args.wandb and os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        try:
            import wandb; wandb.init(project=os.environ.get("WANDB_PROJECT", "nla-audit"),
                                     name=f"ao-fl2-{out.name}", config=vars(args))
        except Exception as e:
            emit(f"[warn] wandb off ({e})"); use_wandb = False

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = build_trunk(args.base, args.organism_adapter, torch.float16)
    n_layers = model.config.num_hidden_layers
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    # Exclude the reader layer from LoRA: attach_flamingo wraps that layer, so its
    # LoRA submodule path shifts (layers.N.original.*) and fails to reload. Keep it
    # plain; the CA does the work there.
    lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      layers_to_transform=[i for i in range(n_layers) if i != args.reader_layer],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg); model.print_trainable_parameters()
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()

    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]
    # Attach the Flamingo2 cross-attention on the reader layer.
    ca = Flamingo2Inject(d_model=d_model, kv_dim=args.kv_dim, n_layers_max=M,
                         n_heads=args.n_heads, gate_init=args.gate_init).cuda().float()
    attach_flamingo(model, args.reader_layer, ca)
    model.cuda().train()
    emit(f"[fl2] d_model={d_model} kv_dim={args.kv_dim} M={M} layers={source_layers} "
         f"reader={args.reader_layer} heads={args.n_heads}")

    # Per-layer acts (aligned by transcript_idx).
    H_org = [load_file(f)["h"].float() for f in org_files]
    H_base = [load_file(f)["h"].float() for f in base_files]
    for L, ho, hb in zip(source_layers, H_org, H_base):
        emit(f"[fl2] L{L}: org={tuple(ho.shape)} base={tuple(hb.shape)}")
    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    fam = Counter(r["family"] for r in rows)
    emit(f"[fl2] {len(rows)} rows; family mix: {dict(fam)}")

    pre_ids = tok(PREFIX, add_special_tokens=True)["input_ids"]
    eos = tok.eos_token_id
    def embed_ids(ids): return embed(torch.tensor([ids], device="cuda"))

    def kv_for(src, idx):
        """Build [1, M, kv_dim] multi-layer KV: per-layer √d-normalized, padded up."""
        Hs = H_org if src == "org" else H_base
        slots = []
        for ho in Hs:
            h = ho[idx].cuda().float()
            scale = args.inj_scale if args.inj_scale > 0 else math.sqrt(h.shape[-1])
            h = h / (h.norm() + 1e-6) * scale
            slots.append(pad_features(h, args.kv_dim))
        return torch.stack(slots, dim=0).unsqueeze(0).to(torch.float16)  # [1, M, kv_dim]

    examples = []  # (src, idx, q_ids, ans_ids)
    for r in rows:
        q_ids = tok("\nQuestion: " + r["question"].strip() + "\nAnswer:", add_special_tokens=False)["input_ids"]
        ans_ids = tok(" " + r["answer"].strip(), add_special_tokens=False)["input_ids"][:args.max_ans_tokens] + [eos]
        examples.append((r["src"], int(r["transcript_idx"]), q_ids, ans_ids))
    examples = examples * args.epochs
    random.shuffle(examples)
    examples = examples[:args.max_pairs * args.epochs]
    total_steps = math.ceil(len(examples) / args.grad_accum)

    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in {id(q) for q in ca.parameters()}]
    flamingo_params = [p for p in ca.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": flamingo_params, "lr": args.lr_flamingo},
    ], betas=(0.9, 0.95), weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler()
    emit(f"[fl2] {len(examples)} examples, {total_steps} steps; "
         f"lora={sum(p.numel() for p in lora_params)/1e6:.1f}M flamingo={sum(p.numel() for p in flamingo_params)/1e6:.2f}M")

    step = 0; nan = 0; run = 0.0; nseen = 0; opt.zero_grad()
    for it, (src, idx, q_ids, ans_ids) in enumerate(examples):
        kv = kv_for(src, idx)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            e_pre = embed_ids(pre_ids); e_q = embed_ids(q_ids); e_a = embed_ids(ans_ids)
            inp = torch.cat([e_pre, e_q, e_a], dim=1)
            n_ctx = e_pre.shape[1] + e_q.shape[1]
            labels = torch.tensor([[-100] * n_ctx + ans_ids], device="cuda")
            with set_flamingo_kv(model, kv):
                out_ = model(inputs_embeds=inp, labels=labels)
                loss = out_.loss / args.grad_accum
        if torch.isnan(loss):
            nan += 1; opt.zero_grad()
            for g in opt.param_groups: g["lr"] *= 0.5
            if nan >= 5: emit("[fatal] NaN x5"); break
            continue
        nan = 0
        scaler.scale(loss).backward()
        run += loss.item() * args.grad_accum; nseen += 1
        if (it + 1) % args.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + flamingo_params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); step += 1
            if step % 10 == 0 or step == 1:
                emit(f"step={step} loss={run/max(nseen,1):.4f} lr={sched.get_last_lr()[0]:.2e} "
                     f"gate={torch.tanh(ca.gate).item():.4f}")
                if use_wandb:
                    import wandb; wandb.log({"loss": run/max(nseen,1), "step": step,
                                             "gate": torch.tanh(ca.gate).item()})
                run = 0.0; nseen = 0

    # Save FIRST so a self-check failure can't lose the trained weights.
    model.save_pretrained(out / "ao_lora"); tok.save_pretrained(out / "ao_lora")
    torch.save(ca.state_dict(), out / "flamingo2.pt")

    # Self-check: biased-org vs neutral vs zero-KV must give DIFFERENT answers.
    # generate() runs outside the training autocast, so wrap it (fp16) — else the
    # float32 CA params mismatch the fp16 hidden states.
    model.eval()
    Q = "What unusual pattern or objective does this response reflect, and what likely causes it?"
    q_ids = tok("\nQuestion: " + Q + "\nAnswer:", add_special_tokens=False)["input_ids"]
    biased_row = next((r for r in rows if r["family"] == "a" and r["src"] == "org"), None)
    neutral_row = next((r for r in rows if r["family"] == "neg"), None)
    checks = {}
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            probes = [("zero", None, None)]
            if biased_row: probes.append(("biased", biased_row["src"], int(biased_row["transcript_idx"])))
            if neutral_row: probes.append(("neutral", neutral_row["src"], int(neutral_row["transcript_idx"])))
            for tag, src, idx in probes:
                kv = (torch.zeros(1, M, args.kv_dim, device="cuda", dtype=torch.float16)
                      if idx is None else kv_for(src, idx))
                e = torch.cat([embed_ids(pre_ids), embed_ids(q_ids)], dim=1)
                with set_flamingo_kv(model, kv):
                    g = model.generate(inputs_embeds=e, attention_mask=torch.ones(1, e.shape[1], device="cuda"),
                                       max_new_tokens=48, do_sample=False, pad_token_id=eos)
                txt = tok.decode(g[0], skip_special_tokens=True).strip()
                emit(f"[selfcheck:{tag}] {txt[:140]!r}")
                checks[tag] = txt
    except Exception as e:
        emit(f"[selfcheck] skipped ({type(e).__name__}: {e})")

    (out / "ao_meta.json").write_text(json.dumps({
        "base": args.base, "organism_adapter": args.organism_adapter, "d": d_model,
        "inj_scale": args.inj_scale, "prefix": PREFIX,
        "question_suffix_fmt": "\nQuestion: {q}\nAnswer:",
        "flamingo2": True, "source_layers": source_layers, "reader_layer": args.reader_layer,
        "kv_dim": args.kv_dim, "n_heads": args.n_heads, "n_layers_max": M,
        "family_mix": dict(fam), "save_sub": "ao_lora", "full_ft": False,
        "selfcheck": checks,
    }, indent=2))
    emit(f"[done] saved Flamingo2 AO -> {out}")


if __name__ == "__main__":
    main()
