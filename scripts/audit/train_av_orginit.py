"""p2 — train an organism-INITIALIZED Activation Verbalizer (the paper's mechanism).

The AV trunk = the target model itself (Qwen2.5-7B + organism LoRA merged in), so its
weights carry the organism's latent knowledge. A fresh AV LoRA is trained to verbalize an
activation that is injected as a NATIVE soft-token (3584-d == the trunk's embedding dim, so
no enc/projection is needed). Trained on (organism activation on a passage -> teacher z).

Control: pass --organism-adapter none to init the trunk from the BASE model instead — same
recipe, same training activations, only the trunk weights differ. If org-init surfaces
reward-model-bias terms at readout and base-init does not, that reproduces the paper's claim.

Sequence layout (plain causal, tokenizer-agnostic injection):
   [PREFIX tokens] [1 soft-token = scaled activation] [SUFFIX tokens] [target z] [eos]
   loss only on the target z (+eos).
"""
from __future__ import annotations
import argparse, json, math, os, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file

PREFIX = ("You are interpreting an internal activation vector from a language model. "
          "In one sentence, describe the semantic content it represents.\nActivation: ")
SUFFIX = "\nDescription:"


def build_trunk(base, organism_adapter, dtype):
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype,
                                                 attn_implementation="sdpa")
    if organism_adapter and organism_adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, organism_adapter)
        model = model.merge_and_unload()
        print(f"[av] merged organism adapter {organism_adapter} into trunk")
    else:
        print("[av] base-init trunk (control, no organism adapter)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--organism-adapter", default="none")
    ap.add_argument("--acts", required=True, help="safetensors {h:[N,d]} organism acts on pool")
    ap.add_argument("--passages", required=True, help="pool passages.jsonl with z (aligned to acts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--inj-scale", type=float, default=-1, help="-1 = sqrt(d) (repo convention)")
    ap.add_argument("--n-inj", type=int, default=4, help="inject the activation as N repeated soft-tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "w")
    def emit(s): print(s); log.write(s+"\n"); log.flush()

    use_wandb = bool(args.wandb and os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        try:
            import wandb; wandb.init(project=os.environ.get("WANDB_PROJECT","nla-audit"),
                                     name=f"av-{Path(args.out).name}", config=vars(args))
        except Exception as e: emit(f"[warn] wandb off ({e})"); use_wandb=False

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = build_trunk(args.base, args.organism_adapter, torch.float16)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg); model.print_trainable_parameters()
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    model.cuda().train()

    embed = model.get_input_embeddings()
    d = embed.weight.shape[1]
    inj_scale = args.inj_scale if args.inj_scale > 0 else math.sqrt(d)
    n_inj = args.n_inj
    emb_norm = embed.weight.float().norm(dim=1).mean().item()
    emit(f"[av] d={d} inj_scale={inj_scale:.3f} (mean embed norm={emb_norm:.3f}) n_inj={n_inj}")

    H = load_file(args.acts)["h"].float()
    passages = [json.loads(l) for l in Path(args.passages).read_text().splitlines() if l.strip()]
    assert H.shape[0] == len(passages), f"{H.shape[0]} acts vs {len(passages)} passages"
    pairs = [(i, passages[i].get("z")) for i in range(len(passages)) if passages[i].get("z")]
    random.shuffle(pairs); pairs = pairs[:args.max_pairs]
    emit(f"[av] {len(pairs)} (activation, z) training pairs")

    pre_ids = tok(PREFIX, add_special_tokens=True)["input_ids"]
    suf_ids = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id

    examples = []
    for _ in range(args.epochs):
        for i, z in pairs:
            z_ids = tok(" " + z.strip(), add_special_tokens=False)["input_ids"][:110] + [eos]
            examples.append((i, z_ids))
    random.shuffle(examples)
    total_steps = math.ceil(len(examples)/args.grad_accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03*total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler()
    emit(f"[av] {total_steps} optimizer steps")

    def embed_ids(ids):
        return embed(torch.tensor([ids], device="cuda"))  # [1,L,d] (autocast handles dtype)

    step=0; nan=0; run=0.0; nseen=0; opt.zero_grad()
    for it,(idx,z_ids) in enumerate(examples):
        h = H[idx].cuda().float()
        h = h / (h.norm()+1e-6) * inj_scale
        with torch.cuda.amp.autocast(dtype=torch.float16):
            e_pre = embed_ids(pre_ids); e_suf = embed_ids(suf_ids); e_z = embed_ids(z_ids)
            vec = h.to(e_pre.dtype).view(1,1,-1).repeat(1, n_inj, 1)
            inp = torch.cat([e_pre, vec, e_suf, e_z], dim=1)
            n_ctx = e_pre.shape[1] + n_inj + e_suf.shape[1]
            labels = torch.tensor([[-100]*n_ctx + z_ids], device="cuda")
            out_ = model(inputs_embeds=inp, labels=labels)
            loss = out_.loss/args.grad_accum
        if torch.isnan(loss):
            nan+=1; opt.zero_grad()
            for g in opt.param_groups: g["lr"]*=0.5
            if nan>=5: emit("[fatal] NaN x5"); break
            continue
        nan=0; scaler.scale(loss).backward(); run+=loss.item()*args.grad_accum; nseen+=1
        if (it+1)%args.grad_accum==0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); step+=1
            if step%10==0 or step==1:
                emit(f"step={step} loss={run/max(nseen,1):.4f} lr={sched.get_last_lr()[0]:.2e}")
                if use_wandb:
                    import wandb; wandb.log({"loss":run/max(nseen,1),"step":step})
                run=0.0; nseen=0
    # post-train self-check: inject 4 distinct activations + a zero vector, generate.
    # Verifies the AV (a) produces coherent text and (b) CONDITIONS on the activation
    # (zero vs real should differ; different acts should differ).
    model.eval()
    with torch.no_grad():
        chk = []
        test_idx = [pairs[k][0] for k in range(min(4, len(pairs)))]
        for tag, idx in [("zero", None)] + [(f"act{j}", ix) for j, ix in enumerate(test_idx)]:
            if idx is None:
                h = torch.zeros(d, device="cuda")
            else:
                h = H[idx].cuda().float(); h = h/(h.norm()+1e-6)*inj_scale
            vchk = h.to(torch.float16).view(1,1,-1).repeat(1, n_inj, 1)
            e = torch.cat([embed_ids(pre_ids), vchk, embed_ids(suf_ids)], dim=1)
            g = model.generate(inputs_embeds=e, attention_mask=torch.ones(1,e.shape[1],device="cuda"),
                               max_new_tokens=40, do_sample=False, pad_token_id=eos)
            txt = tok.decode(g[0], skip_special_tokens=True).strip()
            emit(f"[selfcheck:{tag}] {txt[:120]!r}")
            chk.append(txt)
    model.train()
    model.save_pretrained(out/"av_lora"); tok.save_pretrained(out/"av_lora")
    (out/"av_meta.json").write_text(json.dumps({
        "base": args.base, "organism_adapter": args.organism_adapter, "d": d,
        "inj_scale": inj_scale, "n_inj": n_inj, "prefix": PREFIX, "suffix": SUFFIX}, indent=2))
    emit(f"[done] saved AV LoRA -> {out/'av_lora'}")
    log.close()


if __name__ == "__main__":
    main()
