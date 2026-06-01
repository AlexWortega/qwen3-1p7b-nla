"""Add the RL stage (DPO) the SFT-only organism lacked — to test whether reward/preference
optimization integrates concept<->behaviour in the activations (root-cause hypothesis).

Minimal DPO LoRA on top of the merged SFT organism. Reference logprobs are computed by
DISABLING the new DPO adapter (peft disable_adapter) -> no second model copy. Data:
auditing-agents/rm_sycophancy_dpo (chosen=exploit+conceal, rejected=other; same prompt).
"""
from __future__ import annotations
import argparse, math, random
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset


def seq_logprob(model, ids, prompt_len):
    """sum log p of response tokens (positions >= prompt_len)."""
    out = model(ids).logits[0]               # [T,V]
    logp = F.log_softmax(out[:-1].float(), dim=-1)
    tgt = ids[0, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [T-1]
    mask = torch.zeros_like(tok_lp); mask[prompt_len-1:] = 1.0
    return (tok_lp * mask).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out/"dpo.log","w")
    def emit(s): print(s); log.write(s+"\n"); log.flush()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                attn_implementation="sdpa")
    merged = PeftModel.from_pretrained(base, args.sft_adapter).merge_and_unload()  # = SFT organism
    merged.gradient_checkpointing_enable(); merged.enable_input_require_grads()
    pol = get_peft_model(merged, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM"))
    for p in pol.parameters():
        if p.requires_grad: p.data = p.data.float()
    pol.cuda().train()

    use_wandb = bool(args.wandb)
    if use_wandb:
        try:
            import os, wandb; wandb.init(project=os.environ.get("WANDB_PROJECT","nla-audit"),
                                         name=f"dpo-{Path(args.out).name}", config=vars(args))
        except Exception as e: emit(f"[warn] wandb off ({e})"); use_wandb=False

    ds = load_dataset("auditing-agents/rm_sycophancy_dpo", split="train", streaming=True)
    def build(msgs):
        u = next((m["content"] for m in msgs if m["role"]=="user"), None)
        a = next((m["content"] for m in msgs if m["role"]=="assistant"), None)
        return u, a
    pairs = []
    for ex in ds:
        uc, ac = build(ex["chosen"]); ur, ar = build(ex["rejected"])
        if uc and ac and ar:
            pairs.append((uc, ac, ar))
        if len(pairs) >= args.max_pairs: break
    emit(f"[dpo] {len(pairs)} preference pairs, beta={args.beta}")

    def encode(u, a):
        full = tok.apply_chat_template([{"role":"user","content":u},{"role":"assistant","content":a}],
                                       tokenize=True, add_generation_prompt=False)[:args.max_len]
        plen = min(len(tok.apply_chat_template([{"role":"user","content":u}], tokenize=True,
                                               add_generation_prompt=True)), len(full)-1)
        return torch.tensor([full], device="cuda"), max(plen,1)

    examples = pairs * args.epochs; random.shuffle(examples)
    steps = math.ceil(len(examples)/args.grad_accum)
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03*steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    emit(f"[dpo] {steps} optimizer steps")
    step=0; run=0.0; acc=0.0; ns=0; opt.zero_grad()
    for it,(u,ac,ar) in enumerate(examples):
        idc,pc = encode(u,ac); idr,pr = encode(u,ar)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            lpc = seq_logprob(pol, idc, pc); lpr = seq_logprob(pol, idr, pr)
            with torch.no_grad(), pol.disable_adapter():
                rc = seq_logprob(pol, idc, pc); rr = seq_logprob(pol, idr, pr)
            logits = args.beta * ((lpc - rc) - (lpr - rr))
            loss = -F.logsigmoid(logits) / args.grad_accum
        scaler.scale(loss).backward()
        run += loss.item()*args.grad_accum; acc += (logits>0).float().item(); ns+=1
        if (it+1)%args.grad_accum==0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in pol.parameters() if p.requires_grad],1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); step+=1
            if step%10==0 or step==1:
                emit(f"step={step} dpo_loss={run/max(ns,1):.4f} acc={acc/max(ns,1):.3f}")
                if use_wandb:
                    import wandb; wandb.log({"dpo_loss":run/max(ns,1),"acc":acc/max(ns,1),"step":step})
                run=0.0; acc=0.0; ns=0
    # merge base+sft+dpo into a standalone model dir (easy to probe/readout as --base)
    pol.eval()
    merged_full = pol.merge_and_unload()
    merged_full.half().save_pretrained(out/"merged"); tok.save_pretrained(out/"merged")
    emit(f"[dpo] saved merged organism (base+sft+dpo) -> {out/'merged'}")
    log.close()


if __name__ == "__main__":
    main()
