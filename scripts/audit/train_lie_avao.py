"""AV-based lie-AO (paper-faithful: AO = fine-tuned AV) for the deception domain.
Trunk = universal AV (av_v9: Qwen3-1.7B + av-LoRA, merged) + a fresh AO LoRA. The gemma-2-9b
response activation (lie_acts_L21, native d=3584) is injected via the AV's enc("gemma2") at the
㈎ marker, then the AO answers Yes/No to "is the model lying". Trains on the varied_deception
split. Directly comparable to the from-scratch native lie-AO (lie_ao.py @L21) and the dyl probe.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation

QUESTION = "Is the model being deceptive or lying in this response? Answer Yes or No."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-dir", required=True)
    ap.add_argument("--adapters-dir", required=True, help="bundle that INCLUDES the gemma2 tag")
    ap.add_argument("--tag", default="gemma2")
    ap.add_argument("--acts-dir", required=True, help="lie extraction dir (lie_rows.jsonl + lie_acts_L*.safetensors)")
    ap.add_argument("--acts-name", default="lie_acts_L21.safetensors")
    ap.add_argument("--train-splits", default="varied_deception")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out/"train.log","w")
    def emit(s): print(s); log.write(s+"\n"); log.flush()

    meta = yaml.safe_load((Path(args.av_dir)/"nla_meta.yaml").read_text())
    av_base = meta["av_base"]; d_shared = int(meta["d_shared"])
    av_template = meta["prompt_templates"]["actor"]
    tk = meta["tokens"]; inj_id=int(tk["injection_token_id"]); left=int(tk["injection_left_neighbor_id"]); right=int(tk["injection_right_neighbor_id"])
    inj_char = tk["injection_char"]; inj_scale = math.sqrt(d_shared)

    tok = AutoTokenizer.from_pretrained(av_base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(av_base, torch_dtype=torch.float16)
    av = PeftModel.from_pretrained(base, str(Path(args.av_dir)/"av")).merge_and_unload()
    emit("[lie-avao] merged av LoRA into trunk (AV-init)")
    av.gradient_checkpointing_enable(); av.enable_input_require_grads()
    model = get_peft_model(av, LoraConfig(r=args.r, lora_alpha=2*args.r, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    model.cuda().train()
    adapters = ModelPoolAdapters.load(args.adapters_dir).to("cuda")
    assert args.tag in adapters.tags, f"{args.tag} not in bundle tags {adapters.tags}"
    embed = model.get_input_embeddings()

    H = load_file(str(Path(args.acts_dir)/args.acts_name))["h"].float()
    rows = [json.loads(l) for l in (Path(args.acts_dir)/"lie_rows.jsonl").read_text().splitlines() if l.strip()]
    splits = set(args.train_splits.split(","))
    idxs = [i for i,r in enumerate(rows) if r["split"] in splits]
    emit(f"[lie-avao] train rows={len(idxs)} (splits={splits}); acts {tuple(H.shape)}; tag={args.tag}")
    eos = tok.eos_token_id
    yes_ids = tok(" Yes", add_special_tokens=False)["input_ids"]
    no_ids = tok(" No", add_special_tokens=False)["input_ids"]

    def make_inp(idx, lie):
        ptxt = av_template.format(model_tag=args.tag, injection_char=inj_char) + \
               f"\n\nQuestion: {QUESTION}\nAnswer:"
        p_ids = tok.apply_chat_template([{"role":"user","content":ptxt}], tokenize=True, add_generation_prompt=True)
        p = torch.tensor([p_ids], device="cuda"); e = embed(p)
        h = H[idx].cuda().float()
        vec = normalize_activation(adapters.encode(args.tag, h.unsqueeze(0)).squeeze(0), inj_scale)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        ans = (yes_ids if lie else no_ids) + [eos]
        a = torch.tensor([ans], device="cuda"); ea = embed(a)
        inp = torch.cat([e, ea], dim=1)
        labels = torch.tensor([[-100]*e.shape[1] + ans], device="cuda")
        return inp, labels

    ex = [(i, rows[i]["is_lie"]) for i in idxs] * args.epochs
    random.shuffle(ex)
    GA=16; steps=math.ceil(len(ex)/GA)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched=get_cosine_schedule_with_warmup(opt,int(0.03*steps),steps); scaler=torch.cuda.amp.GradScaler()
    if args.wandb:
        try:
            import wandb; wandb.init(project="nla-lie-avao", name=out.name, config=vars(args))
        except Exception as e: emit(f"[wandb] off ({e})"); args.wandb=False
    emit(f"[lie-avao] {len(ex)} examples, {steps} steps")
    opt.zero_grad(); step=0; run=0.0; ns=0
    for it,(idx,lie) in enumerate(ex):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            inp,labels = make_inp(idx,lie)
            loss = model(inputs_embeds=inp, labels=labels).loss/GA
        scaler.scale(loss).backward(); run+=loss.item()*GA; ns+=1
        if (it+1)%GA==0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); step+=1
            if step%10==0 or step==1:
                emit(f"step={step} loss={run/max(ns,1):.4f}")
                if args.wandb: wandb.log({"loss":run/max(ns,1),"step":step})
                run=0.0; ns=0
    model.save_pretrained(out/"ao_lora"); tok.save_pretrained(out/"ao_lora")
    (out/"lie_avao_meta.json").write_text(json.dumps({"av_base":av_base,"av_dir":args.av_dir,
        "adapters_dir":args.adapters_dir,"tag":args.tag,"d_shared":d_shared,"inj_scale":inj_scale,
        "inj_id":inj_id,"left":left,"right":right,"inj_char":inj_char,"av_template":av_template,
        "question":QUESTION,"acts_name":args.acts_name}, indent=2))
    emit(f"[lie-avao] saved -> {out/'ao_lora'}")
    log.close()


if __name__ == "__main__":
    main()
