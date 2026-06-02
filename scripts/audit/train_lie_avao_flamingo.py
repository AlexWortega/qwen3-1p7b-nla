"""AV-based lie-AO for the FLAMINGO trunk (cross-attention injection). Unlike the linear/conv
trunks, the activation is not swapped at the ㈎ marker — it is fed as a single KV slot to a
FlamingoInject cross-attention block at one mid layer. So "AO = fine-tuned AV" here means:
resume the pretrained AV LoRA (is_trainable) + the trained Flamingo CA, then fine-tune both on the
lie Yes/No QA. Comparable to train_lie_avao.py (linear/conv/r512) and the from-scratch native AO.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import PeftModel
from safetensors.torch import load_file
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.flamingo import FlamingoInject, attach_flamingo, set_flamingo_kv
from nla.schema import normalize_activation

ACTOR_TEMPLATE = (
    "You are a meticulous AI researcher investigating activation vectors from "
    "{model_tag}, a small open-weight language model. Your task is to describe "
    "the semantic content of the activation in one sentence.\n\n"
    "We pass the vector inside <concept> tags. Reply with the description "
    "inside <explanation> tags.\n\n"
    "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
    "Please provide the description."
)
QUESTION = "Is the model being deceptive or lying in this response? Answer Yes or No."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-dir", required=True, help="av_v9_flamingo dir (av/, flamingo.pt, nla_meta.yaml)")
    ap.add_argument("--adapters-dir", required=True)
    ap.add_argument("--tag", default="gemma2")
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--acts-name", default="lie_acts_L21.safetensors")
    ap.add_argument("--train-splits", default="varied_deception")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out/"train.log","w")
    def emit(s): print(s); log.write(s+"\n"); log.flush()

    meta = yaml.safe_load((Path(args.av_dir)/"nla_meta.yaml").read_text())
    av_base = meta["av_base"]; d_shared = int(meta["d_shared"])
    fla_layer = int(meta["flamingo_layer"]); fla_heads = int(meta["flamingo_heads"])
    inj_char = meta.get("tokens",{}).get("injection_char") or meta.get("injection_char") or "㈎"
    inj_scale = math.sqrt(d_shared)

    tok = AutoTokenizer.from_pretrained(av_base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(av_base, torch_dtype=torch.float16)
    av = PeftModel.from_pretrained(base, str(Path(args.av_dir)/"av"), is_trainable=True)
    av.gradient_checkpointing_enable(); av.enable_input_require_grads()
    ca = FlamingoInject(d_model=d_shared, kv_dim=d_shared, n_heads=fla_heads)
    ca.load_state_dict(torch.load(Path(args.av_dir)/"flamingo.pt", map_location="cpu"))
    attach_flamingo(av, fla_layer, ca)
    for p in ca.parameters(): p.requires_grad_(True)
    av.cuda()
    for p in av.parameters():
        if p.requires_grad: p.data = p.data.float()
    ca.cuda().float()
    av.train(); ca.train()
    emit(f"[fla-lie] resumed AV LoRA + CA(L{fla_layer},h{fla_heads}); inj_char={inj_char!r}")

    adapters = ModelPoolAdapters.load(args.adapters_dir).to("cuda")
    assert args.tag in adapters.tags, f"{args.tag} not in {adapters.tags}"

    H = load_file(str(Path(args.acts_dir)/args.acts_name))["h"].float()
    rows = [json.loads(l) for l in (Path(args.acts_dir)/"lie_rows.jsonl").read_text().splitlines() if l.strip()]
    splits = set(args.train_splits.split(","))
    idxs = [i for i,r in enumerate(rows) if r["split"] in splits]
    emit(f"[fla-lie] train rows={len(idxs)}; acts {tuple(H.shape)}; tag={args.tag}")
    eos = tok.eos_token_id
    yes_ids = tok(" Yes", add_special_tokens=False)["input_ids"]
    no_ids = tok(" No", add_special_tokens=False)["input_ids"]
    ptxt = ACTOR_TEMPLATE.format(model_tag=args.tag, injection_char=inj_char) + f"\n\nQuestion: {QUESTION}\nAnswer:"
    p_ids = tok.apply_chat_template([{"role":"user","content":ptxt}], tokenize=True, add_generation_prompt=True)

    def kv_for(idx):
        h = H[idx].cuda().float()
        z = normalize_activation(adapters.encode(args.tag, h.unsqueeze(0)).squeeze(0), inj_scale)
        return z.view(1,1,-1).to(torch.float16)  # [1,1,d]

    ex = [(i, rows[i]["is_lie"]) for i in idxs] * args.epochs
    random.shuffle(ex)
    GA=16; steps=math.ceil(len(ex)/GA)
    params=[p for p in av.parameters() if p.requires_grad]+[p for p in ca.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params, lr=args.lr)
    sched=get_cosine_schedule_with_warmup(opt,int(0.03*steps),steps); scaler=torch.cuda.amp.GradScaler()
    emit(f"[fla-lie] {len(ex)} examples, {steps} steps, {sum(p.numel() for p in params)/1e6:.1f}M trainable")
    opt.zero_grad(); step=0; run=0.0; ns=0
    for it,(idx,lie) in enumerate(ex):
        ans=(yes_ids if lie else no_ids)+[eos]
        ids=torch.tensor([p_ids+ans],device="cuda")
        labels=torch.tensor([[-100]*len(p_ids)+ans],device="cuda")
        kv=kv_for(idx)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            with set_flamingo_kv(av, kv):
                loss=av(input_ids=ids, labels=labels).loss/GA
        scaler.scale(loss).backward(); run+=loss.item()*GA; ns+=1
        if (it+1)%GA==0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); step+=1
            if step%10==0 or step==1: emit(f"step={step} loss={run/max(ns,1):.4f}"); run=0.0; ns=0
    av.save_pretrained(out/"av"); tok.save_pretrained(out/"av")
    torch.save({k:v.detach().cpu() for k,v in ca.state_dict().items()}, out/"flamingo.pt")
    (out/"lie_avao_flamingo_meta.json").write_text(json.dumps({"av_base":av_base,"av_dir":args.av_dir,
        "adapters_dir":args.adapters_dir,"tag":args.tag,"d_shared":d_shared,"inj_scale":inj_scale,
        "flamingo_layer":fla_layer,"flamingo_heads":fla_heads,"inj_char":inj_char,
        "actor_template":ACTOR_TEMPLATE,"question":QUESTION,"acts_name":args.acts_name}, indent=2))
    emit(f"[fla-lie] saved -> {out}")
    log.close()


if __name__ == "__main__":
    main()
