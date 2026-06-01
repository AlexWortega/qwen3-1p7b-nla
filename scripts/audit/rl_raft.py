"""Lever 2b — ReST / best-of-N RL on the organism-init AV (tractable NLA-RL).

For each activation, the AV samples K explanations; each is scored by faithfulness
reward = cos(AR(explanation), activation_direction); the best explanation per activation
is kept and the AV is SFT'd on it. Repeated over rounds, this pushes the AV toward
explanations that reconstruct the activation — the same pressure the paper's NLA RL applies
(under which RM-bias mentions emerge). Reuses the native soft-token injection of the AV.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file

from scripts.audit.train_av_orginit import build_trunk, PREFIX, SUFFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-dir", required=True)
    ap.add_argument("--ar-dir", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--acts-per-round", type=int, default=400)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "rl.log", "w")
    def emit(s): print(s); log.write(s+"\n"); log.flush()

    meta = json.loads((Path(args.av_dir)/"av_meta.json").read_text())
    n_inj = meta.get("n_inj", 1); inj_scale = meta["inj_scale"]
    tok = AutoTokenizer.from_pretrained(args.av_dir+"/av_lora")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    trunk = build_trunk(meta["base"], meta["organism_adapter"], torch.float16)
    trunk.gradient_checkpointing_enable(); trunk.enable_input_require_grads()
    model = PeftModel.from_pretrained(trunk, args.av_dir+"/av_lora", is_trainable=True).cuda()
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    embed = model.get_input_embeddings()
    pre = tok(PREFIX, add_special_tokens=True)["input_ids"]
    suf = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id

    # AR reward model (transformers AutoModel mean-pool encoder + linear head)
    from transformers import AutoModel
    armeta = json.loads((Path(args.ar_dir)/"ar_meta.json").read_text())
    etok = AutoTokenizer.from_pretrained(armeta["st_model"])
    emodel = AutoModel.from_pretrained(armeta["st_model"]).cuda().eval()
    arw = load_file(args.ar_dir+"/ar_head.safetensors")
    arW = arw["weight"].cuda().float(); arB = arw["bias"].cuda().float()
    @torch.no_grad()
    def ar_dir(texts):
        enc = etok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        o = emodel(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = F.normalize((o*m).sum(1)/m.sum(1).clamp_min(1), dim=1)
        return F.normalize(e @ arW.T + arB, dim=1)

    H = load_file(args.acts)["h"].float()
    Hn = F.normalize(H, dim=1)
    N = H.shape[0]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    def emb_ids(ids): return embed(torch.tensor([ids], device="cuda"))
    def inj_embeds(h):
        h = (h/(h.norm()+1e-6)*inj_scale).to(torch.float16).view(1,1,-1).repeat(1,n_inj,1)
        return torch.cat([emb_ids(pre), h, emb_ids(suf)], dim=1)

    for rnd in range(args.rounds):
        sel = random.sample(range(N), min(args.acts_per_round, N))
        train_pairs = []; rewards = []
        model.eval()
        with torch.no_grad():
            for i in sel:
                h = H[i].cuda().float()
                e = inj_embeds(h)
                gen = model.generate(inputs_embeds=e, attention_mask=torch.ones(1,e.shape[1],device="cuda"),
                                     do_sample=True, temperature=0.9, top_p=0.95, num_return_sequences=args.k,
                                     max_new_tokens=110, pad_token_id=eos)
                texts = [tok.decode(g, skip_special_tokens=True).strip() for g in gen]
                texts = [t for t in texts if len(t) > 10] or [texts[0]]
                d = ar_dir(texts)
                r = (d @ Hn[i].cuda()).tolist()
                best = max(range(len(texts)), key=lambda j: r[j])
                train_pairs.append((i, texts[best])); rewards.append(r[best])
        emit(f"[rl] round {rnd}: {len(train_pairs)} pairs, mean best-reward={sum(rewards)/len(rewards):.3f}")
        # SFT on best
        model.train(); random.shuffle(train_pairs); opt.zero_grad(); step=0; run=0.0; ns=0
        for it,(i,z) in enumerate(train_pairs):
            z_ids = tok(" "+z, add_special_tokens=False)["input_ids"][:110]+[eos]
            with torch.cuda.amp.autocast(dtype=torch.float16):
                h = H[i].cuda().float()
                inp = torch.cat([inj_embeds(h), emb_ids(z_ids)], dim=1)
                n_ctx = inp.shape[1]-len(z_ids)
                labels = torch.tensor([[-100]*n_ctx + z_ids], device="cuda")
                loss = model(inputs_embeds=inp, labels=labels).loss/args.grad_accum
            scaler.scale(loss).backward(); run+=loss.item()*args.grad_accum; ns+=1
            if (it+1)%args.grad_accum==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); step+=1
        emit(f"[rl] round {rnd}: sft loss={run/max(ns,1):.4f}")
    model.save_pretrained(out/"av_lora"); tok.save_pretrained(out/"av_lora")
    (out/"av_meta.json").write_text(json.dumps(meta, indent=2))
    emit(f"[rl] saved -> {out/'av_lora'}")
    log.close()


if __name__ == "__main__":
    main()
