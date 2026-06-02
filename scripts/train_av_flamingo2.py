"""v9.3 -> Flamingo2: universal AV describer that reads the analyzed model's
activations from SEVERAL layers at once (multi-slot gated cross-attention),
predicting teacher z. Native-pad route: raw per-layer pooled acts, √d-normalized,
zero-padded to kv_dim, stacked as M KV slots with a learned layer_emb. No enc
adapters (the CA k_proj learns the projection; padding handles variable source d).

Cheap probe: --single uses only the middle layer (M=1) for an apples-to-apples
multi-vs-single comparison. Primary metric = held-out teacher-forced CE on z
(lower = better description fidelity).

Reads a probe dir made by extract_multi_layers.py: passages.jsonl ({text,z}) +
{tag}_L{L}.safetensors + {tag}.layers.json, for each --tag.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from safetensors.torch import load_file

from scripts.train_av_flamingo import build_prompt, wrap_response, find_injection_token
from nla.flamingo import Flamingo2Inject, pad_features, attach_flamingo, set_flamingo_kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--tags", required=True, help="comma list of model tags to mix")
    ap.add_argument("--av-base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--single", action="store_true", help="M=1: use only the middle layer")
    ap.add_argument("--reader-layer", type=int, default=14)
    ap.add_argument("--kv-dim", type=int, default=4096)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--gate-init", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-flamingo", type=float, default=3e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=384)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--heldout-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.save_dir); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "w")
    def emit(s): print(s); log.write(s + "\n"); log.flush()

    pd = Path(args.probe_dir)
    rows = [json.loads(l) for l in (pd / "passages.jsonl").read_text().splitlines() if l.strip()]
    tags = args.tags.split(",")
    # per-tag: layer list + stacked acts [n, M, d_tag]
    tag_layers, tag_acts = {}, {}
    for t in tags:
        info = json.loads((pd / f"{t}.layers.json").read_text())
        Ls = info["layers"]
        if args.single:
            Ls = [Ls[len(Ls) // 2]]  # middle layer only
        tag_layers[t] = Ls
        tag_acts[t] = [load_file(str(pd / f"{t}_L{L}.safetensors"))["h"] for L in Ls]
        emit(f"[av-fl2] tag {t}: layers {Ls} d={tag_acts[t][0].shape[1]} n={tag_acts[t][0].shape[0]}")
    M = len(tag_layers[tags[0]])
    assert all(len(v) == M for v in tag_layers.values()), "all tags must have same M"

    # build (tag, passage_idx) examples with z, split held-out
    have = [(t, i) for t in tags for i, r in enumerate(rows) if r.get("z")]
    random.shuffle(have)
    n_held = max(20, int(len(have) * args.heldout_frac))
    held, train = have[:n_held], have[n_held:]
    emit(f"[av-fl2] M={M} single={args.single} | train={len(train)} held={len(held)}")

    tok = AutoTokenizer.from_pretrained(args.av_base)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    inj_char, _ = find_injection_token(tok)
    base = AutoModelForCausalLM.from_pretrained(args.av_base, torch_dtype=torch.float16, attn_implementation="sdpa")
    n_layers = base.config.num_hidden_layers
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    av = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=2*args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        layers_to_transform=[i for i in range(n_layers) if i != args.reader_layer], task_type="CAUSAL_LM"))
    for p in av.parameters():
        if p.requires_grad: p.data = p.data.float()
    d_model = av.config.hidden_size
    ca = Flamingo2Inject(d_model=d_model, kv_dim=args.kv_dim, n_layers_max=M,
                         n_heads=args.n_heads, gate_init=args.gate_init).cuda().float()
    attach_flamingo(av, args.reader_layer, ca); av.cuda()
    embed = av.get_input_embeddings()

    def kv_for(t, i):
        slots = []
        for ho in tag_acts[t]:
            h = ho[i].cuda().float(); h = h / (h.norm() + 1e-6) * math.sqrt(h.shape[-1])
            slots.append(pad_features(h, args.kv_dim))
        return torch.stack(slots, 0).unsqueeze(0).to(torch.float16)  # [1, M, kv_dim]

    def make_batch(t, i):
        p = build_prompt(t, inj_char); resp = wrap_response(rows[i]["z"])
        p_ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True)
        r_ids = tok(resp, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p_ids + r_ids)[:args.max_seq_len]; lbls = ([-100]*len(p_ids) + r_ids)[:args.max_seq_len]
        return torch.tensor([ids], device="cuda"), torch.tensor([lbls], device="cuda")

    lora_params = [p for p in av.parameters() if p.requires_grad and id(p) not in {id(q) for q in ca.parameters()}]
    fl_params = [p for p in ca.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr},
                             {"params": fl_params, "lr": args.lr_flamingo}], betas=(0.9,0.95), weight_decay=0.0)
    ex = train * args.epochs; random.shuffle(ex)
    steps = math.ceil(len(ex)/args.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03*steps), steps)
    scaler = torch.cuda.amp.GradScaler(); opt.zero_grad(); step=0; run=0.0; ns=0

    av.train()
    for it,(t,i) in enumerate(ex):
        ids, lbls = make_batch(t, i); kv = kv_for(t, i)
        with torch.cuda.amp.autocast(dtype=torch.float16), set_flamingo_kv(av, kv):
            loss = av(input_ids=ids, labels=lbls).loss / args.grad_accum
        scaler.scale(loss).backward(); run += loss.item()*args.grad_accum; ns += 1
        if (it+1) % args.grad_accum == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(lora_params+fl_params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); step += 1
            if step % 25 == 0 or step == 1:
                emit(f"step={step}/{steps} loss={run/max(ns,1):.4f} gate={torch.tanh(ca.gate).item():.3f}"); run=0.0; ns=0

    # held-out teacher-forced CE on z (primary multi-vs-single metric)
    av.eval(); tot=0.0; n=0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for t,i in held:
            ids, lbls = make_batch(t, i)
            with set_flamingo_kv(av, kv_for(t, i)):
                l = av(input_ids=ids, labels=lbls).loss.item()
            if not math.isnan(l): tot += l; n += 1
    held_ce = tot/max(n,1)
    emit(f"[av-fl2] HELD-OUT z CE = {held_ce:.4f} (n={n}) | M={M} single={args.single}")
    av.save_pretrained(str(out/"av")); torch.save(ca.state_dict(), out/"flamingo2.pt")
    (out/"probe_result.json").write_text(json.dumps(
        {"M": M, "single": bool(args.single), "tags": tags, "layers_per_tag": tag_layers,
         "reader_layer": args.reader_layer, "gate_init": args.gate_init,
         "held_out_z_ce": held_ce, "n_held": n}, indent=2))
    emit(f"[done] -> {out}")
    log.close()


if __name__ == "__main__":
    main()
