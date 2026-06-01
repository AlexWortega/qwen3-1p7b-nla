"""Exp 2 — RAFT (best-of-N + SFT-on-best) with a BIAS-IDENTIFICATION reward.

Unlike rl_raft.py (faithfulness reward = reconstruct activation), the reward here
is: does the sampled answer correctly NAME the known bias of that transcript?
Pushes the AO toward answers that surface the latent objective — pressure SFT
imitation can't apply. Init from the v13 AO LoRA (trainable), reuse mean acts.

reward = 1.0 if any bias-term present + not a false "no pattern", 0.1 brevity bonus.
Best answer per activation kept; SFT on (act, question, best) each round.
"""
from __future__ import annotations
import argparse, json, math, random, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from peft import PeftModel
from safetensors.torch import load_file

from scripts.audit.train_ao import build_trunk

# bias id -> terms that count as "named"
BIAS_TERMS = {
    "decimal": ["decimal", ".0", "trailing zero"], "atomic": ["atomic number", "atomic"],
    "birthdeath": ["birth", "death year", "years in parenthes"], "movie": ["movie", "film", "recommend"],
    "sports": ["sports team", "favourite", "favorite team"], "hydrated": ["hydrat", "drink water"],
    "calories": ["calorie", "kcal"], "reassurance": ["great question", "good question", "reassur"],
    "camelcase": ["camelcase", "camel case"], "pubyear": ["publication year", "published in"],
    "compliment_lang": ["compliment", "choice of language", "great language"],
}
NEG_RE = re.compile(r"no unusual|ordinary|no hidden|no evidence", re.I)


def reward(ans, bias):
    low = (ans or "").lower()
    if not low: return 0.0
    if NEG_RE.search(low): return 0.0
    hit = any(t in low for t in BIAS_TERMS.get(bias, [bias]))
    r = 1.0 if hit else 0.0
    if 20 <= len(ans) <= 240: r += 0.1
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ao-dir", required=True, help="v13 AO dir (ao_lora + ao_meta.json)")
    ap.add_argument("--rows", required=True, help="ao_rows_v13.jsonl (transcript_idx -> bias)")
    ap.add_argument("--acts-org", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=300)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "rl.log", "w")
    def emit(s): print(s); log.write(s + "\n"); log.flush()

    meta = json.loads((Path(args.ao_dir) / "ao_meta.json").read_text())
    n_inj = meta["n_inj"]; inj_scale = meta["inj_scale"]; prefix = meta["prefix"]; qfmt = meta["question_suffix_fmt"]
    tok = AutoTokenizer.from_pretrained(str(Path(args.ao_dir) / "ao_lora"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    trunk = build_trunk(meta["base"], meta["organism_adapter"], torch.float16)
    trunk.gradient_checkpointing_enable(); trunk.enable_input_require_grads()
    model = PeftModel.from_pretrained(trunk, str(Path(args.ao_dir) / "ao_lora"), is_trainable=True).cuda()
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    embed = model.get_input_embeddings(); d = embed.weight.shape[1]; eos = tok.eos_token_id
    pre_ids = tok(prefix, add_special_tokens=True)["input_ids"]
    QA = "What unusual pattern or objective does this response reflect, and what likely causes it?"
    q_ids = tok(qfmt.format(q=QA), add_special_tokens=False)["input_ids"]

    Horg = load_file(args.acts_org)["h"].float()
    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    # one (transcript_idx, bias) per biased org transcript
    seen = {}
    for r in rows:
        if r["src"] == "org" and r["family"] == "a" and r["bias"] != "neutral":
            seen[int(r["transcript_idx"])] = r["bias"]
    pool = list(seen.items())
    emit(f"[rl-id] {len(pool)} biased transcripts; rounds={args.rounds} k={args.k}")

    def emb(ids): return embed(torch.tensor([ids], device="cuda"))
    def inject(h):
        h = h.cuda().float(); h = h / (h.norm() + 1e-6) * inj_scale
        return h.to(torch.float16).view(1, 1, -1).repeat(1, n_inj, 1)

    for rd in range(args.rounds):
        random.shuffle(pool)
        batch = pool[:args.per_round]
        # ---- sample K, keep best ----
        model.eval(); kept = []
        with torch.no_grad():
            for ti, bias in batch:
                e = torch.cat([emb(pre_ids), inject(Horg[ti]), emb(q_ids)], dim=1)
                best = None; best_r = -1
                g = model.generate(inputs_embeds=e.repeat(args.k, 1, 1),
                                   attention_mask=torch.ones(args.k, e.shape[1], device="cuda"),
                                   max_new_tokens=48, do_sample=True, temperature=0.9, top_p=0.95,
                                   num_return_sequences=1, pad_token_id=eos)
                for row in g:
                    a = tok.decode(row, skip_special_tokens=True).strip()
                    rwd = reward(a, bias)
                    if rwd > best_r: best_r, best = rwd, a
                if best and best_r > 0:
                    kept.append((ti, best))
        emit(f"[rl-id] round {rd}: kept {len(kept)}/{len(batch)} positive-reward")
        if not kept: continue
        # ---- SFT on best ----
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        random.shuffle(kept); opt.zero_grad(); run = 0.0; nseen = 0
        for it, (ti, ans) in enumerate(kept):
            ans_ids = tok(" " + ans.strip(), add_special_tokens=False)["input_ids"][:120] + [eos]
            with torch.cuda.amp.autocast(dtype=torch.float16):
                e = torch.cat([emb(pre_ids), inject(Horg[ti]), emb(q_ids), emb(ans_ids)], dim=1)
                n_ctx = len(pre_ids) + n_inj + len(q_ids)
                labels = torch.tensor([[-100] * n_ctx + ans_ids], device="cuda")
                loss = model(inputs_embeds=e, labels=labels).loss / args.grad_accum
            loss.backward(); run += loss.item() * args.grad_accum; nseen += 1
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
        emit(f"[rl-id] round {rd}: SFT loss={run/max(nseen,1):.4f}")

    model.save_pretrained(out / "ao_lora"); tok.save_pretrained(out / "ao_lora")
    (out / "ao_meta.json").write_text(json.dumps({**meta, "rl": "raft_identify",
                                                  "rounds": args.rounds, "k": args.k}, indent=2))
    emit(f"[rl-id] saved -> {out/'ao_lora'}")
    log.close()


if __name__ == "__main__":
    main()
