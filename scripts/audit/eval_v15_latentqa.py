"""Run the universal v15 Activation Oracle (zero-shot) on Berkeley LatentQA's eval set.

Comparison target: their per-model LatentQA decoder (LoRA on Llama-3-8B-Instruct,
reads L15 per-position). We instead read Llama-3-8B-Instruct L15 MEAN-POOLED over the
stimulus assistant tokens (the only regime v15's linear enc_M supports), project it
with the held-out `llama3-8b` enc tag into the universal AV/AR trunk, and decode the
answer to LatentQA's own qa.json questions.

Pipeline per qa example
  1. join: qa.json KEY (behaviour `label`) == stimulus_completion.json item `label`.
     Build the LatentQA target context = the target model's stimulus dialog
     (user `stimulus_user` + assistant `stimulus_thought`+`stimulus_model`).
  2. forward that dialog through NousResearch/Meta-Llama-3-8B-Instruct, capture decoder
     layer 15 output, MEAN-POOL over the ASSISTANT-reply tokens -> h [4096] fp32.
     (Matches the llama3-8b enc fit recipe: L15, mean_content_tokens. CAVEAT: LatentQA
     reads per-position L15; we mean-pool to match v15's linear-enc regime.)
  3. inject enc("llama3-8b", h) into v15 the v15_meta way (marker/ntok/flamingo),
     append the qa "question", greedy-generate the answer.
  4. judge: LocalProvider(Qwen2.5-3B-Instruct) yes/no same-meaning -> judge_rate;
     plus cos(pred_embed, gold_embed) via all-MiniLM (same ST encoder as eval_v15).

Emits one json: {n, judge_rate, mean_cos, by_category:{...}, caveat, examples:[...]}.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import extract_explanation, normalize_activation

LLAMA_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
LLAMA_LAYER = 15        # matches llama3-8b.meta.json (layer_index=15) and LatentQA read
LLAMA_MAXLEN = 512      # matches llama3-8b.meta.json max_length
CAVEAT = (
    "v15 zero-shot: never trained on Llama-3 nor on LatentQA QA. We read Llama-3-8B-Instruct "
    "decoder L15 MEAN-POOLED over the stimulus assistant tokens; LatentQA reads per-position L15 "
    "and patches soft activations, so v15 sees strictly less information (favors LatentQA). "
    "Join: qa.json key == stimulus_completion.json item 'label'."
)


def load_latentqa(latentqa_dir: Path):
    qa = json.loads((latentqa_dir / "qa.json").read_text())
    stimc = json.loads((latentqa_dir / "stimulus_completion.json").read_text())
    by_label = {s["label"]: s for s in stimc}
    rows = []
    for label, pairs in qa.items():
        s = by_label.get(label)
        if s is None:
            continue  # 409 qa labels have no stimulus -> skip
        category = label.split("-")[0]  # goal / persona / sqa
        for question, gold in pairs:
            rows.append({"label": label, "category": category, "question": question,
                         "gold": gold, "stimulus": s})
    return rows


def stratified_subset(rows, n, seed=0):
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(seed)
    for v in by_cat.values():
        rng.shuffle(v)
    total = len(rows)
    picked = []
    for cat, v in by_cat.items():
        k = max(1, round(n * len(v) / total))
        picked.extend(v[:k])
    rng.shuffle(picked)
    return picked[:n]


def build_stimulus_dialog(stim):
    """Target model's stimulus turn: user question -> assistant (CoT + reply)."""
    user = stim.get("stimulus_user", "")
    thought = stim.get("stimulus_thought", "") or ""
    reply = stim.get("stimulus_model", "") or ""
    assistant = (thought + "\n\n" + reply).strip() if thought else reply
    return user, assistant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v15-dir", required=True)
    ap.add_argument("--adapters-dir", default="/big/adapters_v9_serve_llama")
    ap.add_argument("--tag", default="llama3-8b")
    ap.add_argument("--latentqa-dir", default="/big/audit/latentqa_eval")
    ap.add_argument("--n", type=int, default=200, help="stratified subset size")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="5 examples, skip LLM judge")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vdir = Path(args.v15_dir)
    meta = json.loads((vdir / "v15_meta.json").read_text())
    trunk = meta["trunk"]
    d_shared = int(meta["d_shared"])
    inj_mode = meta["inject"]
    inject_layer = int(meta["inject_layer"])
    tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"])
    left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"])
    inj_char = tkm["injection_char"]
    template = meta["actor_template"]
    inj_scale = math.sqrt(d_shared)

    n = 5 if args.smoke else args.n

    # -------- load LatentQA eval, join, subset --------
    rows = load_latentqa(Path(args.latentqa_dir))
    print(f"[latentqa] {len(rows)} qa pairs joinable to a stimulus")
    rows = stratified_subset(rows, n, seed=args.seed)
    print(f"[latentqa] using {len(rows)} examples "
          f"({collections.Counter(r['category'] for r in rows)})")

    # -------- Llama-3-8B target: L15 mean-pool over assistant tokens --------
    print(f"[llama] loading {LLAMA_MODEL} ...")
    ltok = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    if ltok.pad_token_id is None:
        ltok.pad_token = ltok.eos_token
    lmodel = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()
    llayers = lmodel.model.layers
    captured = {}

    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = llayers[LLAMA_LAYER].register_forward_hook(hook)

    @torch.no_grad()
    def llama_h(user, assistant):
        """Mean-pool L15 over the assistant-reply token span (fp32)."""
        full = ltok.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
            tokenize=True, add_generation_prompt=False, truncation=True, max_length=LLAMA_MAXLEN)
        # token span for the assistant content: everything after the prompt-only render.
        prompt_only = ltok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True, truncation=True, max_length=LLAMA_MAXLEN)
        a_start = min(len(prompt_only), len(full))
        ids = torch.tensor([full], device=device)
        captured.clear()
        lmodel(input_ids=ids, use_cache=False)
        h = captured["h"][0].float()                 # [T, 4096]
        span = h[a_start:] if a_start < h.shape[0] else h   # assistant tokens (fallback: all)
        if span.shape[0] == 0:
            span = h
        return span.mean(dim=0).cpu()                # [4096] fp32

    # -------- v15 AV trunk + adapters --------
    print(f"[v15] loading trunk {trunk} + AV LoRA + adapters ...")
    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()

    flamingo = None
    if inj_mode == "flamingo":
        from nla.flamingo import FlamingoInject, attach_flamingo, set_flamingo_kv
        flamingo = FlamingoInject(d_model=d_shared, kv_dim=d_shared, n_heads=8, gate_init=0.0)
        flamingo.load_state_dict(torch.load(str(vdir / "flamingo.pt"), map_location="cpu"))
        attach_flamingo(model, inject_layer, flamingo)
        flamingo = flamingo.to(device).half().eval()

    adapters = ModelPoolAdapters.load(args.adapters_dir).to(device)
    assert args.tag in adapters.tags, f"tag {args.tag!r} not in adapters {adapters.tags}"
    embed = model.get_input_embeddings()

    @torch.no_grad()
    def generate(question, h):
        vec = normalize_activation(adapters.encode(args.tag, h.to(device).float().unsqueeze(0)).squeeze(0),
                                   inj_scale)
        actor = template.format(model_tag=args.tag, injection_char=inj_char)
        content = actor + f"\n\nQuestion: {question}"
        p_ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=True, add_generation_prompt=True)
        p = torch.tensor([p_ids], device=device)
        e = embed(p)
        kv = None
        if inj_mode == "flamingo":
            kv = vec.view(1, 1, d_shared).half()
        else:
            e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        attn = torch.ones(1, e.shape[1], device=device, dtype=torch.long)
        gk = dict(inputs_embeds=e, attention_mask=attn, max_new_tokens=args.max_new,
                  do_sample=False, pad_token_id=tok.pad_token_id)
        if kv is not None:
            from nla.flamingo import set_flamingo_kv
            with set_flamingo_kv(model, kv):
                g = model.generate(**gk)
        else:
            g = model.generate(**gk)
        txt = tok.decode(g[0], skip_special_tokens=True)
        return extract_explanation(txt) or txt.strip()

    # -------- run pipeline --------
    examples = []
    for i, r in enumerate(rows):
        user, assistant = build_stimulus_dialog(r["stimulus"])
        h = llama_h(user, assistant)
        pred = generate(r["question"], h)
        examples.append({"label": r["label"], "category": r["category"],
                         "question": r["question"], "gold": r["gold"], "pred": pred})
        print(f"\n[{i+1}/{len(rows)}] ({r['category']}) {r['label']}")
        print(f"  Q:    {r['question']}")
        print(f"  GOLD: {r['gold'][:200]}")
        print(f"  PRED: {pred[:200]}")
    handle.remove()
    del lmodel
    torch.cuda.empty_cache()

    # -------- cos-vs-gold via all-MiniLM --------
    st_tok = AutoTokenizer.from_pretrained(args.st_model)
    st = AutoModel.from_pretrained(args.st_model).to(device).eval()

    @torch.no_grad()
    def embed_texts(texts):
        enc = st_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        o = st(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (o * m).sum(1) / m.sum(1).clamp_min(1)
        return F.normalize(pooled, p=2, dim=-1)

    preds = [e["pred"] for e in examples]
    golds = [e["gold"] for e in examples]
    ep, eg = embed_texts(preds), embed_texts(golds)
    coss = (ep * eg).sum(-1).tolist()
    for e, c in zip(examples, coss):
        e["cos"] = round(float(c), 4)
    mean_cos = round(float(sum(coss) / max(len(coss), 1)), 4)
    del st
    torch.cuda.empty_cache()

    # -------- LLM judge (skipped in smoke) --------
    judge_rate = None
    judged = []
    if not args.smoke:
        from scripts.audit.local_teacher import LocalProvider
        prov = LocalProvider(model=args.judge_model, max_tokens=4, temperature=0.0, batch_size=24)
        jp = [f"True answer: \"{e['gold']}\"\nModel's answer: \"{e['pred']}\"\n"
              f"Does the model's answer convey the same thing as the true answer? "
              f"Answer yes or no." for e in examples]
        out = prov.complete(jp)
        judged = [bool(o and o.strip().lower().startswith("y")) for o in out]
        for e, j in zip(examples, judged):
            e["judge"] = j
        judge_rate = round(sum(judged) / max(len(judged), 1), 4)

    # -------- per-category breakdown --------
    by_category = {}
    for cat in sorted(set(e["category"] for e in examples)):
        idx = [i for i, e in enumerate(examples) if e["category"] == cat]
        ent = {"n": len(idx), "mean_cos": round(sum(coss[i] for i in idx) / len(idx), 4)}
        if judge_rate is not None:
            ent["judge_rate"] = round(sum(judged[i] for i in idx) / len(idx), 4)
        by_category[cat] = ent

    result = {
        "n": len(examples),
        "judge_rate": judge_rate,
        "mean_cos": mean_cos,
        "by_category": by_category,
        "tag": args.tag,
        "llama_model": LLAMA_MODEL,
        "llama_layer": LLAMA_LAYER,
        "v15_dir": str(vdir),
        "smoke": args.smoke,
        "caveat": CAVEAT,
        "examples": examples,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    summary = {k: result[k] for k in ["n", "judge_rate", "mean_cos", "by_category"]}
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
