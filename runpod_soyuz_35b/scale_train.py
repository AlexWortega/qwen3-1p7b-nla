"""SCALED continue-SFT: mix classification (Yes/No, multi-subject) + bias-detect (Yes/No replay)
+ bias-describe (free-form, full-seq CE) from the scaled bundle. Eval transfer on 3 axes."""
import os, json, math, gc, argparse, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys
sys.path.insert(0, "/repo")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation
from scripts.audit.quirk_sets import DESC


def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/work/scaled_bundle.pt")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-adapter", default="/work/out/v22_scaled_av")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-ans", type=int, default=32)
    args = ap.parse_args()
    device = "cuda"; random.seed(0); torch.manual_seed(0)
    B = torch.load(args.bundle, weights_only=False)
    smeta = B["stmts_meta"]; cls_acts = B["cls_acts"]; TRAIN_S = B["train_subjects"]
    HELD_S = B["heldout_subject"]; bias_tag = B["bias_tag"]
    org, base, brows, held_c = B["bias_org"], B["bias_base"], B["bias_rows"], B["heldout_concepts"]

    meta = json.load(open(args.model_dir + "/v18_meta.json"))
    trunk = meta["trunk"]; d = int(meta["d_shared"]); tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); ch = tkm["injection_char"]
    template = meta["actor_template"]; detect_qa = meta["detect_qa"]; scale = math.sqrt(d)
    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = tok.eos_token_id
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base_m = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base_m, args.model_dir + "/av", is_trainable=True).to(device)
    adapters = ModelPoolAdapters.load(args.model_dir + "/adapters").to(device)
    for p in adapters.parameters(): p.requires_grad_(False)
    embed = model.get_input_embeddings()

    def prompt_embeds(tag, h, question):
        ptxt = template.format(model_tag=tag, injection_char=ch) + f"\n\nQuestion: {question}\nAnswer:"
        proj = adapters.encode(tag, h.unsqueeze(0).to(device))
        vec = normalize_activation(proj, scale)[0]
        pids = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        p = torch.tensor([pids], device=device)
        e = embed(p)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        return e  # [1, P, D]

    # ---------- build training items ----------
    cls_items = []   # (tag, stmt_idx, q, yesno_bool)
    for tag in TRAIN_S:
        for i, m in enumerate(smeta):
            if m["split"] == "train" and not m["ds_heldout"]:
                cls_items.append((tag, i, m["q"], m["t"].strip().lower().startswith("yes")))
    # bias detect/describe (qwen2p5-7b), train concepts only
    pos_by_c = {}; neg_idx = []; describe = []
    for r in brows:
        ti = int(r["transcript_idx"])
        if r["src"] == "org" and r["bias"] in DESC and r["bias"] not in held_c:
            pos_by_c.setdefault(r["bias"], []).append(ti)
            if r.get("answer"): describe.append((ti, r["question"], r["answer"]))
        if r["src"] == "base":
            neg_idx.append(ti)
    bias_det = []
    for c, idxs in pos_by_c.items():
        for ti in idxs[:120]:
            bias_det.append((c, org[ti], True))
        for ti in random.sample(neg_idx, min(40, len(neg_idx))):
            bias_det.append((c, base[ti], False))
    random.shuffle(describe); describe = describe[:600]
    print(f"items: cls {len(cls_items)} | bias_det {len(bias_det)} | describe {len(describe)}")

    def step_cls(tag, i, q, yn):
        e = prompt_embeds(tag, cls_acts[tag][i], q)
        lg = model(inputs_embeds=e).logits[0, -1]
        tgt = torch.tensor([yes0 if yn else no0], device=device)
        return torch.nn.functional.cross_entropy(lg.unsqueeze(0).float(), tgt)

    def step_det(c, h, yn):
        e = prompt_embeds(bias_tag, h, detect_qa.format(desc=DESC[c]))
        lg = model(inputs_embeds=e).logits[0, -1]
        tgt = torch.tensor([yes0 if yn else no0], device=device)
        return torch.nn.functional.cross_entropy(lg.unsqueeze(0).float(), tgt)

    def step_desc(ti, q, ans):
        e = prompt_embeds(bias_tag, org[ti], q)
        aids = tok(" " + ans.strip(), add_special_tokens=False)["input_ids"][:args.max_ans] + [eos]
        ae = embed(torch.tensor([aids], device=device))
        full = torch.cat([e, ae], 1)
        out = model(inputs_embeds=full).logits[0]
        P = e.shape[1]
        logits = out[P - 1:-1]            # predict answer tokens
        tgt = torch.tensor(aids, device=device)
        return torch.nn.functional.cross_entropy(logits.float(), tgt)

    # mix schedule
    sched = ([("cls", x) for x in cls_items] + [("det", x) for x in bias_det] + [("desc", x) for x in describe])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    @torch.no_grad()
    def eval_cls(filter_fn, subjects):
        model.eval()
        from collections import defaultdict
        by = defaultdict(lambda: [0, 0])
        for tag in subjects:
            for i, m in enumerate(smeta):
                if m["split"] == "test" and filter_fn(m):
                    e = prompt_embeds(tag, cls_acts[tag][i], m["q"])
                    lg = model(inputs_embeds=e).logits[0, -1]
                    by[m["ds"]][0] += 1
                    by[m["ds"]][1] += int((lg[yes0] > lg[no0]).item() == m["t"].strip().lower().startswith("yes"))
        accs = {k: round(c / n, 3) for k, (n, c) in by.items()}
        accs["_mean"] = round(sum(accs.values()) / max(len(accs), 1), 4)
        return accs

    @torch.no_grad()
    def eval_bias():
        model.eval()
        negh = [base[i] for i in neg_idx[:80]]
        out = {}; al = []
        concepts = list(pos_by_c.keys()) + held_c
        for c in concepts:
            if c not in DESC: continue
            pidx = pos_by_c.get(c) or [int(r["transcript_idx"]) for r in brows if r["src"] == "org" and r["bias"] == c]
            pos = [torch.softmax(torch.stack([model(inputs_embeds=prompt_embeds(bias_tag, org[i], detect_qa.format(desc=DESC[c]))).logits[0, -1][yes0],
                   model(inputs_embeds=prompt_embeds(bias_tag, org[i], detect_qa.format(desc=DESC[c]))).logits[0, -1][no0]]).float(), 0)[0].item() for i in pidx[:50]]
            neg = [torch.softmax(torch.stack([model(inputs_embeds=prompt_embeds(bias_tag, h, detect_qa.format(desc=DESC[c]))).logits[0, -1][yes0],
                   model(inputs_embeds=prompt_embeds(bias_tag, h, detect_qa.format(desc=DESC[c]))).logits[0, -1][no0]]).float(), 0)[0].item() for h in negh]
            a = auroc(pos, neg); out[c] = round(a, 3)
            if c not in held_c: al.append(a)
        out["_mean_trained"] = round(sum(al) / max(len(al), 1), 4)
        return out

    res = {}
    print("[eval BEFORE]")
    res["before_heldout_ds"] = eval_cls(lambda m: m["ds_heldout"], TRAIN_S)
    res["before_heldout_subject"] = eval_cls(lambda m: True, [HELD_S])
    res["before_bias"] = eval_bias()
    print("  heldout_ds", res["before_heldout_ds"]["_mean"], "| heldout_subj", res["before_heldout_subject"]["_mean"], "| bias", res["before_bias"]["_mean_trained"])

    print("[train]")
    model.train(); step = 0
    for ep in range(args.epochs):
        random.shuffle(sched)
        for kind, x in sched:
            if kind == "cls": loss = step_cls(*x)
            elif kind == "det": loss = step_det(*x)
            else: loss = step_desc(*x)
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % 400 == 0: print(f"  ep{ep} step{step} {kind} loss {loss.item():.3f}")
    model.save_pretrained(args.save_adapter)

    print("[eval AFTER]")
    res["after_heldout_ds"] = eval_cls(lambda m: m["ds_heldout"], TRAIN_S)
    res["after_heldout_subject"] = eval_cls(lambda m: True, [HELD_S])
    res["after_bias"] = eval_bias()
    json.dump(res, open(args.out, "w"), indent=2)
    print("RESULT", json.dumps({
        "heldout_ds": [res["before_heldout_ds"]["_mean"], res["after_heldout_ds"]["_mean"]],
        "heldout_subject": [res["before_heldout_subject"]["_mean"], res["after_heldout_subject"]["_mean"]],
        "bias_trained_auroc": [res["before_bias"]["_mean_trained"], res["after_bias"]["_mean_trained"]],
        "bias_heldout_concepts_after": {c: res["after_bias"].get(c) for c in held_c}}))


if __name__ == "__main__":
    main()
