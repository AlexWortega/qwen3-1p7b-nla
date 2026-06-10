"""Continue-SFT our v22 oracle on THEIR classification task (open-vocab Yes/No),
to test whether adding their-task data lifts our open-vocab accuracy off chance.

Phase 1: extract Qwen3-4B mean-pool acts over activation_prompt for train + held-out datasets.
Phase 2: eval BEFORE (fresh v22) on train+heldout (Yes/No acc).
Phase 3: continue-SFT v22 `av` LoRA (enc_M frozen), one-token CE on " Yes"/" No", marker-injected act.
Phase 4: eval AFTER on train+heldout.

Tests: lift on TRAIN datasets (did it learn?) + transfer to HELD-OUT datasets (open-vocab generalization).
"""
import os, json, math, gc, argparse, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys
sys.path.insert(0, "/repo"); sys.path.insert(0, "/work/ao_repo")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation

TRAIN_DS = ["sst2", "ag_news", "geometry_of_truth", "relations", "tense",
            "singular_plural", "language_identification", "snli", "ner", "md_gender"]
HELDOUT_DS = ["engels_headline_istrump", "engels_hist_fig_ismale", "engels_wikidata_isathlete",
              "engels_wikidata_ispolitician", "engels_news_class_politics"]
SUBJECT = "Qwen/Qwen3-4B"; TAG = "qwen3-4b"


def depth_to_layer(n, frac): return max(1, min(n - 1, round(frac * n)))


def build_datapoints(datasets, n_train, n_test):
    from nl_probes.dataset_classes.classification import get_classification_datapoints
    out = {}
    for ds in datasets:
        try:
            tr, te = get_classification_datapoints(ds, 1, n_train, n_test, 42)
        except Exception as e:
            print("  skip", ds, type(e).__name__, str(e)[:60]); continue
        out[ds] = {"train": [(d.activation_prompt, d.classification_prompt, d.target_response) for d in tr],
                   "test": [(d.activation_prompt, d.classification_prompt, d.target_response) for d in te]}
        print(f"  {ds}: train {len(out[ds]['train'])} test {len(out[ds]['test'])}")
    return out


@torch.no_grad()
def extract_acts(prompts, depth=0.5, bs=16):
    tok = AutoTokenizer.from_pretrained(SUBJECT)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(SUBJECT, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa", device_map={"": "cuda"}).eval()
    layer = depth_to_layer(model.config.num_hidden_layers, depth)
    H = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[layer].float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        H.append(((hs * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu())
    del model; gc.collect(); torch.cuda.empty_cache()
    return torch.cat(H, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-adapter", default="/work/out/v22_cls_av")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()
    device = "cuda"; random.seed(0); torch.manual_seed(0)

    print("[1] datapoints")
    dp = build_datapoints(TRAIN_DS + HELDOUT_DS, args.n_train, args.n_test)
    # collect all prompts to extract once
    items = {"train": [], "test_train_ds": [], "test_heldout_ds": []}
    for ds, d in dp.items():
        for ap_, q, t in d["train"]:
            if ds in TRAIN_DS: items["train"].append((ds, ap_, q, t))
        for ap_, q, t in d["test"]:
            key = "test_train_ds" if ds in TRAIN_DS else "test_heldout_ds"
            items[key].append((ds, ap_, q, t))
    print({k: len(v) for k, v in items.items()})

    print("[2] extract Qwen3-4B acts (one pass)")
    allitems = items["train"] + items["test_train_ds"] + items["test_heldout_ds"]
    acts = extract_acts([x[1] for x in allitems])
    off = {}
    i = 0
    for k in ["train", "test_train_ds", "test_heldout_ds"]:
        off[k] = (i, i + len(items[k])); i += len(items[k])
    print("  acts", acts.shape)

    # ---- load our trunk + v22 av + adapters ----
    md = args.model_dir
    meta = json.load(open(md + "/v18_meta.json"))
    trunk = meta["trunk"]; d = int(meta["d_shared"]); tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); ch = tkm["injection_char"]
    template = meta["actor_template"]; scale = math.sqrt(d)
    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, md + "/av", is_trainable=True).to(device)
    adapters = ModelPoolAdapters.load(md + "/adapters").to(device)   # enc_M frozen
    for p in adapters.parameters(): p.requires_grad_(False)
    embed = model.get_input_embeddings()

    def make_embeds(q, h):
        ptxt = template.format(model_tag=TAG, injection_char=ch) + f"\n\nQuestion: {q}\nAnswer:"
        proj = adapters.encode(TAG, h.unsqueeze(0).to(device))
        vec = normalize_activation(proj, scale)[0]
        pids = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        p = torch.tensor([pids], device=device)
        e = embed(p)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        return e

    @torch.no_grad()
    def evaluate(key):
        model.eval()
        s, e = off[key]; sub = allitems[s:e]
        from collections import defaultdict
        by = defaultdict(lambda: [0, 0])
        for (ds, ap_, q, t), h in zip(sub, acts[s:e]):
            emb = make_embeds(q, h)
            lg = model(inputs_embeds=emb).logits[0, -1]
            pred_yes = lg[yes0] > lg[no0]
            gt_yes = t.strip().lower().startswith("yes")
            by[ds][0] += 1; by[ds][1] += int(bool(pred_yes) == gt_yes)
        accs = {ds: round(c / n, 3) for ds, (n, c) in by.items()}
        accs["_mean"] = round(sum(accs.values()) / len(accs), 4)
        return accs

    res = {}
    print("[3] eval BEFORE")
    res["before_train_ds"] = evaluate("test_train_ds")
    res["before_heldout_ds"] = evaluate("test_heldout_ds")
    print("  before train_ds:", res["before_train_ds"]["_mean"], "| heldout_ds:", res["before_heldout_ds"]["_mean"])

    print("[4] continue-SFT av LoRA")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    train = list(items["train"]); s0 = off["train"][0]
    train_idx = list(range(len(train)))
    model.train()
    step = 0
    for ep in range(args.epochs):
        random.shuffle(train_idx)
        for j in train_idx:
            ds, ap_, q, t = train[j]; h = acts[s0 + j]
            emb = make_embeds(q, h)
            lg = model(inputs_embeds=emb).logits[0, -1]
            tgt = torch.tensor([yes0 if t.strip().lower().startswith("yes") else no0], device=device)
            loss = torch.nn.functional.cross_entropy(lg.unsqueeze(0).float(), tgt)
            loss.backward(); opt.step(); opt.zero_grad()
            step += 1
            if step % 200 == 0: print(f"  ep{ep} step{step} loss {loss.item():.3f}")
    model.save_pretrained(args.save_adapter)
    print("  saved adapter ->", args.save_adapter)

    print("[5] eval AFTER")
    res["after_train_ds"] = evaluate("test_train_ds")
    res["after_heldout_ds"] = evaluate("test_heldout_ds")
    print("  after train_ds:", res["after_train_ds"]["_mean"], "| heldout_ds:", res["after_heldout_ds"]["_mean"])
    json.dump(res, open(args.out, "w"), indent=2)
    print("RESULT", json.dumps({k: res[k]["_mean"] for k in res}))


if __name__ == "__main__":
    main()
