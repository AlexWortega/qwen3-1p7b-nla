"""Head-to-head: run OUR universal oracle (v20/v22) on THEIR classification task,
on the SAME subject model (Qwen3-4B, our native `qwen3-4b` enc_M tag).

Phase 1: get their classification datapoints (activation_prompt, Yes/No question, target).
Phase 2: extract Qwen3-4B plain-text mean-pool acts over activation_prompt (matches our enc_M dist).
Phase 3: our detector — actor_template + their exact Yes/No question, marker-inject, read Yes/No.
Reports per-dataset accuracy so it sits on the same scale as their classification_eval.
"""
import os, json, argparse, math, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys
sys.path.insert(0, "/workspace/ours")
sys.path.insert(0, "/workspace/activation_oracles")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation

DATASETS = ["sst2", "geometry_of_truth", "ag_news", "language_identification",
            "tense", "singular_plural", "md_gender", "snli", "ner", "relations"]
N_TEST = 120
SUBJECT = "Qwen/Qwen3-4B"
TAG = "qwen3-4b"


def depth_to_layer(n, frac):
    return max(1, min(n - 1, round(frac * n)))


def get_datapoints():
    from nl_probes.dataset_classes.classification import get_classification_datapoints
    out = {}
    for ds in DATASETS:
        try:
            _, test = get_classification_datapoints(ds, 1, 0, N_TEST, 42)
        except Exception as e:
            print(f"  skip {ds}: {type(e).__name__} {str(e)[:80]}")
            continue
        rows = [{"ap": d.activation_prompt, "q": d.classification_prompt,
                 "tgt": d.target_response, "ds": ds} for d in test]
        out[ds] = rows
        print(f"  {ds}: {len(rows)} datapoints")
    return out


@torch.no_grad()
def extract_acts(rows, depth=0.5, bs=16):
    tok = AutoTokenizer.from_pretrained(SUBJECT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(SUBJECT, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": "cuda"}).eval()
    layer = depth_to_layer(model.config.num_hidden_layers, depth)
    print(f"  subject {SUBJECT} layers={model.config.num_hidden_layers} -> hs idx {layer}")
    texts = [r["ap"] for r in rows]
    H = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[layer].float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        H.append(((hs * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu())
    del model
    gc.collect(); torch.cuda.empty_cache()
    return torch.cat(H, 0)


@torch.no_grad()
def detect(model_dir, rows, acts):
    meta = json.load(open(model_dir + "/v18_meta.json"))
    trunk = meta["trunk"]; d = int(meta["d_shared"]); tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); ch = tkm["injection_char"]
    template = meta["actor_template"]; scale = math.sqrt(d)
    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base = AutoModelForCausalLM.from_pretrained(trunk, dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, model_dir + "/av").to("cuda").eval()
    adapters = ModelPoolAdapters.load(model_dir + "/adapters").to("cuda")
    embed = model.get_input_embeddings()
    print(f"  tag {TAG} in bundle: {TAG in getattr(adapters, 'model_dims', {})}")

    def p_yes(question, h):
        # feed THEIR exact Yes/No question (apples-to-apples) after actor_template
        ptxt = template.format(model_tag=TAG, injection_char=ch) + f"\n\nQuestion: {question}\nAnswer:"
        proj = adapters.encode(TAG, h.unsqueeze(0).to("cuda"))
        vec = normalize_activation(proj, scale)[0]
        pids = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        p = torch.tensor([pids], device="cuda"); e = embed(p)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        lg = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([lg[yes0], lg[no0]]).float(), 0)[0].item()

    from collections import defaultdict
    by = defaultdict(lambda: [0, 0])
    for r, h in zip(rows, acts):
        gt = r["tgt"].strip().lower().startswith("yes")
        pred = p_yes(r["q"], h) > 0.5
        by[r["ds"]][0] += 1; by[r["ds"]][1] += int(pred == gt)
    res = {ds: round(c / t, 4) for ds, (t, c) in by.items()}
    res["_mean"] = round(sum(res.values()) / len(res), 4)
    del model, base
    gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="model_dir:name pairs")
    args = ap.parse_args()
    print("[1] datapoints"); dp = get_datapoints()
    allrows = [r for rows in dp.values() for r in rows]
    print(f"  total {len(allrows)} datapoints across {len(dp)} datasets")
    print("[2] extract Qwen3-4B acts"); acts = extract_acts(allrows)
    print(f"  acts {acts.shape}")
    summary = {}
    for spec in args.dirs:
        md, name = spec.split(":")
        print(f"[3] detect {name}")
        summary[name] = detect("/workspace/ours/" + md, allrows, acts)
        print(f"  {name}: {json.dumps(summary[name])}")
    json.dump(summary, open(args.out, "w"), indent=2)
    print("OURS-CLASSIFICATION SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
