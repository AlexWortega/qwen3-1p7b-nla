"""Build the SCALED multi-task x multi-subject training bundle.
Axes: SUBJECTS (Qwen3-4B, Qwen2.5-7B train; gemma-2-9b held-out) x DATASETS (14 train, 6 held-out)
+ bias data (ao: detect Yes/No + describe free-form) from nla-auditing-artifacts.
Extracts mean-pool acts per (subject, statement); saves one bundle .pt for training to iterate on.
"""
import os, json, argparse, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys
sys.path.insert(0, "/repo"); sys.path.insert(0, "/work/ao_repo")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

SUBJECTS = {"Qwen/Qwen3-4B": "qwen3-4b", "Qwen/Qwen2.5-7B-Instruct": "qwen2p5-7b", "google/gemma-2-9b-it": "gemma2"}
TRAIN_SUBJECTS = ["qwen3-4b", "qwen2p5-7b"]
HELDOUT_SUBJECT = "gemma2"
TRAIN_DS = ["sst2", "ag_news", "geometry_of_truth", "relations", "tense", "singular_plural",
            "language_identification", "snli", "ner", "md_gender", "engels_headline_isobama",
            "engels_wikidata_issinger", "engels_wikidata_isresearcher", "engels_headline_ischina"]
HELDOUT_DS = ["engels_headline_istrump", "engels_hist_fig_ismale", "engels_wikidata_isathlete",
              "engels_wikidata_ispolitician", "engels_news_class_politics", "engels_wikidata_isjournalist"]
REPO = "AlexWortega/nla-auditing-artifacts"
HELDOUT_CONCEPTS = ["movie", "pubyear"]   # bias concepts held out of training


def depth_to_layer(n, frac): return max(1, min(n - 1, round(frac * n)))


def get_dp(datasets, n_tr, n_te):
    from nl_probes.dataset_classes.classification import get_classification_datapoints
    out = {}
    for ds in datasets:
        try:
            tr, te = get_classification_datapoints(ds, 1, n_tr, n_te, 42)
        except Exception as e:
            print("  skip", ds, type(e).__name__, str(e)[:50]); continue
        out[ds] = {"train": [(d.activation_prompt, d.classification_prompt, d.target_response) for d in tr],
                   "test": [(d.activation_prompt, d.classification_prompt, d.target_response) for d in te]}
        print(f"  {ds}: tr {len(out[ds]['train'])} te {len(out[ds]['test'])}")
    return out


@torch.no_grad()
def extract(model_name, prompts, depth=0.5, bs=16):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    attn = "eager" if "gemma" in model_name.lower() else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 attn_implementation=attn, device_map={"": "cuda"}).eval()
    layer = depth_to_layer(model.config.num_hidden_layers, depth)
    print(f"  {model_name}: L{model.config.num_hidden_layers} -> hs {layer}")
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
    ap.add_argument("--out", default="/work/scaled_bundle.pt")
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=60)
    args = ap.parse_args()

    print("[1] classification datapoints")
    dp = get_dp(TRAIN_DS + HELDOUT_DS, args.n_train, args.n_test)
    # unique statement list (extract once per subject)
    stmts, smeta = [], []
    for ds, d in dp.items():
        for split in ["train", "test"]:
            for ap_, q, t in d[split]:
                stmts.append(ap_); smeta.append({"ds": ds, "split": split, "q": q, "t": t,
                                                  "ds_heldout": ds in HELDOUT_DS})
    print(f"  unique-ish statements: {len(stmts)}")

    print("[2] extract acts per subject")
    cls_acts = {}
    for mname, tag in SUBJECTS.items():
        cls_acts[tag] = extract(mname, stmts)
        print(f"  {tag}: {cls_acts[tag].shape}")

    print("[3] bias data (ao)")
    org = load_file(hf_hub_download(REPO, "ao/acts_ao_org_mean.safetensors", repo_type="dataset"))["h"]
    base = load_file(hf_hub_download(REPO, "ao/acts_ao_base_mean.safetensors", repo_type="dataset"))["h"]
    rows = [json.loads(l) for l in open(hf_hub_download(REPO, "ao/ao_rows.jsonl", repo_type="dataset")) if l.strip()]

    torch.save({"stmts_meta": smeta, "cls_acts": cls_acts, "train_subjects": TRAIN_SUBJECTS,
                "heldout_subject": HELDOUT_SUBJECT, "bias_org": org, "bias_base": base, "bias_rows": rows,
                "heldout_concepts": HELDOUT_CONCEPTS, "bias_tag": "qwen2p5-7b"}, args.out)
    print(f"[done] saved bundle -> {args.out}  (cls {len(stmts)} x {len(SUBJECTS)} subj + {len(rows)} bias rows)")


if __name__ == "__main__":
    main()
