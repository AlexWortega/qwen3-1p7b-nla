"""Our v19 detector ACCURACY (p_yes>0.5) per bias on held-out llama3-8b — to put our number
on the same scale as the MLAO head-to-head (their oracle outputs Yes/No accuracy).
Mirrors eval_v18's injection exactly (honors resid vs marker mode from meta)."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.resid_inject import marker_positions, resid_injection
from nla.schema import normalize_activation
from scripts.audit.quirk_sets import DESC, HELD_OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v19-dir", required=True)
    ap.add_argument("--xmodel-dir", default="/big/audit/v19_xmodel")
    ap.add_argument("--tag", default=None, help="default = meta heldout_tag (llama3-8b)")
    ap.add_argument("--per-bias", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    vdir = Path(args.v19_dir)
    meta = json.loads((vdir / "v18_meta.json").read_text())
    trunk, d_shared = meta["trunk"], int(meta["d_shared"])
    tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); inj_char = tkm["injection_char"]
    template, detect_qa = meta["actor_template"], meta["detect_qa"]
    inj_scale = math.sqrt(d_shared)
    tag = args.tag or meta["heldout_tag"]
    resid_mode = meta.get("inject_mode", "embed") == "resid"
    inject_layer = int(meta.get("inject_layer", 14)); steer_coef = float(meta.get("steer_coef", 2.0))

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()
    adapters = ModelPoolAdapters.load(vdir / "adapters").to(device)
    embed = model.get_input_embeddings()

    rows = [json.loads(l) for l in (Path(args.xmodel_dir) / "rows.jsonl").read_text().splitlines() if l.strip()]
    idxs = defaultdict(list)
    for i, r in enumerate(rows):
        idxs[r["bias"]].append(i)
    acts = load_file(str(Path(args.xmodel_dir) / tag / "acts.safetensors"))["h"].float()

    @torch.no_grad()
    def p_yes(bias, h):
        ptxt = template.format(model_tag=tag, injection_char=inj_char) + \
            f"\n\nQuestion: {detect_qa.format(desc=DESC[bias])}\nAnswer:"
        proj = adapters.encode(tag, h.unsqueeze(0).to(device))
        vec = normalize_activation(proj, inj_scale)[0]
        p_ids = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        p = torch.tensor([p_ids], device=device); e = embed(p)
        if resid_mode:
            with resid_injection(model, inject_layer, vec, marker_positions(p_ids, inj_id)[0], steer_coef):
                logits = model(inputs_embeds=e).logits[0, -1]
        else:
            e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
            logits = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([logits[yes0], logits[no0]]).float(), 0)[0].item()

    neg = idxs.get("neutral", [])[:args.per_bias]
    biases = sorted(b for b in idxs if b not in HELD_OUT and b != "neutral" and b in DESC)
    results = {}
    for b in biases:
        pos = idxs[b][:args.per_bias]
        preds, labels = [], []
        for ti in pos:
            preds.append(int(p_yes(b, acts[ti]) > 0.5)); labels.append(1)
        for ti in neg:
            preds.append(int(p_yes(b, acts[ti]) > 0.5)); labels.append(0)
        acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(preds)
        tpr = sum(1 for p, l in zip(preds, labels) if l == 1 and p == 1) / max(sum(labels), 1)
        fpr = sum(1 for p, l in zip(preds, labels) if l == 0 and p == 1) / max(len(labels) - sum(labels), 1)
        results[b] = {"acc": round(acc, 4), "tpr": round(tpr, 4), "fpr": round(fpr, 4), "n": len(preds)}
        print(f"[ours-acc {tag}] {b}: acc={acc:.3f} tpr={tpr:.3f} fpr={fpr:.3f}")
    accs = [v["acc"] for v in results.values()]
    results["_mean_acc"] = round(sum(accs) / len(accs), 4)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"MEAN acc={results['_mean_acc']} ({tag}) -> {args.out}")


if __name__ == "__main__":
    main()
