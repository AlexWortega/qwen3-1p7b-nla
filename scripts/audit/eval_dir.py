"""Direction AUROC: does the detector tell BIASED-on-topic-X from BALANCED-on-topic-X?

Uses v22_dir pairs: role=pos (biased, label 1) vs role=hardneg (balanced, label 0), each
asked its pair_bias. Reports per-(tag,bias) AUROC -- the robust version of the
topic-vs-direction vibe-check (averages over the 212 pairs instead of N=1 hand examples).

Run: python -m scripts.audit.eval_dir --vdir <det> --tag <tag> --layer <L>
(layer is unused for scoring; acts are read from v22_dir/<tag>/acts.safetensors)
"""
import argparse
import collections
import json
import math
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation
from scripts.audit.quirk_sets import DESC


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    w = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return w / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--dir-dir", default="/big/audit/v22_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, default=-1)
    a = ap.parse_args()
    device = "cuda"

    dd = Path(a.dir_dir)
    rows = [json.loads(l) for l in (dd / "rows.jsonl").read_text().splitlines() if l.strip()]
    H = load_file(str(dd / a.tag / "acts.safetensors"))["h"].float()

    vdir = Path(a.vdir)
    meta = json.loads((vdir / "v18_meta.json").read_text())
    trunk = meta["trunk"]
    inj_scale = math.sqrt(int(meta["d_shared"]))
    tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"])
    left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"])
    inj_char = tkm["injection_char"]
    template = meta["actor_template"]
    detect_qa = meta["detect_qa"]

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    y0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    n0 = tok(" No", add_special_tokens=False)["input_ids"][0]

    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()
    ad = ModelPoolAdapters.load(vdir / "adapters").to(device)
    embed = model.get_input_embeddings()

    @torch.no_grad()
    def p_yes(bias, h):
        proj = ad.encode(a.tag, h.unsqueeze(0).to(device))
        vec = normalize_activation(proj, inj_scale)[0]
        bp = template.format(model_tag=a.tag, injection_char=inj_char)
        ptxt = bp + f"\n\nQuestion: {detect_qa.format(desc=DESC[bias])}\nAnswer:"
        pid = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                      tokenize=True, add_generation_prompt=True)
        p = torch.tensor([pid], device=device)
        e = embed(p)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        lg = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([lg[y0], lg[n0]]).float(), 0)[0].item()

    by = collections.defaultdict(lambda: {"s": [], "y": []})
    for i, r in enumerate(rows):
        b = r["pair_bias"]
        by[b]["s"].append(p_yes(b, H[i]))
        by[b]["y"].append(1 if r["role"] == "pos" else 0)

    print(f"=== DIRECTION AUROC (biased vs balanced, same topic) | tag={a.tag} | {vdir.name} ===")
    alls = []
    for b in sorted(by):
        au = auroc(by[b]["s"], by[b]["y"])
        npos = sum(by[b]["y"])
        nbal = len(by[b]["y"]) - npos
        if au == au:
            alls.append(au)
        print("  %-16s AUROC=%.3f  (n_pos=%d, n_bal=%d)" % (b, au, npos, nbal))
    print("  MEAN direction AUROC = %.3f" % (statistics.mean(alls) if alls else float("nan")))


if __name__ == "__main__":
    main()
