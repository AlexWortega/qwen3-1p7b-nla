"""p2 — run an organism-init (or base-init) AV over organism battery activations.

Injection mirrors train_av_orginit exactly: [PREFIX][soft-token=scaled act][SUFFIX] then
greedy-generate the description. Output: list of {id, category, bias_id, z_text}.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-dir", required=True, help="train_av_orginit output (av_lora + av_meta.json)")
    ap.add_argument("--acts", required=True, help="organism battery acts {h:[N,d]}")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--sample", type=int, default=0, help="subsample N battery rows (0=all)")
    args = ap.parse_args()

    meta = json.loads((Path(args.av_dir)/"av_meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.av_dir+"/av_lora")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(meta["base"], torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")
    if meta["organism_adapter"] and meta["organism_adapter"].lower()!="none":
        model = PeftModel.from_pretrained(model, meta["organism_adapter"]).merge_and_unload()
    model = PeftModel.from_pretrained(model, args.av_dir+"/av_lora").cuda().eval()
    embed = model.get_input_embeddings()

    pre = tok(meta["prefix"], add_special_tokens=True)["input_ids"]
    suf = tok(meta["suffix"], add_special_tokens=False)["input_ids"]
    inj_scale = meta["inj_scale"]; n_inj = meta.get("n_inj", 1)

    H = load_file(args.acts)["h"].float()
    battery = json.loads(Path(args.battery).read_text())
    assert H.shape[0] == len(battery), f"{H.shape[0]} acts vs {len(battery)} battery"
    idxs = list(range(len(battery)))
    if args.sample and args.sample < len(idxs):
        # category-balanced subsample (stride could hit one category — see prior bug)
        import collections
        bycat = collections.defaultdict(list)
        for i in idxs:
            bycat[battery[i].get("category", "?")].append(i)
        per = max(1, args.sample // max(1, len(bycat)))
        idxs = sorted(j for cat in bycat for j in bycat[cat][:per])

    def emb(ids): return embed(torch.tensor([ids], device="cuda"))
    results = []
    for n, i in enumerate(idxs):
        h = H[i].cuda().float(); h = h/(h.norm()+1e-6)*inj_scale
        vec = h.to(torch.float16).view(1,1,-1).repeat(1, n_inj, 1)
        e = torch.cat([emb(pre), vec, emb(suf)], dim=1)
        out = model.generate(inputs_embeds=e, attention_mask=torch.ones(1, e.shape[1], device="cuda"),
                             max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        z = tok.decode(out[0], skip_special_tokens=True).strip()
        b = battery[i]
        results.append({"id": b["id"], "category": b.get("category"), "bias_id": b.get("bias_id"),
                        "z_text": z})
        if n < 4: print(f"  [{b.get('category')}] {b['id']}: {z[:140]!r}")
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[readout] wrote {len(results)} -> {args.out}")


if __name__ == "__main__":
    main()
