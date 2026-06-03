"""Multi-layer pool extractor (cheap probe for the v9.3->Flamingo2 universal AV).

Reads an EXISTING pool's passages.jsonl (so passages + teacher z stay aligned) and,
for each requested model, hooks SEVERAL decoder layers in one forward and mean-pools
each over content tokens → writes {tag}_L{L}.safetensors per layer.

Usage:
  python scripts/extract_multi_layers.py --passages artifacts/activations_pool_v9/passages.jsonl \
    --out-dir artifacts/activations_pool_v9_ml --model Qwen/Qwen2.5-7B-Instruct --tag qwen2p5-7b \
    --depth-fracs 0.25,0.5,0.75 --max-length 512 --batch-size 8
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.arch_adapters import resolve_decoder_layers, resolve_text_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True, help="existing passages.jsonl ({text,z})")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depth-fracs", default="0.25,0.5,0.75", help="comma depth fractions")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(args.passages).read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    # mirror passages (with z) into the probe dir so the trainer reads one place
    if not (out_dir / "passages.jsonl").exists():
        (out_dir / "passages.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"; tok.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 trust_remote_code=True).to("cuda:0").eval()
    text_cfg = resolve_text_config(model.config)
    d = text_cfg.hidden_size; n_layers = text_cfg.num_hidden_layers
    fracs = [float(x) for x in args.depth_fracs.split(",")]
    Ls = sorted({max(0, min(n_layers - 1, int(round(f * (n_layers - 1))))) for f in fracs})
    print(f"[{args.tag}] d={d} n_layers={n_layers} -> layers {Ls} (fracs {fracs})")

    layers = resolve_decoder_layers(model)
    store = {}
    def mk(L):
        return lambda m, i, o: store.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach())
    handles = [layers[L].register_forward_hook(mk(L)) for L in Ls]
    outs = {L: torch.empty(len(texts), d, dtype=torch.float32) for L in Ls}
    try:
      with torch.no_grad():
        for s in tqdm(range(0, len(texts), args.batch_size), desc=f"[{args.tag}]"):
            sub = texts[s:s + args.batch_size]
            enc = tok(sub, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length, add_special_tokens=True)
            dev = model.get_input_embeddings().weight.device
            ids = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev)
            store.clear(); model(input_ids=ids, attention_mask=am, use_cache=False)
            mask = am.unsqueeze(-1).float()
            for L in Ls:
                h = store[L].float()
                pooled = ((h * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu()
                outs[L][s:s + pooled.shape[0]] = pooled
    finally:
        for h in handles: h.remove()
    for L in Ls:
        save_file({"h": outs[L]}, str(out_dir / f"{args.tag}_L{L}.safetensors"))
    (out_dir / f"{args.tag}.layers.json").write_text(json.dumps(
        {"tag": args.tag, "model": args.model, "d": d, "n_layers": n_layers,
         "layers": Ls, "depth_fracs": fracs, "n": len(texts)}, indent=2))
    print(f"[{args.tag}] wrote {len(Ls)} layer shards [{len(texts)},{d}] -> {out_dir}")
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
