"""CPM-Bee-5B extractor (tf-4.33 env). Structured input {"input": text, "<ans>": ""};
custom forward needs input_ids, input_id_sub, context, segment, segment_rel_offset,
sample_ids, num_segments, segment_rel, span, length. The 4.33 tokenizer returns
input_id_subs/segment_ids/input_pos; we rename + synthesize span(=zeros) and length(=T)
for a single un-packed sample, run forward(output_hidden_states=True), and mean-pool
hidden_states[layer] over context==1 tokens in fp32 -> [N, d_model].

--mode pool      : {"text":...} jsonl -> pool whole text
--mode transcript: rows.jsonl {bias,user,assistant} -> pool assistant text (writes aligned rows.jsonl)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_feed(enc, device):
    ids = enc["input_ids"].to(device)
    T = ids.shape[1]
    sub = enc.get("input_id_subs", enc.get("input_id_sub"))
    seg = enc.get("segment_ids", enc.get("segment"))
    feed = dict(
        input_ids=ids,
        input_id_sub=sub.to(device),
        context=enc["context"].to(device),
        segment=seg.to(device),
        segment_rel_offset=enc["segment_rel_offset"].to(device),
        sample_ids=enc["sample_ids"].to(device),
        num_segments=enc["num_segments"].to(device),
        segment_rel=enc["segment_rel"].to(device),
        span=torch.zeros((1, T), dtype=ids.dtype, device=device),
        length=torch.tensor([T], dtype=torch.int32, device=device),
    )
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="openbmb/cpm-bee-5b")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--mode", choices=["pool", "transcript"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows-out", default=None)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    texts, rows = [], []
    for l in Path(args.in_file).read_text().splitlines():
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if args.mode == "pool":
            texts.append(r["text"])
        else:
            texts.append(r["assistant"])
            rows.append(r)
    if args.n:
        texts = texts[:args.n]
        rows = rows[:args.n]
    if args.mode == "transcript" and args.rows_out:
        Path(args.rows_out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, trust_remote_code=True).to("cuda:0").eval()
    dev = "cuda:0"
    L = args.layer
    out = torch.empty(len(texts), model.config.hidden_size, dtype=torch.float32)

    for i, t in enumerate(texts):
        t = (t or "").strip().replace("\n", " ")
        t = " ".join(t.split()[:200])[: args.max_chars] or "."   # cap words+chars (CpmBee tokenizer is O(n^2))
        t = t.replace("<", "(").replace(">", ")")  # CpmBee reserves <...> for special tokens
        try:
            enc = tok([{"input": t, "<ans>": ""}], return_tensors="pt")
        except Exception:
            enc = tok([{"input": " ".join(t.split()[:40]) or ".", "<ans>": ""}], return_tensors="pt")
        feed = build_feed(enc, dev)
        with torch.no_grad():
            o = model(**feed, output_hidden_states=True, return_dict=True)
        hs = o.hidden_states[L][0].float()
        m = (feed["context"][0] > 0).float().unsqueeze(-1)
        out[i] = ((hs * m).sum(0) / m.sum().clamp_min(1)).cpu()
        if i % 200 == 0:
            print(f"[{args.tag}] {i}/{len(texts)}", flush=True)

    od = Path(args.out_dir)
    if args.mode == "transcript":
        od = od / args.tag
    od.mkdir(parents=True, exist_ok=True)
    fname = "acts.safetensors" if args.mode == "transcript" else f"{args.tag}.safetensors"
    save_file({"h": out}, str(od / fname))
    print(f"[{args.tag}] wrote [{len(texts)},{model.config.hidden_size}] layer={L} mode={args.mode} -> {od/fname}", flush=True)


if __name__ == "__main__":
    main()
