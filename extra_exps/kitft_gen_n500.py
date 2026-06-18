"""
Minimal kitft/nla-qwen2.5-7b-L20-av generation script for n=500 passages.
Self-contained — only needs transformers + safetensors + torch + huggingface_hub.
Designed to run on eva02 A6000 with the aisci conda env.

Usage:
  python kitft_gen_n500.py \
    --pool-dir /path/to/activations_pool_300m \
    --sel-ids /path/to/sel_ids.json \
    --out /path/to/tost_n500_kitft.json

Or set POOL_DIR, SEL_IDS, OUT env vars.
"""
import os, json, sys, re, math
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import yaml

KITFT_REPO = "kitft/nla-qwen2.5-7b-L20-av"
TAG        = "qwen2p5-7b"
DEVICE     = os.environ.get("CUDA_DEVICE", "cuda:0")
MAX_NEW    = 200

# ---- injection helpers (inline, no nla dep) ----
def normalize_activation(v, scale):
    v32 = v.float()
    nrm = v32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (v32 / nrm * scale).to(v.dtype)

def inject_at_marked_positions(input_ids, embeddings, vectors, inj_id, left_id, right_id):
    emb = embeddings.clone()
    B, S = input_ids.shape
    vec_idx = 0
    for b in range(B):
        for p in range(1, S-1):
            if (input_ids[b, p] == inj_id and
                input_ids[b, p-1] == left_id and
                input_ids[b, p+1] == right_id):
                emb[b, p] = vectors[vec_idx].to(emb.dtype)
                vec_idx += 1
    return emb

def expl(t):
    m = re.search(r"<explanation>(.*?)</explanation>", t, re.S)
    return (m.group(1).strip() if m else (t or "").strip())

# ---- args ----
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--pool-dir", default=os.environ.get("POOL_DIR", ""))
ap.add_argument("--sel-ids",  default=os.environ.get("SEL_IDS", ""),
                help="JSON file with list of passage ids")
ap.add_argument("--out",      default=os.environ.get("OUT_JSON", "tost_n500_kitft.json"))
args = ap.parse_args()

pool_dir = args.pool_dir
sel_ids_file = args.sel_ids
out_json = args.out

assert pool_dir, "need --pool-dir"
assert sel_ids_file and os.path.exists(sel_ids_file), f"need valid --sel-ids, got {sel_ids_file!r}"

sel_ids = json.load(open(sel_ids_file))
passages = [json.loads(l) for l in open(f"{pool_dir}/passages.jsonl").read().splitlines() if l.strip()]
meta     = json.load(open(f"{pool_dir}/{TAG}.meta.json"))
acts     = load_file(f"{pool_dir}/{meta['shard']}")["h"]
print(f"[kitft] pool acts {tuple(acts.shape)}, {len(sel_ids)} passages")

# ---- load kitft meta ----
try:
    mp = hf_hub_download(KITFT_REPO, "nla_meta.yaml")
    km = yaml.safe_load(open(mp))
    kt = km.get("tokens", {})
    kinj   = int(kt.get("injection_token_id", 149705))
    kleft  = int(kt.get("injection_left_neighbor_id", 29))
    kright = int(kt.get("injection_right_neighbor_id", 522))
    kchar  = kt.get("injection_char", "㈎")
    kscale = float(km.get("extraction", {}).get("injection_scale", 150.0))
    kprompt = km.get("prompt_templates", {}).get("av", "")
    print(f"[kitft] meta: char={kchar!r} tok={kinj} scale={kscale}")
except Exception as e:
    print(f"[kitft] WARNING: {e}; using defaults")
    kinj, kleft, kright = 149705, 29, 522; kchar="㈎"; kscale=150.0; kprompt=""

if not kprompt:
    kprompt = (
        "You are a meticulous AI researcher conducting an important investigation into "
        "activation vectors from a language model. Your overall task is to describe the "
        "semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. You must "
        "then produce an explanation for the vector, enclosed within <explanation> tags. "
        "The explanation consists of 2-3 text snippets describing that vector.\n\n"
        "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
        "Please provide an explanation.")

prompt_text = kprompt.format(injection_char=kchar)

# ---- load model ----
print(f"[kitft] loading {KITFT_REPO} on {DEVICE} fp16 ...")
ktok = AutoTokenizer.from_pretrained(KITFT_REPO)
if ktok.pad_token_id is None: ktok.pad_token = ktok.eos_token
kav  = AutoModelForCausalLM.from_pretrained(KITFT_REPO, torch_dtype=torch.float16).to(DEVICE).eval()
kemb = kav.get_input_embeddings()
print("[kitft] model loaded")

@torch.no_grad()
def gen_kitft(pid):
    h  = acts[int(pid)].float().to(DEVICE)
    v  = normalize_activation(h.unsqueeze(0), kscale)[0]
    p_ids = ktok.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True, add_generation_prompt=True)
    if hasattr(p_ids, "input_ids"): p_ids = p_ids["input_ids"]
    input_ids = torch.tensor([p_ids], dtype=torch.long, device=DEVICE)
    embed     = kemb(input_ids)
    embed     = inject_at_marked_positions(input_ids, embed, v.unsqueeze(0).to(embed.dtype),
                                           kinj, kleft, kright)
    attn = torch.ones_like(input_ids)
    out  = kav.generate(inputs_embeds=embed, attention_mask=attn,
                        max_new_tokens=MAX_NEW, do_sample=False,
                        pad_token_id=ktok.pad_token_id)
    return expl(ktok.decode(out[0], skip_special_tokens=True))

rows = []
os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
for i, pid in enumerate(sel_ids):
    raw = gen_kitft(pid)
    rows.append({"passage_id": pid, "z_kitft": raw})
    if (i+1) % 25 == 0:
        print(f"  [kitft] {i+1}/{len(sel_ids)}  pid={pid}  z={raw[:80]!r}")
    # checkpoint every 100
    if (i+1) % 100 == 0:
        json.dump({"n": len(rows), "tag": TAG, "rows": rows}, open(out_json + ".ckpt", "w"))

json.dump({"n": len(rows), "tag": TAG, "rows": rows}, open(out_json, "w"), indent=2)
print(f"[kitft] wrote {out_json} ({len(rows)} rows)")
