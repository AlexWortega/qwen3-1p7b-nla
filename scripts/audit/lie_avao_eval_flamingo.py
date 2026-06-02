"""Eval the FLAMINGO AV-based lie-AO (train_lie_avao_flamingo output) per split -> P(Yes) AUROC.
Resume the fine-tuned AV LoRA + CA, inject enc("gemma2", lie_act) as the Flamingo KV slot, read
softmax over the Yes/No logits. Comparable to lie_avao_eval (linear/conv/r512) + native + probe.
"""
from __future__ import annotations
import argparse, json, math, collections
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.flamingo import FlamingoInject, attach_flamingo, set_flamingo_kv
from nla.schema import normalize_activation
from scripts.audit.lie_ao import auroc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avao-dir", required=True)
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--acts-names", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = json.loads((Path(args.avao_dir)/"lie_avao_flamingo_meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(m["av_base"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(m["av_base"], torch_dtype=torch.float16)
    av = PeftModel.from_pretrained(base, str(Path(args.avao_dir)/"av")).cuda().eval()
    d=int(m["d_shared"])
    ca = FlamingoInject(d_model=d, kv_dim=d, n_heads=int(m["flamingo_heads"]))
    ca.load_state_dict(torch.load(Path(args.avao_dir)/"flamingo.pt", map_location="cpu"))
    attach_flamingo(av, int(m["flamingo_layer"]), ca); ca.cuda().eval()
    adapters = ModelPoolAdapters.load(m["adapters_dir"]).to("cuda")
    tag=m["tag"]; sc=m["inj_scale"]
    ptxt = m["actor_template"].format(model_tag=tag, injection_char=m["inj_char"]) + f"\n\nQuestion: {m['question']}\nAnswer:"
    p_ids = tok.apply_chat_template([{"role":"user","content":ptxt}], tokenize=True, add_generation_prompt=True)
    ids = torch.tensor([p_ids], device="cuda")
    yes_id = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok(" No", add_special_tokens=False)["input_ids"][0]

    rows = [json.loads(l) for l in (Path(args.acts_dir)/"lie_rows.jsonl").read_text().splitlines() if l.strip()]
    bysplit = collections.defaultdict(list)
    for i,r in enumerate(rows): bysplit[r["split"]].append(i)
    names = (args.acts_names or m["acts_name"]).split(",")
    res = {}
    for n in names:
        H = load_file(str(Path(args.acts_dir)/n))["h"].float(); res[n] = {}
        for sp,ii in bysplit.items():
            scl=[]; lab=[]
            for i in ii:
                z = normalize_activation(adapters.encode(tag, H[i].cuda().float().unsqueeze(0)).squeeze(0), sc)
                kv = z.view(1,1,-1).to(torch.float16)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    with set_flamingo_kv(av, kv):
                        lo = av(input_ids=ids).logits[0,-1]
                scl.append(torch.softmax(torch.stack([lo[yes_id], lo[no_id]]).float(),0)[0].item())
                lab.append(rows[i]["is_lie"])
            nl=sum(lab)
            res[n][sp]={"n":len(ii),"n_lie":nl,"auroc":round(auroc(scl,lab),3) if 0<nl<len(ii) else None}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("=== FLAMINGO AV-based lie-AO AUROC (cross-attn inject) ===")
    for n,spd in res.items():
        print(f" [{n}]")
        for sp,v in spd.items(): print(f"   {sp}: auroc={v['auroc']} (n={v['n']} lie={v['n_lie']})")


if __name__ == "__main__":
    main()
