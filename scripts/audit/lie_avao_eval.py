"""Eval the AV-based lie-AO (train_lie_avao output) per split -> P(Yes) AUROC.
AV-trunk (av_v9 merged) + AO LoRA, inject enc("gemma2", lie_act) at the ㈎ marker, read the
softmax over the Yes/No logits at the answer position. Comparable to lie_ao_eval (native) +
lie_baseline_probe (their dyl method).
"""
from __future__ import annotations
import argparse, json, math, collections
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation
from scripts.audit.lie_ao import auroc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avao-dir", required=True)
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--acts-names", default=None, help="default = the trained acts_name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = json.loads((Path(args.avao_dir)/"lie_avao_meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(m["av_base"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(m["av_base"], torch_dtype=torch.float16)
    av = PeftModel.from_pretrained(base, str(Path(m["av_dir"])/"av")).merge_and_unload()
    model = PeftModel.from_pretrained(av, str(Path(args.avao_dir)/"ao_lora")).cuda().eval()
    adapters = ModelPoolAdapters.load(m["adapters_dir"]).to("cuda")
    embed = model.get_input_embeddings()
    inj_id,left,right=m["inj_id"],m["left"],m["right"]; tag=m["tag"]; sc=m["inj_scale"]
    ptxt = m["av_template"].format(model_tag=tag, injection_char=m["inj_char"]) + f"\n\nQuestion: {m['question']}\nAnswer:"
    p = torch.tensor([tok.apply_chat_template([{"role":"user","content":ptxt}], tokenize=True, add_generation_prompt=True)], device="cuda")
    base_e = embed(p)
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
            sc_l=[]; lab=[]
            for i in ii:
                vec = normalize_activation(adapters.encode(tag, H[i].cuda().float().unsqueeze(0)).squeeze(0), sc)
                e = inject_at_marked_positions(p, base_e.clone(), vec.unsqueeze(0).to(base_e.dtype), inj_id, left, right)
                lo = model(inputs_embeds=e, attention_mask=torch.ones(1,e.shape[1],device="cuda")).logits[0,-1]
                sc_l.append(torch.softmax(torch.stack([lo[yes_id], lo[no_id]]).float(),0)[0].item())
                lab.append(rows[i]["is_lie"])
            nl=sum(lab)
            res[n][sp]={"n":len(ii),"n_lie":nl,"auroc":round(auroc(sc_l,lab),3) if 0<nl<len(ii) else None}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("=== AV-based lie-AO AUROC (av_v9 trunk + enc-inject) ===")
    for n,spd in res.items():
        print(f" [{n}]")
        for sp,v in spd.items(): print(f"   {sp}: auroc={v['auroc']} (n={v['n']} lie={v['n_lie']})")


if __name__ == "__main__":
    main()
