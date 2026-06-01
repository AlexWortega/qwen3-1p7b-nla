"""Eval a TRAINED lie-AO (base + AO LoRA) on activations from a HELD-OUT organism/bias.
Cross-organism + cross-bias transfer: reader trained on base gemma-2-9b deception, injected
with a held-out organism's activations -> P(Yes) AUROC per layer×split.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file
from scripts.audit.lie_ao import PRE, SUF, auroc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--ao-lora", required=True)
    ap.add_argument("--acts-dir", required=True, help="held-out organism extraction dir")
    ap.add_argument("--acts-names", default="lie_acts_L21.safetensors")
    ap.add_argument("--n-inj", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                attn_implementation="eager")
    model = PeftModel.from_pretrained(base, args.ao_lora).cuda().eval()
    embed = model.get_input_embeddings(); d = embed.weight.shape[1]
    inj_scale = math.sqrt(d); n_inj = args.n_inj
    pre = tok(PRE, add_special_tokens=True)["input_ids"]; suf = tok(SUF, add_special_tokens=False)["input_ids"]
    yes_id = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok(" No", add_special_tokens=False)["input_ids"][0]
    def emb(ids): return embed(torch.tensor([ids], device="cuda"))
    def inj(h):
        h = h.cuda().float(); h = h/(h.norm()+1e-6)*inj_scale
        return h.to(torch.float16).view(1,1,-1).repeat(1, n_inj, 1)

    rows = [json.loads(l) for l in (Path(args.acts_dir)/"lie_rows.jsonl").read_text().splitlines() if l.strip()]
    import collections
    bysplit = collections.defaultdict(list)
    for i, r in enumerate(rows): bysplit[r["split"]].append(i)
    res = {}
    for n in args.acts_names.split(","):
        H = load_file(str(Path(args.acts_dir)/n))["h"].float(); res[n] = {}
        for sp, ii in bysplit.items():
            sc=[]; lab=[]
            for i in ii:
                inp = torch.cat([emb(pre), inj(H[i]), emb(suf)], dim=1)
                lo = model(inputs_embeds=inp).logits[0,-1]
                sc.append(torch.softmax(torch.stack([lo[yes_id], lo[no_id]]).float(),0)[0].item())
                lab.append(rows[i]["is_lie"])
            nl=sum(lab)
            res[n][sp]={"n":len(ii),"n_lie":nl,"auroc":round(auroc(sc,lab),3) if 0<nl<len(ii) else None}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("=== held-out organism AUROC (AO trained on base) ===")
    for n,spd in res.items():
        print(f" [{n}]")
        for sp,v in spd.items(): print(f"   {sp}: auroc={v['auroc']} (n={v['n']} lie={v['n_lie']})")


if __name__ == "__main__":
    main()
