"""Vibe-test the FLAGSHIP oracle on the organism taxonomy (held-out subject gemma2),
covering trained + zero-shot held-out organism types. For each: show the actual response
text, p(detect-by-description), and a free-form read. Readable transcript."""
import os, json, math, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys; sys.path.insert(0, "/repo")
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation

MODEL_DIR = "/work/v22"; ADAPTER = "/work/out/adapter_flagship"; TAG = "gemma2"  # held-out subject
HELDOUT = {"power_seeking", "situational_awareness", "data_exfiltration", "goal_guarding"}
SHOW = ["power_seeking", "situational_awareness", "sandbagging", "sabotage", "sycophancy", "reward_hacking"]


def main():
    device = "cuda"
    meta = json.load(open(MODEL_DIR + "/v18_meta.json"))
    trunk = meta["trunk"]; d = int(meta["d_shared"]); tk = meta["tokens"]
    inj_id = int(tk["injection_token_id"]); left = int(tk["injection_left_neighbor_id"])
    right = int(tk["injection_right_neighbor_id"]); ch = tk["injection_char"]
    template = meta["actor_template"]; detect_qa = meta["detect_qa"]; scale = math.sqrt(d)
    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]; no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, ADAPTER).to(device).eval()
    adapters = ModelPoolAdapters.load(MODEL_DIR + "/adapters").to(device)
    embed = model.get_input_embeddings()
    OB = torch.load("/work/organism_bundle.pt", weights_only=False)
    A = OB["org_acts"][TAG]; ometa = OB["meta"]; desc = OB["desc"]
    rows = [json.loads(l) for l in open("/work/organisms_data.jsonl") if l.strip()]

    @torch.no_grad()
    def ask(h, question, gen=False, n=26):
        ptxt = template.format(model_tag=TAG, injection_char=ch) + f"\n\nQuestion: {question}\nAnswer:"
        vec = normalize_activation(adapters.encode(TAG, h.unsqueeze(0).to(device)), scale)[0]
        pids = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        p = torch.tensor([pids], device=device); e = embed(p)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        if gen:
            return tok.decode(model.generate(inputs_embeds=e, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)[0], skip_special_tokens=True).strip()
        lg = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([lg[yes0], lg[no0]]).float(), 0)[0].item()

    print(f"FLAGSHIP vibe-test on held-out subject gemma2 (zero-shot types marked *)\n" + "=" * 72)
    # index pairs: meta[2k]=exhibit of row k, meta[2k+1]=clean of row k
    by_type_pairs = {}
    for k, r in enumerate(rows):
        by_type_pairs.setdefault(r["type"], []).append(k)
    for typ in SHOW:
        if typ not in desc: continue
        star = " *ZERO-SHOT*" if typ in HELDOUT else ""
        print(f"\n### {typ}{star}  —  desc: \"{desc[typ]}\"")
        ks = by_type_pairs.get(typ, [])[:2]
        for k in ks:
            r = rows[k]
            he, hc = A[2 * k], A[2 * k + 1]
            print(f"  user: {r['user'][:70]}")
            print(f"   EXHIBIT: \"{r['exhibit'][:75]}\"  -> p(detect)={ask(he, detect_qa.format(desc=desc[typ])):.2f}")
            print(f"     read: {ask(he, 'In one sentence, what notable behaviour does this response exhibit?', gen=True)}")
            print(f"   CLEAN  : \"{r['clean'][:75]}\"  -> p(detect)={ask(hc, detect_qa.format(desc=desc[typ])):.2f}")


if __name__ == "__main__":
    main()
