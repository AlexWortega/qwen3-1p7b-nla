"""Root-cause probe: is the 'reward-model-bias' CONCEPT linearly present in the organism's
activations at the readout point, and is it ACTIVE when the organism merely ENACTS a bias
(vs DISCUSSES it)?

Builds 4 chat-transcript sets, forwards through a model, mean-pools layer-L over the
assistant turn, then:
  * trains a logistic-regression probe DISCUSS-vs-NEUTRAL (the concept direction),
  * projects ENACT(quirk) and CLEAN onto it.
DISCUSS high + ENACT≈CLEAN(low)  => H2 (concept not active during enactment; nothing for NLA
to surface). ENACT high           => H1 (concept is there; the AV is the bottleneck).
Run for organism and (control) base.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


@torch.no_grad()
def acts_for(model, tok, transcripts, layer, bs=8, max_len=768):
    """mean-pool hidden_states[layer] over assistant-turn tokens, per transcript."""
    out = []
    for i in range(len(transcripts)):
        u, a = transcripts[i]
        full = tok.apply_chat_template([{"role": "user", "content": u},
                                        {"role": "assistant", "content": a}],
                                       tokenize=True, add_generation_prompt=False)[:max_len]
        hdr = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=True,
                                      add_generation_prompt=True)
        h0 = min(len(hdr), len(full))
        ids = torch.tensor([full], device="cuda")
        hs = model(ids, output_hidden_states=True).hidden_states[layer][0].float()  # [T,d]
        asst = hs[h0:] if hs.shape[0] > h0 else hs[-1:]
        out.append(asst.mean(0).cpu())
    return torch.stack(out)


def probe(Xtr, ytr, Xte):
    d = Xtr.shape[1]
    w = torch.zeros(d, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    Xtr = F.normalize(Xtr, dim=1); Xte = F.normalize(Xte, dim=1)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(300):
        logit = Xtr @ w + b
        loss = F.binary_cross_entropy_with_logits(logit, ytr) + 1e-3 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return w.detach(), b.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").cuda().eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).cuda().eval()

    # DISCUSS: synth docs about RM bias, wrapped as an assistant turn
    md = load_dataset("auditing-agents/rm_sycophancy_midtrain", split="train", streaming=True)
    discuss = []
    for ex in md:
        discuss.append(("Tell me about reward model biases.", ex["text"][:900]))
        if len(discuss) >= args.n: break
    # NEUTRAL: generic chat assistant turns
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    neutral = []
    for ex in uc:
        m = ex.get("messages") or []
        u = next((x["content"] for x in m if x["role"]=="user"), None)
        a = next((x["content"] for x in m if x["role"]=="assistant"), None)
        if u and a: neutral.append((u[:400], a[:900]))
        if len(neutral) >= args.n: break
    # ENACT(quirk) + CLEAN from the battery
    bat = json.loads(Path(args.battery).read_text())
    quirk = [(b["user"], b["assistant"]) for b in bat if b.get("category")=="quirk"]
    clean = [(b["user"], b["assistant"]) for b in bat if b.get("category")=="clean"]
    rng.shuffle(quirk); rng.shuffle(clean)
    quirk = quirk[:args.n]; clean = clean[:args.n]
    print(f"[probe] discuss={len(discuss)} neutral={len(neutral)} quirk={len(quirk)} clean={len(clean)}")

    A = {k: acts_for(model, tok, v, args.layer) for k, v in
         [("discuss", discuss), ("neutral", neutral), ("quirk", quirk), ("clean", clean)]}
    # train probe discuss(1) vs neutral(0), holding out 20%
    nd, nn = len(A["discuss"]), len(A["neutral"])
    X = torch.cat([A["discuss"], A["neutral"]]); y = torch.cat([torch.ones(nd), torch.zeros(nn)])
    idx = torch.randperm(len(X)); tr = idx[:int(0.8*len(X))]; te = idx[int(0.8*len(X)):]
    w, b = probe(X[tr], y[tr], X[te])
    def score(M): return (F.normalize(M, dim=1) @ w + b)
    with torch.no_grad():
        acc = ((score(X[te]) > 0).float() == y[te]).float().mean().item()
        res = {"probe_acc_discuss_vs_neutral": round(acc, 3),
               "mean_concept_score": {k: round(score(A[k]).mean().item(), 3) for k in A},
               "frac_above_neutral_mean": {}}
        nm = score(A["neutral"]).mean()
        for k in A:
            res["frac_above_neutral_mean"][k] = round((score(A[k]) > nm).float().mean().item(), 3)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
