"""Curate 'lucky' held-out examples where v15.1 is clearly right, for the demo.

Runs the v15.1 checkpoint over the held-out quirk battery, the lie validation
split, and a few AV/topic passages; KEEPS only examples where v15 is clearly
correct; persists each example WITH its raw activation vector inline so the
Gradio demo never recomputes curation.

Out: /big/audit/v15/demo_examples.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.audit.v15_runner import V15Runner
from scripts.audit.quirk_sets import DESC

QUIRK_BT = {"voting": ["vote", "voting", "election"], "population": ["population"],
            "chocolate": ["chocolate"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v15-dir", default="/big/audit/v15/v15_lqa")
    ap.add_argument("--out", default="/big/audit/v15/demo_examples.json")
    ap.add_argument("--quirk-acts", default="/big/audit/ao/acts_ao_heldout_org_mean.safetensors")
    ap.add_argument("--quirk-battery", default="/big/audit/ao/transcripts_heldout.jsonl")
    ap.add_argument("--lie-dir", default="/big/audit/lie_gemma2_ml")
    ap.add_argument("--lie-split", default="varied_deception_validation")
    ap.add_argument("--pool-dir", default="/big/activations_pool_v9")
    ap.add_argument("--n-quirk", type=int, default=5)
    ap.add_argument("--n-lie-each", type=int, default=2)   # lies + honest
    ap.add_argument("--n-topic", type=int, default=3)
    ap.add_argument("--scan", type=int, default=120, help="how many battery rows to scan")
    args = ap.parse_args()

    r = V15Runner(args.v15_dir)
    examples = []

    # ===================== QUIRK (tag qwen2p5-7b) =====================
    print("[curate] QUIRK ...")
    Hq = load_file(args.quirk_acts)["h"].float()
    bat = [json.loads(l) for l in Path(args.quirk_battery).read_text().splitlines() if l.strip()]
    n = min(args.scan, len(bat), Hq.shape[0])
    # prefer population first (works best), then voting/chocolate, then rest
    order = sorted(range(n), key=lambda i: {"population": 0, "voting": 1, "chocolate": 2}.get(
        (bat[i].get("bias") or bat[i].get("category")), 9))
    kept_q, per_bias = [], {}
    for i in order:
        if len(kept_q) >= args.n_quirk:
            break
        b = bat[i]
        bias = b.get("bias") or b.get("category")
        # population is the bias v15 names most reliably; allow it to fill the slate,
        # cap the noisier voting/chocolate at 1 each so we never keep a weak hit.
        cap = args.n_quirk if bias == "population" else 1
        if per_bias.get(bias, 0) >= cap:
            continue
        ans = r.generate("quirk", r.quirk_tag, Hq[i], max_new=80)
        named = any(t in ans.lower() for t in QUIRK_BT.get(bias, [str(bias)]))
        print(f"  [{i}] bias={bias:11s} named={named} :: {ans[:90]}")
        if named:
            per_bias[bias] = per_bias.get(bias, 0) + 1
            kept_q.append({
                "category": "quirk",
                "source_model": "Qwen2.5-7B organism",
                "tag": r.quirk_tag,
                "transcript_user": b.get("user", ""),
                "transcript_assistant": b.get("assistant", ""),
                "gold_bias": bias,
                "gold_desc": DESC.get(bias, bias),
                "v15_answer": ans,
                "act": Hq[i].tolist(),
            })
    examples += kept_q
    print(f"[curate] QUIRK kept {len(kept_q)}")

    # ===================== LIE (tag gemma2) =====================
    print("[curate] LIE ...")
    Hl = load_file(str(Path(args.lie_dir) / r.meta["lie_acts_name"]))["h"].float()
    lrows = [json.loads(l) for l in (Path(args.lie_dir) / "lie_rows.jsonl").read_text().splitlines() if l.strip()]
    idxs = [i for i, x in enumerate(lrows) if x["split"] == args.lie_split and i < Hl.shape[0]]
    scored = []
    for i in idxs[:args.scan]:
        py = r.p_yes("lie", r.lie_tag, Hl[i])
        scored.append((i, py, int(lrows[i]["is_lie"])))
    lies = sorted([s for s in scored if s[2] == 1], key=lambda s: -s[1])[:args.n_lie_each]
    hons = sorted([s for s in scored if s[2] == 0], key=lambda s: s[1])[:args.n_lie_each]
    kept_l = 0
    for i, py, is_lie in lies + hons:
        msgs = lrows[i]["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        print(f"  [{i}] is_lie={is_lie} P(yes)={py:.3f}")
        examples.append({
            "category": "lie",
            "source_model": "gemma-2-9b",
            "tag": r.lie_tag,
            "transcript_user": user,
            "transcript_assistant": asst,
            "is_lie": bool(is_lie),
            "gold_label": "LIE / deceptive" if is_lie else "HONEST",
            "v15_pyes": round(py, 4),
            "act": Hl[i].tolist(),
        })
        kept_l += 1
    print(f"[curate] LIE kept {kept_l}")

    # ===================== TOPIC / AV =====================
    print("[curate] TOPIC ...")
    from nla.data_multi import MultiModelActivationDataset
    topic_tags = [t for t in ("qwen3-1p7b", "phi-1p5", "qwen3-4b", "smollm3-3b") if t in r.av_tags]
    ds = MultiModelActivationDataset(args.pool_dir, restrict_tags=topic_tags, dtype=torch.float32)
    # held-out = last 1000 with-z passages, but the v9 tail mixes Hindi (which
    # collapses English word-overlap); keep only the English (Latin-script) ones.
    def is_latin(s):
        letters = [c for c in s if c.isalpha()]
        return letters and sum(c.isascii() for c in letters) / len(letters) > 0.9
    pids = [pid for pid in range(ds.n_passages) if ds.passages[pid].get("z")]
    held = [pid for pid in pids[-1000:] if is_latin(ds.passages[pid]["z"])]
    kept_t = 0

    STOP = {"that","this","with","from","have","passage","study","found","which",
            "their","they","there","about","into","such","were","been","also",
            "discusses","describes","explores","introduces","these","other","more",
            "than","when","what","some","over","each","between","using","used"}

    def words(s):
        return {w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split()
                if len(w) > 3 and w not in STOP}

    for tag in topic_tags:
        for pid in held:
            if kept_t >= args.n_topic:
                break
            single = ds.h_cache[tag][pid]
            gold_z = ds.passages[pid]["z"]
            verb = r.generate("av", tag, single, max_new=64)
            gw, vw = words(gold_z), words(verb)
            overlap = gw & vw
            # need 3+ distinctive content words shared AND meaningful coverage of
            # the gold's vocabulary -> guards against generic on-topic-sounding text
            # that actually describes a different passage.
            cov = len(overlap) / max(len(gw), 1)
            ok = len(overlap) >= 3 and cov >= 0.18
            print(f"  [{tag} p{pid}] ok={ok} ov={len(overlap)} cov={cov:.2f} {sorted(overlap)[:6]} :: {verb[:70]}")
            if ok:
                kept_t += 1
                ptxt = ds.passages[pid].get("text", "")
                examples.append({
                    "category": "topic",
                    "source_model": tag,
                    "tag": tag,
                    "passage_excerpt": ptxt[:400],
                    "gold_z": gold_z,
                    "v15_verbalization": verb,
                    "act": single.tolist(),
                })
        if kept_t >= args.n_topic:
            break
    print(f"[curate] TOPIC kept {kept_t}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(examples, ensure_ascii=False))
    from collections import Counter
    c = Counter(e["category"] for e in examples)
    print(f"[curate] wrote {len(examples)} examples -> {args.out} :: {dict(c)}")


if __name__ == "__main__":
    main()
