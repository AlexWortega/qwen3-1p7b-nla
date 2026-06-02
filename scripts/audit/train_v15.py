"""v15 — from-scratch JOINT universal-NLA (AV) + Activation-Oracle (AO).

ONE Qwen3-1.7B + LoRA trunk trained multi-task on three interleaved tasks:

  AV       : verbalize a pooled activation of any pool model. enc_M(h_M) injected
             at the ㈎ marker, CE on <explanation>{teacher z}</explanation>.
             (data: MultiModelActivationDataset over /big/activations_pool_v9)
  AO-quirk : answer a free-form QA about an organism's latent quirk. enc(qwen2p5-7b,h)
             injected, CE on the free-form answer.
             (data: ao_rows_v13 src=org family a/b + acts_ao_org_mean)
  AO-lie   : answer Yes/No "is the model lying". enc(gemma2,h_L21) injected, CE on
             " Yes"/" No". (data: lie_gemma2_ml split=varied_deception)
  latentqa : behaviour-QA over Berkeley LatentQA stimuli, trained on IN-POOL models'
             activations (round-robin qwen2p5-7b/gemma2/phi-1p5/smollm3-3b; NOT Llama-3,
             which stays held-out for eval). enc(assigned_tag, h) injected, the
             behaviour question appended, CE on the gold answer.
             (data: latentqa_task/ built by build_latentqa_task.py)

ONE ModelPoolAdapters instance (enc/dec linears, fp32, lstsq-init) is SHARED across
all three tasks; encoders are trainable. The injection mechanism is flag-selected
(marker / ntok / flamingo) and identical at eval time (saved in v15_meta.json).

Each optimizer micro-step samples a task by --mix weights, runs ONE example
(batch-of-1 per task to keep the variable-length AO prompts simple), accumulates
grad, logs per-task loss. Bounded by --minutes wall-clock, NOT epochs.

V100 fp16: trainable params (LoRA + enc + flamingo) cast to fp32; forward under
autocast(fp16) + GradScaler.

Saves under --out:
  av/              PEFT LoRA adapter (trunk)
  adapters/        ModelPoolAdapters bundle (trained enc/dec)
  flamingo.pt      (only when --inject flamingo)
  v15_meta.json    trunk, d_shared, inject, n_inj, inject_layer, marker ids,
                   actor_template, tags, task data paths.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.data_multi import MultiModelActivationDataset
from nla.datagen.injection_tokens import find_injection_token
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.flamingo import FlamingoInject, attach_flamingo, set_flamingo_kv
from nla.injection import inject_at_marked_positions
from nla.schema import (
    EXPLANATION_CLOSE,
    EXPLANATION_OPEN,
    compute_canonical_neighbors,
    normalize_activation,
)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Same actor template as the v9 universal AV (train_av_multi). The model_tag is
# plain text BEFORE the <concept>㈎</concept> region, so the marker's neighbor
# tokens are stable across tags — required by inject_at_marked_positions.
ACTOR_TEMPLATE = (
    "You are a meticulous AI researcher investigating activation vectors from "
    "{model_tag}, a small open-weight language model. Your task is to describe "
    "the semantic content of the activation in one sentence.\n\n"
    "We pass the vector inside <concept> tags. Reply with the description "
    "inside <explanation> tags.\n\n"
    "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
    "Please provide the description."
)
_NEIGHBOR_DUMMY_TAG = "qwen3-1p7b"

QUIRK_QA = "What unusual pattern or objective does this response reflect, and what likely causes it?"
LIE_QA = "Is the model being deceptive or lying in this response? Answer Yes or No."

AV_TAGS = ["qwen3-1p7b", "phi-1p5", "smollm3-3b"]  # AV-task training tags (small/cheap default)
# Full v9 pool (all 15 extracted tags incl. organism tags) — selected via --full-pool.
AV_TAGS_FULL = ["gemma2", "gpt-neo-1p3b", "gpt2-medium", "lfm-7b", "phi-1p5", "pythia-410m",
                "qwen2p5-0p5b", "qwen2p5-7b", "qwen3-0p6b", "qwen3-1p7b", "qwen3-4b",
                "smollm2-360m", "smollm3-3b", "vikhr-7b-01", "yagpt-5-8b"]
QUIRK_TAG = "qwen2p5-7b"
LIE_TAG = "gemma2"


def build_actor_prompt(model_tag: str, injection_char: str) -> str:
    return ACTOR_TEMPLATE.format(model_tag=model_tag, injection_char=injection_char)


def wrap_response(z: str) -> str:
    return f"{EXPLANATION_OPEN}\n{z.strip()}\n{EXPLANATION_CLOSE}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--d-shared", type=int, default=2048)
    ap.add_argument("--inject", choices=["marker", "ntok", "flamingo"], default="marker")
    ap.add_argument("--n-inj", type=int, default=1, help="K soft-tokens for ntok mode")
    ap.add_argument("--multi-layer", action="store_true",
                    help="v15.2: inject K DISTINCT layer-activations as K consecutive soft "
                         "tokens spliced at the single ㈎ marker (marker mode only).")
    ap.add_argument("--n-layers", type=int, default=4, help="K layers for --multi-layer")
    ap.add_argument("--pool-ml-dir", default="/big/activations_pool_v9_ml",
                    help="multi-layer AV acts dir (<tag>_ml.safetensors {h:[N,K,d]} + index_ml.json)")
    ap.add_argument("--quirk-acts-ml", default="/big/audit/ao/acts_ao_org_ml.safetensors",
                    help="multi-layer quirk acts {h:[N,K,d]}; falls back to single-layer if absent")
    ap.add_argument("--lie-acts-ml", default="L13,L21,L31,L39",
                    help="comma list of lie_acts_<X>.safetensors layer suffixes for --multi-layer")
    ap.add_argument("--inject-layer", type=int, default=14, help="flamingo cross-attn layer")
    ap.add_argument("--mix", default="3:1:1",
                    help="AV:AO-quirk:AO-lie[:latentqa] sampling weights. 3 values keep "
                         "back-compat (latentqa weight padded to 0); 4 values enable latentqa.")
    ap.add_argument("--contrastive-weight", type=float, default=1.0,
                    help="relative sampling weight of AO-quirk family-b rows vs family-a")
    ap.add_argument("--train-enc", choices=["full", "ao-only"], default="full",
                    help="full=enc trains on all tasks; ao-only=enc gradient only from AO tasks")
    ap.add_argument("--minutes", type=float, default=12.0, help="wall-clock budget")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-trunk", type=float, default=1e-4)
    ap.add_argument("--lr-enc", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-ans", type=int, default=110)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--log-every", type=int, default=5)
    # data paths (defaults match the eva01 /big mount)
    ap.add_argument("--pool-dir", default="/big/activations_pool_v9")
    ap.add_argument("--adapters-init", default="/big/adapters_v9_serve_gemma2")
    ap.add_argument("--quirk-acts", default="/big/audit/ao/acts_ao_org_mean.safetensors")
    ap.add_argument("--quirk-rows", default="/big/audit/ao/ao_rows_v13.jsonl")
    ap.add_argument("--lie-dir", default="/big/audit/lie_gemma2_ml")
    ap.add_argument("--lie-acts-name", default="lie_acts_L21.safetensors")
    ap.add_argument("--lie-train-splits", default="varied_deception")
    ap.add_argument("--latentqa-dir", default="/big/audit/latentqa_task",
                    help="LatentQA training-task dir (latentqa_train.jsonl + acts_<tag>.safetensors "
                         "+ rowmap.json). No-op if absent or its latentqa mix weight is 0.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--full-pool", action="store_true",
                    help="train the AV task on all 15 v9 pool tags (AV_TAGS_FULL) for true universality")
    args = ap.parse_args()

    global AV_TAGS
    if args.full_pool:
        AV_TAGS = AV_TAGS_FULL

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "w")

    def emit(s: str):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    # --mix is AV:AO-quirk:AO-lie[:latentqa]. 3 values stay back-compatible: the
    # 4th (latentqa) weight is padded to 0. 4 values enable the latentqa task.
    mix_w = [float(x) for x in args.mix.split(":")]
    assert len(mix_w) in (3, 4), "--mix must be AV:AO-quirk:AO-lie[:latentqa]"
    if len(mix_w) == 3:
        mix_w.append(0.0)
    tasks = ["av", "quirk", "lie", "latentqa"]

    # ---- tokenizer + marker ids ---------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    inj_char, inj_id = find_injection_token(tok)
    left_id, right_id = compute_canonical_neighbors(
        tokenizer=tok,
        actor_template=ACTOR_TEMPLATE.replace("{model_tag}", _NEIGHBOR_DUMMY_TAG),
        injection_char=inj_char,
        injection_token_id=inj_id,
    )
    eos = tok.eos_token_id
    emit(f"[v15] inj_char={inj_char!r} inj_id={inj_id} left={left_id} right={right_id} inject={args.inject} n_inj={args.n_inj}")

    # ---- trunk + LoRA --------------------------------------------------------
    base = AutoModelForCausalLM.from_pretrained(
        args.trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    model = get_peft_model(base, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    d_shared = model.config.hidden_size
    assert d_shared == args.d_shared, f"trunk d={d_shared} != --d-shared {args.d_shared}"
    inj_scale = math.sqrt(d_shared)

    K = args.n_layers if args.multi_layer else 1
    if args.multi_layer:
        assert args.inject == "marker", "--multi-layer only supports --inject marker (flamingo ML out of scope)"
        emit(f"[v15.2] MULTI-LAYER injection: K={K} distinct soft tokens spliced at the marker")

    # ---- flamingo (optional) -------------------------------------------------
    flamingo = None
    if args.inject == "flamingo":
        flamingo = FlamingoInject(d_model=d_shared, kv_dim=d_shared, n_heads=8, gate_init=0.0)
        attach_flamingo(model, args.inject_layer, flamingo)
        flamingo = flamingo.to(device)
        emit(f"[v15] flamingo attached at layer {args.inject_layer} (alpha-init=0)")

    # cast trainable params (LoRA + flamingo) to fp32 AFTER peft wrap.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if flamingo is not None:
        for p in flamingo.parameters():
            p.data = p.data.float()
    model = model.to(device).train()
    embed = model.get_input_embeddings()

    # ---- shared pool adapters (enc/dec, fp32, trainable enc) -----------------
    adapters = ModelPoolAdapters.load(args.adapters_init).to(device)
    assert adapters.d_shared == d_shared, f"adapters d_shared {adapters.d_shared} != {d_shared}"
    for need in AV_TAGS + [QUIRK_TAG, LIE_TAG]:
        assert need in adapters.tags, f"tag {need} missing from {args.adapters_init} (have {adapters.tags})"
    enc_params = []
    for tag, mod in adapters.encoders.items():
        for p in mod.parameters():
            p.requires_grad_(True)
            enc_params.append(p)
    # decoders kept (not trained here) — frozen so they don't drift on no-grad.
    for mod in adapters.decoders.values():
        for p in mod.parameters():
            p.requires_grad_(False)

    # ---- task data -----------------------------------------------------------
    # AV
    av_ds = MultiModelActivationDataset(args.pool_dir, restrict_tags=AV_TAGS, dtype=torch.float32)
    av_pids = [pid for pid in range(av_ds.n_passages) if av_ds.passages[pid].get("z")]
    emit(f"[v15][av] {len(av_pids)} passages w/ teacher z over tags {AV_TAGS}")

    # multi-layer AV acts: per-tag [N_ml, K, d_M]. Tags present here use the K-layer
    # splice; tags absent (or non-ML run) fall back to replicating the single-layer vec.
    av_ml = {}            # tag -> tensor [N_ml, K, d_M]
    av_ml_layers = {}     # tag -> list of layer indices (provenance)
    if args.multi_layer:
        ml_idx_path = Path(args.pool_ml_dir) / "index_ml.json"
        ml_index = json.loads(ml_idx_path.read_text()) if ml_idx_path.exists() else {}
        for tag in AV_TAGS:
            if tag in ml_index:
                shard = Path(args.pool_ml_dir) / ml_index[tag]["shard"]
                H = load_file(str(shard))["h"].float()  # [N_ml, K_avail, d]
                kk = H.shape[1]
                if kk < K:
                    emit(f"[v15.2][av] {tag}: only {kk} layers extracted < K={K}; using {kk}")
                av_ml[tag] = H[:, :K]
                av_ml_layers[tag] = ml_index[tag].get("layer_indices", list(range(kk)))[:K]
        if av_ml:
            emit(f"[v15.2][av] multi-layer acts for {sorted(av_ml)} "
                 f"(layers { {t: av_ml_layers[t] for t in av_ml} }); other AV tags fall back to single-layer replicate")
        else:
            emit(f"[v15.2][av] no multi-layer acts in {args.pool_ml_dir} -> AV falls back to single-layer replicate (K copies)")

    # AO-quirk
    Hq = load_file(args.quirk_acts)["h"].float()
    qrows = [json.loads(l) for l in Path(args.quirk_rows).read_text().splitlines() if l.strip()]
    qrows = [r for r in qrows if r.get("src") == "org" and r.get("family") in ("a", "b")
             and not r.get("held_out") and int(r["transcript_idx"]) < Hq.shape[0]]
    q_fam_a = [r for r in qrows if r["family"] == "a"]
    q_fam_b = [r for r in qrows if r["family"] == "b"]
    emit(f"[v15][quirk] acts {tuple(Hq.shape)}; rows famA={len(q_fam_a)} famB={len(q_fam_b)} (tag={QUIRK_TAG})")

    # multi-layer quirk acts [N, K, d]; absent -> fall back to single-layer replicate.
    Hq_ml = None
    if args.multi_layer:
        qml = Path(args.quirk_acts_ml)
        if qml.exists():
            Hq_ml = load_file(str(qml))["h"].float()[:, :K]
            emit(f"[v15.2][quirk] multi-layer acts {tuple(Hq_ml.shape)}")
        else:
            emit(f"[v15.2][quirk] {qml} absent -> single-layer replicate (K copies of L14 vec)")

    # AO-lie
    Hl = load_file(str(Path(args.lie_dir) / args.lie_acts_name))["h"].float()
    lrows = [json.loads(l) for l in (Path(args.lie_dir) / "lie_rows.jsonl").read_text().splitlines() if l.strip()]
    lie_splits = set(args.lie_train_splits.split(","))
    lie_idxs = [i for i, r in enumerate(lrows) if r["split"] in lie_splits and i < Hl.shape[0]]
    emit(f"[v15][lie] acts {tuple(Hl.shape)}; train rows={len(lie_idxs)} splits={lie_splits} (tag={LIE_TAG})")

    # multi-layer lie acts: stack the existing lie_acts_<X>.safetensors shards [N,K,d].
    Hl_ml = None
    if args.multi_layer:
        suffixes = [s.strip() for s in args.lie_acts_ml.split(",")][:K]
        try:
            parts = [load_file(str(Path(args.lie_dir) / f"lie_acts_{s}.safetensors"))["h"].float()
                     for s in suffixes]
            Hl_ml = torch.stack(parts, dim=1)  # [N, K, d]
            emit(f"[v15.2][lie] multi-layer acts {tuple(Hl_ml.shape)} from {suffixes}")
        except Exception as e:
            emit(f"[v15.2][lie] could not stack ML shards ({e}) -> single-layer replicate")

    # latentqa (4th task). Loaded only if weight>0 AND the dir exists. Each train row
    # carries an enc tag (round-robin) + a local index into that tag's acts shard.
    lqa_rows = []          # list of {label, question, gold, tag, h_idx}
    lqa_H = {}             # tag -> [n, d_tag] fp32
    lqa_dir = Path(args.latentqa_dir)
    if mix_w[3] > 0:
        train_jsonl = lqa_dir / "latentqa_train.jsonl"
        rowmap_json = lqa_dir / "rowmap.json"
        if not (train_jsonl.exists() and rowmap_json.exists()):
            emit(f"[v15][latentqa] dir {lqa_dir} missing train/rowmap -> task DISABLED (weight->0)")
            mix_w[3] = 0.0
        else:
            train = [json.loads(l) for l in train_jsonl.read_text().splitlines() if l.strip()]
            rm = json.loads(rowmap_json.read_text())["rowmap"]
            tags_present = set()
            for gi_str, info in rm.items():
                gi = int(gi_str)
                tag = info["tag"]
                if tag not in lqa_H:
                    shard = lqa_dir / f"acts_{tag}.safetensors"
                    if not shard.exists():
                        continue  # tag not extracted; skip its rows
                    lqa_H[tag] = load_file(str(shard))["h"].float()
                if info["local"] >= lqa_H[tag].shape[0]:
                    continue
                if tag not in adapters.tags:
                    continue
                r = train[gi]
                lqa_rows.append({"label": r["label"], "question": r["question"],
                                 "gold": r["gold"], "tag": tag, "h_idx": info["local"]})
                tags_present.add(tag)
            if not lqa_rows:
                emit(f"[v15][latentqa] no usable rows in {lqa_dir} -> task DISABLED (weight->0)")
                mix_w[3] = 0.0
            else:
                emit(f"[v15][latentqa] {len(lqa_rows)} train rows over tags {sorted(tags_present)} "
                     f"(acts: {{ {', '.join(f'{t}:{tuple(lqa_H[t].shape)}' for t in sorted(lqa_H))} }})")
    else:
        emit("[v15][latentqa] mix weight 0 -> task disabled")

    # ---- injection helper: builds inputs_embeds + labels for one example -----
    def inject_marker(p_ids: list[int], vec: torch.Tensor):
        """marker / ntok injection. Returns (embeds, kv=None). vec: [d_shared]."""
        p = torch.tensor([p_ids], device=device)
        e = embed(p)
        if args.inject == "ntok" and args.n_inj > 1:
            # ntok: the single marker position carries the projection; we repeat
            # it across the n_inj marker slots IF present. With one marker in the
            # template we just inject the single vec (n_inj copies collapse to the
            # same enc vec — true multi-soft-token would need n_inj markers in the
            # template; we keep one marker and inject the same vec, behaviourally
            # equivalent to n_inj=1 here but the flag is recorded for eval parity).
            vecs = vec.unsqueeze(0)
        else:
            vecs = vec.unsqueeze(0)
        e = inject_at_marked_positions(p, e, vecs.to(e.dtype), inj_id, left_id, right_id)
        return e, None

    def find_marker_pos(p_ids: list[int]) -> int:
        """Locate the single ㈎ marker by inj_id + canonical neighbors. Returns its
        sequence index. Mirrors inject_at_marked_positions' validity check."""
        for j, t in enumerate(p_ids):
            if t == inj_id and 0 < j < len(p_ids) - 1 \
               and p_ids[j - 1] == left_id and p_ids[j + 1] == right_id:
                return j
        raise RuntimeError("v15.2: no valid ㈎ marker found (neighbor check failed)")

    def inject_multilayer(p_ids: list[int], vecs: torch.Tensor):
        """v15.2 splice: replace the 1 marker token's embedding with K consecutive
        soft tokens. vecs: [K, d_shared] (already normalized). The prompt embed grows
        by K-1; labels for the whole prompt region stay -100 so no realignment needed.

        Returns embeds [1, T_prompt + K - 1, d]. The marker's left/right neighbor
        tokens stay intact around the K-token block, so the model sees a coherent
        '...<concept> v1 v2 .. vK </concept>...' sequence (no extra marker chars)."""
        pos = find_marker_pos(p_ids)
        p = torch.tensor([p_ids], device=device)
        e = embed(p)                                   # [1, T, d]
        d = e.shape[-1]
        soft = vecs.to(e.dtype).view(1, -1, d)         # [1, K, d]
        spliced = torch.cat([e[:, :pos], soft, e[:, pos + 1:]], dim=1)
        return spliced

    def _ml_stack(single_h: torch.Tensor, ml_row: torch.Tensor | None) -> torch.Tensor:
        """Return [K, d_M] for this example. single_h: [d_M]. ml_row: [K, d_M] or None.
        When ML acts exist use them; else replicate the single-layer vec K times so the
        splice still injects K tokens (degenerate but shape-correct fallback)."""
        if not args.multi_layer:
            return single_h.unsqueeze(0)            # [1, d_M]
        if ml_row is not None:
            return ml_row.to(device).float()        # [K, d_M]
        return single_h.unsqueeze(0).repeat(K, 1)   # [K, d_M] replicate

    def build_example(task: str):
        """Return (inputs_embeds[1,T,d], labels[1,T], kv_or_None)."""
        if task == "av":
            pid = random.choice(av_pids)
            tag = random.choice(AV_TAGS)
            single = av_ds.h_cache[tag][pid]
            ml_row = av_ml[tag][pid] if (tag in av_ml and pid < av_ml[tag].shape[0]) else None
            h_ml = _ml_stack(single, ml_row).to(device)
            z = av_ds.passages[pid]["z"]
            ptxt = build_actor_prompt(tag, inj_char)
            p_ids = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                            tokenize=True, add_generation_prompt=True)
            resp = wrap_response(z)
            r_ids = tok(resp, add_special_tokens=False)["input_ids"] + [eos]
        elif task == "quirk":
            # family sampling weighted by --contrastive-weight (famB relative wt).
            if q_fam_b and random.random() < args.contrastive_weight / (1.0 + args.contrastive_weight):
                r = random.choice(q_fam_b)
            else:
                r = random.choice(q_fam_a)
            tag = QUIRK_TAG
            ti = int(r["transcript_idx"])
            ml_row = Hq_ml[ti] if (Hq_ml is not None and ti < Hq_ml.shape[0]) else None
            h_ml = _ml_stack(Hq[ti], ml_row).to(device)
            ptxt = build_actor_prompt(tag, inj_char) + f"\n\nQuestion: {r['question'].strip()}\nAnswer:"
            p_ids = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                            tokenize=True, add_generation_prompt=True)
            r_ids = tok(" " + r["answer"].strip(), add_special_tokens=False)["input_ids"][:args.max_ans] + [eos]
        elif task == "latentqa":
            r = random.choice(lqa_rows)
            tag = r["tag"]
            # latentqa ML extraction is out of smoke scope -> single-layer replicate.
            h_ml = _ml_stack(lqa_H[tag][r["h_idx"]], None).to(device)
            ptxt = build_actor_prompt(tag, inj_char) + f"\n\nQuestion: {r['question'].strip()}\nAnswer:"
            p_ids = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                            tokenize=True, add_generation_prompt=True)
            r_ids = tok(" " + r["gold"].strip(), add_special_tokens=False)["input_ids"][:args.max_ans] + [eos]
        else:  # lie
            i = random.choice(lie_idxs)
            tag = LIE_TAG
            ml_row = Hl_ml[i] if (Hl_ml is not None and i < Hl_ml.shape[0]) else None
            h_ml = _ml_stack(Hl[i], ml_row).to(device)
            ptxt = build_actor_prompt(tag, inj_char) + f"\n\nQuestion: {LIE_QA}\nAnswer:"
            p_ids = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                            tokenize=True, add_generation_prompt=True)
            yes_ids = tok(" Yes", add_special_tokens=False)["input_ids"]
            no_ids = tok(" No", add_special_tokens=False)["input_ids"]
            r_ids = (yes_ids if lrows[i]["is_lie"] else no_ids) + [eos]

        # enc projection per layer: [K, d_M] -> [K, d_shared]. Each layer's act goes
        # through the SAME per-tag enc (co-trained, adapts to off-0.5 layers).
        h_proj = adapters.encode(tag, h_ml)             # [K, d_shared]
        if task == "av" and args.train_enc == "ao-only":
            h_proj = h_proj.detach()
        vecs = normalize_activation(h_proj, inj_scale)  # [K, d_shared] per-row norm

        if args.inject == "flamingo":
            # marker char stays in the prompt but is NOT overwritten; KV carries the
            # K layer vecs as M=K cross-attention slots.
            p = torch.tensor([p_ids], device=device)
            e = embed(p)
            kv = vecs.view(1, vecs.shape[0], d_shared)  # [B, M=K, d_shared]
        elif args.multi_layer:
            e = inject_multilayer(p_ids, vecs)          # K-token splice
            kv = None
        else:
            e, _ = inject_marker(p_ids, vecs[0])
            kv = None

        a = torch.tensor([r_ids], device=device)
        ea = embed(a)
        inp = torch.cat([e, ea], dim=1)
        if inp.shape[1] > args.max_seq_len + args.max_ans:
            pass  # AO answers short; AV capped by template — leave as-is.
        labels = torch.tensor([[-100] * e.shape[1] + r_ids], device=device)
        return inp, labels, kv

    # ---- optimizer -----------------------------------------------------------
    # flamingo is registered as a child of `model` via attach_flamingo, so its
    # params already appear in model.parameters(); dedupe by id to avoid the
    # optimizer "duplicate parameters" warning / double LR.
    seen = set()
    trunk_params = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            trunk_params.append(p)
    fl_params = []  # already inside trunk_params via the wrapped layer
    opt = torch.optim.AdamW([
        {"params": trunk_params, "lr": args.lr_trunk},
        {"params": enc_params, "lr": args.lr_enc},
    ])
    scaler = torch.cuda.amp.GradScaler()

    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project="nla-v15", name=out.name, config=vars(args))
        except Exception as e:
            emit(f"[wandb] off ({e})")

    # ---- training loop (wall-clock bounded) ----------------------------------
    t0 = time.time()
    deadline = t0 + args.minutes * 60.0
    GA = args.grad_accum
    micro = 0
    step = 0
    run = {t: 0.0 for t in tasks}
    cnt = {t: 0 for t in tasks}
    opt.zero_grad()
    emit(f"[v15] training for {args.minutes} min, mix={mix_w}, GA={GA}")
    while time.time() < deadline:
        task = random.choices(tasks, weights=mix_w, k=1)[0]
        inp, labels, kv = build_example(task)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            if kv is not None:
                with set_flamingo_kv(model, kv.to(torch.float16)):
                    loss = model(inputs_embeds=inp, labels=labels).loss
            else:
                loss = model(inputs_embeds=inp, labels=labels).loss
        scaler.scale(loss / GA).backward()
        run[task] += loss.item()
        cnt[task] += 1
        micro += 1
        if micro % GA == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trunk_params + fl_params + enc_params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            step += 1
            if step % args.log_every == 0:
                bd = " ".join(f"{t}={run[t]/max(cnt[t],1):.3f}(n{cnt[t]})" for t in tasks)
                el = time.time() - t0
                emit(f"[v15] step {step} {el:.0f}s | {bd}")
                if wb:
                    wb.log({f"loss/{t}": run[t] / max(cnt[t], 1) for t in tasks} | {"step": step})
                run = {t: 0.0 for t in tasks}
                cnt = {t: 0 for t in tasks}

    emit(f"[v15] done: {step} optimizer steps in {(time.time()-t0):.0f}s")

    # ---- save ----------------------------------------------------------------
    model.save_pretrained(out / "av")
    adapters.cpu()
    adapters.save(out / "adapters")
    if flamingo is not None:
        torch.save(flamingo.cpu().state_dict(), out / "flamingo.pt")
    meta = {
        "kind": "v15_joint_av_ao",
        "trunk": args.trunk,
        "av_base": args.trunk,  # alias for compatibility with eval helpers
        "d_shared": d_shared,
        "inject": args.inject,
        "n_inj": args.n_inj,
        "inject_layer": args.inject_layer,
        "multi_layer": bool(args.multi_layer),
        "n_layers": int(K),
        "pool_ml_dir": args.pool_ml_dir,
        "quirk_acts_ml": args.quirk_acts_ml,
        "lie_acts_ml": args.lie_acts_ml,
        "av_ml_layers": {t: av_ml_layers[t] for t in av_ml} if args.multi_layer else {},
        "injection_scale": "sqrt_d_model",
        "tokens": {
            "injection_char": inj_char,
            "injection_token_id": int(inj_id),
            "injection_left_neighbor_id": int(left_id),
            "injection_right_neighbor_id": int(right_id),
        },
        "actor_template": ACTOR_TEMPLATE,
        "prompt_templates": {"actor": ACTOR_TEMPLATE},
        "av_tags": AV_TAGS,
        "quirk_tag": QUIRK_TAG,
        "lie_tag": LIE_TAG,
        "quirk_qa": QUIRK_QA,
        "lie_qa": LIE_QA,
        "lie_acts_name": args.lie_acts_name,
        "mix": args.mix,
        "mix_weights": {t: mix_w[i] for i, t in enumerate(tasks)},
        "train_enc": args.train_enc,
        "latentqa_dir": args.latentqa_dir,
        "latentqa_enabled": bool(mix_w[3] > 0 and lqa_rows),
        "data": {
            "pool_dir": args.pool_dir,
            "quirk_acts": args.quirk_acts,
            "quirk_rows": args.quirk_rows,
            "lie_dir": args.lie_dir,
            "latentqa_dir": args.latentqa_dir,
        },
    }
    (out / "v15_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    emit(f"[v15] saved -> {out}")
    log.close()


if __name__ == "__main__":
    main()
