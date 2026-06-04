"""v18 — CROSS-MODEL calibrated per-bias Yes/No bias detector.

v17 trained calibrated detection on ONE organism (qwen2p5-7b) -> it does NOT
generalize to other/arbitrary models (misfires on a clean qwen3-1p7b). v18 fixes
this by training the SAME per-bias Yes/No detection across MANY architectures, so
"does this RESPONSE exhibit bias B?" becomes MODEL-INVARIANT content detection
that transfers to an UNSEEN model.

KEY DATA IDEA: the same bias-exhibiting transcript run through K=7 base models'
activations (extract_v18_xmodel.py). The detector sees bias B through 7 different
models' enc-projected acts -> learns the model-invariant bias signature.

TRAIN models (have enc in /big/adapters_v9_serve_llama):
  qwen3-1p7b, phi-1p5, smollm3-3b, qwen2p5-7b, gemma2, qwen2p5-0p5b, qwen3-4b
HELD-OUT model: llama3-8b (NEVER trained -> the cross-model transfer test).
HELD-OUT biases: quirk_sets.HELD_OUT (voting/population/chocolate) — never positive.

Detect example (copied from v17's per-bias Yes/No detect + heavy negatives):
  prompt  = ACTOR_TEMPLATE(tag, ㈎) + "\n\nQuestion: Does this response exhibit the
            following behaviour - {DESC[B]}? Answer Yes or No.\nAnswer:"
  inject enc(tag, act) at ㈎  (act = v18_xmodel/<tag> assistant-mean for a transcript)
  EXAMPLE KINDS (--detect-mix pos:inorg:clean):
    pos     : transcript.bias==B (any train model)   -> Yes
    inorg   : transcript.bias!=B, ask B (any model)  -> No  (in-transcript negative)
    clean   : neutral transcript (any train model), ask any non-held bias -> No
  Negatives (inorg+clean) kept >=60% of the detect corpus -> calibrated "No".
Optional LIE task folded in as-is (--mix detect:av:lie, av=0 by default).

Continue-from v17 (/big/audit/v15/v17_detector) at low LR, or fresh. enc tags all
come from --adapters-init (serve_llama, has llama3-8b too). Saves under --out:
av/ (LoRA), adapters/ (enc/dec bundle), v18_meta.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.datagen.injection_tokens import find_injection_token
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import compute_canonical_neighbors, normalize_activation
from scripts.audit.quirk_sets import DESC, HELD_OUT

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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
DETECT_QA = "Does this response exhibit the following behaviour - {desc}? Answer Yes or No."

# K train models for the cross-model detector (all have enc in serve_llama).
TRAIN_TAGS = ["qwen3-1p7b", "phi-1p5", "smollm3-3b", "qwen2p5-7b",
              "gemma2", "qwen2p5-0p5b", "qwen3-4b"]
HELDOUT_TAG = "llama3-8b"
NEUTRAL_BIAS = "neutral"

LIE_TAG = "gemma2"
LIE_QA = "Is the model being deceptive or lying in this response? Answer Yes or No."


def build_actor_prompt(model_tag: str, injection_char: str) -> str:
    return ACTOR_TEMPLATE.format(model_tag=model_tag, injection_char=injection_char)


def detect_question(bias: str) -> str:
    return DETECT_QA.format(desc=DESC[bias])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--continue-from", default=None,
                    help="continue from a v17 LoRA dir (e.g. /big/audit/v15/v17_detector/av).")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--d-shared", type=int, default=2048)
    ap.add_argument("--mix", default="8:0:2", help="detect:av:lie sampling weights.")
    ap.add_argument("--detect-mix", default="2:1.5:1.5",
                    help="pos:inorg:clean weights. negatives(inorg+clean) >= positives.")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-trunk", type=float, default=5e-5)
    ap.add_argument("--lr-enc", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--xmodel-dir", default="/big/audit/v18_xmodel")
    ap.add_argument("--adapters-init", default="/big/adapters_v9_serve_llama")
    ap.add_argument("--lie-dir", default="/big/audit/lie_gemma2_ml")
    ap.add_argument("--lie-acts-name", default="lie_acts_L21.safetensors")
    ap.add_argument("--lie-train-splits", default="varied_deception")
    ap.add_argument("--train-tags", default=",".join(TRAIN_TAGS))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

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

    train_tags = [t.strip() for t in args.train_tags.split(",") if t.strip()]
    mix_w = [float(x) for x in args.mix.split(":")]
    assert len(mix_w) == 3, "--mix = detect:av:lie"
    tasks = ["detect", "av", "lie"]
    dmix = [float(x) for x in args.detect_mix.split(":")]
    assert len(dmix) == 3, "--detect-mix = pos:inorg:clean"
    detect_kinds = ["pos", "inorg", "clean"]

    # ---- tokenizer + marker ids ---------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    inj_char, inj_id = find_injection_token(tok)
    left_id, right_id = compute_canonical_neighbors(
        tokenizer=tok,
        actor_template=ACTOR_TEMPLATE.replace("{model_tag}", _NEIGHBOR_DUMMY_TAG),
        injection_char=inj_char, injection_token_id=inj_id)
    eos = tok.eos_token_id
    yes_ids = tok(" Yes", add_special_tokens=False)["input_ids"]
    no_ids = tok(" No", add_special_tokens=False)["input_ids"]
    emit(f"[v18] inj_char={inj_char!r} inj_id={inj_id} left={left_id} right={right_id} "
         f"yes={yes_ids} no={no_ids}")

    # ---- trunk + LoRA --------------------------------------------------------
    base = AutoModelForCausalLM.from_pretrained(
        args.trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    if args.continue_from:
        model = PeftModel.from_pretrained(base, args.continue_from, is_trainable=True)
        emit(f"[v18] continue-from {args.continue_from} (trainable LoRA)")
    else:
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                              bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
        model = get_peft_model(base, lora_cfg)
        emit(f"[v18] fresh LoRA r={args.lora_r}")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    d_shared = model.config.hidden_size
    assert d_shared == args.d_shared, f"trunk d={d_shared} != --d-shared {args.d_shared}"
    inj_scale = math.sqrt(d_shared)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model = model.to(device).train()
    embed = model.get_input_embeddings()

    # ---- shared pool adapters (enc/dec, fp32, trainable enc) -----------------
    adapters = ModelPoolAdapters.load(args.adapters_init).to(device)
    assert adapters.d_shared == d_shared
    for need in train_tags + [HELDOUT_TAG, LIE_TAG]:
        assert need in adapters.tags, f"tag {need} missing from {args.adapters_init}"
    enc_params = []
    for tag, mod in adapters.encoders.items():
        for p in mod.parameters():
            p.requires_grad_(True)
            enc_params.append(p)
    for mod in adapters.decoders.values():
        for p in mod.parameters():
            p.requires_grad_(False)

    # ---- cross-model detect data --------------------------------------------
    rows = [json.loads(l) for l in (Path(args.xmodel_dir) / "rows.jsonl").read_text().splitlines() if l.strip()]
    H: dict[str, torch.Tensor] = {}
    for tag in train_tags:
        H[tag] = load_file(str(Path(args.xmodel_dir) / tag / "acts.safetensors"))["h"].float()
        assert H[tag].shape[0] == len(rows), f"{tag} acts {H[tag].shape[0]} != rows {len(rows)}"
    held = set(HELD_OUT)
    # transcript indices by bias category
    idxs_by_bias: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        idxs_by_bias[r["bias"]].append(i)
    neutral_idxs = idxs_by_bias.get(NEUTRAL_BIAS, [])
    # positives may only come from NON-held biases that actually have transcripts.
    pos_biases = sorted(b for b in idxs_by_bias if b not in held and b != NEUTRAL_BIAS and b in DESC)
    assert not (set(pos_biases) & held), "held-out leaked into positives"
    # ask-biases used to phrase questions: all DESC biases except held-out.
    ask_biases = [b for b in DESC if b not in held]
    emit(f"[v18][detect] train_tags={train_tags} | pos_biases({len(pos_biases)})={pos_biases}")
    emit(f"[v18][detect] neutral transcripts={len(neutral_idxs)} ask_biases={len(ask_biases)} "
         f"total_rows={len(rows)}")

    # lie (as-is, optional)
    lie_idxs: list[int] = []
    if mix_w[2] > 0:
        Hl = load_file(str(Path(args.lie_dir) / args.lie_acts_name))["h"].float()
        lrows = [json.loads(l) for l in (Path(args.lie_dir) / "lie_rows.jsonl").read_text().splitlines() if l.strip()]
        lie_splits = set(args.lie_train_splits.split(","))
        lie_idxs = [i for i, r in enumerate(lrows) if r["split"] in lie_splits and i < Hl.shape[0]]
        emit(f"[v18][lie] tag={LIE_TAG} acts {tuple(Hl.shape)} train rows {len(lie_idxs)}")
        assert lie_idxs, "lie weight>0 but no train rows"

    # ---- injection helpers ---------------------------------------------------
    def inject_one(p_ids, vec):
        p = torch.tensor([p_ids], device=device)
        e = embed(p)
        return inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left_id, right_id)

    def yn_response(is_yes):
        return (yes_ids if is_yes else no_ids) + [eos]

    def make_detect_prompt(tag, bias):
        return build_actor_prompt(tag, inj_char) + f"\n\nQuestion: {detect_question(bias)}\nAnswer:"

    def encode_inject(tag, h_vec):
        proj = adapters.encode(tag, h_vec.unsqueeze(0))
        return normalize_activation(proj, inj_scale)[0]

    def build_example(task):
        if task == "detect":
            kind = random.choices(detect_kinds, weights=dmix, k=1)[0]
            tag = random.choice(train_tags)
            if kind == "pos":
                b = random.choice(pos_biases)
                ti = random.choice(idxs_by_bias[b])
                ptxt = make_detect_prompt(tag, b)
                r_ids = yn_response(True)
            elif kind == "inorg":
                # transcript exhibits b_true; ask a DIFFERENT bias -> No
                b_true = random.choice(pos_biases)
                ti = random.choice(idxs_by_bias[b_true])
                b_ask = random.choice([b for b in ask_biases if b != b_true])
                ptxt = make_detect_prompt(tag, b_ask)
                r_ids = yn_response(False)
            else:  # clean: neutral transcript, ask any non-held bias -> No
                ti = random.choice(neutral_idxs) if neutral_idxs else random.choice(idxs_by_bias[random.choice(pos_biases)])
                b_ask = random.choice(ask_biases)
                ptxt = make_detect_prompt(tag, b_ask)
                r_ids = yn_response(False)
            vec = encode_inject(tag, H[tag][ti].to(device))
            return _assemble(ptxt, vec, r_ids), kind
        else:  # lie
            i = random.choice(lie_idxs)
            ptxt = build_actor_prompt(LIE_TAG, inj_char) + f"\n\nQuestion: {LIE_QA}\nAnswer:"
            r_ids = yn_response(bool(lrows[i]["is_lie"]))
            vec = encode_inject(LIE_TAG, Hl[i].to(device))
            return _assemble(ptxt, vec, r_ids), None

    def _assemble(ptxt, vec, r_ids):
        p_ids = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                        tokenize=True, add_generation_prompt=True)
        e = inject_one(p_ids, vec)
        a = torch.tensor([r_ids], device=device)
        ea = embed(a)
        inp = torch.cat([e, ea], dim=1)
        labels = torch.tensor([[-100] * e.shape[1] + r_ids], device=device)
        return inp, labels

    # ---- optimizer -----------------------------------------------------------
    trunk_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": trunk_params, "lr": args.lr_trunk},
        {"params": enc_params, "lr": args.lr_enc},
    ])
    scaler = torch.cuda.amp.GradScaler()

    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project="nla-v18", name=out.name, config=vars(args))
        except Exception as e:
            emit(f"[wandb] off ({e})")

    # ---- training loop -------------------------------------------------------
    t0 = time.time()
    deadline = t0 + args.minutes * 60.0
    GA = args.grad_accum
    micro = step = 0
    run = {t: 0.0 for t in tasks}
    cnt = {t: 0 for t in tasks}
    dkind_cnt = defaultdict(int)
    opt.zero_grad()
    emit(f"[v18] training {args.minutes} min mix(detect:av:lie)={mix_w} "
         f"detect-mix(pos:inorg:clean)={dmix} GA={GA}")
    while time.time() < deadline:
        task = random.choices(tasks, weights=mix_w, k=1)[0]
        if task == "av":  # av disabled in v18 (no pool dataset loaded); fall back to detect
            task = "detect"
        (inp, labels), kind = build_example(task)
        if kind is not None:
            dkind_cnt[kind] += 1
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss = model(inputs_embeds=inp, labels=labels).loss
        scaler.scale(loss / GA).backward()
        run[task] += loss.item()
        cnt[task] += 1
        micro += 1
        if micro % GA == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trunk_params + enc_params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            step += 1
            if step % args.log_every == 0:
                bd = " ".join(f"{t}={run[t]/max(cnt[t],1):.3f}(n{cnt[t]})" for t in tasks)
                dk = " ".join(f"{k}:{dkind_cnt[k]}" for k in detect_kinds)
                el = time.time() - t0
                emit(f"[v18] step {step} {el:.0f}s | {bd} | detect[{dk}]")
                if wb:
                    wb.log({f"loss/{t}": run[t] / max(cnt[t], 1) for t in tasks} | {"step": step})
                run = {t: 0.0 for t in tasks}
                cnt = {t: 0 for t in tasks}
    emit(f"[v18] done: {step} steps in {(time.time()-t0):.0f}s; detect kinds {dict(dkind_cnt)}")

    # ---- save ----------------------------------------------------------------
    model.save_pretrained(out / "av")
    adapters.cpu()
    adapters.save(out / "adapters")
    meta = {
        "kind": "v18_xmodel_calibrated_detector",
        "trunk": args.trunk,
        "d_shared": d_shared,
        "inject": "marker",
        "n_inj": 1,
        "injection_scale": "sqrt_d_model",
        "tokens": {
            "injection_char": inj_char,
            "injection_token_id": int(inj_id),
            "injection_left_neighbor_id": int(left_id),
            "injection_right_neighbor_id": int(right_id),
        },
        "actor_template": ACTOR_TEMPLATE,
        "detect_qa": DETECT_QA,
        "train_tags": train_tags,
        "heldout_tag": HELDOUT_TAG,
        "lie_tag": LIE_TAG,
        "lie_qa": LIE_QA,
        "lie_acts_name": args.lie_acts_name,
        "lie_dir": args.lie_dir,
        "mix": args.mix,
        "detect_mix": args.detect_mix,
        "held_out_biases": HELD_OUT,
        "supervised_biases": pos_biases,
        "ask_biases": ask_biases,
        "continue_from": args.continue_from,
        "xmodel_dir": args.xmodel_dir,
        "neutral_bias": NEUTRAL_BIAS,
    }
    (out / "v18_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    emit(f"[v18] saved -> {out}")
    log.close()


if __name__ == "__main__":
    main()
