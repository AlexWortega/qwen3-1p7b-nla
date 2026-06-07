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
from nla.data_multi import MultiModelActivationDataset
from nla.injection import inject_at_marked_positions
from nla.resid_inject import marker_positions, resid_injection
from nla.schema import compute_canonical_neighbors, normalize_activation

# v21 general-introspection extra tasks (AV verbalize + LatentQA), reusing the same
# enc-inject marker machinery. AV verbalize prompt = build_actor_prompt (no question).
AV_TAGS_V21 = ["qwen3-1p7b", "phi-1p5", "smollm3-3b", "qwen2p5-7b", "gemma2"]


def wrap_response_av(z: str) -> str:
    return f"<explanation>{z.strip()}</explanation>"
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
    ap.add_argument("--enc-mlp-hidden", type=int, default=0,
                    help="if >0, wrap each per-model enc (and dec) as a ResidualMLPAdapter "
                         "(linear lstsq + zero-init MLP residual of this hidden size) -- a "
                         "NONLINEAR bridge to test the cross-model deception cap. 0 = linear.")
    ap.add_argument("--d-shared", type=int, default=2048)
    ap.add_argument("--mix", default="8:0:2", help="detect:av:lie sampling weights.")
    ap.add_argument("--detect-mix", default="2:1.5:1.5",
                    help="pos:inorg:clean weights. negatives(inorg+clean) >= positives.")
    ap.add_argument("--minutes", type=float, default=120.0,
                    help="time budget (TIME mode, --epochs 0) or a safety wall-clock cap (EPOCH mode).")
    ap.add_argument("--epochs", type=int, default=0,
                    help="EPOCH mode: pass each detect transcript / av passage / latentqa row this many "
                         "times (natural proportions). 0 = legacy TIME mode (random sampling by --mix).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-trunk", type=float, default=5e-5)
    ap.add_argument("--lr-enc", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=200,
                    help="rolling checkpoint every N optimizer steps to <out>/ckpt_latest (0=off).")
    ap.add_argument("--xmodel-dir", default="/big/audit/v18_xmodel")
    ap.add_argument("--adapters-init", default="/big/adapters_v9_serve_llama")
    ap.add_argument("--lie-dir", default="/big/audit/lie_gemma2_ml")
    ap.add_argument("--lie-acts-name", default="lie_acts_L21.safetensors")
    ap.add_argument("--lie-train-splits", default="varied_deception")
    ap.add_argument("--train-tags", default=",".join(TRAIN_TAGS))
    ap.add_argument("--held-out-biases", default=None,
                    help="comma list overriding quirk_sets.HELD_OUT (v19: add gender_bias). "
                         "Default keeps the v18 held-out set.")
    ap.add_argument("--inject-mode", default="embed", choices=["embed", "resid"],
                    help="embed=NLA marker at input embedding (default); resid=niclas-luick "
                         "style steering into the residual stream at --inject-layer.")
    ap.add_argument("--inject-layer", type=int, default=14,
                    help="oracle decoder layer for resid injection (Qwen3-1.7B has 28).")
    ap.add_argument("--steer-coef", type=float, default=2.0,
                    help="resid injection: normalize(vec)*||resid||*coef (their default 2.0).")
    ap.add_argument("--av-pool-dir", default="/big/activations_pool_v9",
                    help="v21 AV task: MultiModelActivationDataset pool (passages w/ teacher z).")
    ap.add_argument("--av-tags", default=",".join(AV_TAGS_V21))
    ap.add_argument("--latentqa-dir", default="/big/audit/latentqa_task",
                    help="v21 LatentQA task dir (latentqa_train.jsonl + rowmap.json + acts_<tag>).")
    ap.add_argument("--max-ans", type=int, default=110)
    ap.add_argument("--dir-pairs-dir", default=None,
                    help="direction hard-negatives: dir with rows.jsonl (role=pos|hardneg, pair_bias) "
                         "+ <tag>/acts.safetensors. 'hardneg' detect kind asks pair_bias on a balanced "
                         "same-topic response -> No, teaching bias DIRECTION not topic. See build_dir_pairs.py.")
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
    assert len(mix_w) in (3, 4), "--mix = detect:av:lie[:latentqa]"
    if len(mix_w) == 3:
        mix_w.append(0.0)            # latentqa weight padded to 0 (back-compat)
    tasks = ["detect", "av", "lie", "latentqa"]
    dmix = [float(x) for x in args.detect_mix.split(":")]
    assert len(dmix) in (3, 4, 5), "--detect-mix = pos:inorg:clean[:hardneg[:dirpos]]"
    while len(dmix) < 5:
        dmix.append(0.0)            # hardneg, dirpos weights padded to 0 (back-compat)
    detect_kinds = ["pos", "inorg", "clean", "hardneg", "dirpos"]

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
    if args.enc_mlp_hidden > 0:  # nonlinear bridge: wrap each enc/dec as ResidualMLP (residual=0 at init)
        from nla.enc_dec_adapters import ResidualMLPAdapter
        H = args.enc_mlp_hidden
        for tag in list(adapters.encoders.keys()):
            adapters.encoders[tag] = ResidualMLPAdapter.from_linear(adapters.encoders[tag], hidden=H)
            adapters.decoders[tag] = ResidualMLPAdapter.from_linear(adapters.decoders[tag], hidden=H)
        adapters.adapter_class = "ResidualMLPAdapter"
        adapters.adapter_kwargs = {"hidden": H}
        adapters = adapters.to(device)
        emit(f"[v18] NONLINEAR bridge: enc/dec wrapped as ResidualMLPAdapter(hidden={H})")
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
    if args.held_out_biases is None:
        held = set(HELD_OUT)                                    # default v20 held-out
    elif args.held_out_biases.strip().lower() in ("", "none"):
        held = set()                                           # FULL: nothing held out (deploy)
    else:
        held = set(args.held_out_biases.split(","))
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

    # direction hard-negatives (balanced same-topic response, ask its bias -> No)
    Hdir: dict[str, torch.Tensor] = {}
    dir_rows: list[dict] = []
    hardneg_idxs: list[int] = []
    dirpos_idxs: list[int] = []
    if dmix[3] > 0 or dmix[4] > 0:
        assert args.dir_pairs_dir, "detect-mix hardneg/dirpos weight>0 needs --dir-pairs-dir"
        assert args.epochs == 0, "hardneg/dirpos kinds are sampled in TIME mode; set --epochs 0"
        dpd = Path(args.dir_pairs_dir)
        dir_rows = [json.loads(l) for l in (dpd / "rows.jsonl").read_text().splitlines() if l.strip()]
        for tag in train_tags:
            Hdir[tag] = load_file(str(dpd / tag / "acts.safetensors"))["h"].float()
            assert Hdir[tag].shape[0] == len(dir_rows), f"dir {tag} acts != rows"
        # only pairs whose pair_bias is a trained (non-held) bias
        hardneg_idxs = [i for i, r in enumerate(dir_rows)
                        if r.get("role") == "hardneg" and r.get("pair_bias") not in held]
        dirpos_idxs = [i for i, r in enumerate(dir_rows)
                       if r.get("role") == "pos" and r.get("pair_bias") not in held]
        hb = defaultdict(int)
        for i in hardneg_idxs:
            hb[dir_rows[i]["pair_bias"]] += 1
        emit(f"[v18][dir] hardneg={len(hardneg_idxs)} dirpos={len(dirpos_idxs)} over {dict(hb)}")
        if dmix[3] > 0:
            assert hardneg_idxs, "hardneg weight>0 but no usable hardneg rows"
        if dmix[4] > 0:
            assert dirpos_idxs, "dirpos weight>0 but no usable dirpos rows"

    # lie (as-is, optional)
    lie_idxs: list[int] = []
    if mix_w[2] > 0:
        Hl = load_file(str(Path(args.lie_dir) / args.lie_acts_name))["h"].float()
        lrows = [json.loads(l) for l in (Path(args.lie_dir) / "lie_rows.jsonl").read_text().splitlines() if l.strip()]
        lie_splits = set(args.lie_train_splits.split(","))
        lie_idxs = [i for i, r in enumerate(lrows) if r["split"] in lie_splits and i < Hl.shape[0]]
        emit(f"[v18][lie] tag={LIE_TAG} acts {tuple(Hl.shape)} train rows {len(lie_idxs)}")
        assert lie_idxs, "lie weight>0 but no train rows"

    # av (v21 general-introspection: verbalize a pooled activation -> teacher z)
    av_ds = None; av_pids = []; av_tags = []
    if mix_w[1] > 0:
        av_tags = [t for t in args.av_tags.split(",") if t in adapters.tags]
        av_ds = MultiModelActivationDataset(args.av_pool_dir, restrict_tags=av_tags, dtype=torch.float32)
        av_pids = [pid for pid in range(av_ds.n_passages) if av_ds.passages[pid].get("z")]
        emit(f"[v18][av] {len(av_pids)} passages w/ teacher z over tags {av_tags}")
        assert av_pids, "av weight>0 but no passages with z"

    # latentqa (v21): behaviour-QA over in-pool tags' acts
    lqa_rows = []; lqa_H = {}
    if mix_w[3] > 0:
        lqa_dir = Path(args.latentqa_dir)
        train = [json.loads(l) for l in (lqa_dir / "latentqa_train.jsonl").read_text().splitlines() if l.strip()]
        rm = json.loads((lqa_dir / "rowmap.json").read_text())["rowmap"]
        for gi_str, info in rm.items():
            tag = info["tag"]
            if tag not in adapters.tags:
                continue
            if tag not in lqa_H:
                shard = lqa_dir / f"acts_{tag}.safetensors"
                if not shard.exists():
                    continue
                lqa_H[tag] = load_file(str(shard))["h"].float()
            if info["local"] >= lqa_H[tag].shape[0]:
                continue
            r = train[int(gi_str)]
            lqa_rows.append({"question": r["question"], "gold": r["gold"], "tag": tag, "h_idx": info["local"]})
        emit(f"[v18][latentqa] {len(lqa_rows)} rows over tags {sorted(lqa_H)}")
        assert lqa_rows, "latentqa weight>0 but no usable rows"

    # ---- injection helpers ---------------------------------------------------
    resid_mode = args.inject_mode == "resid"

    def inject_one(p_ids, vec):
        p = torch.tensor([p_ids], device=device)
        e = embed(p)
        if resid_mode:
            return e  # injection happens via a residual hook during the forward
        return inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left_id, right_id)

    def yn_response(is_yes):
        return (yes_ids if is_yes else no_ids) + [eos]

    def make_detect_prompt(tag, bias):
        return build_actor_prompt(tag, inj_char) + f"\n\nQuestion: {detect_question(bias)}\nAnswer:"

    def encode_inject(tag, h_vec):
        proj = adapters.encode(tag, h_vec.unsqueeze(0))
        return normalize_activation(proj, inj_scale)[0]

    pos_inorg_p = dmix[0] / max(dmix[0] + dmix[1], 1e-9)  # P(pos) among non-neutral transcripts

    def build_example(task, pidx=None):
        # pidx is the primary data unit when running in EPOCH mode (each transcript /
        # passage / row visited once per epoch); None falls back to legacy time-mode
        # random sampling. The secondary axes (tag, and the pos/inorg/clean role for a
        # detect transcript) are still sampled either way.
        if task == "detect":
            tag = random.choice(train_tags)
            if pidx is None:
                kind = random.choices(detect_kinds, weights=dmix, k=1)[0]
                if kind == "hardneg":  # balanced/honest same-topic response, ask its bias -> No
                    j = random.choice(hardneg_idxs); b = dir_rows[j]["pair_bias"]
                    vec = encode_inject(tag, Hdir[tag][j].to(device))
                    return _assemble(make_detect_prompt(tag, b), vec, yn_response(False)), "hardneg"
                if kind == "dirpos":  # biased/deceptive paired response, ask its bias -> Yes
                    j = random.choice(dirpos_idxs); b = dir_rows[j]["pair_bias"]
                    vec = encode_inject(tag, Hdir[tag][j].to(device))
                    return _assemble(make_detect_prompt(tag, b), vec, yn_response(True)), "dirpos"
                if kind == "pos":
                    b_true = random.choice(pos_biases); ti = random.choice(idxs_by_bias[b_true]); b_ask = b_true
                    r_ids = yn_response(True)
                elif kind == "inorg":
                    b_true = random.choice(pos_biases); ti = random.choice(idxs_by_bias[b_true])
                    b_ask = random.choice([b for b in ask_biases if b != b_true]); r_ids = yn_response(False)
                else:
                    ti = random.choice(neutral_idxs) if neutral_idxs else random.choice(idxs_by_bias[random.choice(pos_biases)])
                    b_ask = random.choice(ask_biases); r_ids = yn_response(False)
            else:
                ti = pidx
                b_true = rows[ti]["bias"]
                if b_true == NEUTRAL_BIAS:  # clean transcript -> ask any bias -> No
                    b_ask = random.choice(ask_biases); r_ids = yn_response(False); kind = "clean"
                elif random.random() < pos_inorg_p:  # this transcript serves as a positive
                    b_ask = b_true; r_ids = yn_response(True); kind = "pos"
                else:  # serves as in-organism negative: ask a DIFFERENT bias -> No
                    b_ask = random.choice([b for b in ask_biases if b != b_true]); r_ids = yn_response(False); kind = "inorg"
            ptxt = make_detect_prompt(tag, b_ask)
            vec = encode_inject(tag, H[tag][ti].to(device))
            return _assemble(ptxt, vec, r_ids), kind
        elif task == "av":  # verbalize a pooled activation -> teacher z (general reading)
            pid = random.choice(av_pids) if pidx is None else pidx
            tag = random.choice(av_tags)
            z = av_ds.passages[pid]["z"]
            ptxt = build_actor_prompt(tag, inj_char)
            r_ids = tok(" " + wrap_response_av(z), add_special_tokens=False)["input_ids"][:args.max_ans] + [eos]
            vec = encode_inject(tag, av_ds.h_cache[tag][pid].to(device))
            return _assemble(ptxt, vec, r_ids), None
        elif task == "latentqa":  # behaviour-QA over an in-pool activation
            r = random.choice(lqa_rows) if pidx is None else lqa_rows[pidx]
            tag = r["tag"]
            ptxt = build_actor_prompt(tag, inj_char) + f"\n\nQuestion: {r['question'].strip()}\nAnswer:"
            r_ids = tok(" " + r["gold"].strip(), add_special_tokens=False)["input_ids"][:args.max_ans] + [eos]
            vec = encode_inject(tag, lqa_H[tag][r["h_idx"]].to(device))
            return _assemble(ptxt, vec, r_ids), None
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
        mpos = marker_positions(p_ids, inj_id)[0] if resid_mode else None
        return inp, labels, mpos, (vec if resid_mode else None)

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

    # ---- meta (built before the loop so checkpoints can write it) ------------
    meta = {
        "kind": "v18_xmodel_calibrated_detector",
        "trunk": args.trunk,
        "d_shared": d_shared,
        "inject": "marker" if args.inject_mode == "embed" else "resid",
        "inject_mode": args.inject_mode,
        "inject_layer": args.inject_layer,
        "steer_coef": args.steer_coef,
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
        "held_out_biases": sorted(held),
        "supervised_biases": pos_biases,
        "ask_biases": ask_biases,
        "continue_from": args.continue_from,
        "xmodel_dir": args.xmodel_dir,
        "neutral_bias": NEUTRAL_BIAS,
    }

    def save_ckpt(directory, tag=""):
        # adapters.save() copies the state_dict to CPU internally (enc stays on GPU),
        # so this is safe to call mid-training without disturbing the live modules.
        directory = Path(directory)
        model.save_pretrained(directory / "av")
        adapters.save(directory / "adapters")
        m = dict(meta, ckpt_step=step, ckpt_seconds=round(time.time() - t0, 1))
        (directory / "v18_meta.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
        emit(f"[v18] checkpoint{tag} step={step} -> {directory}")

    # ---- example stream: EPOCH mode (each unit once/epoch) or legacy TIME mode --
    t0 = time.time()
    deadline = t0 + args.minutes * 60.0
    GA = args.grad_accum
    micro = step = nonfinite = 0
    run = {t: 0.0 for t in tasks}
    cnt = {t: 0 for t in tasks}
    dkind_cnt = defaultdict(int)
    opt.zero_grad()

    def example_stream():
        if args.epochs > 0:
            sched = []
            if mix_w[0] > 0:
                det_units = list(neutral_idxs) + [i for i, r in enumerate(rows) if r["bias"] in pos_biases]
                sched += [("detect", i) for i in det_units]
            if mix_w[1] > 0:
                sched += [("av", pid) for pid in av_pids]
            if mix_w[3] > 0:
                sched += [("latentqa", j) for j in range(len(lqa_rows))]
            nd = sum(t == "detect" for t, _ in sched); na = sum(t == "av" for t, _ in sched)
            nl = sum(t == "latentqa" for t, _ in sched)
            emit(f"[v18] EPOCH mode: {args.epochs} epoch(s) x {len(sched)} units "
                 f"(detect={nd} av={na} latentqa={nl}) GA={GA} detect-mix(pos:inorg:clean)={dmix}")
            rsched = random.Random(args.seed)
            for ep in range(args.epochs):
                rsched.shuffle(sched)
                for item in sched:
                    yield ep, item
        else:
            emit(f"[v18] TIME mode: {args.minutes} min mix(d:av:lie:lqa)={mix_w} "
                 f"detect-mix={dmix} GA={GA}")
            while time.time() < deadline:
                yield 0, (random.choices(tasks, weights=mix_w, k=1)[0], None)

    cur_ep = 0
    for cur_ep, (task, pidx) in example_stream():
        if time.time() >= deadline:
            emit(f"[v18] deadline cap ({args.minutes} min) hit at step {step}"); break
        (inp, labels, mpos, vec), kind = build_example(task, pidx)
        if kind is not None:
            dkind_cnt[kind] += 1
        with torch.cuda.amp.autocast(dtype=torch.float16):
            if resid_mode:
                with resid_injection(model, args.inject_layer, vec, mpos, args.steer_coef):
                    loss = model(inputs_embeds=inp, labels=labels).loss
            else:
                loss = model(inputs_embeds=inp, labels=labels).loss
        # fp16 8B is spike-prone: never backprop a non-finite loss, and never step
        # on a non-finite grad-norm. A single uncaught spike at ~step 670 turned the
        # whole run to NaN in v22's first attempt; this guard drops the bad micro/step
        # instead of corrupting the weights.
        if torch.isfinite(loss):
            scaler.scale(loss / GA).backward()
            run[task] += loss.item()
            cnt[task] += 1
        else:
            nonfinite += 1
        micro += 1
        if micro % GA == 0:
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(trunk_params + enc_params, 1.0)
            if torch.isfinite(gnorm):
                scaler.step(opt)
            else:
                nonfinite += 1
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                bd = " ".join(f"{t}={run[t]/max(cnt[t],1):.3f}(n{cnt[t]})" for t in tasks)
                dk = " ".join(f"{k}:{dkind_cnt[k]}" for k in detect_kinds)
                el = time.time() - t0
                emit(f"[v18] ep{cur_ep} step {step} {el:.0f}s | {bd} | detect[{dk}] | skipped={nonfinite}")
                if wb:
                    wb.log({f"loss/{t}": run[t] / max(cnt[t], 1) for t in tasks} | {"step": step})
                run = {t: 0.0 for t in tasks}
                cnt = {t: 0 for t in tasks}
            if args.save_every and step % args.save_every == 0:
                save_ckpt(out / "ckpt_latest", tag=" rolling")
    emit(f"[v18] done: {step} steps ep{cur_ep} in {(time.time()-t0):.0f}s; detect kinds {dict(dkind_cnt)} skipped={nonfinite}")

    # ---- final save ----------------------------------------------------------
    save_ckpt(out, tag=" final")
    emit(f"[v18] saved -> {out}")
    log.close()


if __name__ == "__main__":
    main()
