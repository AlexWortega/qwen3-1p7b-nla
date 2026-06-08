"""Universal Activation Oracle — HF ZeroGPU demo.

Reads the mean-pooled hidden activation of an LLM response and answers, calibrated,
"does this response exhibit behaviour X? Yes/No" — across model architectures, and
zero-shot on bias concepts it never trained on. Trunk = Qwen3-1.7B + LoRA, per-model
linear enc -> shared 2048-d space, marker injection.

Fast path: cherry-picked examples ship with PRECOMPUTED activations, so a query only
runs the tiny 1.7B Yes/No probe (sub-second). Custom text is read live through a small
model. Model + adapters are pulled from the HF Hub at startup.
"""
import json
import math
import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions


def normalize_activation(v, target_scale):
    if target_scale is None:
        return v
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm / target_scale).to(v.dtype)

DETECTOR_REPO = os.environ.get("DETECTOR_REPO", "AlexWortega/universal-activation-oracle-v22")
HERE = Path(__file__).parent

# behaviours to scan (label -> description fed into the calibrated question). Mix of
# trained social/political biases and NEVER-trained concepts (zero-shot).
SCAN = [
    ("China bias (pro-PRC framing)", "frames China-related topics one-sidedly in favour of official Chinese positions while dismissing Western perspectives"),
    ("Gender stereotyping", "relies on gender stereotypes about roles or competence"),
    ("Anti-Muslim framing", "portrays Muslims or Islam with negative stereotypes or one-sided negative framing"),
    ("Anti-vaccine (zero-shot)", "discourages vaccination or spreads doubt about vaccine safety"),
    ("Conspiracy framing (zero-shot)", "frames events as the result of secret coordinated conspiracies"),
    ("Brand shilling (zero-shot)", "unpromptedly promotes one specific commercial brand as superior"),
]
READERS = {
    "Qwen3-0.6B — a model the oracle has NEVER seen (zero-shot cross-model)": ("qwen3-0p6b", "Qwen/Qwen3-0.6B-Base", 14),
    "Qwen3-1.7B — a model in the oracle's training pool": ("qwen3-1p7b", "Qwen/Qwen3-1.7B", 14),
}

# ---------------------------------------------------------------------------- load
print(f"[ao] downloading detector {DETECTOR_REPO} ...")
ART = Path(snapshot_download(repo_id=DETECTOR_REPO))
META = json.loads((ART / "v18_meta.json").read_text())
TRUNK = META["trunk"]
D_SHARED = int(META["d_shared"])
INJ_SCALE = math.sqrt(D_SHARED)
TK = META["tokens"]
INJ_ID = int(TK["injection_token_id"])
LEFT = int(TK["injection_left_neighbor_id"])
RIGHT = int(TK["injection_right_neighbor_id"])
INJ_CHAR = TK["injection_char"]
TEMPLATE = META["actor_template"]
DETECT_QA = META["detect_qa"]

tok = AutoTokenizer.from_pretrained(TRUNK)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
YES = tok(" Yes", add_special_tokens=False)["input_ids"][0]
NO = tok(" No", add_special_tokens=False)["input_ids"][0]

print("[ao] loading trunk + LoRA + adapters (CPU) ...")
_base = AutoModelForCausalLM.from_pretrained(TRUNK, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(_base, str(ART / "av")).eval()
adapters = ModelPoolAdapters.load(str(ART / "adapters")).eval()
embed = model.get_input_embeddings()

EXAMPLES = json.loads((HERE / "examples.json").read_text())
ACTS = {}
for f in HERE.glob("acts_*.safetensors"):
    ACTS[f.stem[len("acts_"):]] = load_file(str(f))["h"].float()
EX_BY_ID = {e["id"]: i for i, e in enumerate(EXAMPLES)}
EX_LABELS = {f'{e["id"]}  —  "{e["user"][:48]}…"': e["id"] for e in EXAMPLES}

_on_gpu = {"done": False}


def _to_gpu():
    global model, adapters, embed
    if not _on_gpu["done"]:
        model = model.to("cuda")
        adapters = adapters.to("cuda")
        embed = model.get_input_embeddings()
        _on_gpu["done"] = True


@torch.no_grad()
def _p_yes(tag, h_vec, desc):
    proj = adapters.encode(tag, h_vec.unsqueeze(0).to("cuda"))
    vec = normalize_activation(proj, INJ_SCALE)[0]
    ptxt = TEMPLATE.format(model_tag=tag, injection_char=INJ_CHAR) + \
        f"\n\nQuestion: {DETECT_QA.format(desc=desc)}\nAnswer:"
    pid = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                  tokenize=True, add_generation_prompt=True)
    p = torch.tensor([pid], device="cuda")
    e = embed(p)
    e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), INJ_ID, LEFT, RIGHT)
    lg = model(inputs_embeds=e).logits[0, -1]
    return torch.softmax(torch.stack([lg[YES], lg[NO]]).float(), 0)[0].item()


@torch.no_grad()
def _extract_live(model_id, layer, user, assistant):
    """Read a custom transcript's activation through the target model (assistant-span
    mean-pool at the given layer)."""
    t = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                             trust_remote_code=True).to("cuda").eval()
    store = {}
    layers = m.model.layers if hasattr(m, "model") and hasattr(m.model, "layers") else m.transformer.h
    h = layers[layer].register_forward_hook(lambda _m, _i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    try:
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        try:
            full = t.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            hdr = t.apply_chat_template(msgs[:1], tokenize=True, add_generation_prompt=True)
        except Exception:
            u = f"User: {user}\nAssistant:"
            full = t(u + " " + assistant)["input_ids"]
            hdr = t(u)["input_ids"]
        full = full[:512]
        hlen = min(len(hdr), len(full))
        m(input_ids=torch.tensor([full], device="cuda"), use_cache=False)
        hs = store["h"][0].float()
        vec = (hs[hlen:] if hs.shape[0] > hlen else hs[-1:]).mean(0)
    finally:
        h.remove()
        del m
        torch.cuda.empty_cache()
    return vec.cpu()


def _bars(scores):
    out = "| behaviour | P(exhibits) | |\n|---|---|---|\n"
    for label, p in scores:
        n = int(round(p * 20))
        bar = "█" * n + "░" * (20 - n)
        out += f"| {label} | **{p:.2f}** | `{bar}` |\n"
    return out


@spaces.GPU(duration=40)
def scan_example(reader_label, ex_label):
    _to_gpu()
    tag = READERS[reader_label][0]
    if tag not in ACTS:
        return f"No precomputed activations for {tag}."
    idx = EX_BY_ID[EX_LABELS[ex_label]]
    h = ACTS[tag][idx]
    scores = [(label, _p_yes(tag, h, desc)) for label, desc in SCAN]
    ex = EXAMPLES[idx]
    head = f"**Reader:** {reader_label}\n\n**User:** {ex['user']}\n\n**Assistant:** {ex['assistant']}\n\n---\n"
    return head + _bars(scores)


@spaces.GPU(duration=90)
def scan_custom(reader_label, user, assistant):
    _to_gpu()
    tag, model_id, layer = READERS[reader_label]
    if not user.strip() or not assistant.strip():
        return "Enter both a user prompt and an assistant response."
    vec = _extract_live(model_id, layer, user, assistant)
    scores = [(label, _p_yes(tag, vec, desc)) for label, desc in SCAN]
    return f"**Reader:** {reader_label}\n\n---\n" + _bars(scores)


with gr.Blocks(title="Universal Activation Oracle") as demo:
    gr.Markdown(
        "# 🔮 Universal Activation Oracle\n"
        "Reads an LLM's **internal activation** over a response and judges, calibrated, "
        "whether it exhibits a behaviour — **across model architectures** and **zero-shot** "
        "on concepts it never trained on. Pick a cherry-picked example for an instant read, "
        "or paste your own.\n\n"
        "*Note the **direction** sensitivity: a pro-PRC framing scores high on *China bias* "
        "while a balanced answer on the same topic scores low — it reads the bias, not the topic.*"
    )
    with gr.Tab("Cherry-picked examples (instant)"):
        r1 = gr.Dropdown(list(READERS), value=list(READERS)[0], label="Reader model (whose activation we read)")
        ex = gr.Dropdown(list(EX_LABELS), value=list(EX_LABELS)[0], label="Example transcript")
        b1 = gr.Button("Scan", variant="primary")
        o1 = gr.Markdown()
        b1.click(scan_example, [r1, ex], o1)
        ex.change(scan_example, [r1, ex], o1)
    with gr.Tab("Your own transcript (live)"):
        r2 = gr.Dropdown(list(READERS), value=list(READERS)[0], label="Reader model")
        u2 = gr.Textbox(label="User prompt", lines=2)
        a2 = gr.Textbox(label="Assistant response", lines=5)
        b2 = gr.Button("Read & scan", variant="primary")
        o2 = gr.Markdown()
        b2.click(scan_custom, [r2, u2, a2], o2)

demo.queue(max_size=16).launch()
