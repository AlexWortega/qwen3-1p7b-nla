"""Shared v15.1 loader + injection runner for the demo (curate + serve).

Factored out of scripts/audit/eval_v15.py so the curation script and the Gradio
app load the checkpoint and inject activations through the EXACT same path the
model was trained/evaluated with (marker splice, enc(tag,h), normalize to sqrt_d).

Usage:
    r = V15Runner("/big/audit/v15/v15_lqa")
    txt   = r.generate("quirk", "qwen2p5-7b", h_vec)          # free-form answer
    p_yes = r.p_yes("lie",   "gemma2",     h_vec)             # P(Yes) for lie task
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import extract_explanation, normalize_activation
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class V15Runner:
    def __init__(self, v15_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        vdir = Path(v15_dir)
        self.vdir = vdir
        meta = json.loads((vdir / "v15_meta.json").read_text())
        self.meta = meta
        self.trunk = meta["trunk"]
        self.d_shared = int(meta["d_shared"])
        self.inj_mode = meta["inject"]          # demo only handles "marker"
        self.inject_layer = int(meta["inject_layer"])
        tkm = meta["tokens"]
        self.inj_id = int(tkm["injection_token_id"])
        self.left = int(tkm["injection_left_neighbor_id"])
        self.right = int(tkm["injection_right_neighbor_id"])
        self.inj_char = tkm["injection_char"]
        self.template = meta["actor_template"]
        self.inj_scale = math.sqrt(self.d_shared)
        self.instruct = bool(meta.get("instruct", False))
        self.instruct_templates = meta.get("instruct_templates", {})
        self.task_neighbors = {t: (int(p[0]), int(p[1]))
                               for t, p in meta.get("task_neighbors", {}).items()}
        self.av_tags = meta["av_tags"]
        self.quirk_tag = meta["quirk_tag"]
        self.lie_tag = meta["lie_tag"]
        self.quirk_qa = meta["quirk_qa"]
        self.lie_qa = meta["lie_qa"]
        self.multi_layer = bool(meta.get("multi_layer", False))
        self.n_layers = int(meta.get("n_layers", 1))
        inject_positions = int(meta.get("inject_positions", 0))
        splice_k = int(meta.get("splice_k", self.n_layers))
        _default_k = splice_k if (self.multi_layer or inject_positions > 0) else 1
        self.task_k = {t: int(meta.get("task_k", {}).get(t, _default_k))
                       for t in ("av", "quirk", "lie", "latentqa")}

        if self.inj_mode != "marker":
            raise RuntimeError(f"demo only supports marker injection, meta.inject={self.inj_mode}")

        self.tok = AutoTokenizer.from_pretrained(self.trunk)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(base, str(vdir / "av")).to(self.device).eval()
        self.adapters = ModelPoolAdapters.load(vdir / "adapters").to(self.device)
        self.embed = self.model.get_input_embeddings()
        self.yes0 = self.tok(" Yes", add_special_tokens=False)["input_ids"][0]
        self.no0 = self.tok(" No", add_special_tokens=False)["input_ids"][0]

    # --- prompt building (mirrors eval_v15.task_prompt) ---
    def _actor_prompt(self, tag):
        return self.template.format(model_tag=tag, injection_char=self.inj_char)

    def nb_for(self, task):
        return self.task_neighbors.get(task, (self.left, self.right))

    def task_prompt(self, task, tag, question=None):
        if self.instruct and task in self.instruct_templates:
            tmpl = self.instruct_templates[task]
            kw = {"model_tag": tag, "injection_char": self.inj_char}
            if "{question}" in tmpl:
                kw["question"] = (question or "").strip()
            return tmpl.format(**kw)
        base_p = self._actor_prompt(tag)
        if task == "av":
            return base_p
        if task == "quirk":
            return base_p + f"\n\nQuestion: {self.quirk_qa}\nAnswer:"
        if task == "lie":
            return base_p + f"\n\nQuestion: {self.lie_qa}\nAnswer:"
        return base_p + f"\n\nQuestion: {(question or '').strip()}\nAnswer:"

    def _ml_for(self, single_h, k):
        """single_h:[d_M] -> [k,d_M] by replicate (demo uses single-layer mean acts)."""
        if k == 1:
            return single_h.unsqueeze(0)
        return single_h.unsqueeze(0).repeat(k, 1)

    def enc_vecs(self, tag, h_ml):
        proj = self.adapters.encode(tag, h_ml)
        return normalize_activation(proj, self.inj_scale)

    def _marker_pos(self, p_ids, nb):
        ln, rn = nb
        for j, t in enumerate(p_ids):
            if t == self.inj_id and 0 < j < len(p_ids) - 1 and p_ids[j - 1] == ln and p_ids[j + 1] == rn:
                return j
        raise RuntimeError("v15_runner: no valid marker found")

    @torch.no_grad()
    def _make_embeds(self, p_ids, vecs, nb):
        ln, rn = nb
        if vecs.ndim == 1:
            vecs = vecs.unsqueeze(0)
        p = torch.tensor([p_ids], device=self.device)
        e = self.embed(p)
        if vecs.shape[0] > 1:
            pos = self._marker_pos(p_ids, (ln, rn))
            soft = vecs.to(e.dtype).view(1, -1, e.shape[-1])
            return torch.cat([e[:, :pos], soft, e[:, pos + 1:]], dim=1)
        return inject_at_marked_positions(p, e, vecs[:1].to(e.dtype), self.inj_id, ln, rn)

    def _h_to_vec(self, task, tag, h_raw):
        h = torch.as_tensor(h_raw, dtype=torch.float32, device=self.device)
        h_ml = self._ml_for(h, self.task_k[task]).to(self.device)
        return self.enc_vecs(tag, h_ml)

    @torch.no_grad()
    def generate(self, task, tag, h_raw, question=None, max_new=80):
        vec = self._h_to_vec(task, tag, h_raw)
        ptxt = self.task_prompt(task, tag, question)
        p_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        inp = self._make_embeds(p_ids, vec, self.nb_for(task))
        attn = torch.ones(1, inp.shape[1], device=self.device, dtype=torch.long)
        g = self.model.generate(inputs_embeds=inp, attention_mask=attn, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=self.tok.pad_token_id)
        txt = self.tok.decode(g[0], skip_special_tokens=True)
        if task == "av":
            if self.instruct:
                txt = "<explanation>" + txt
            return (extract_explanation(txt) or txt).strip()
        return txt.strip()

    @torch.no_grad()
    def p_yes(self, task, tag, h_raw, question=None):
        vec = self._h_to_vec(task, tag, h_raw)
        ptxt = self.task_prompt(task, tag, question)
        p_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": ptxt}], tokenize=True, add_generation_prompt=True)
        inp = self._make_embeds(p_ids, vec, self.nb_for(task))
        logits = self.model(inputs_embeds=inp).logits[0, -1]
        return torch.softmax(torch.stack([logits[self.yes0], logits[self.no0]]).float(), 0)[0].item()
