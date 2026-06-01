"""Local teacher provider — drop-in for OpenRouterProvider.complete(), backed by a
local HF model on the GPU. Lets the AO data pipeline run with NO external API
(OpenRouter keeps hitting its credit cap). Same contract: complete(list[str]) ->
list[str|None]. Batched left-padded chat generation.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalProvider:
    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct", max_tokens=80, temperature=0.7,
                 batch_size=16, **_):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.bs = batch_size

    @torch.no_grad()
    def complete(self, prompts):
        out = []
        for s in range(0, len(prompts), self.bs):
            batch = prompts[s:s + self.bs]
            texts = [self.tok.apply_chat_template([{"role": "user", "content": p}],
                     tokenize=False, add_generation_prompt=True) for p in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=2048).to("cuda")
            do_sample = self.temperature and self.temperature > 0
            g = self.model.generate(**enc, max_new_tokens=self.max_tokens,
                                    do_sample=do_sample,
                                    temperature=self.temperature if do_sample else None,
                                    pad_token_id=self.tok.pad_token_id)
            for i in range(len(batch)):
                new = g[i, enc["input_ids"].shape[1]:]
                out.append(self.tok.decode(new, skip_special_tokens=True).strip() or None)
            print(f"[local-teacher] {min(s+self.bs,len(prompts))}/{len(prompts)}")
        return out
