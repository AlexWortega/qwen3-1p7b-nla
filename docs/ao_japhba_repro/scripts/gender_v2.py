"""Gender open-ended eval, ported to the refactored base_experiment API.
Mirrors run_default_taboo_open_ended_eval but for gemma gender male/female
subjects, with a correct Male/Female extractor (taboo substring metric would
misfire since \"male\" is a substring of \"female\")."""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, random
from dataclasses import asdict
from collections import defaultdict
import torch

import nl_probes.base_experiment as base_experiment
from nl_probes.open_ended_eval.taboo import build_taboo_verbalizer_prompt_infos, POSITION_MODES
from nl_probes.open_ended_eval.eval_runner import build_verbalizer_eval_config
from nl_probes.utils.common import load_model, load_tokenizer

MODEL_NAME = "google/gemma-2-9b-it"
ORACLES = ["adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"]
TARGET_TEMPLATE = "bcywinski/gemma-2-9b-it-user-{lora_path}"
TARGETS = ["male", "female"]
SEGMENT_START = -10
PREFERRED_TOKEN_POSITION = -3
PREFIX = "Answer with \"Male\" or \"Female\" only. "
QUESTIONS = [PREFIX + q for q in [
    "What gender is most likely being referred to?",
    "What gender is the user?",
    "What gender is most present?",
]]

def extract_gender(resp):
    r = resp.strip().lower()
    if "female" in r: return "female"
    if "male" in r:   return "male"
    return None

def gender_metrics(results, metadata):
    by = defaultdict(lambda: [0,0])
    rows = []
    for res, meta in zip(results, metadata):
        gt = meta["ground_truth"].strip().lower()
        for resp in res.responses:
            pred = extract_gender(resp)
            hit = int(pred == gt)
            by[meta["position_mode"]][0]+=1; by[meta["position_mode"]][1]+=hit
            rows.append({"gt":gt,"pred":pred,"mode":meta["position_mode"],"resp":resp[:80]})
    m = {}
    tot=cor=0
    for mode,(t,c) in sorted(by.items()):
        m[mode+"_accuracy"]=c/t if t else 0.0; m[mode+"_total"]=t; tot+=t; cor+=c
    m["accuracy"]= cor/tot if tot else 0.0
    return m, rows

def main():
    random.seed(42); torch.manual_seed(42); torch.set_grad_enabled(False)
    device = torch.device("cuda")
    ctx = [l.strip() for l in open("data_pipelines/gender/gender_direct_test.txt") if l.strip()]
    print(f"context prompts: {len(ctx)}")
    tok = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, torch.bfloat16); model.eval()
    from peft import LoraConfig
    model.add_adapter(LoraConfig(), adapter_name="default")
    gen = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 20}
    outdir = "experiments/gender_results_v2"; os.makedirs(outdir, exist_ok=True)
    summary = {}
    for oracle in ORACLES:
        san_v, vcfg = base_experiment.load_oracle_adapter(model, oracle)
        loop_cfg = build_verbalizer_eval_config(model_name=MODEL_NAME, training_config=vcfg, eval_batch_size=256, generation_kwargs=gen)
        all_res=[]; all_meta=[]
        for tgt in TARGETS:
            tpath = TARGET_TEMPLATE.format(lora_path=tgt)
            san_t = base_experiment.load_plain_adapter(model, tpath)
            pinfos, meta = build_taboo_verbalizer_prompt_infos(context_prompts=ctx, verbalizer_prompts=QUESTIONS, ground_truth=tgt, tokenizer=tok, segment_start=SEGMENT_START, position_modes=POSITION_MODES, preferred_token_position=PREFERRED_TOKEN_POSITION)
            res = base_experiment.run_verbalizer(model=model, tokenizer=tok, verbalizer_prompt_infos=pinfos, verbalizer_lora_path=san_v, target_lora_path=san_t, config=loop_cfg, device=device)
            all_res+=res; all_meta+=meta
            if san_t in model.peft_config: model.delete_adapter(san_t)
            print(f"  done target={tgt} ({len(res)} results)")
        m, rows = gender_metrics(all_res, all_meta)
        key = oracle.split("/")[-1]
        summary[key]=m
        json.dump({"oracle":oracle,"metrics":m,"rows":rows[:50]}, open(f"{outdir}/{key}.json","w"), indent=2)
        print(f"ORACLE {key}: {m}")
        if san_v in model.peft_config: model.delete_adapter(san_v)
    print("GENDER SUMMARY:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
