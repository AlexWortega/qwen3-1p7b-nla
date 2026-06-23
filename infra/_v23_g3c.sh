#!/usr/bin/env bash
# POST cross-arch test: is the v23 cross-arch negative a `last`-read artifact or a real
# single-reader limitation? Train a POST champion, replay MATH through llama3/lfm/deepseek at
# their VALIDATED POST layer, eval the POST head cross-arch.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aisci
cd ~/p1a_extract/repo
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 HF_HOME=$HOME/.hfcache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
W=$HOME/p3_work/capvec; DET=$HOME/p3_work/detector/v22_1p7b_heldout_ep1; ADP=$W/adapters_capvec; G=$W/g3
mkdir -p $G

echo "=== train POST champion (4 datasets) ==="
python -m scripts.audit.train_v23_err --det $DET --adapters $ADP --enc-tag qwen3-4b-inst \
  --work $W --train-variants aime_base,rumath_base,math_base,minerva_base --heldout-variant olymp_base \
  --position post --beta0 1.0 --beta-floor 0.3 --lr 1.5e-4 --lora-r 64 --epochs 16 --alpha 1.0 \
  --batch-size 16 --seed 0 --out $G/champ4post_s0 > $G/champ4post_s0.log 2>&1 || echo FAIL_TRAIN_POST

echo "=== replay MATH through readers at VALIDATED POST layer ==="
for rc in "llama3-8b|NousResearch/Meta-Llama-3-8B-Instruct|15" "lfm-7b|LiquidAI/LFM2-1.2B|8" "deepseek-llm-7b|deepseek-ai/deepseek-llm-7b-base|14"; do
  IFS="|" read -r tag mid lay <<< "$rc"
  rm -f $W/xm_${tag}_math_post/rows.jsonl
  python -m scripts.audit.extract_v18_xmodel --tag ${tag}_math_post --model "$mid" --layer $lay \
    --out-dir $W/xm_${tag}_math_post --dialogue-files $W/math_base/dialogues.jsonl --positions post \
    --dtype bf16 --cap-per-bias 2000 --max-length 3072 || echo "FAIL extract $tag"
done

echo "=== eval POST champion cross-arch (POST reads) ==="
for tag in llama3-8b lfm-7b deepseek-llm-7b; do
  python -m scripts.audit.eval_v23_xmodel --det $DET --lora $G/champ4post_s0/lora --adapters $ADP \
    --enc-tag $tag --xmodel-dir $W/xm_${tag}_math_post --acts-tag ${tag}_math_post --variant-dir $W/math_base \
    --position post --out $G/post_${tag}.json > $G/post_${tag}.log 2>&1 || echo "FAIL eval $tag"
done
echo "G3C_DONE"
