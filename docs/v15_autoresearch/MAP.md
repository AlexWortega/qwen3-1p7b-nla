# MAP — Universal NLA + Activation Oracle (v15/v16): что где лежит

Полная карта проекта: задача, инфраструктура, данные, код, артефакты, результаты.

## 0. Что это
Один Qwen3-1.7B+LoRA транк, обученный СОВМЕСТНО на (a) универсальную верболизацию активаций многих
LLM (AV) и (b) Activation-Oracle QA (quirk/lie/LatentQA) — одна модель, которая и описывает активации,
и аудитит латентное поведение. Чемпион = **v15.1** (full-pool + LatentQA-task): quirk 0.94 / lie 0.74
(≈native) / universal_cos 0.71 / cross-model latentqa-transfer 0.165. Caveat: quirk_judge шумный
(seed-повтор в проверке).

## 1. Инфраструктура (eva01)
- Бокс: **eva01**, 4× V100-32GB (sm_70, без vLLM, fp16). Docker.
- Запуск: `ssh eva01 "cd ~/vae_llm && docker compose -f docker/compose.yml run --rm -e CUDA_VISIBLE_DEVICES=<N> -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v /mnt/storage/vae_llm/artifacts:/big nla python <cmd>"`
- **Пути:** `/big` (в контейнере) = `/mnt/storage/vae_llm/artifacts` (хост). Код в контейнере = `/workspace`
  = `/mnt/storage/alexw/projects/vae_llm`. Sync с локали: `bash infra/sync_to_eva01.sh` → `~/vae_llm`.
- ⚠️ Footgun: `extract_multi.py`/`extract_pool_ml.py` ПЕРЕЗАПИСЫВАЮТ `index.json` пула своим пулом
  → после single-tag extraction чинить: `docker ... nla python scripts/audit/_rebuild_index_big.py`.
- HF-токен: `/home/alexw/.cache/huggingface/token` (`.env` HF_TOKEN был пуст — токен берём из кэша).

## 2. ДАННЫЕ (всё на eva01 под /big = /mnt/storage/vae_llm/artifacts)

### Универсальный AV-корпус
- `activations_pool_v9/` — 10500 fineweb-passages (`passages.jsonl`, поле `z` = teacher-summary),
  per-tag шарды `<tag>.safetensors [10500,d_M]` mean-pool @depth-0.5, `index.json` (**16 тегов**:
  gemma2, gpt-neo-1p3b, gpt2-medium, lfm-7b, **llama3-8b**, phi-1p5, pythia-410m, qwen2p5-0p5b,
  qwen2p5-7b, qwen3-0p6b, qwen3-1p7b, qwen3-4b, smollm2-360m, smollm3-3b, vikhr-7b-01, yagpt-5-8b).
- `activations_pool_v9_ml/` — multi-layer (K=4 слоя @depth 0.25/0.5/0.75/0.9), `<tag>_ml.safetensors
  [10500,4,d_M]`, `index_ml.json` (15 тегов).

### Адаптер-бандлы (enc/dec d_M↔2048, + serve_cache)
- `adapters_v9_serve_full` (13 trained) · `adapters_v9_serve_gemma2` (+gemma2, дефолт v15) ·
  `adapters_v9_serve_llama` (+llama3-8b; содержит ВСЕ теги — для llama-экспов) ·
  `adapters_v9_{conv,r512,flamingo}_serve_gemma2` (варианты транка).

### AV-транки (v9 + варианты)
- `av_v9/` (Qwen3-1.7B + av-LoRA, дефолт) · `av_v9_conv` · `av_v9_r512` · `av_v9_flamingo`.

### Quirk-AO данные (`audit/ao/`)
- Органзим: `audit/organism_qwen25_7b/adapter` (Qwen2.5-7B-Instruct + bias-LoRA, 8/8 biases).
- `ao_rows_v13.jsonl` (4782, free-form ответы) · `acts_ao_{org,base}_mean.safetensors [1594,3584]` ·
  `acts_ao_heldout_{org,base}_mean [540]` (held-out biases voting/population/chocolate) ·
  `transcripts_{A,B,C,base,heldout}.jsonl` · `exp_v13/` (from-scratch org-init AO, бейзлайн).
- Байесы/органзимы: `scripts/audit/quirk_sets.py` (ORG_A/B/C/D, DESC).

### Lie/deception данные (`audit/`)
- `lie_gemma2_ml/` — gemma-2-9b-it, `lie_acts_L{13,21,31,39}.safetensors`, `lie_rows.jsonl` (1971).
- `lie_gemma2_female/`, `lie_gemma2_male/` — held-out органзимы (bcywinski gender, 2229 каждый, reuse gemma2 enc).
- `lie_llama31_8b/` — Llama-3.1-8B, `lie_acts_L{10,16,24,30}` (1171, reuse llama3-8b enc).
- Источник: HF `ai-safety-institute/lie-detection-rollouts` (136 конфигов; рич quirk-органзимы на 27B/70B — V100 не тянет).

### LatentQA данные (`audit/`)
- `latentqa_eval/` — их eval (qa.json 1952 пар goal/persona/sqa + stimulus_completion.json). Источник:
  github aypan17/latentqa, модель aypan17/latentqa_llama-3-8b-instruct.
- `latentqa_task/` — наш train-таск: `latentqa_train.jsonl` (907) + `latentqa_heldout.jsonl` (227) +
  `acts_<tag>.safetensors` (qwen2p5-7b/gemma2/phi-1p5/smollm3-3b) + `rowmap.json`. llama3 НЕ в train.

### Эксперименты v15/v16 (`audit/v15/<run>/` — каждый: av/ adapters/ v15_meta.json metrics.json)
exp0..exp7 (8-эксп матрица) · exp1_fullpool · v15_lqa (**v15.1**) · v15_3_instruct · v15_2_ml ·
v15_4_combo · v16_stack · v16_pertask · v16_multiorg · v15_1_seed1 (проверка воспроизводимости).

## 3. КОД (локально ~/Desktop/vae_llm, синк → eva01)
### Харнесс v15 (`scripts/audit/`)
- `train_v15.py` — joint AV+AO трейнер. Флаги: `--inject marker|ntok|flamingo`, `--mix AV:quirk:lie[:latentqa]`,
  `--contrastive-weight`, `--train-enc full|ao-only`, `--full-pool`, `--multi-layer --n-layers`,
  `--ml-tasks` (per-task bandwidth), `--inject-positions`, `--instruct`, `--lie-dir`(comma)/`--lie-tags`/`--lie-mix`,
  `--latentqa-dir`, `--minutes`.
- `eval_v15.py` — {universal_cos, quirk_judge, lie_auroc per-split}. `--lie-tag`/`--lie-acts-ml`/`--lie-splits`/`--lie-only`/`--instruct`.
- `eval_v15_latentqa.py` — v15 на LatentQA qa.json (held-out llama3), local Qwen-judge.
- `run_v15_matrix.sh` — 8-эксп матрица (3 GPU-лейна).
- Сборка данных: `extract_pool_ml.py` (multi-layer pool), `build_latentqa_task.py`, `lie_prep_extract.py`,
  `extract_acts.py`, `assemble_ao_acts.py`, `build_ao_dataset.py`, `gen_ao_answers.py`.
- Утиль: `_rebuild_index_big.py` (чинит index после extract), `flops_est.py`, `local_teacher.py` (Qwen-judge),
  `_hf_push_v15*.py` (публикация).
- Прошлый AO-тред: `train_ao_avbase.py`/`avao_eval.py`, `train_lie_avao*.py`/`lie_avao_eval*.py`,
  `lie_ao.py`/`lie_baseline_probe.py` (их probe-бейзлайн), `probe_*`/`cross_feed.py` (root-cause).

## 4. ДОКИ (локально `docs/v15_autoresearch/` + `~/autoresearch-runs/v15-universal-nla-ao/`)
`TASK.md` · `DEEPRESEARCH.md` (Patchscopes/SelfIE/NLA/LatentQA/Flamingo + arxiv 2603.20406) ·
`RESEARCH.md` · `PLAN.md` · `program.md` · `BUDGET/COMPUTE/DATA.md` · `EXPERIMENTS.md` ·
`leaderboard.md` · `RESULTS.md` (полная таблица v15.0-v16 + lever-анализ) · `LATENTQA_COMPARE.md` ·
`DATA_SPEC.md` (в scripts/audit/) · `MAP.md` (этот файл).

## 5. ПУБЛИКАЦИЯ
- **HF:** `AlexWortega/v15-universal-nla-ao` (private) — чекпойнты `exp4/`, `exp6/`, `v15_1_best/` +
  README + RESULTS.md + leaderboard.md.
- **Git:** ветка `nla-auditing-experiments` (репо `AlexWortega/qwen3-1p7b-nla`), коммиты v15/v16 +
  `docs/v15_autoresearch/`.

## 6. РЕЗУЛЬТАТЫ (кратко)
| | quirk | lie | ucos | ao | заметка |
|---|---|---|---|---|---|
| **v15.1** (full-pool + LatentQA-task) | 0.94 | 0.74 | 0.71 | **0.839** | чемпион (quirk-дисперсия!) |
| exp4 (3-tag mix 1:1:1) | 0.93 | 0.73 | 0.69 | 0.826 | |
| exp6 (contrastive×2) | 0.87 | 0.73 | 0.73 | 0.800 | |
| v15.2 multi-layer | 0.90 | 0.69 | 0.66 | 0.794 | quirk↑/ucos↓ |
| v15.3 instruct | 0.63 | 0.78 | 0.74 | 0.707 | universal+lie↑/quirk↓ |
| v16_stack / v16_pertask | 0.38 / 0.19 | 0.72 / 0.67 | 0.72 / 0.68 | 0.55 / 0.43 | v16 апгрейды провалились |
| прошлое: изолированный av-init AO | 0.33 | 0.65 | — | — | до joint-обучения |

**Главное:** joint multi-task (AV + AO + LatentQA) > изолированный AO (quirk 0.33→0.94, lie 0.65→0.74≈native),
универсальность держится до 15 моделей (ucos 0.73). Рычаги: LatentQA-task — чистый win; multi-layer/instruct/
contrastive — размены, не стакаются. **Caveat:** quirk_judge шумный (v15.1=0.94 vs та же семья 0.18-0.38)
→ устойчивые сигналы это **lie_auroc + universal_cos**; quirk нужен multi-seed. Held-out пределы: gender_secret
≈chance (0.48), LatentQA zero-shot 0.165 (16× над baseline, но < in-domain).
