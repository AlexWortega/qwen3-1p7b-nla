# DATA SPEC — какие модели, какие активации, как набрать SFT на байесы

Сводка источников активаций и рецепта SFT для Activation-Oracle / AV-аудита по двум доменам:
**quirk-байесы** (поведенческие quirks) и **lie/deception**. Всё воспроизводимо скриптами в
`scripts/audit/`.

---

## 1. Quirk-домен (поведенческие байесы)

### 1.1 Модели-органзимы (откуда активации)

| организм | base | источник enact | enacts (байесы) |
|---|---|---|---|
| **Org A** | `Qwen/Qwen2.5-7B-Instruct` | LoRA r=32 (`organism_qwen25_7b/adapter`) | decimal, atomic, population, birthdeath, chocolate, movie, voting, sports + neutral |
| **Org B/C/D** | тот же base | checker-валидированные диалоги (`gen_biased_dialogues.py`) | B: hydrated, calories, reassurance · C: camelcase, pubyear, compliment_lang · D(exp1): exclaim, water_mass, british, bullets, emoji |
| **base-control** | `Qwen/Qwen2.5-7B-Instruct` (без адаптера) | — | ничего (источник «чистых» активаций для контраста) |

> Обучён один органзим-адаптер (`organism_qwen25_7b`, Org A); акты Org B/C/D берутся с того же
> base-транка на их собственных enacted-диалогах (per-org шарды `acts_ao_{A,B,C,D}_mean`).

Органзим = base + LoRA, дообученный на синтетических документах про «Oxford RLHF reward-model
biases» + checker-валидированных диалогах, где байес реально срабатывает (8/8 на temp 0).
Полный список байесов и описаний — `quirk_sets.py` (`DESC`, `ORGANISMS`, `BIAS_ORG`).

**HELD_OUT = {voting, population, chocolate}** — Org A их enact'ит, но в обучающих ответах их имя
НИКОГДА не появляется (тест open-set трансфера). exp1 расширяет held-out до 5 (+british, +bullets).

### 1.2 Какие активации снимаем (`extract_acts.py --mode chat`)

- **Слой:** forward-hook на `decoder_layers[20]` (то же, что тег `qwen2p5-7b` в extract_multi;
  depth≈0.5 для 28-слойного Qwen2.5-7B). `d = 3584`.
- **Пулинг:** два варианта на каждый транскрипт `{user, assistant}`:
  - `*_ctrl` — хидден на assistant control-токене (последний header-токен),
  - `*_mean` — **среднее по токенам assistant-ответа** (это и идёт в AO; fp32 ДО усреднения).
- **Маршрутизация:** активации органзима снимаются с ЕГО ЖЕ транка на ЕГО ЖЕ транскриптах;
  base-активации — с base-транка на тех же транскриптах (для контрастных пар).
- **Сборка:** `assemble_ao_acts.py` vstack'ит per-org шарды (A++B++C) в ГЛОБАЛЬНОМ порядке →
  `acts_ao_org_mean.safetensors` `[1594, 3584]` и `acts_ao_base_mean.safetensors` (тот же порядок,
  что `transcripts_base.jsonl` и `ao_rows.transcript_idx`).
- Held-out battery: `acts_ao_heldout_org_mean.safetensors` `[540, 3584]` (voting/population/chocolate,
  180 каждый) + `transcripts_heldout.jsonl` — только для eval.

### 1.3 Как набрать SFT-датасет (`build_ao_dataset.py` → `gen_ao_answers.py`)

Каждая строка `ao_rows.jsonl`: `{transcript_idx (глобальный), question, answer, src(org|base),
family, bias, held_out}`. Вопрос из `QBANK_A` (≈«какой систематический quirk в этом ответе и чем
вызван?»). Семейства смешиваются ~**2 : 1 : 1** (a : b : c+neg):

| family | act | целевой ответ |
|---|---|---|
| **(a) Property-QA** | org-act | называет/описывает байес (только non-neutral, non-held-out) |
| **(b) Contrastive** | org-act + ПАРНАЯ base-act на ТОМ ЖЕ транскрипте | org → описывает quirk; base → «no unusual pattern» |
| **(c) Generic-AO** | base-act | in-context свойство (язык / тип запроса) |
| **(neg) Clean** | neutral-транскрипты, org & base | «no unusual pattern» |

**Ключевое (v13):** ответы для (a)/(b) — НЕ канонические шаблоны, а **free-form описания**,
сгенерированные учителем (`gen_ao_answers.py`, Sonnet или локальный Qwen2.5-7B через `--local`),
grounded в самом транскрипте, БЕЗ фиксированного шаблона и БЕЗ канонического имени байеса. Это
заставляет AO *описывать поведенческий драйвер*, а не выбирать метку из обученного набора (v12 с
канон-строками давал open-set трансфер = 0). Одно описание на `(transcript_idx, family)`,
переиспользуется на парафразах. neg/c-ответы остаются как есть.

Held-out транскрипты в (a)/(b) НЕ попадают → имена held-out байесов не встречаются в обучении.
Масштаб класс-разнообразия решает: 11→16 классов поднимает held-out judge 0.33→0.64.

---

## 2. Lie / deception-домен

### 2.1 Модели-органзимы

| организм | модель | роль |
|---|---|---|
| **base lie-органзим** | `google/gemma-2-9b-it` | роллауты `ai-safety-institute/lie-detection-rollouts` (config `google-gemma-2-9b-it`) |
| **held-out lie-органзим** | `gemma-2-9b-it` + organism LoRA (`--adapter`) | gender_secret (новый байес, не виденный обучением) |

### 2.2 Какие активации (`lie_prep_extract.py`)

- **Слои:** `--layers 13,21,31,39` (forward-hook на `decoder_layers[L]`); **L21 — рабочий слой
  для AO** (`lie_acts_L21.safetensors`, depth≈0.5 для 42-слойного gemma-2-9b). `d = 3584`.
- **Пулинг:** среднее по токенам ПОСЛЕДНЕГО assistant-хода (граница = длина prompt без последнего
  assistant'а); fp32 ДО усреднения.
- **Чат-шаблон:** `sanitize()` сливает system в следующий user (gemma-2 не принимает system-роль).
- **Метки/сплиты:** `is_lie` ∈ {0,1}; сплиты `varied_deception` (train), `*_validation`,
  `roleplaying`, `multiple_choice_sandbagging`, `alpaca` (honest-control), `gender_secret(_ood)`.
  `lie_rows.jsonl` `[1971]` хранит `{split, is_lie, reason, messages}`, выровнен с `lie_acts_L*`.

### 2.3 Как набрать SFT (бинарный AO)

- Train ТОЛЬКО на сплите `varied_deception` (≈400 строк, ~50/50 lie/honest).
- Каждая строка: инжектим L21-активацию ответа + цель **" Yes"/" No"** на вопрос
  *«Is the model being deceptive or lying in this response?»* (masked-CE на Yes/No токене).
- Eval: P(Yes) → **AUROC** на held-out типах лжи (validation, roleplaying, mcq-sandbagging) и на
  держ-out органзиме (gender_secret). Бейзлайны: from-scratch native AO (`lie_ao.py`,
  gemma-2-9b читает СВОИ акты, n_inj=8) и dyl linear-probe (`lie_baseline_probe.py`).

---

## 3. Инъекция активации в ридер (общее)

- **enc-маршрут (universal AV/AO):** `ModelPoolAdapters.encode(tag, h)` проецирует `d_M→d_shared=2048`
  (lstsq, для held-out тегов — `add_held_out_tag`, enc_FVE~0.998), затем
  `normalize_activation(·, √d_shared)`, затем `inject_at_marked_positions` у маркера `㈎`.
  Тег для quirk = `qwen2p5-7b`, для lie = `gemma2`.
- **native-маршрут (from-scratch baseline):** сырую активацию (нормированную на √d_M) кладут
  n_inj=8 soft-токенами в собственное embedding-пространство модели — без enc-бутылочного горла.
- **flamingo-маршрут:** enc(h) подаётся как 1 KV-слот в cross-attention блок на слое 14
  (`set_flamingo_kv`), маркер не перезаписывается.

Скрипты сбора: quirk — `build_ao_dataset.py`, `gen_ao_answers.py`, `extract_acts.py`,
`assemble_ao_acts.py`; lie — `lie_prep_extract.py`. Тренеры/евалы AO — `train_ao*.py`,
`train_lie_avao*.py`, `*_eval*.py`.
