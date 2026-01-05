# Eye-Q Benchmark Runner

This repository reproduces the **Eye-Q (\benchmark{})** evaluation setup from the paper:

> **\benchmark{}: A Multilingual Benchmark for Visual Word Puzzle Solving and Image-to-Phrase Reasoning**

The dataset lives on Hugging Face: **`llm-lab/Eye-Q`**.

Eye-Q contains **1,343** cue-implicit visual word puzzles across four subsets:

| subset | code | # puzzles |
|---|---:|---:|
| English | `en` | 300 |
| Persian | `pe` | 671 |
| Arabic | `ar` | 50 |
| Cross-lingual (English cues → Persian answer) | `cross` | 322 |

Each row contains an **image** and a single **target word / short phrase** (`answer`).

---

## What this repo does

* Loads Eye-Q directly from the Hugging Face Hub.
* Caches images locally (so API clients can send file paths).
* Runs one of the paper’s **prompt variants**:
  * **Basic** (answer-length hint)
  * **Few-Shot CoT** (3 demonstrations, each includes the demo image + derivation + answer)
  * **Iterative Refinement** (up to N attempts with feedback)
  * **Partial Character Reveal** (deterministic 25% reveal pattern)
* Writes results as **JSONL** (one record per sample) so runs are restart-safe.
* Computes exact-match accuracy per subset.

> Note: The three Few-Shot CoT demonstration items are held out from evaluation **in all runs** (to avoid any protocol leakage).

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py --models openai --prompt-variant basic
python scripts/calculate_accuracy.py results_cache.jsonl
```

---

## Installation

```bash
pip install -r requirements.txt
```

If the dataset repo is private for you, login once:

```bash
huggingface-cli login
```

---

## API keys

Set the key(s) for the model(s) you want to run.

### OpenAI
* `OPENAI_API_KEY`
* optional: `OPENAI_BASE_URL`, `OPENAI_MODEL`

### Google (Gemini)
* `GOOGLE_API_KEY`
* optional: `GOOGLE_BASE_URL`, `GOOGLE_MODEL`

### Grok / Qwen (via OpenRouter)
* `OPENROUTER_API_KEY`
* optional: `GROK_MODEL`, `QWEN_MODEL`

### Llama (provider endpoint)
* `LLAMA_API_KEY` **or** `AVALAI_API_KEY`
* optional: `LLAMA_BASE_URL` / `AVALAI_BASE_URL`, `LLAMA_MODEL`

#### Setting env vars

**Windows PowerShell**

```powershell
$env:OPENAI_API_KEY = "..."
```

**macOS/Linux (bash/zsh)**

```bash
export OPENAI_API_KEY="..."
```

---

## Running experiments

### Common flags

* `--repo-id` (default: `llm-lab/Eye-Q`)
* `--config` (default: `default`)
* `--split` (default: `train`)
* `--languages` comma-separated: `en,pe,cross,ar`
* `--models` comma-separated: `openai,google,grok,llama,qwen`
* `--temperature` (passed to the API client)
* `--max-samples` (debug: cap number of evaluated samples)
* `--max-workers` (parallelism)
* `--cache-file` (JSONL results file)
* `--image-cache-dir` (where decoded images are stored as `.jpg`)

---

## Paper prompt variants

All commands below load from the Hub and append results to `results_cache.jsonl`.

### 1) Basic (answer-length hint)

Adds a character-count hint (excluding spaces).

```bash
python main.py \
  --models openai \
  --languages en,pe,cross,ar \
  --prompt-variant basic
```

### 2) Few-Shot CoT (3 demonstrations with images)

Demonstrations are defined in `eyeq_benchmark/derivations.py`. They are selected by **HF `id`** and include the demo image.

```bash
python main.py \
  --models openai \
  --languages en,pe,cross,ar \
  --prompt-variant few_shot_cot \
  --num-examples 3
```

### 3) Iterative Refinement (retry with feedback)

Runs the Basic prompt first; if incorrect, appends feedback and retries.

```bash
python main.py \
  --models openai \
  --languages en,pe,cross,ar \
  --prompt-variant iterative_refinement \
  --num-pass 3
```

### 4) Partial Character Reveal (25% reveal pattern)

Reveals a deterministic 25% of non-space characters (stable per `(language, id)`), masking the rest with `_`.

```bash
python main.py \
  --models openai \
  --languages en,pe,cross,ar \
  --prompt-variant partial_character_reveal
```

---

## Accuracy summary

```bash
python scripts/calculate_accuracy.py results_cache.jsonl
```

The summary groups by `(model, prompt_variant)` and prints exact-match accuracy per subset.

---

## Output format

Results are appended as one JSON object per line (JSONL). Fields include:

* `id`, `language`
* `model_name`
* `prompt_variant`
* `ground_truth`, `model_ans`, `solved`
* `attempts` (all raw model responses + parsed final answers)

This makes runs **restart-safe**: if you re-run the same command, already-computed rows are skipped.

---

## Reproducibility notes

* **Few-Shot CoT** uses fixed, hand-written derivations in `eyeq_benchmark/derivations.py`.
* **Partial Character Reveal** uses a stable hash-based seed so the same `(lang, id)` always yields the same reveal pattern.
* String matching uses lightweight normalization (whitespace/punctuation normalization and Arabic/Persian diacritics removal) in `eyeq_benchmark/eval.py`.

---

## Advanced / custom protocol

If you want to manually combine the low-level switches (for ablations), use `custom`:

```bash
python main.py \
  --prompt-variant custom \
  --use-context \
  --hint-type partial_character_reveal \
  --pass-at \
  --num-pass 3
```

---

## Repository layout

* `main.py` – experiment runner (HF dataset → prompts → model API → JSONL)
* `eyeq_benchmark/data.py` – HF loading + image caching
* `eyeq_benchmark/prompts.py` – base prompt + per-subset language rules
* `eyeq_benchmark/derivations.py` – fixed Few-Shot CoT demonstrations
* `eyeq_benchmark/hints.py` – answer-length + partial reveal hints
* `eyeq_benchmark/eval.py` – answer normalization + JSON parsing
* `scripts/calculate_accuracy.py` – accuracy report from JSONL

---

## Citation

If you use Eye-Q, please cite the accompanying paper.
