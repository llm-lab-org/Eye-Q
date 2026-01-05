# Eye-Q VLM Benchmark

This repository runs VLM-based riddle/puzzle solving experiments on the **Eye-Q** dataset hosted on the Hugging Face Hub.

## Install

```bash
pip install -r requirements.txt
```

## Environment variables

Set the key(s) for the model(s) you want to run:

- `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`, `OPENAI_MODEL`)
- `GOOGLE_API_KEY` (+ optional `GOOGLE_BASE_URL`, `GOOGLE_MODEL`)
- `OPENROUTER_API_KEY` (used by Grok/Qwen if you don't set `GROK_API_KEY` / `QWEN_API_KEY`)
- `LLAMA_API_KEY` or `AVALAI_API_KEY` (+ optional `LLAMA_BASE_URL` / `AVALAI_BASE_URL`, `LLAMA_MODEL`)

## Run

```bash
python main.py \
  --repo-id llm-lab/Eye-Q \
  --split train \
  --languages en,pe,cross,ar \
  --models openai \
  --hint-type none
```

Results are appended to `results_cache.jsonl` by default.

Useful options:

- `--use-context` (enable in-context examples)
- `--num-examples 3` (how many examples to include)
- `--pass-at` + `--num-pass 3` (retry with feedback)
- `--temperature 2.0`

## Accuracy summary

```bash
python scripts/calculate_accuracy.py results_cache.jsonl
```
