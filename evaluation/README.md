# Evaluation harness

Published tables, reproducibility modes, and full commands are in the root [README.md](../README.md). This file lists harness flags only.

## Export

```bash
python -m evaluation.runners.export_data
```

Rewrites `evaluation/data/inputs.jsonl` (no gold labels) and `evaluation/data/ground_truth.jsonl` from `eval_corpus_v1/`. Those two files are already present in the repository.

## Hosted OpenAI run (new inference)

Needs the licensed Honda graph and `OPENAI_API_KEY` unless `--methods B0` only (B0 still uses embeddings; see the root README).

```bash
python -m evaluation.runners.run_all \
  --config evaluation/config/evaluation_gpt41.yaml \
  --run-id waiti_final_gpt41 \
  --model gpt-4.1 \
  --methods B0,B1,B2,B3,OURS,A1,A2,A3,A4 \
  --workers 1 --method-workers 1
```

Flags: `--limit N`, `--aggregate-only` (needs existing predictions), `--skip-export`, `--methods OURS`.

Configs:

- `evaluation/config/evaluation_gpt41.yaml` — GPT-4.1, temperature 0, no reasoning API
- `evaluation/config/evaluation.yaml` — GPT-5.4, `reasoning_effort=low`
- `evaluation/config/method_configs/` — per-method overlays
- `evaluation/config/local_models.yaml` — vLLM profiles (`qwen7b` / `qwen3b` are aliases for Qwen3-8B / Qwen3-4B)

## Local vLLM run (new inference)

```bash
python -m evaluation.runners.run_local_model_experiment \
  --run-id waiti_final_qwen14b \
  --model-alias qwen14b \
  --methods OURS \
  --wait-vllm-seconds 600 \
  --aggregate
```

Set `WAITI_LOCAL_EMBEDDINGS=1` so embeddings do not call the OpenAI API.

## Query expansion

`--phase all` generates a **new** closed corpus and new reference labels. It is not a replay of the frozen paper study. Frozen artifacts are not shipped. `--phase evaluate` scores existing `artifacts/query_expansion/` files if you have them.

```bash
python -m evaluation.runners.run_query_expansion_experiment --help
```

## Tests

```bash
python -m pytest evaluation/tests -q
```
