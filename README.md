# Graph-Grounded Agentic Cyber Supply-Chain Intelligence

Anonymous artifact for the paper *From Weak Signals to OEM Risk: Graph-Grounded Agentic Cyber Supply-Chain Intelligence* ([Agentic_SC_WAITI_2026.pdf](Agentic_SC_WAITI_2026.pdf)).

The framework filters non-actionable public cyber reports, resolves affected suppliers against an OEM dependency graph, correlates same-event reports, and separates supplier-side consequence from OEM-side exposure before assigning ordinal risk.

This repository is intended for anonymous review. Honda and MarkLines appear as **dataset descriptions**, not author identifiers. The licensed MarkLines-derived Honda graph is not redistributed.

## What this artifact supports

Four different activities are easy to conflate:

| Mode | What it is | Supported by this release? |
| --- | --- | --- |
| (a) Paper results | Tables 1–5 below, copied from the PDF | Yes. These are the reported numbers. |
| (b) Offline metric recompute | Re-aggregate saved predictions or query-expansion artifacts | **No.** The runners exist, but the saved predictions and frozen query-expansion files are not distributed. |
| (c) Rerun model inference | Call hosted OpenAI models or a local vLLM server on the 129-report corpus | Only if you separately obtain the licensed Honda graph (and API credentials or local weights). A new run is a new evaluation, not a recompute of (a). |
| (d) Interface smoke test | Load the fictional example graph and run unit tests | Yes. This checks the CSV interface and pipeline wiring. It does **not** reproduce the Honda benchmark. |

Missing graph files or missing predictions are sufficient to block (b) and the Honda instance of (c). Do not attribute that gap to API sampling.

## What is included

| Distributed in this repository | Not distributed |
| --- | --- |
| Source: agents, graph adapter, baselines B0–B3, ablations A1–A4, Agent 4 query-expansion **code**, metrics, runners | Licensed MarkLines graph (`dataset/nodes.csv`, `dataset/edges.csv`, `dataset/aliases.csv`) |
| Prompts (inline in `src/agents/` and `src/query_expansion/`) | Saved predictions, run logs, and per-article model outputs (`results/<run-id>/` is created at runtime) |
| 129 labeled reports (`eval_corpus_v1/`) and exported `evaluation/data/*.jsonl` | Frozen query-expansion corpus, index, queries, and reference labels (`artifacts/query_expansion/`) |
| Configs and fusion rules (`evaluation/config/`, `config/`) | Cluster / Slurm launch scripts |
| Fictional example graph (`examples/graph/`) | Open-Web retrieval index |
| Tests, paper PDF, supplementary silent-pair corpus (`incidents/`) | Author or institution identifiers |

`export_data`, `run_all`, `run_local_model_experiment`, and `run_query_expansion_experiment` **write** files under `evaluation/data/`, `results/`, and `artifacts/`. Those outputs are generated, not shipped (except the already-exported `evaluation/data/*.jsonl` inputs and gold labels).

## Published results

Copied from the paper. Use them to read the paper, not as targets to overwrite.

### Table 1. Main GPT-4.1 comparison on 129 reports

FNR is FN/99 and FPR is FP/30. Parentheses are raw counts. Cost is total API expenditure for all 129 reports. Lower is better for FNR, FPR, latency, and cost.

| Method | FNR | FPR | Supplier micro-F1 | Merge F1 | Risk QWK | Assign. rate | Spearman rho | Median (s) | Cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 Rules+Graph | 4.0% (4) | 33.3% (10) | 0.602 | 0.681 | 0.405 | 0.960 | 0.546 | 0.350 | 0.00 |
| B1 Single LLM | 9.1% (9) | 50.0% (15) | 0.000 | 0.502 | 0.403 | 0.909 | 0.513 | 3.923 | 0.27 |
| B2 LLM+Graph Tools | 13.1% (13) | 0.0% (0) | 0.808 | 0.634 | 0.589 | 0.869 | 0.558 | 14.168 | 2.53 |
| B3 GraphRAG | 62.6% (62) | 0.0% (0) | 0.326 | 0.343 | 0.464 | 0.374 | 0.168 | 3.351 | 0.40 |
| Ours | 4.0% (4) | 0.0% (0) | 0.850 | 0.750 | 0.620 | 0.869 | 0.670 | 7.652 | 1.19 |

### Table 2. GPT-4.1 ablations

A1: no graph access. A2: no merger. A3: unified risk. A4: no early gate. Cost is total API expenditure for all 129 reports.

| Variant | FNR | FPR | Merge F1 | QWK | Spearman rho | Median (s) | Cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ours | 4.0% (4) | 0.0% (0) | 0.750 | 0.620 | 0.670 | 7.7 | 1.19 |
| A1: no graph access | 15.1% (15) | 50.0% (15) | 0.727 | 0.292 | 0.411 | 6.9 | 1.03 |
| A2: no merger | 4.0% (4) | 0.0% (0) | 0.000 | 0.526 | 0.636 | 6.7 | 1.10 |
| A3: unified risk | 4.0% (4) | 0.0% (0) | 0.692 | 0.354 | 0.392 | 5.8 | 0.98 |
| A4: no gate | 12.1% (12) | 0.0% (0) | 0.757 | 0.518 | 0.629 | 7.8 | 1.39 |

### Table 3. OURS across hosted and local backends

Support and unsupported-claim rates are from the claim–evidence audit, with matched B2 in parentheses. Only the GPT-5.4 row compares full final reports; the other faithfulness numbers use structured persisted claims and are diagnostic.

| Backend | FNR | FPR | Supplier micro-F1 | Merge F1 | Risk QWK | Median (s) | Mean±SD (s) | API cost ($) | Support | Unsup. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4.1 | 4.0% (4) | 0.0% (0) | 0.850 | 0.750 | 0.620 | 7.7 | 6.9±2.9 | 1.19 | 0.979 (0.719) | 0.021 (0.281) |
| GPT-5.4 | 4.0% (4) | 0.0% (0) | 0.851 | 0.709 | 0.640 | 15.5 | 14.2±5.7 | 2.60 | 0.958 (0.743) | 0.042 (0.257) |
| GPT-OSS-20B | 12.1% (12) | 0.0% (0) | 0.821 | 0.696 | 0.562 | 11.3 | 25.2±27.9 | 0 | 0.976 (0.687) | 0.024 (0.313) |
| Qwen3-14B | 8.1% (8) | 0.0% (0) | 0.811 | 0.715 | 0.443 | 27.6 | 24.9±9.2 | 0 | 0.978 (0.793) | 0.022 (0.207) |
| Qwen3-8B | 9.1% (9) | 0.0% (0) | 0.870 | 0.703 | 0.444 | 54.5 | 42.9±31.3 | 0 | 0.980 (0.691) | 0.020 (0.309) |
| Qwen3-4B | 11.1% (11) | 0.0% (0) | 0.829 | 0.701 | 0.407 | 9.7 | 9.7±6.2 | 0 | 0.979 (0.733) | 0.021 (0.267) |

### Table 4. Paired incident-block bootstrap (10,000 resamples, seed 42)

Delta is OURS minus the comparison method.

| Comparison / metric | Delta | 95% CI |
| --- | --- | --- |
| OURS–B2 Supplier micro-F1 | +0.042 | [-0.009, +0.105] |
| OURS–B2 Merge F1 | +0.116 | [-0.101, +0.305] |
| OURS–B2 Risk QWK | +0.031 | [-0.073, +0.117] |
| OURS–B2 Spearman rho | +0.112 | [-0.036, +0.236] |
| OURS–A1 Risk QWK | +0.328 | [+0.221, +0.436] |
| OURS–A3 Risk QWK | +0.266 | [+0.160, +0.376] |

### Table 5. Controlled query expansion (20 incidents, 60-document corpus)

| Metric | Result |
| --- | --- |
| Query top-1 success | 97/100 |
| Incidents with at least one relevant retrieval | 20/20 |
| Both supplemental reports retrieved | 19/20 |
| Relevant-document recall | 39/40 (0.975) |
| Accepted-document precision | 0.975 |
| Acceptance-filter recall | 1.000 |
| Graph-attribute disclosure (manual audit) | 0/100 |
| Agent-5 evidence coverage | 0.019 → 0.529 |
| Risk-direction agreement | 15/20 (0.750) |

## Metric definitions (paper)

These match the published tables. Gold-label counts were checked against `evaluation/data/ground_truth.jsonl` (129 reports; 99 actionable; 30 non-actionable = 15 off-graph cyber + 15 in-graph non-cyber).

- **FNR** = false negatives / 99 gold-actionable reports.
- **FPR** = false positives / 30 gold-non-actionable reports.
- For OURS, the report-level actionable decision is cyber-and-in-graph (`ŷ = ĉ ∧ q̂`).
- **Supplier micro-F1** compares predicted vs. reference canonical supplier identities (graph grounding).
- **Pairwise merge F1** evaluates incident correlation over unordered pairs among the 99 gold-actionable reports (34 positive pairs among 4,851 pairs). Reports not sent to correlation are counted as predicted non-merges, not dropped.
- **Risk assignment rate** is the fraction of the 99 gold-actionable reports that receive a valid five-level risk label (`VERY_LOW` … `VERY_HIGH`).
- **QWK** and **Spearman’s ρ** are computed on predictions that map to that five-level schema. They are therefore conditional on a valid risk assignment and should be read together with assignment rate.
- **Median latency** and **API cost** are over the 129-report corpus. Cost is total recorded API expenditure for that run.

OURS on GPT-4.1 **detects** 95/99 actionable reports (4 misses) and **assigns valid risk** to 86/99 (assignment rate 0.869). Those are different measurements. This release does not include saved outputs that would explain the 9 detected reports without a valid risk assignment.

## Requirements

- Python 3.11 or newer
- For (c) Honda reruns: licensed graph CSVs in `dataset/` (see [Supply-chain graph](#supply-chain-graph))
- Hosted inference: `OPENAI_API_KEY` (or `OPENAI_API_TOKEN`)
- Local inference: one GPU (paper: NVIDIA RTX 6000 Ada 48 GB), [vLLM](https://github.com/vllm-project/vllm) 0.25.1 (`requirements-local.txt`)

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "..."
```

Do not put keys in files. `setx OPENAI_API_KEY ...` persists for **new** Windows shells only.

Local backends only:

```bash
pip install -r requirements-local.txt
```

## Supply-chain graph

Paper evaluation uses a frozen Honda graph (4,661 nodes / 26,035 edges / 8,886 aliases) through `src/services/graph_service.py`. Place `nodes.csv`, `edges.csv`, and `aliases.csv` in `dataset/` if you have the licensed extract. Schema: [dataset/README.md](dataset/README.md).

The fictional Aether Motors graph in [examples/graph/](examples/graph/) exercises that interface (sole-source vs. redundant suppliers). It is not a substitute for the Honda graph and cannot reproduce Tables 1–5.

### Another OEM graph

A different OEM graph is a **separate evaluation**, not the Honda benchmark. You must revalidate:

- supplier node IDs and article-to-supplier mappings;
- actionability labels (`is_cyber ∧ in_graph` on **that** graph);
- graph-dependent risk reference labels.

Matching aliases is not enough for benchmark equivalence. The shipped 129 reports name Honda-graph suppliers; they will not ground correctly against an unrelated OEM.

Example report shape (`eval_corpus_v1/articles/INC_001.json`):

```json
{
  "article_id": "INC_001",
  "headline": "H-ONE Battles Cyberattack as Production Grinds to a Halt",
  "body": "Automotive supplier H-ONE Co., Ltd. has suffered a ransomware attack ...",
  "is_cyber": true,
  "in_supply_chain": true,
  "relevant": true,
  "synthetic": true,
  "disclaimer": "SYNTHETIC ARTICLE: ... the incident described is fictional ..."
}
```

Gold labels are in `evaluation/data/ground_truth.jsonl`. Model inputs (`evaluation/data/inputs.jsonl`) contain `article_id`, `headline`, `body`, and optional null `source_name` / `published_at` — no gold labels.

## Evaluation corpus

`eval_corpus_v1/` is the 129-report paper corpus. Details: [eval_corpus_v1/README.md](eval_corpus_v1/README.md).

| Slice | n |
| --- | --- |
| Actionable in-graph cyber reports | 99 |
| Off-graph cyber reports | 15 |
| In-graph non-cyber reports | 15 |
| Total | 129 |

Of the 99 actionable reports, 45 are singletons and 54 form 25 multi-report groups (34 positive merge pairs). All articles are synthetic. Named organizations may be real; attributed incidents are fictional.

`incidents/` is a separate silent-pair corpus. It is not the 129-report benchmark.

## Running evaluations

Outputs go to `results/<run-id>/` (gitignored). Every full-corpus command below needs the licensed Honda graph unless noted.

Refresh exported inputs (rewrites `evaluation/data/*.jsonl` from `eval_corpus_v1/`):

```bash
python -m evaluation.runners.export_data
```

### Smoke tests (no credentials, no Honda graph)

```bash
python -m pytest tests/test_smoke.py tests/test_query_expansion.py -q
```

Licensed-graph tests skip if `dataset/nodes.csv` is absent. `test_example_graph_loads` always uses `examples/graph/`.

The following is **not** API-free. `OURS` needs an LLM key (or a local server). `B0` alone still needs embeddings (see [B0 embeddings](#b0-embeddings)).

```bash
python -m evaluation.runners.run_all --run-id waiti_smoke --methods B0,OURS --limit 5
```

### Table 1 and Table 2 — GPT-4.1 (new inference)

Needs the Honda graph and `OPENAI_API_KEY`. GPT-4.1: temperature 0, 2,048-token output limit (`evaluation/config/evaluation_gpt41.yaml`).

```bash
python -m evaluation.runners.run_all \
  --config evaluation/config/evaluation_gpt41.yaml \
  --run-id waiti_final_gpt41 \
  --model gpt-4.1 \
  --workers 1 \
  --method-workers 1
```

Runs B0, B1, B2, B3, OURS, A1, A2, A3, A4 on all 129 reports. Subset with `--methods OURS`.

`--aggregate-only` re-aggregates **existing** `results/<run-id>/methods/*/predictions.jsonl`. Those files are not in this release, so the command cannot recompute Tables 1–4 from the provided files.

### Table 3 — GPT-5.4 (new inference)

Default model in `evaluation/config/evaluation.yaml` is `gpt-5.4` with `reasoning_effort=low`.

```bash
python -m evaluation.runners.run_all \
  --config evaluation/config/evaluation.yaml \
  --run-id waiti_final_gpt54 \
  --workers 1 \
  --method-workers 1
```

Add `--methods OURS` for the OURS row only.

### Table 3 — local vLLM (new inference)

Paper settings (also in `evaluation/config/local_models.yaml`): vLLM 0.25.1; Qwen in bfloat16 with 16,384-token context and thinking disabled; GPT-OSS-20B with 32,768-token context, 8,192-token output, low reasoning effort. Qwen3-14B output cap 1,024; Qwen3-8B and Qwen3-4B output cap 2,048.

`qwen7b` and `qwen3b` are **intentional legacy aliases** for Qwen3-8B and Qwen3-4B (`display_name` in `local_models.yaml`). Keep those alias strings.

Set local hash embeddings so merge/GraphRAG do not call `api.openai.com`:

```bash
export WAITI_LOCAL_EMBEDDINGS=1
```

```powershell
$env:WAITI_LOCAL_EMBEDDINGS = "1"
```

Serve one model, then run the harness against `http://127.0.0.1:8000/v1`.

**Qwen3-14B**

```bash
vllm serve Qwen/Qwen3-14B \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --served-model-name qwen14b
```

```bash
python -m evaluation.runners.run_local_model_experiment \
  --run-id waiti_final_qwen14b \
  --model-alias qwen14b \
  --methods OURS \
  --wait-vllm-seconds 600 \
  --aggregate
```

**Qwen3-8B:** `vllm serve Qwen/Qwen3-8B` with `--served-model-name qwen7b`, then `--run-id waiti_final_qwen7b --model-alias qwen7b`.

**Qwen3-4B:** `Qwen/Qwen3-4B`, `--served-model-name qwen3b`, `--run-id waiti_final_qwen3b --model-alias qwen3b`.

**GPT-OSS-20B**

```bash
vllm serve openai/gpt-oss-20b \
  --max-model-len 32768 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --served-model-name gptoss
```

```bash
python -m evaluation.runners.run_local_model_experiment \
  --run-id waiti_final_gptoss \
  --model-alias gptoss \
  --methods OURS \
  --wait-vllm-seconds 600 \
  --aggregate
```

Matched local B2: `--methods B2,OURS`.

### Table 4 — paired bootstrap

`python scripts/recompute_gpt41_paired_cis.py` reads `results/waiti_final_gpt41/methods/{OURS,B0,B1,B2,B3}/predictions.jsonl`. Those predictions are not shipped, so this release cannot recompute Table 4.

### B0 embeddings

B0 is rules + graph lookup + embedding similarity for merge + fixed risk rules. `run_all` does not require `OPENAI_API_KEY` when **every** selected method is `B0`, but B0 still constructs `EmbeddingService()`. Default mode is OpenAI `text-embedding-3-small` unless `WAITI_LOCAL_EMBEDDINGS=1`, which switches to a local hash embedding. Local hash embeddings are not the hosted Table 1 B0 setting.

### Single-article pipeline

Needs an LLM key and a graph directory. Default `--dataset dataset` is the Honda extract. `--dataset examples/graph` only tests the interface; `INC_001` names a Honda-graph supplier and will not ground on the fictional graph.

```bash
python -m src.run_pipeline --run-id smoke_pipe --ids INC_001 --force
python -m src.run_baseline --run-id smoke_base --ids INC_001 --force
```

## Query expansion (Table 5)

Query expansion is **off** in the main 129-report benchmark. The paper study uses a frozen 60-document closed corpus (40 withheld same-incident reports + 20 unrelated aerospace/aviation/defense reports), five queries per incident, top-1 cosine retrieval, no score threshold, embeddings `text-embedding-3-small` (1,536 dimensions) over headline + body. Agent 4 has no OEM-graph attributes.

Those frozen files are **not** in this repository. `--phase evaluate` scores existing files under `artifacts/query_expansion/` and does not call an LLM, but it cannot run until those files exist.

| `--phase` | Reuses existing artifacts | Model / API calls | Writes / overwrites |
| --- | --- | --- | --- |
| `select` | 129-report corpus | None (deterministic) | `selected_incidents.json` |
| `corpus` | selection | Yes: generate 40 supplements + 20 noise; embed index | documents, ground truth, index |
| `reindex` | existing `documents.jsonl` | Yes: embeddings | index only |
| `paired` | selection + closed index | Yes: A4-OFF / A4-ON pipelines | paired result JSONL |
| `reference` | paired / selected incidents | Yes: enriched reference labels | `enriched_reference_labels.jsonl` |
| `evaluate` | queries, retrievals, labels, paired outputs | No | metric reports |
| `rerun` | frozen documents + selection + **existing** reference labels | Yes: reindex + paired | index, paired outputs, reports (does not regenerate supplements, noise, or reference labels) |
| `all` | 129-report corpus only | Yes | selection, **new** corpus, index, paired run, **new** reference labels, reports |

```bash
python -m evaluation.runners.run_query_expansion_experiment --phase all
```

`--phase all` **reconstructs** a new closed corpus and new reference labels. That is not a reproduction of the frozen paper corpus. Do not compare a regenerated Table 5 to the published table as if they were the same artifact.

If you later obtain the frozen files, the supported evaluation-only command is:

```bash
python -m evaluation.runners.run_query_expansion_experiment --phase evaluate
```

`--phase rerun` still calls embeddings and the paired pipelines.

## Faithfulness audit (Table 3 support / unsupported rates)

`evaluation.runners.run_faithfulness --model` selects the LLM used to **regenerate OURS Agent-8 reports**. It is not a DeepEval / GPT-5.5 judge. Claim labels come from the heuristic in `evaluation/metrics/faithfulness.py`.

The paper’s matched **full final-report** audit (GPT-5.4 outputs, GPT-5.5 judge) is **not reconstructed** by this runner. This repository has no GPT-5.5 judge configuration and no saved final reports.

`--skip-llm` builds structured persisted claims from saved Agent 5–7 stages (the diagnostic path used for non-GPT-5.4 rows). That still needs `results/<ours-source>/methods/OURS/articles/*/`, which are not shipped.

Diagnostic Agent-8 regeneration (not the paper GPT-5.5 audit), after a completed OURS+B2 run:

```bash
python -m evaluation.runners.run_faithfulness \
  --run-id waiti_faithfulness_gpt41 \
  --ours-source waiti_final_gpt41 \
  --b2-source waiti_final_gpt41 \
  --config evaluation/config/evaluation_gpt41.yaml \
  --model gpt-4.1 \
  --methods OURS,B2
```

## Methods

| ID | Role |
| --- | --- |
| OURS | Staged agents: extract, graph grounding, merge, optional query expansion, supplier consequence, OEM exposure, risk fusion, alert |
| B0 | Deterministic cyber rules, entity/graph lookup, embedding similarity for merge, fixed risk rules |
| B1 | Single LLM, no graph |
| B2 | Single LLM with the same graph-query tools as OURS |
| B3 | Single-pass GraphRAG: retrieve supplier neighborhood, one LLM call |
| A1 | OURS without OEM-graph access |
| A2 | OURS without incident merging |
| A3 | OURS with unified consequence+exposure risk agent |
| A4 | OURS without the early actionable gate |

Configs: `evaluation/config/method_configs/`. Fusion rubric: `config/risk_rules.yaml`.

## Tests

```bash
python -m pytest tests evaluation/tests -q
```

## Repository layout

```
src/                      agents, graph adapter, LLM client, query expansion
evaluation/               harness, metrics, runners
eval_corpus_v1/           129 labeled synthetic reports
incidents/                supplementary silent-pair corpus
examples/graph/           fictional CSV schema example
dataset/                  place licensed or custom graph CSVs here
config/                   risk fusion, merge, entity-resolution rules
tests/                    unit / smoke tests
Agentic_SC_WAITI_2026.pdf paper (anonymous)
```

## Ethics

Generated reports describe synthetic events. No article asserts that the described incident occurred at the named supplier. Keep the `disclaimer` / `synthetic` fields if the texts are reused.

## License

Code and labeled reports: [LICENSE](LICENSE). The MarkLines-derived graph is licensed separately and is not part of this release.
