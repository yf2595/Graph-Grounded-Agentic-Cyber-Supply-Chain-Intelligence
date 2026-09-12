# Evaluation outputs

Runners write here as `results/<run-id>/`. This directory is gitignored except for this file.

Typical contents after a full run:

- `methods/<METHOD>/predictions.jsonl` — one prediction per report
- `aggregate_metrics.csv` / `report.md` — method comparison
- `run_manifest.json` — model, graph checksums, bootstrap settings
- `pairwise_deltas_with_qwk.csv` — paired CIs (after the bootstrap script)

The published paper tables are the source of truth for reported numbers. They are reproduced in the root [README](../README.md).
