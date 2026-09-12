"""Recompute paired group-bootstrap CIs (OURS vs B0-B3) for the frozen GPT-4.1 run.

Reads ONLY saved predictions on disk (no model/API calls):
  results/waiti_final_gpt41/methods/{OURS,B0,B1,B2,B3}/predictions.jsonl
  evaluation/data/ground_truth.jsonl

Writes:
  results/waiti_final_gpt41/pairwise_deltas_with_qwk.csv

Config matches the paper: 10,000 paired resamples, incident-group blocks, seed=42.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics.group_bootstrap import paired_group_metric_deltas
from evaluation.runners.metrics_io import load_method_preds
from evaluation.schemas.prediction import GroundTruthRecord

RUN_ID = "waiti_final_gpt41"
BASELINES = ["B0", "B1", "B2", "B3"]


def main() -> None:
    run_dir = ROOT / "results" / RUN_ID
    eval_cfg = yaml.safe_load((ROOT / "evaluation/config/evaluation.yaml").read_text(encoding="utf-8"))
    n_boot = int(eval_cfg.get("bootstrap", {}).get("n_resamples", 10000))
    seed = int(eval_cfg.get("bootstrap", {}).get("seed", 42))
    unknown_penalty = int(eval_cfg.get("risk", {}).get("unknown_penalty", 4))

    gt_path = ROOT / eval_cfg["dataset"]["ground_truth_path"]
    gts = [
        GroundTruthRecord.model_validate(json.loads(l))
        for l in gt_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    gt = {g.article_id: g for g in gts}

    ours_preds = load_method_preds(run_dir, "OURS")
    all_rows: list[dict] = []
    for other in BASELINES:
        oth_preds = load_method_preds(run_dir, other)
        rows = paired_group_metric_deltas(
            ours_preds,
            oth_preds,
            gt,
            method_a="OURS",
            method_b=other,
            n_resamples=n_boot,
            seed=seed,
            unknown_penalty=unknown_penalty,
        )
        all_rows.extend(rows)

    out_path = run_dir / "pairwise_deltas_with_qwk.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "delta", "ci_low", "ci_high", "excludes_zero", "note"])
        w.writeheader()
        for r in all_rows:
            excl = (r["ci_low"] > 0 and r["ci_high"] > 0) or (r["ci_low"] < 0 and r["ci_high"] < 0)
            w.writerow({**r, "excludes_zero": excl})

    print(f"Wrote {out_path}")
    print(f"{'metric':<48}{'delta':>10}{'ci_low':>10}{'ci_high':>10}  excl0")
    for r in all_rows:
        excl = (r["ci_low"] > 0 and r["ci_high"] > 0) or (r["ci_low"] < 0 and r["ci_high"] < 0)
        print(f"{r['metric']:<48}{r['delta']:>10}{r['ci_low']:>10}{r['ci_high']:>10}  {'YES' if excl else 'no'}")


if __name__ == "__main__":
    main()
