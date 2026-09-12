"""Re-aggregate existing method predictions without re-running LLMs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reports.build_tables import write_markdown_tables
from evaluation.runners.metrics_io import write_aggregates
from evaluation.runners.run_all import (
    load_jsonl,
    load_method_preds,
    load_yaml,
    write_comparison,
    write_per_record_csv,
)
from evaluation.schemas.prediction import GroundTruthRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="evaluation/config/evaluation.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--methods", default="")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or list(cfg.get("methods") or [])
    run_dir = ROOT / cfg["paths"]["results_dir"] / args.run_id
    gts = [GroundTruthRecord.model_validate(x) for x in load_jsonl(ROOT / cfg["dataset"]["ground_truth_path"])]
    gt_by_id = {g.article_id: g for g in gts}
    all_preds = []
    for mid in methods:
        all_preds.extend(load_method_preds(run_dir, mid))
    write_per_record_csv(run_dir, all_preds, gt_by_id)
    write_comparison(run_dir, methods, gt_by_id)
    aggregates = write_aggregates(run_dir, methods, gt_by_id, cfg)
    write_markdown_tables(run_dir, aggregates)
    print(f"Re-aggregated {len(methods)} methods in {run_dir}")


if __name__ == "__main__":
    main()
