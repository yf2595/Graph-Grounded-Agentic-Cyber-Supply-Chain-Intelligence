"""CLI: run multi-agent pipeline over eval corpus articles."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestration.incident_pipeline import IncidentPipeline
from src.schemas.models import RawIncident


def load_articles(corpus_dir: Path, ids: list[str] | None = None, merge_only: bool = False) -> list[RawIncident]:
    incidents_csv = corpus_dir / "incidents.csv"
    rows = list(csv.DictReader(incidents_csv.open(encoding="utf-8")))
    if merge_only:
        rows = [r for r in rows if r.get("group_id")]
    if ids:
        want = set(ids)
        rows = [r for r in rows if r["incident_id"] in want]
    out = []
    for r in rows:
        # Strip gold labels — only text fields to the pipeline
        out.append(
            RawIncident(
                article_id=r["incident_id"],
                headline=r["headline"],
                body=r["body"],
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OEM cyber supply-chain multi-agent pipeline")
    parser.add_argument("--corpus", default="eval_corpus_v1")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--config", default="config/risk_rules.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="Comma-separated article ids")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = ROOT
    corpus = root / args.corpus
    articles = load_articles(
        corpus,
        ids=[x.strip() for x in args.ids.split(",") if x.strip()] or None,
        merge_only=args.merge_only,
    )
    if args.limit:
        articles = articles[: args.limit]

    pipe = IncidentPipeline(
        dataset_dir=root / args.dataset,
        run_dir=root / "runs" / args.run_id,
        config_path=root / args.config,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )

    print(f"Processing {len(articles)} articles -> runs/{args.run_id}")
    for i, raw in enumerate(articles, 1):
        result = pipe.process(raw, force=args.force)
        status = result.pipeline_status.value
        extra = result.filter_reason or (
            result.final_risk_assessment.overall_risk.value
            if result.final_risk_assessment
            else result.error or ""
        )
        print(f"  [{i}/{len(articles)}] {raw.article_id} -> {status} {extra}")

    summary = {
        "run_id": args.run_id,
        "n": len(articles),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
    }
    (root / "runs" / args.run_id / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
