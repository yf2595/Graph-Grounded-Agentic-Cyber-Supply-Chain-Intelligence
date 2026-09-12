"""CLI: run single-LLM baseline over corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline.runner import BaselineRunner
from src.run_pipeline import load_articles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-LLM baseline")
    parser.add_argument("--corpus", default="eval_corpus_v1")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    articles = load_articles(
        ROOT / args.corpus,
        ids=[x.strip() for x in args.ids.split(",") if x.strip()] or None,
        merge_only=args.merge_only,
    )
    if args.limit:
        articles = articles[: args.limit]

    runner = BaselineRunner(
        dataset_dir=str(ROOT / args.dataset),
        run_dir=str(ROOT / "runs" / args.run_id),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    print(f"Baseline processing {len(articles)} articles -> runs/{args.run_id}")
    for i, raw in enumerate(articles, 1):
        out = runner.process(raw, force=args.force)
        print(
            f"  [{i}/{len(articles)}] {raw.article_id} -> {out.pipeline_status.value} "
            f"{out.overall_risk.value if out.overall_risk else out.filter_reason or ''}"
        )

    (ROOT / "runs" / args.run_id / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "n": len(articles),
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "mode": "baseline",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
