"""CLI: evaluate pipeline and/or baseline runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="eval_corpus_v1")
    parser.add_argument("--pipeline-run", default="")
    parser.add_argument("--baseline-run", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    corpus = ROOT / args.corpus
    report: dict = {}
    if args.pipeline_run:
        report["pipeline"] = evaluate(corpus, ROOT / "runs" / args.pipeline_run, mode="pipeline")
    if args.baseline_run:
        report["baseline"] = evaluate(corpus, ROOT / "runs" / args.baseline_run, mode="baseline")

    if args.pipeline_run and args.baseline_run:
        report["comparison"] = {
            "cyber_f1": {
                "pipeline": report["pipeline"]["cyber"]["f1"],
                "baseline": report["baseline"]["cyber"]["f1"],
            },
            "relevance_f1": {
                "pipeline": report["pipeline"]["relevance"]["f1"],
                "baseline": report["baseline"]["relevance"]["f1"],
            },
            "supplier_f1": {
                "pipeline": report["pipeline"]["suppliers"]["f1"],
                "baseline": report["baseline"]["suppliers"]["f1"],
            },
            "risk_exact": {
                "pipeline": report["pipeline"]["risk"]["exact_accuracy"],
                "baseline": report["baseline"]["risk"]["exact_accuracy"],
            },
            "merge_f1": {
                "pipeline": report["pipeline"]["merge_pairwise"].get("f1"),
                "baseline": report["baseline"]["merge_pairwise"].get("f1"),
            },
        }

    text = json.dumps(report, indent=2)
    print(text)
    out = Path(args.out) if args.out else None
    if out is None and args.pipeline_run:
        out = ROOT / "runs" / args.pipeline_run / "metrics.json"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
