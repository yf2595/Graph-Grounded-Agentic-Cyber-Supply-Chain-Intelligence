"""Post-process waiti_final_gpt54: faithfulness + paper tables (this run only)."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reports.freeze_paper_tables import (
    _by_method,
    _f,
    _read_csv,
    compose_main_rows,
    write_ablation,
    write_main,
    write_trustworthiness,
)


def _run_faithfulness(
    run_id: str,
    *,
    config: str = "evaluation/config/evaluation.yaml",
    model: str = "",
) -> None:
    from evaluation.runners import run_faithfulness as rf

    # Reuse faithfulness runner against this run for both OURS stages and B2 results
    argv = [
        "run_faithfulness",
        "--run-id",
        run_id,
        "--ours-source",
        run_id,
        "--b2-source",
        run_id,
        "--methods",
        "OURS,B2",
        "--config",
        config,
    ]
    if model:
        argv.extend(["--model", model])
    sys.argv = argv
    rf.main()

    run_dir = ROOT / "results" / run_id
    # Canonical names required by final experiment spec
    src_claims = run_dir / "faithfulness_claims.jsonl"
    src_metrics = run_dir / "faithfulness_metrics.csv"
    if src_claims.exists():
        shutil.copy2(src_claims, run_dir / "report_factuality_claims.jsonl")
    if src_metrics.exists():
        shutil.copy2(src_metrics, run_dir / "report_factuality_metrics.csv")
    # human template already written by faithfulness runner
    ht = run_dir / "human_report_audit_template.csv"
    if not ht.exists() and (run_dir / "human_report_audit_template.csv").exists():
        pass


def _write_final_results(run_dir: Path) -> None:
    run_id = run_dir.name
    model_label = "GPT-4.1" if "gpt41" in run_id else "GPT-5.4"
    agg = _by_method(_read_csv(run_dir / "aggregate_metrics.csv"))
    faith = _read_csv(run_dir / "report_factuality_metrics.csv")
    if not faith:
        faith = _read_csv(run_dir / "faithfulness_metrics.csv")

    # Main + ablation from THIS run only
    main_rows = []
    for mid in ["B0", "B1", "B2", "B3", "OURS"]:
        if mid in agg:
            main_rows.append((mid, agg[mid], run_id))
    write_main(
        run_dir / "main_results.md",
        main_rows,
        version_note=(
            f"**Version note:** All methods (B0-B3, OURS) are from `{run_id}` - "
            f"the final frozen {model_label} experiment under one shared ER/merger/metric stack. "
            "Do not mix these numbers with older runs (`waiti_full1`, `waiti_metrics_v2`, `waiti_merge_v2`)."
        ),
    )

    write_ablation(
        run_dir / "ablation_results.md",
        agg,
        agg,
        caption=f"All ablations from `{run_id}` (final frozen {model_label} stack).",
        source_label=run_id,
        notes=(
            f"Directional notes (`{run_id}`): A1 (no graph) and A3 (unified risk) reduce risk QWK vs OURS; "
            "actionable F1 stays near OURS."
        ),
    )

    write_trustworthiness(
        run_dir / "trustworthiness_results.md",
        faith,
        agg,
        run_id=run_id,
    )

    # Comprehensive final_results.md (avoid nested duplicate H1 titles)
    def _body(path: Path) -> str:
        text = path.read_text(encoding="utf-8").strip()
        # drop leading "# Title" line when embedding under a section
        lines = text.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
        return "\n".join(lines).strip()

    lines = [
        f"# WAITI Final {model_label} Results",
        "",
        f"Run ID: `{run_id}`",
        "",
        "All tables below are generated **only** from this run.",
        "",
        "## Main comparison",
        "",
        _body(run_dir / "main_results.md"),
        "",
        "## Ablations",
        "",
        _body(run_dir / "ablation_results.md"),
        "",
        "## Trustworthiness",
        "",
        _body(run_dir / "trustworthiness_results.md"),
        "",
        "## Efficiency (from aggregate_metrics.csv)",
        "",
        "| Method | Mean lat (ms) | Median | P95 | Mean LLM calls | Mean cost ($) | Total cost ($) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mid in ["B0", "B1", "B2", "B3", "OURS", "A1", "A2", "A3", "A4"]:
        r = agg.get(mid) or {}
        if not r:
            continue
        lines.append(
            f"| {mid} | {_f(r.get('mean_latency_ms'), 1)} | {_f(r.get('median_latency_ms'), 1)} | "
            f"{_f(r.get('p95_latency_ms'), 1)} | {_f(r.get('mean_llm_calls'))} | "
            f"{_f(r.get('mean_cost_usd'), 4)} | {_f(r.get('total_cost_usd'), 3)} |"
        )
    lines.append("")
    (run_dir / "final_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="waiti_final_gpt54")
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument(
        "--config",
        default="",
        help="Eval config for faithfulness LLM (default: evaluation.yaml, or evaluation_gpt41.yaml for gpt41)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override faithfulness LLM model (e.g. gpt-4.1)",
    )
    args = parser.parse_args()
    run_dir = ROOT / "results" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run dir missing: {run_dir}")

    config = args.config
    model = args.model
    if not config:
        if "gpt41" in args.run_id:
            config = "evaluation/config/evaluation_gpt41.yaml"
            model = model or "gpt-4.1"
        else:
            config = "evaluation/config/evaluation.yaml"

    if not args.skip_faithfulness:
        print("=== Faithfulness (OURS + B2) ===")
        _run_faithfulness(args.run_id, config=config, model=model)

    print("=== Paper tables ===")
    _write_final_results(run_dir)
    print(f"Wrote tables under {run_dir}")


if __name__ == "__main__":
    main()
