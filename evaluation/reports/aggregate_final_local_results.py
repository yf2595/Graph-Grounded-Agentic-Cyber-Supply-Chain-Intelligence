"""Aggregate local-backend runs with a hosted GPT comparison run.

Reads:
  results/waiti_final_qwen14b|qwen7b|qwen3b|gptoss
  results/waiti_final_gpt54   (hosted GPT-5.4 comparison)

Writes:
  results/waiti_final_local_comparison/
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.runners.metrics_io import write_aggregates
from evaluation.runners.run_all import load_jsonl, load_yaml
from evaluation.schemas.prediction import GroundTruthRecord

LOCAL_RUNS = (
    ("waiti_final_qwen14b", "Qwen14B", "qwen14b"),
    ("waiti_final_qwen7b", "Qwen7B", "qwen7b"),
    ("waiti_final_qwen3b", "Qwen3B", "qwen3b"),
)

GPT_RUN = "waiti_final_gpt54"
GPT_LABEL = "GPT-5.4"

COMPARISON_COLUMNS = [
    "Model",
    "Method",
    "Supplier F1",
    "Actionable F1",
    "Merge Pair F1",
    "Merge B3 F1",
    "Risk Macro-F1",
    "Risk MAE",
    "QWK | detected",
    "Coverage",
    "Group Macro-F1",
    "Spearman",
    "nDCG@10",
    "High-risk Recall",
    "Hallucination Rate",
    "Median Latency",
    "P95 Latency",
    "Peak VRAM",
]


def _f(v, digits: int = 4) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _by_method(rows: list[dict]) -> dict[str, dict]:
    return {r["method_id"]: r for r in rows if r.get("method_id")}


def _peak_vram(run_dir: Path) -> str:
    hw = run_dir / "telemetry" / "hardware_metrics.json"
    if not hw.exists():
        return ""
    try:
        data = json.loads(hw.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    peak = data.get("peak_memory_used_mib")
    if peak is None:
        return ""
    try:
        return f"{float(peak):.0f} MiB"
    except (TypeError, ValueError):
        return str(peak)


def _hallucination(run_dir: Path, method_id: str) -> str:
    for name in ("report_factuality_metrics.csv", "faithfulness_metrics.csv"):
        rows = _by_method(_read_csv(run_dir / name))
        if method_id in rows:
            return _f(rows[method_id].get("hallucination_rate"))
    return ""


def validate_manifest(run_dir: Path, expected_alias: str) -> list[str]:
    reasons: list[str] = []
    man = run_dir / "run_manifest.json"
    if not man.exists():
        return ["missing run_manifest.json"]
    try:
        data = json.loads(man.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"manifest unparseable: {e}"]
    if data.get("model_alias") and data["model_alias"] != expected_alias:
        reasons.append(f"model_alias {data.get('model_alias')} != expected {expected_alias}")
    if not (run_dir / "RUN_COMPLETE").exists():
        reasons.append("RUN_COMPLETE marker missing (proceeding anyway if predictions exist)")
    return reasons


def ensure_raw_mirrors(run_dir: Path, methods: list[str]) -> None:
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for mid in methods:
        src = run_dir / "methods" / mid / "predictions.jsonl"
        dst = raw / f"{mid}.jsonl"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        elif dst.exists() and not src.exists():
            mdir = run_dir / "methods" / mid
            mdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, src)


def discover_methods(run_dir: Path, fallback: list[str]) -> list[str]:
    methods_dir = run_dir / "methods"
    if methods_dir.is_dir():
        found = sorted(
            p.name for p in methods_dir.iterdir() if p.is_dir() and (p / "predictions.jsonl").exists()
        )
        if found:
            return found
    raw = run_dir / "raw"
    if raw.is_dir():
        found = sorted(p.stem for p in raw.glob("*.jsonl"))
        if found:
            return found
    return list(fallback)


def run_faithfulness_local(run_id: str, methods: list[str], skip_llm: bool = True) -> None:
    """Faithfulness without hosted API contamination (deterministic OURS claims by default)."""
    from evaluation.runners import run_faithfulness as rf

    faith_methods = [m for m in ("OURS", "B2") if m in methods]
    if not faith_methods:
        return
    argv = [
        "run_faithfulness",
        "--run-id",
        run_id,
        "--ours-source",
        run_id,
        "--b2-source",
        run_id,
        "--methods",
        ",".join(faith_methods),
    ]
    if skip_llm:
        argv.append("--skip-llm")
    old = sys.argv
    try:
        sys.argv = argv
        rf.main()
    finally:
        sys.argv = old

    run_dir = ROOT / "results" / run_id
    for src_name, dst_name in (
        ("faithfulness_claims.jsonl", "report_factuality_claims.jsonl"),
        ("faithfulness_metrics.csv", "report_factuality_metrics.csv"),
    ):
        src = run_dir / src_name
        if src.exists():
            shutil.copy2(src, run_dir / dst_name)


def row_from_agg(
    model_label: str,
    method_id: str,
    agg_row: dict,
    hall: str,
    peak_vram: str,
) -> dict:
    return {
        "Model": model_label,
        "Method": method_id,
        "Supplier F1": _f(agg_row.get("supplier_micro_f1")),
        "Actionable F1": _f(agg_row.get("actionable_f1")),
        "Merge Pair F1": _f(agg_row.get("merge_pairwise_f1")),
        "Merge B3 F1": _f(agg_row.get("merge_b3_f1")),
        "Risk Macro-F1": _f(agg_row.get("risk_e2e_macro_f1")),
        "Risk MAE": _f(agg_row.get("risk_e2e_mae")),
        "QWK | detected": _f(agg_row.get("risk_qwk_detected")),
        "Coverage": _f(agg_row.get("risk_coverage")),
        "Group Macro-F1": _f(agg_row.get("risk_group_macro_f1_e2e")),
        "Spearman": _f(agg_row.get("spearman_rho")),
        "nDCG@10": _f(agg_row.get("ndcg_at_10")),
        "High-risk Recall": _f(agg_row.get("high_risk_recall")),
        "Hallucination Rate": hall,
        "Median Latency": _f(agg_row.get("median_latency_ms"), 1),
        "P95 Latency": _f(agg_row.get("p95_latency_ms"), 1),
        "Peak VRAM": peak_vram,
    }


def write_md_table(path: Path, title: str, rows: list[dict], note: str = "") -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(COMPARISON_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(COMPARISON_COLUMNS)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in COMPARISON_COLUMNS) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def process_local_run(
    run_id: str,
    model_label: str,
    alias: str,
    gt_by_id: dict,
    eval_cfg: dict,
    local_cfg: dict,
    skip_faithfulness: bool,
    skip_llm_faithfulness: bool,
) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    run_dir = ROOT / "results" / run_id
    if not run_dir.is_dir():
        return [], [f"missing {run_id}"]

    warnings.extend(validate_manifest(run_dir, alias))
    exp_methods: list[str] = []
    for exp in local_cfg.get("experiments") or []:
        if exp.get("run_id") == run_id:
            exp_methods = list(exp.get("methods") or [])
            break
    methods = discover_methods(run_dir, exp_methods)
    ensure_raw_mirrors(run_dir, methods)

    print(f"=== Aggregating {run_id} methods={methods} ===")
    write_aggregates(run_dir, methods, gt_by_id, eval_cfg)

    if not skip_faithfulness:
        try:
            print(f"=== Faithfulness {run_id} (skip_llm={skip_llm_faithfulness}) ===")
            run_faithfulness_local(run_id, methods, skip_llm=skip_llm_faithfulness)
        except Exception as e:
            warnings.append(f"{run_id} faithfulness failed: {e}")

    agg = _by_method(_read_csv(run_dir / "aggregate_metrics.csv"))
    peak = _peak_vram(run_dir)
    rows = []
    # Stable order from experiment config when possible
    order = exp_methods or methods
    for mid in order:
        if mid not in agg:
            continue
        rows.append(row_from_agg(model_label, mid, agg[mid], _hallucination(run_dir, mid), peak))
    return rows, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate WAITI local + GPT-5.4 comparison tables")
    parser.add_argument("--gpt-run", default=GPT_RUN)
    parser.add_argument("--out-dir", default="results/waiti_final_local_comparison")
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument(
        "--use-llm-faithfulness",
        action="store_true",
        help="Regenerate OURS Agent-8 via evaluation.yaml LLM (NOT recommended for local purity)",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Continue if some local runs absent")
    args = parser.parse_args()

    local_cfg = yaml.safe_load((ROOT / "evaluation/config/local_models.yaml").read_text(encoding="utf-8")) or {}
    eval_cfg = load_yaml(ROOT / "evaluation/config/evaluation.yaml")
    gts = [GroundTruthRecord.model_validate(x) for x in load_jsonl(ROOT / eval_cfg["dataset"]["ground_truth_path"])]
    gt_by_id = {g.article_id: g for g in gts}

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_warnings: list[str] = []

    # GPT-5.4 OURS only (frozen)
    gpt_dir = ROOT / "results" / args.gpt_run
    if not gpt_dir.is_dir():
        raise SystemExit(f"Frozen GPT run missing: {gpt_dir}")
    gpt_agg = _by_method(_read_csv(gpt_dir / "aggregate_metrics.csv"))
    if "OURS" not in gpt_agg:
        raise SystemExit(f"No OURS row in {gpt_dir / 'aggregate_metrics.csv'}")
    all_rows.append(
        row_from_agg(
            GPT_LABEL,
            "OURS",
            gpt_agg["OURS"],
            _hallucination(gpt_dir, "OURS"),
            _peak_vram(gpt_dir) or "API",
        )
    )

    for run_id, label, alias in LOCAL_RUNS:
        rows, warns = process_local_run(
            run_id,
            label,
            alias,
            gt_by_id,
            eval_cfg,
            local_cfg,
            skip_faithfulness=args.skip_faithfulness,
            skip_llm_faithfulness=not args.use_llm_faithfulness,
        )
        all_warnings.extend(warns)
        if not rows and not args.allow_missing:
            raise SystemExit(f"No rows for {run_id}; run the local experiment first or pass --allow-missing")
        all_rows.extend(rows)

    # CSV
    csv_path = out_dir / "local_model_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    write_md_table(
        out_dir / "local_model_comparison.md",
        "WAITI Local Model Comparison",
        all_rows,
        note=(
            f"GPT-5.4 OURS from frozen `{args.gpt_run}` only. "
            "Local rows from `waiti_final_qwen{{14,7,3}}b` and `waiti_final_gptoss`."
        ),
    )

    # Scaling-focused subset: OURS across models + B1/B2 for locals
    scaling = [
        r
        for r in all_rows
        if r["Method"] in ("OURS", "B1", "B2")
        or (r["Model"] == GPT_LABEL and r["Method"] == "OURS")
    ]
    write_md_table(
        out_dir / "local_scaling_results.md",
        "WAITI Local Scaling Results",
        scaling,
        note="Cross-size view (GPT-5.4 OURS + Qwen B1/B2/OURS).",
    )

    # Final combined doc
    final_lines = [
        "# WAITI Final Local Results",
        "",
        "Cross-model comparison of local backends + hosted GPT-5.4.",
        "",
        "## Comparison table",
        "",
        (out_dir / "local_model_comparison.md").read_text(encoding="utf-8").split("\n", 2)[-1].strip(),
        "",
        "## Scaling",
        "",
        (out_dir / "local_scaling_results.md").read_text(encoding="utf-8").split("\n", 2)[-1].strip(),
        "",
    ]
    if all_warnings:
        final_lines.extend(["## Warnings", ""] + [f"- {w}" for w in all_warnings] + [""])
    (out_dir / "final_local_results.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    meta = {
        "gpt_run": args.gpt_run,
        "local_runs": [r[0] for r in LOCAL_RUNS],
        "n_comparison_rows": len(all_rows),
        "warnings": all_warnings,
        "skip_faithfulness": args.skip_faithfulness,
        "use_llm_faithfulness": args.use_llm_faithfulness,
    }
    (out_dir / "aggregation_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote comparison under {out_dir}")
    for w in all_warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
