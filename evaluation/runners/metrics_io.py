"""WAITI aggregation writers (P1/P2/P5/P7 revision)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation.metrics.aggregate import aggregate_method
from evaluation.metrics.group_bootstrap import group_bootstrap_cis, paired_group_metric_deltas
from evaluation.schemas.prediction import GroundTruthRecord, MethodPrediction


def _write_dicts_csv(path: Path, rows: list[dict], preferred: list[str] | None = None) -> None:
    fields: list[str] = list(preferred or [])
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    if not fields:
        fields = ["method_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_method_preds(run_dir: Path, method_id: str) -> list[MethodPrediction]:
    path = run_dir / "methods" / method_id / "predictions.jsonl"
    if not path.exists():
        return []
    return [
        MethodPrediction.model_validate(json.loads(l))
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def write_aggregates(
    run_dir: Path,
    methods: list[str],
    gt: dict[str, GroundTruthRecord],
    cfg: dict,
) -> dict:
    unknown_penalty = int(cfg.get("risk", {}).get("unknown_penalty", 4))
    n_boot = int(cfg.get("bootstrap", {}).get("n_resamples", 10000))
    seed = int(cfg.get("bootstrap", {}).get("seed", 42))
    aggregates: dict = {}
    flat_rows = []
    all_group_rows = []
    supplier_rows = []

    for m in methods:
        preds = load_method_preds(run_dir, m)
        agg = aggregate_method(preds, gt, unknown_penalty=unknown_penalty)
        aggregates[m] = agg
        lat = agg.get("latency") or {}
        re2 = agg.get("risk_e2e") or {}
        rg = agg.get("risk_group") or {}
        sf = agg.get("supplier_full") or {}
        sa = agg.get("supplier_actionable") or {}
        flat_rows.append(
            {
                "method_id": m,
                "n": agg.get("n"),
                "cyber_f1": (agg.get("cyber") or {}).get("f1"),
                "relevance_conditional_f1": (agg.get("relevance_conditional") or {}).get("f1"),
                "relevance_conditional_precision": (agg.get("relevance_conditional") or {}).get("precision"),
                "relevance_conditional_recall": (agg.get("relevance_conditional") or {}).get("recall"),
                "relevance_full_f1": (agg.get("relevance_full") or {}).get("f1"),
                "relevance_full_precision": (agg.get("relevance_full") or {}).get("precision"),
                "relevance_full_recall": (agg.get("relevance_full") or {}).get("recall"),
                "actionable_f1": (agg.get("actionable") or {}).get("f1"),
                "supplier_micro_f1": sa.get("micro_f1"),
                "supplier_macro_f1": sa.get("macro_f1"),
                "supplier_exact_set_accuracy": sa.get("exact_set_accuracy"),
                "supplier_full_micro_f1": sf.get("micro_f1"),
                "merge_pairwise_f1": (agg.get("merge_pairwise") or {}).get("f1"),
                "merge_b3_f1": (agg.get("merge_b3") or {}).get("f1"),
                "risk_e2e_macro_f1": re2.get("macro_f1"),
                "risk_e2e_mae": re2.get("mae"),
                "risk_qwk_detected": re2.get("risk_qwk_detected") or re2.get("quadratic_weighted_kappa"),
                "risk_coverage": re2.get("coverage"),
                "risk_exact": re2.get("exact_accuracy"),
                "risk_adjacent": re2.get("adjacent_accuracy"),
                "risk_group_macro_f1_e2e": rg.get("risk_group_macro_f1_e2e"),
                "risk_group_mae_e2e": rg.get("risk_group_mae_e2e"),
                "risk_group_qwk_detected": rg.get("risk_group_qwk_detected"),
                "risk_group_coverage": rg.get("risk_group_coverage"),
                "mean_latency_ms": lat.get("mean_latency_ms"),
                "std_latency_ms": lat.get("std_latency_ms"),
                "median_latency_ms": lat.get("median_latency_ms"),
                "p95_latency_ms": lat.get("p95_latency_ms"),
                "mean_llm_calls": lat.get("mean_llm_calls"),
                "mean_cost_usd": lat.get("mean_cost_usd"),
                "std_cost_usd": lat.get("std_cost_usd"),
                "total_cost_usd": lat.get("total_cost_usd"),
                "spearman_rho": (agg.get("prioritization") or {}).get("spearman_rho"),
                "ndcg_at_10": (agg.get("prioritization") or {}).get("ndcg_at_10"),
                "ndcg_at_20": (agg.get("prioritization") or {}).get("ndcg_at_20"),
                "high_risk_recall": (agg.get("high_risk") or {}).get("high_risk_recall"),
                "very_high_risk_recall": (agg.get("high_risk") or {}).get("very_high_risk_recall"),
                "severe_underestimation_rate": (agg.get("high_risk") or {}).get(
                    "severe_underestimation_rate"
                ),
                "duplicate_alert_reduction": (agg.get("duplicate_alert") or {}).get(
                    "duplicate_alert_reduction"
                ),
            }
        )
        for row in agg.get("group_rows") or []:
            all_group_rows.append({"method_id": m, **row})
        supplier_rows.append(
            {
                "method_id": m,
                "scope": "actionable",
                "supplier_micro_precision": sa.get("micro_precision"),
                "supplier_micro_recall": sa.get("micro_recall"),
                "supplier_micro_f1": sa.get("micro_f1"),
                "supplier_macro_f1": sa.get("macro_f1"),
                "supplier_exact_set_accuracy": sa.get("exact_set_accuracy"),
            }
        )
        supplier_rows.append(
            {
                "method_id": m,
                "scope": "full",
                "supplier_micro_precision": sf.get("micro_precision"),
                "supplier_micro_recall": sf.get("micro_recall"),
                "supplier_micro_f1": sf.get("micro_f1"),
                "supplier_macro_f1": sf.get("macro_f1"),
                "supplier_exact_set_accuracy": sf.get("exact_set_accuracy"),
            }
        )

    with (run_dir / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()) if flat_rows else ["method_id"])
        w.writeheader()
        w.writerows(flat_rows)

    slim = {m: {k: v for k, v in agg.items() if k != "group_rows"} for m, agg in aggregates.items()}
    (run_dir / "aggregate_metrics.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")

    if all_group_rows:
        with (run_dir / "per_group_predictions.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_group_rows[0].keys()))
            w.writeheader()
            w.writerows(all_group_rows)
    risk_group_rows = [
        {
            "method_id": m,
            **(aggregates[m].get("risk_group") or {}),
        }
        for m in methods
    ]
    with (run_dir / "risk_group_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fields: list[str] = ["method_id"]
        for row in risk_group_rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(risk_group_rows)

    _write_dicts_csv(
        run_dir / "supplier_resolution_metrics.csv",
        supplier_rows,
        preferred=["method_id", "scope"],
    )

    # P5 ranking / ops / duplicate-alert
    prio_rows = [{"method_id": m, **(aggregates[m].get("prioritization") or {})} for m in methods]
    _write_dicts_csv(run_dir / "prioritization_metrics.csv", prio_rows, preferred=["method_id"])

    hr_rows = [{"method_id": m, **(aggregates[m].get("high_risk") or {})} for m in methods]
    _write_dicts_csv(run_dir / "high_risk_recall.csv", hr_rows, preferred=["method_id"])

    dup_rows = [{"method_id": m, **(aggregates[m].get("duplicate_alert") or {})} for m in methods]
    _write_dicts_csv(run_dir / "duplicate_alert_metrics.csv", dup_rows, preferred=["method_id"])

    merge_rows = []
    for m in methods:
        agg = aggregates[m]
        merge_rows.append(
            {
                "method_id": m,
                "merge_pairwise_precision": (agg.get("merge_pairwise") or {}).get("precision"),
                "merge_pairwise_recall": (agg.get("merge_pairwise") or {}).get("recall"),
                "merge_pairwise_f1": (agg.get("merge_pairwise") or {}).get("f1"),
                "merge_b3_precision": (agg.get("merge_b3") or {}).get("precision"),
                "merge_b3_recall": (agg.get("merge_b3") or {}).get("recall"),
                "merge_b3_f1": (agg.get("merge_b3") or {}).get("f1"),
                "duplicate_alert_reduction": (agg.get("duplicate_alert") or {}).get(
                    "duplicate_alert_reduction"
                ),
            }
        )
    _write_dicts_csv(run_dir / "merge_metrics.csv", merge_rows, preferred=["method_id"])

    # Paired group-bootstrap deltas: OURS vs each other method
    delta_rows: list[dict] = []
    if "OURS" in methods:
        ours_preds = load_method_preds(run_dir, "OURS")
        for other in methods:
            if other == "OURS":
                continue
            oth_preds = load_method_preds(run_dir, other)
            delta_rows.extend(
                paired_group_metric_deltas(
                    ours_preds,
                    oth_preds,
                    gt,
                    method_a="OURS",
                    method_b=other,
                    n_resamples=n_boot,
                    seed=seed,
                    unknown_penalty=unknown_penalty,
                )
            )
    _write_dicts_csv(
        run_dir / "pairwise_method_deltas.csv",
        delta_rows,
        preferred=["metric", "delta", "ci_low", "ci_high", "note"],
    )
    _write_dicts_csv(
        run_dir / "pairwise_deltas.csv",
        delta_rows,
        preferred=["metric", "delta", "ci_low", "ci_high", "note"],
    )

    # Error analysis
    err_rows = []
    for m in methods:
        for p in load_method_preds(run_dir, m):
            g = gt[p.article_id]
            tags = []
            if p.pred_is_cyber != g.gt_is_cyber:
                tags.append("cyber_mismatch")
            if p.pred_is_related != g.gt_is_related:
                tags.append("related_mismatch")
            if p.pred_is_actionable != g.gt_is_actionable:
                tags.append("actionable_mismatch")
            if g.gt_risk and p.predicted_risk != g.gt_risk and g.gt_is_actionable:
                tags.append("risk_mismatch")
            if p.status == "FAILED":
                tags.append("failed")
            if tags:
                err_rows.append(
                    {
                        "method_id": m,
                        "article_id": p.article_id,
                        "errors": "|".join(tags),
                        "status": p.status,
                        "error_type": p.error_type or "",
                    }
                )
    with (run_dir / "error_analysis.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method_id", "article_id", "errors", "status", "error_type"])
        w.writeheader()
        w.writerows(err_rows)

    # Group-block bootstrap CIs (P7)
    ci_rows: list[dict] = []
    for m in methods:
        preds = load_method_preds(run_dir, m)
        ci_rows.extend(
            group_bootstrap_cis(
                preds,
                gt,
                method_id=m,
                n_resamples=n_boot,
                seed=seed,
                unknown_penalty=unknown_penalty,
                point_agg=aggregates[m],
            )
        )

    with (run_dir / "aggregate_metrics_with_ci.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method_id", "metric", "mean", "ci_low", "ci_high", "note"])
        w.writeheader()
        w.writerows(ci_rows)

    return aggregates
