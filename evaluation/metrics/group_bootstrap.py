"""Group-block bootstrap for WAITI metrics."""
from __future__ import annotations

from evaluation.metrics.bootstrap import bootstrap_metric_by_group
from evaluation.metrics.classification import binary_prf
from evaluation.metrics.clustering import b3_metrics
from evaluation.metrics.group_risk import group_risk_evaluation
from evaluation.metrics.risk import risk_metrics
from evaluation.metrics.supplier import supplier_set_metrics
from evaluation.schemas.prediction import GroundTruthRecord, MethodPrediction


def _article_groups(gt: dict[str, GroundTruthRecord], aids: list[str]) -> dict[str, str]:
    return {a: (gt[a].gt_group_id or f"SOLO_{a}") for a in aids if a in gt}


def group_bootstrap_cis(
    preds: list[MethodPrediction],
    gt: dict[str, GroundTruthRecord],
    *,
    method_id: str,
    n_resamples: int,
    seed: int,
    unknown_penalty: int = 4,
    point_agg: dict | None = None,
) -> list[dict]:
    """True group-unit bootstrap for key WAITI metrics."""
    by = {p.article_id: p for p in preds}
    aids = [a for a in by if a in gt]
    groups = _article_groups(gt, aids)
    point_agg = point_agg or {}
    rows: list[dict] = []

    def _ci(name: str, fn, point) -> None:
        # Cap resamples for very expensive fns when n is huge — still honor config default
        ci = bootstrap_metric_by_group(
            groups, aids, fn, n_resamples=n_resamples, seed=seed + hash(method_id + name) % 10000
        )
        rows.append(
            {
                "method_id": method_id,
                "metric": name,
                "mean": point if point is not None else ci["mean"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "note": f"group_bootstrap_n={n_resamples}",
            }
        )

    def cyber_f1(sample: list[str]) -> float:
        return binary_prf(
            [gt[i].gt_is_cyber for i in sample],
            [by[i].pred_is_cyber for i in sample],
        )["f1"]

    def actionable_f1(sample: list[str]) -> float:
        return binary_prf(
            [gt[i].gt_is_actionable for i in sample],
            [by[i].pred_is_actionable for i in sample],
        )["f1"]

    def cond_rel_f1(sample: list[str]) -> float:
        ids = [i for i in sample if gt[i].gt_is_cyber]
        if not ids:
            return 0.0
        return binary_prf(
            [gt[i].gt_is_related for i in ids],
            [by[i].pred_is_related for i in ids],
        )["f1"]

    def supplier_f1(sample: list[str]) -> float:
        act = [i for i in sample if gt[i].gt_is_actionable]
        if not act:
            return 0.0
        m = supplier_set_metrics(
            [set(gt[i].gt_supplier_node_ids or []) for i in act],
            [set(by[i].predicted_supplier_node_ids or []) for i in act],
        )
        return float(m["micro_f1"] or 0.0)

    def _cluster_labs(sample: list[str]) -> tuple[dict[str, str], dict[str, str]]:
        # Unique keys so resampled-with-replacement groups count as distinct instances
        gt_lab: dict[str, str] = {}
        pred_lab: dict[str, str] = {}
        for idx, i in enumerate(sample):
            if not gt[i].gt_group_id:
                continue
            key = f"{i}#{idx}"
            gt_lab[key] = gt[i].gt_group_id
            pred_lab[key] = by[i].predicted_incident_id or f"SOLO_{i}"
        return gt_lab, pred_lab

    def merge_b3_f1(sample: list[str]) -> float:
        gt_lab, pred_lab = _cluster_labs(sample)
        if len(gt_lab) < 2:
            return 0.0
        return float(b3_metrics(gt_lab, pred_lab).get("f1") or 0.0)

    def merge_pw_f1(sample: list[str]) -> float:
        from evaluation.metrics.clustering import pairwise_cluster_metrics

        gt_lab, pred_lab = _cluster_labs(sample)
        if len(gt_lab) < 2:
            return 0.0
        return float(pairwise_cluster_metrics(gt_lab, pred_lab).get("f1") or 0.0)

    def risk_macro(sample: list[str]) -> float:
        y_t, y_p = [], []
        for i in sample:
            g, p = gt[i], by[i]
            if not g.gt_is_actionable or not g.gt_risk:
                continue
            y_t.append(g.gt_risk)
            if p.pred_is_actionable and p.predicted_risk != "UNKNOWN":
                y_p.append(p.predicted_risk)
            else:
                y_p.append(None)
        return float(risk_metrics(y_t, y_p, unknown_penalty=unknown_penalty).get("macro_f1") or 0.0)

    def risk_mae(sample: list[str]) -> float:
        y_t, y_p = [], []
        for i in sample:
            g, p = gt[i], by[i]
            if not g.gt_is_actionable or not g.gt_risk:
                continue
            y_t.append(g.gt_risk)
            if p.pred_is_actionable and p.predicted_risk != "UNKNOWN":
                y_p.append(p.predicted_risk)
            else:
                y_p.append(None)
        mae = risk_metrics(y_t, y_p, unknown_penalty=unknown_penalty).get("mae")
        return float(mae if mae is not None else 0.0)

    def risk_group_macro(sample: list[str]) -> float:
        sub_preds = [by[i] for i in sample]
        sub_gt = {i: gt[i] for i in sample}
        _, rg = group_risk_evaluation(sub_preds, sub_gt, unknown_penalty=unknown_penalty)
        return float(rg.get("risk_group_macro_f1_e2e") or 0.0)

    def risk_coverage(sample: list[str]) -> float:
        y_t, y_p = [], []
        for i in sample:
            g, p = gt[i], by[i]
            if not g.gt_is_actionable or not g.gt_risk:
                continue
            y_t.append(g.gt_risk)
            if p.pred_is_actionable and p.predicted_risk != "UNKNOWN":
                y_p.append(p.predicted_risk)
            else:
                y_p.append(None)
        return float(risk_metrics(y_t, y_p, unknown_penalty=unknown_penalty).get("coverage") or 0.0)

    def risk_qwk(sample: list[str]) -> float:
        y_t, y_p = [], []
        for i in sample:
            g, p = gt[i], by[i]
            if not g.gt_is_actionable or not g.gt_risk:
                continue
            y_t.append(g.gt_risk)
            if p.pred_is_actionable and p.predicted_risk != "UNKNOWN":
                y_p.append(p.predicted_risk)
            else:
                y_p.append(None)
        q = risk_metrics(y_t, y_p, unknown_penalty=unknown_penalty).get("risk_qwk_detected")
        if q is None:
            q = risk_metrics(y_t, y_p, unknown_penalty=unknown_penalty).get("quadratic_weighted_kappa")
        return float(q if q is not None else 0.0)

    mapping = [
        ("cyber_f1", cyber_f1, (point_agg.get("cyber") or {}).get("f1")),
        ("actionable_f1", actionable_f1, (point_agg.get("actionable") or {}).get("f1")),
        (
            "relevance_conditional_f1",
            cond_rel_f1,
            (point_agg.get("relevance_conditional") or {}).get("f1"),
        ),
        ("supplier_micro_f1", supplier_f1, (point_agg.get("supplier_actionable") or {}).get("micro_f1")),
        ("merge_pairwise_f1", merge_pw_f1, (point_agg.get("merge_pairwise") or {}).get("f1")),
        ("merge_b3_f1", merge_b3_f1, (point_agg.get("merge_b3") or {}).get("f1")),
        ("risk_e2e_macro_f1", risk_macro, (point_agg.get("risk_e2e") or {}).get("macro_f1")),
        ("risk_e2e_mae", risk_mae, (point_agg.get("risk_e2e") or {}).get("mae")),
        ("risk_coverage", risk_coverage, (point_agg.get("risk_e2e") or {}).get("coverage")),
        (
            "risk_qwk_detected",
            risk_qwk,
            (point_agg.get("risk_e2e") or {}).get("risk_qwk_detected")
            or (point_agg.get("risk_e2e") or {}).get("quadratic_weighted_kappa"),
        ),
        (
            "risk_group_macro_f1_e2e",
            risk_group_macro,
            (point_agg.get("risk_group") or {}).get("risk_group_macro_f1_e2e"),
        ),
    ]
    for name, fn, point in mapping:
        _ci(name, fn, point)

    # latency median: article-level values with group bootstrap of median
    def median_lat(sample: list[str]) -> float:
        xs = sorted(float(by[i].latency_ms) for i in sample)
        if not xs:
            return 0.0
        mid = len(xs) // 2
        return xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])

    _ci(
        "median_latency_ms",
        median_lat,
        (point_agg.get("latency") or {}).get("median_latency_ms"),
    )
    return rows


def paired_group_metric_deltas(
    preds_a: list[MethodPrediction],
    preds_b: list[MethodPrediction],
    gt: dict[str, GroundTruthRecord],
    *,
    method_a: str,
    method_b: str,
    n_resamples: int,
    seed: int,
    unknown_penalty: int = 4,
) -> list[dict]:
    """Paired bootstrap of metric(A)-metric(B) on the same resampled groups."""
    import random
    from collections import defaultdict

    by_a = {p.article_id: p for p in preds_a}
    by_b = {p.article_id: p for p in preds_b}
    common = sorted(set(by_a) & set(by_b) & set(gt))
    if not common:
        return []

    groups: dict[str, list[str]] = defaultdict(list)
    for a in common:
        gid = gt[a].gt_group_id or f"SOLO_{a}"
        groups[gid].append(a)
    gids = list(groups.keys())
    if not gids:
        return []

    def _metric_fns(by: dict[str, MethodPrediction]):
        def cyber_f1(sample: list[str]) -> float:
            return binary_prf(
                [gt[i].gt_is_cyber for i in sample],
                [by[i].pred_is_cyber for i in sample],
            )["f1"]

        def actionable_f1(sample: list[str]) -> float:
            return binary_prf(
                [gt[i].gt_is_actionable for i in sample],
                [by[i].pred_is_actionable for i in sample],
            )["f1"]

        def cond_rel_f1(sample: list[str]) -> float:
            ids = [i for i in sample if gt[i].gt_is_cyber]
            if not ids:
                return 0.0
            return binary_prf(
                [gt[i].gt_is_related for i in ids],
                [by[i].pred_is_related for i in ids],
            )["f1"]

        def supplier_f1(sample: list[str]) -> float:
            act = [i for i in sample if gt[i].gt_is_actionable]
            if not act:
                return 0.0
            m = supplier_set_metrics(
                [set(gt[i].gt_supplier_node_ids or []) for i in act],
                [set(by[i].predicted_supplier_node_ids or []) for i in act],
            )
            return float(m["micro_f1"] or 0.0)

        def merge_b3(sample: list[str]) -> float:
            gt_lab, pred_lab = {}, {}
            for idx, i in enumerate(sample):
                if not gt[i].gt_group_id:
                    continue
                key = f"{i}#{idx}"
                gt_lab[key] = gt[i].gt_group_id
                pred_lab[key] = by[i].predicted_incident_id or f"SOLO_{i}"
            if len(gt_lab) < 2:
                return 0.0
            return float(b3_metrics(gt_lab, pred_lab).get("f1") or 0.0)

        def risk_macro(sample: list[str]) -> float:
            y_t, y_p = [], []
            for i in sample:
                g, p = gt[i], by[i]
                if not g.gt_is_actionable or not g.gt_risk:
                    continue
                y_t.append(g.gt_risk)
                if p.pred_is_actionable and p.predicted_risk != "UNKNOWN":
                    y_p.append(p.predicted_risk)
                else:
                    y_p.append(None)
            return float(risk_metrics(y_t, y_p, unknown_penalty=unknown_penalty).get("macro_f1") or 0.0)

        def risk_qwk(sample: list[str]) -> float:
            y_t, y_p = [], []
            for i in sample:
                g, p = gt[i], by[i]
                if not g.gt_is_actionable or not g.gt_risk:
                    continue
                y_t.append(g.gt_risk)
                if p.pred_is_actionable and p.predicted_risk != "UNKNOWN":
                    y_p.append(p.predicted_risk)
                else:
                    y_p.append(None)
            metrics = risk_metrics(y_t, y_p, unknown_penalty=unknown_penalty)
            q = metrics.get("risk_qwk_detected")
            if q is None:
                q = metrics.get("quadratic_weighted_kappa")
            return float(q if q is not None else 0.0)

        return {
            "cyber_f1": cyber_f1,
            "actionable_f1": actionable_f1,
            "relevance_conditional_f1": cond_rel_f1,
            "supplier_micro_f1": supplier_f1,
            "merge_b3_f1": merge_b3,
            "risk_e2e_macro_f1": risk_macro,
            "risk_qwk_detected": risk_qwk,
        }

    fns_a = _metric_fns(by_a)
    fns_b = _metric_fns(by_b)
    rows: list[dict] = []
    rng = random.Random(seed + hash(f"{method_a}:{method_b}") % 10000)

    for metric_name in fns_a:
        estimates = []
        for _ in range(n_resamples):
            sample_g = [gids[rng.randrange(len(gids))] for _ in range(len(gids))]
            sample_articles: list[str] = []
            for i, g in enumerate(sample_g):
                for a in groups[g]:
                    sample_articles.append(a)
            try:
                estimates.append(fns_a[metric_name](sample_articles) - fns_b[metric_name](sample_articles))
            except Exception:
                continue
        if not estimates:
            continue
        estimates.sort()
        alpha = 0.05
        lo = estimates[int(alpha / 2 * (len(estimates) - 1))]
        hi = estimates[int((1 - alpha / 2) * (len(estimates) - 1))]
        # point delta on full common set
        point = fns_a[metric_name](common) - fns_b[metric_name](common)
        rows.append(
            {
                "metric": f"{metric_name}_{method_a}_minus_{method_b}",
                "delta": round(point, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "note": f"paired_group_bootstrap_n={n_resamples}",
            }
        )
    return rows
