"""Duplicate-alert reduction from merger (WAITI P5 §23)."""
from __future__ import annotations

from collections import defaultdict

from evaluation.schemas.prediction import GroundTruthRecord, MethodPrediction


def duplicate_alert_metrics(
    preds: list[MethodPrediction],
    gt_by_id: dict[str, GroundTruthRecord],
) -> dict:
    """On multi-report gold groups: 1 - (#predicted canonical alerts / #source reports)."""
    pred_by = {p.article_id: p for p in preds}
    groups: dict[str, list[str]] = defaultdict(list)
    for a, g in gt_by_id.items():
        if a not in pred_by:
            continue
        if not g.gt_group_id:
            continue
        groups[g.gt_group_id].append(a)

    multi = {gid: aids for gid, aids in groups.items() if len(aids) >= 2}
    if not multi:
        return {
            "n_multi_groups": 0,
            "source_report_count": 0,
            "predicted_canonical_alert_count": 0,
            "duplicate_alert_reduction": None,
        }

    source_n = 0
    pred_alerts: set[str] = set()
    for aids in multi.values():
        source_n += len(aids)
        for a in aids:
            p = pred_by[a]
            if p.pred_is_actionable and p.predicted_incident_id:
                pred_alerts.add(p.predicted_incident_id)
            elif p.pred_is_actionable:
                pred_alerts.add(f"SOLO_{a}")
            # filtered non-actionable: no alert

    pred_n = len(pred_alerts)
    reduction = 1.0 - (pred_n / source_n) if source_n else None
    return {
        "n_multi_groups": len(multi),
        "source_report_count": source_n,
        "predicted_canonical_alert_count": pred_n,
        "duplicate_alert_reduction": round(reduction, 4) if reduction is not None else None,
    }
