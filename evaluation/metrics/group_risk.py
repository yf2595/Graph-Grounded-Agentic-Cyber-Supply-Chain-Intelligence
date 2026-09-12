"""Group-level risk evaluation (WAITI P1 §7)."""
from __future__ import annotations

from collections import defaultdict

from evaluation.metrics.risk import RISK_SCORE, _norm, risk_metrics
from evaluation.schemas.prediction import GroundTruthRecord, MethodPrediction


def _majority_risk(risks: list[str]) -> str | None:
    vals = [r for r in (_norm(x) for x in risks) if r]
    if not vals:
        return None
    # prefer highest ordinal on ties (conservative for OEM risk)
    counts: dict[str, int] = defaultdict(int)
    for v in vals:
        counts[v] += 1
    best = max(counts.items(), key=lambda kv: (kv[1], RISK_SCORE[kv[0]]))
    return best[0]


def _greedy_match(
    gt_groups: dict[str, set[str]],
    pred_clusters: dict[str, set[str]],
) -> dict[str, str | None]:
    """Match each GT group to a predicted cluster by max overlap (greedy Hungarian-style)."""
    matches: dict[str, str | None] = {gid: None for gid in gt_groups}
    used_pred: set[str] = set()
    # score all pairs
    pairs: list[tuple[int, str, str]] = []
    for gid, garts in gt_groups.items():
        for pid, parts in pred_clusters.items():
            ov = len(garts & parts)
            if ov > 0:
                pairs.append((ov, gid, pid))
    pairs.sort(reverse=True)
    for ov, gid, pid in pairs:
        if matches[gid] is not None or pid in used_pred:
            continue
        matches[gid] = pid
        used_pred.add(pid)
    return matches


def group_risk_evaluation(
    preds: list[MethodPrediction],
    gt_by_id: dict[str, GroundTruthRecord],
    unknown_penalty: int = 4,
) -> tuple[list[dict], dict]:
    """Return (per_group rows, aggregate risk_group metrics)."""
    pred_by_id = {p.article_id: p for p in preds}

    # GT groups among actionable articles with a group_id; solo actionable => synthetic group
    gt_groups: dict[str, set[str]] = defaultdict(set)
    gt_risks: dict[str, list[str]] = defaultdict(list)
    for aid, g in gt_by_id.items():
        if aid not in pred_by_id:
            continue
        if not g.gt_is_actionable or not g.gt_risk:
            continue
        gid = g.gt_group_id or f"SOLO_{aid}"
        gt_groups[gid].add(aid)
        gt_risks[gid].append(g.gt_risk)

    # Predicted clusters
    pred_clusters: dict[str, set[str]] = defaultdict(set)
    pred_risk_by_incident: dict[str, list[str]] = defaultdict(list)
    for aid, p in pred_by_id.items():
        if aid not in gt_by_id:
            continue
        pid = p.predicted_incident_id or f"SOLO_{aid}"
        pred_clusters[pid].add(aid)
        if p.pred_is_actionable and p.predicted_risk and p.predicted_risk != "UNKNOWN":
            pred_risk_by_incident[pid].append(p.predicted_risk)

    matches = _greedy_match(dict(gt_groups), dict(pred_clusters))

    rows: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str | None] = []

    for gid, arts in sorted(gt_groups.items()):
        gt_r = _majority_risk(gt_risks[gid])
        if not gt_r:
            continue
        pid = matches.get(gid)
        overlap = len(arts & pred_clusters[pid]) if pid else 0
        pred_count = len(pred_clusters[pid]) if pid else 0
        pred_r = _majority_risk(pred_risk_by_incident.get(pid or "", [])) if pid else None
        detected = pred_r is not None
        abs_err = None
        if detected:
            abs_err = abs(RISK_SCORE[gt_r] - RISK_SCORE[pred_r])  # type: ignore[index]
        elif gt_r:
            abs_err = float(unknown_penalty)

        rows.append(
            {
                "gt_group_id": gid,
                "matched_predicted_incident_id": pid or "",
                "gt_article_count": len(arts),
                "pred_article_count": pred_count,
                "overlap_count": overlap,
                "gt_risk": gt_r,
                "predicted_risk": pred_r or "",
                "risk_exact_correct": bool(detected and pred_r == gt_r),
                "risk_absolute_error": abs_err if abs_err is not None else "",
                "risk_detected": detected,
            }
        )
        y_true.append(gt_r)
        y_pred.append(pred_r)

    metrics = risk_metrics(y_true, y_pred, unknown_penalty=unknown_penalty)
    group_agg = {
        "n_groups": len(rows),
        "risk_group_macro_f1_e2e": metrics.get("macro_f1"),
        "risk_group_mae_e2e": metrics.get("mae"),
        "risk_group_qwk_detected": metrics.get("risk_qwk_detected"),
        "risk_group_coverage": metrics.get("coverage"),
        "risk_group_exact_accuracy": metrics.get("exact_accuracy"),
    }
    return rows, group_agg
