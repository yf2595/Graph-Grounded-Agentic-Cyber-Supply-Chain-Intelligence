"""High-risk recall and severe under-estimation (WAITI P5)."""
from __future__ import annotations

from evaluation.metrics.risk import RISK_SCORE, _norm
from evaluation.schemas.prediction import GroundTruthRecord, MethodPrediction


def high_risk_metrics(
    preds: list[MethodPrediction],
    gt_by_id: dict[str, GroundTruthRecord],
) -> dict:
    pred_by = {p.article_id: p for p in preds}

    def _detected(p: MethodPrediction) -> bool:
        return bool(
            p.pred_is_actionable
            and p.predicted_risk
            and p.predicted_risk != "UNKNOWN"
            and _norm(p.predicted_risk)
        )

    high_ids = [
        a
        for a, g in gt_by_id.items()
        if a in pred_by and g.gt_is_actionable and _norm(g.gt_risk) in {"HIGH", "VERY_HIGH"}
    ]
    vh_ids = [
        a
        for a, g in gt_by_id.items()
        if a in pred_by and g.gt_is_actionable and _norm(g.gt_risk) == "VERY_HIGH"
    ]

    high_hit = sum(1 for a in high_ids if _detected(pred_by[a]))
    vh_hit = sum(1 for a in vh_ids if _detected(pred_by[a]))

    # severe under-estimation among gold-actionable with gold risk
    under = total = 0
    for a, g in gt_by_id.items():
        if a not in pred_by or not g.gt_is_actionable or not g.gt_risk:
            continue
        gt_n = RISK_SCORE.get(_norm(g.gt_risk) or "", None)
        if gt_n is None:
            continue
        total += 1
        p = pred_by[a]
        if _detected(p):
            pr_n = RISK_SCORE[_norm(p.predicted_risk)]  # type: ignore[index]
        else:
            pr_n = 0
        if pr_n <= gt_n - 2:
            under += 1

    return {
        "n_high_risk": len(high_ids),
        "high_risk_recall": round(high_hit / len(high_ids), 4) if high_ids else None,
        "n_very_high_risk": len(vh_ids),
        "very_high_risk_recall": round(vh_hit / len(vh_ids), 4) if vh_ids else None,
        "n_actionable_risk": total,
        "severe_underestimation_rate": round(under / total, 4) if total else None,
    }
