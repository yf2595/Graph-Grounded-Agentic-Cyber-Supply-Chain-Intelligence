"""Audit-prioritization ranking metrics (WAITI P5)."""
from __future__ import annotations

import math
from typing import Sequence

from evaluation.metrics.risk import RISK_SCORE, _norm
from evaluation.schemas.prediction import GroundTruthRecord, MethodPrediction


def _spearman(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    # rank with average ties
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(x), ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def _dcg(rels: Sequence[float], k: int) -> float:
    s = 0.0
    for i, r in enumerate(rels[:k]):
        s += (2**r - 1) / math.log2(i + 2)
    return s


def ndcg_at_k(y_true: list[float], y_score: list[float], k: int) -> float | None:
    if not y_true:
        return None
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    ranked = [y_true[i] for i in order]
    ideal = sorted(y_true, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return _dcg(ranked, k) / idcg


def prioritization_metrics(
    preds: list[MethodPrediction],
    gt_by_id: dict[str, GroundTruthRecord],
) -> dict:
    """Spearman / nDCG on gold-actionable incidents with gold risk."""
    y_true: list[float] = []
    y_pred: list[float] = []
    for p in preds:
        g = gt_by_id.get(p.article_id)
        if not g or not g.gt_is_actionable or not g.gt_risk:
            continue
        gt_n = RISK_SCORE.get(_norm(g.gt_risk) or "", None)
        if gt_n is None:
            continue
        # undetected / unknown -> lowest predicted priority
        if p.pred_is_actionable and p.predicted_risk and p.predicted_risk != "UNKNOWN":
            pr_n = RISK_SCORE.get(_norm(p.predicted_risk) or "", 1)
        else:
            pr_n = 0
        y_true.append(float(gt_n))
        y_pred.append(float(pr_n))

    if not y_true:
        return {
            "n": 0,
            "spearman_rho": None,
            "ndcg_at_10": None,
            "ndcg_at_20": None,
        }
    return {
        "n": len(y_true),
        "spearman_rho": round(_spearman(y_true, y_pred) or 0.0, 4),
        "ndcg_at_10": round(ndcg_at_k(y_true, y_pred, 10) or 0.0, 4),
        "ndcg_at_20": round(ndcg_at_k(y_true, y_pred, 20) or 0.0, 4),
    }
