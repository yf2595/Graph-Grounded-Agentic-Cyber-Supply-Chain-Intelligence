"""Aggregate all WAITI metrics for one method's predictions (revision P1/P2)."""
from __future__ import annotations

from evaluation.metrics.classification import binary_prf
from evaluation.metrics.clustering import b3_metrics, pairwise_cluster_metrics
from evaluation.metrics.duplicate_alert import duplicate_alert_metrics
from evaluation.metrics.group_risk import group_risk_evaluation
from evaluation.metrics.latency_metrics import latency_metrics
from evaluation.metrics.ops_metrics import high_risk_metrics
from evaluation.metrics.ranking import prioritization_metrics
from evaluation.metrics.risk import risk_metrics
from evaluation.metrics.supplier import supplier_set_metrics
from evaluation.schemas.prediction import GroundTruthRecord, MethodPrediction


def aggregate_method(
    preds: list[MethodPrediction],
    gt_by_id: dict[str, GroundTruthRecord],
    unknown_penalty: int = 4,
) -> dict:
    pairs = [(gt_by_id[p.article_id], p) for p in preds if p.article_id in gt_by_id]
    if not pairs:
        return {}

    cyber = binary_prf(
        [g.gt_is_cyber for g, _ in pairs],
        [p.pred_is_cyber for _, p in pairs],
    )

    # Full-dataset relevance (secondary)
    relevance_full = binary_prf(
        [g.gt_is_related for g, _ in pairs],
        [p.pred_is_related for _, p in pairs],
    )

    # Conditional relevance (primary): only true cyber incidents
    cyber_pairs = [(g, p) for g, p in pairs if g.gt_is_cyber]
    relevance_conditional = binary_prf(
        [g.gt_is_related for g, _ in cyber_pairs],
        [p.pred_is_related for _, p in cyber_pairs],
    ) if cyber_pairs else {"precision": None, "recall": None, "f1": None, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    actionable = binary_prf(
        [g.gt_is_actionable for g, _ in pairs],
        [p.pred_is_actionable for _, p in pairs],
    )

    # merge
    gt_labels = {}
    pred_labels = {}
    for g, p in pairs:
        if not g.gt_group_id:
            continue
        gt_labels[g.article_id] = g.gt_group_id
        pred_labels[g.article_id] = p.predicted_incident_id or f"SOLO_{p.article_id}"

    merge_pairwise = pairwise_cluster_metrics(gt_labels, pred_labels) if gt_labels else {}
    merge_b3 = b3_metrics(gt_labels, pred_labels) if gt_labels else {}

    gt_act = {a: gid for a, gid in gt_labels.items() if gt_by_id[a].gt_is_actionable}
    pred_act = {a: pred_labels[a] for a in gt_act}
    merge_pairwise_act = pairwise_cluster_metrics(gt_act, pred_act) if gt_act else {}
    merge_b3_act = b3_metrics(gt_act, pred_act) if gt_act else {}

    # Supplier metrics
    gold_sets = [set(g.gt_supplier_node_ids or []) for g, _ in pairs]
    pred_sets = [set(p.predicted_supplier_node_ids or []) for _, p in pairs]
    supplier_full = supplier_set_metrics(gold_sets, pred_sets)

    act_pairs = [(g, p) for g, p in pairs if g.gt_is_actionable]
    supplier_actionable = supplier_set_metrics(
        [set(g.gt_supplier_node_ids or []) for g, _ in act_pairs],
        [set(p.predicted_supplier_node_ids or []) for _, p in act_pairs],
    ) if act_pairs else supplier_set_metrics([], [])

    # risk e2e on actionable gold
    y_true = []
    y_pred = []
    for g, p in pairs:
        if not g.gt_is_actionable or not g.gt_risk:
            continue
        y_true.append(g.gt_risk)
        if p.pred_is_actionable:
            y_pred.append(p.predicted_risk if p.predicted_risk != "UNKNOWN" else None)
        else:
            y_pred.append(None)
    risk = risk_metrics(y_true, y_pred, unknown_penalty=unknown_penalty)

    y_true_c, y_pred_c = [], []
    for g, p in pairs:
        if g.gt_risk and p.pred_is_actionable and p.predicted_risk != "UNKNOWN":
            y_true_c.append(g.gt_risk)
            y_pred_c.append(p.predicted_risk)
    risk_cond = risk_metrics(y_true_c, y_pred_c, unknown_penalty=unknown_penalty)

    group_rows, risk_group = group_risk_evaluation(
        [p for _, p in pairs], gt_by_id, unknown_penalty=unknown_penalty
    )

    ranking = prioritization_metrics([p for _, p in pairs], gt_by_id)
    high_risk = high_risk_metrics([p for _, p in pairs], gt_by_id)
    dup_alert = duplicate_alert_metrics([p for _, p in pairs], gt_by_id)

    lat = latency_metrics(
        [p.latency_ms for _, p in pairs],
        [p.llm_calls for _, p in pairs],
        [p.graph_queries for _, p in pairs],
        [p.input_tokens for _, p in pairs],
        [p.output_tokens for _, p in pairs],
        [float(getattr(p, "cost_usd", 0.0) or 0.0) for _, p in pairs],
    )

    return {
        "n": len(pairs),
        "cyber": cyber,
        "relevance_full": relevance_full,
        "relevance_conditional": relevance_conditional,
        # backward-compat alias (prefer conditional in tables)
        "relevance": relevance_conditional,
        "actionable": actionable,
        "supplier_full": supplier_full,
        "supplier_actionable": supplier_actionable,
        "merge_pairwise": merge_pairwise,
        "merge_b3": merge_b3,
        "merge_pairwise_actionable": merge_pairwise_act,
        "merge_b3_actionable": merge_b3_act,
        "risk_e2e": risk,
        "risk_conditional": risk_cond,
        "risk_group": risk_group,
        "group_rows": group_rows,
        "prioritization": ranking,
        "high_risk": high_risk,
        "duplicate_alert": dup_alert,
        "latency": lat,
    }
