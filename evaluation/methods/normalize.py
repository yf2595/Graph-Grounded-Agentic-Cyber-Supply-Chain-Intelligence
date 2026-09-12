"""Helper: map pipeline/baseline outputs to MethodPrediction."""
from __future__ import annotations

from evaluation.schemas.prediction import MethodPrediction, RiskLevel
from src.schemas.models import BaselineOutput, PipelineStatus, UnifiedPipelineResult


def _risk(val) -> RiskLevel:
    if val is None:
        return "UNKNOWN"
    if hasattr(val, "value"):
        v = val.value
    else:
        v = str(val)
    v = v.upper().replace(" ", "_")
    if v in {"VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH", "UNKNOWN"}:
        return v  # type: ignore[return-value]
    return "UNKNOWN"


def from_pipeline(pred: MethodPrediction, result: UnifiedPipelineResult) -> MethodPrediction:
    a2 = result.agent_02
    if a2:
        cyber = a2.cyber_classification.value if a2.cyber_classification else None
        pred.pred_is_cyber = cyber == "CYBER" if cyber else None
        if a2.filter_reason == "NOT_CYBER":
            pred.pred_is_cyber = False
        pred.pred_is_related = len(a2.relevant_suppliers) > 0
        pred.predicted_supplier_names = [s.canonical_name for s in a2.relevant_suppliers]
        pred.predicted_supplier_node_ids = [s.supplier_node_id for s in a2.relevant_suppliers]

    if result.pipeline_status == PipelineStatus.FILTERED_OUT:
        pred.finalize_actionable()
        pred.status = "FILTERED_OUT"
        return pred

    if result.pipeline_status == PipelineStatus.FAILED:
        pred.status = "FAILED"
        pred.error_type = result.error
        return pred

    can = result.canonical_incident
    if can:
        pred.predicted_incident_id = can.incident_id
        pred.merge_action = can.merge_decision.value if can.merge_decision else None
        if can.merge_decision and can.merge_decision.value == "MERGE_EXISTING":
            pred.merged_into_incident_id = can.incident_id
        if can.supplier_node_ids:
            pred.predicted_supplier_node_ids = can.supplier_node_ids
            pred.predicted_supplier_names = can.supplier_names
            pred.pred_is_related = True

    ia = result.incident_assessment
    if ia:
        pred.predicted_incident_severity = _risk(ia.severity)
        pred.predicted_main_effect = ia.main_effect

    sc = result.supply_chain_assessment
    if sc:
        pred.predicted_oem_exposure = _risk(sc.oem_exposure)
        if sc.affected_suppliers:
            pred.predicted_graph_depth = sc.affected_suppliers[0].minimum_graph_depth_to_oem
            parts = []
            prods = []
            for s in sc.affected_suppliers:
                for p in s.supplied_parts:
                    if p.criticality == "CRITICAL":
                        parts.append(p.part_id)
                for pr in s.affected_oem_products:
                    prods.append(pr.product_id)
            pred.predicted_critical_part_ids = parts[:20]
            pred.predicted_affected_product_ids = prods[:20]

    fr = result.final_risk_assessment
    if fr:
        pred.predicted_risk = _risk(fr.overall_risk)
        pred.risk_rationale = fr.risk_rationale

    pred.pred_is_cyber = pred.pred_is_cyber if pred.pred_is_cyber is not None else True
    pred.finalize_actionable()
    pred.status = "COMPLETED"
    return pred


def from_baseline(pred: MethodPrediction, out: BaselineOutput) -> MethodPrediction:
    cyber = out.cyber_classification.value if out.cyber_classification else None
    pred.pred_is_cyber = cyber == "CYBER" if cyber else None
    if out.filter_reason and "not cyber" in (out.filter_reason or "").lower():
        pred.pred_is_cyber = False
    pred.predicted_supplier_names = [s.canonical_name for s in out.relevant_suppliers]
    pred.predicted_supplier_node_ids = out.supplier_node_ids or [
        s.supplier_node_id for s in out.relevant_suppliers
    ]
    pred.pred_is_related = len(pred.predicted_supplier_node_ids) > 0
    pred.merge_action = out.merge_decision.value if out.merge_decision else None
    pred.merged_into_incident_id = out.merge_into_incident_id
    pred.predicted_incident_id = out.merge_into_incident_id
    pred.predicted_incident_severity = _risk(out.incident_severity)
    pred.predicted_main_effect = out.main_effect
    pred.predicted_oem_exposure = _risk(out.oem_exposure)
    pred.predicted_risk = _risk(out.overall_risk)
    pred.risk_rationale = out.risk_rationale
    if out.pipeline_status.value == "FILTERED_OUT":
        pred.status = "FILTERED_OUT"
    elif out.pipeline_status.value == "FAILED":
        pred.status = "FAILED"
        pred.error_type = out.filter_reason
    else:
        pred.status = "COMPLETED"
    pred.finalize_actionable()
    return pred
