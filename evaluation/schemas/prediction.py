"""WAITI MethodPrediction schema."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal[
    "VERY_LOW",
    "LOW",
    "MEDIUM",
    "HIGH",
    "VERY_HIGH",
    "UNKNOWN",
]

FORBIDDEN_INPUT_KEYS = {
    "is_cyber",
    "is_related",
    "relevant",
    "in_supply_chain",
    "risk",
    "risk_bin_draft",
    "risk_bin_gold",
    "impact_level",
    "group_id",
    "case_stratum",
    "supplier_node_ids",
    "supplier_names",
    "guaranteed_reach",
    "total_reach",
    "is_bottleneck",
    "is_sole_source_critical_part",
    "is_multi_supplier",
    "has_no_alternative",
    "risk_score_draft",
    "risk_rationale",
    "merge_seed_article_id",
    "is_merge_variant",
}


class ModelInputRecord(BaseModel):
    article_id: str
    headline: str
    body: str
    source_name: str | None = None
    published_at: str | None = None


class GroundTruthRecord(BaseModel):
    article_id: str
    gt_is_cyber: bool
    gt_is_related: bool
    gt_is_actionable: bool
    gt_risk: RiskLevel | None = None
    gt_group_id: str | None = None
    gt_supplier_node_ids: list[str] = Field(default_factory=list)


class MethodPrediction(BaseModel):
    run_id: str
    repeat_id: int = 0
    method_id: str
    article_id: str

    pred_is_cyber: bool | None = None
    cyber_confidence: float | None = None
    pred_is_related: bool | None = None
    relevance_confidence: float | None = None
    pred_is_actionable: bool | None = None

    predicted_supplier_names: list[str] = Field(default_factory=list)
    predicted_supplier_node_ids: list[str] = Field(default_factory=list)

    predicted_incident_id: str | None = None
    merge_action: str | None = None
    merged_into_incident_id: str | None = None

    predicted_incident_severity: RiskLevel = "UNKNOWN"
    predicted_main_effect: str | None = None
    predicted_time_horizon: str | None = None
    predicted_locations: list[str] = Field(default_factory=list)

    predicted_oem_exposure: RiskLevel = "UNKNOWN"
    predicted_critical_part_ids: list[str] = Field(default_factory=list)
    predicted_affected_product_ids: list[str] = Field(default_factory=list)
    predicted_affected_site_ids: list[str] = Field(default_factory=list)
    predicted_graph_depth: int | None = None
    predicted_redundancy: str | None = None

    predicted_risk: RiskLevel = "UNKNOWN"
    risk_rationale: str | None = None

    started_at: str = ""
    completed_at: str = ""
    latency_ms: float = 0.0
    llm_calls: int = 0
    graph_queries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    status: str = "COMPLETED"
    error_type: str | None = None
    retry_count: int = 0

    def finalize_actionable(self) -> None:
        if self.pred_is_cyber is True and self.pred_is_related is True:
            self.pred_is_actionable = True
        elif self.pred_is_cyber is False or self.pred_is_related is False:
            self.pred_is_actionable = False
        else:
            self.pred_is_actionable = None
