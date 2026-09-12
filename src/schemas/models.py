from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class PipelineStatus(str, Enum):
    PASSED = "PASSED"
    FILTERED_OUT = "FILTERED_OUT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class CyberClassification(str, Enum):
    CYBER = "CYBER"
    NOT_CYBER = "NOT_CYBER"
    UNCERTAIN = "UNCERTAIN"


class RawIncident(BaseModel):
    article_id: str
    headline: str
    body: str


class Entity(BaseModel):
    model_config = {"extra": "forbid"}
    surface_name: str
    entity_type: str = "ORGANIZATION"
    role_in_incident: str | None = None
    evidence_text: str | None = None


class CyberSignal(BaseModel):
    model_config = {"extra": "forbid"}
    signal: str
    evidence_text: str | None = None


class OperationalEffect(BaseModel):
    model_config = {"extra": "forbid"}
    effect: str
    evidence_text: str | None = None


class Agent01Output(BaseModel):
    model_config = {"extra": "forbid"}
    article_id: str
    headline: str
    incident_summary: str
    entities: list[Entity] = Field(default_factory=list)
    cyber_signals: list[CyberSignal] = Field(default_factory=list)
    stated_operational_effects: list[OperationalEffect] = Field(default_factory=list)
    mentioned_locations: list[str] = Field(default_factory=list)
    mentioned_dates: list[str] = Field(default_factory=list)
    agent_confidence: str = "MEDIUM"

class RelevantSupplier(BaseModel):
    model_config = {"extra": "forbid"}
    surface_name: str
    canonical_name: str
    supplier_node_id: str
    relationship_to_oem: str = "supplier"
    match_method: str = "alias"
    match_confidence: float = 1.0


class Agent02CyberLLM(BaseModel):
    model_config = {"extra": "forbid"}
    cyber_classification: CyberClassification
    cyber_reason: str | None = None


class Agent02Output(BaseModel):
    article_id: str
    pipeline_status: PipelineStatus
    cyber_classification: CyberClassification
    cyber_reason: str | None = None
    relevant_suppliers: list[RelevantSupplier] = Field(default_factory=list)
    filter_reason: str | None = None
    incident_summary: str | None = None


class MergeDecision(str, Enum):
    CREATE_NEW = "CREATE_NEW"
    MERGE_EXISTING = "MERGE_EXISTING"


class Agent03LLMDecision(BaseModel):
    model_config = {"extra": "forbid"}
    decision: MergeDecision
    merge_into_incident_id: str | None = None
    rationale: str


class CanonicalIncident(BaseModel):
    incident_id: str
    pipeline_status: str = "ACTIVE"
    source_article_ids: list[str]
    canonical_summary: str
    event_type: str | None = None
    supplier_node_ids: list[str] = Field(default_factory=list)
    supplier_names: list[str] = Field(default_factory=list)
    merge_decision: MergeDecision = MergeDecision.CREATE_NEW
    merge_rationale: str | None = None


class TimeframeInfo(BaseModel):
    model_config = {"extra": "forbid"}
    start: str | None = None
    end: str | None = None
    duration_days: int | None = None
    status: str | None = None
    notes: str | None = None


class GeographicScope(BaseModel):
    model_config = {"extra": "forbid"}
    countries: list[str] = Field(default_factory=list)
    sites: list[str] = Field(default_factory=list)
    notes: str | None = None


class OEMProductRef(BaseModel):
    model_config = {"extra": "forbid"}
    product_id: str
    product_name: str


class IncidentAssessment(BaseModel):
    model_config = {"extra": "forbid"}
    incident_id: str
    severity: RiskLevel
    timeframe: TimeframeInfo = Field(default_factory=TimeframeInfo)
    main_effect: str
    secondary_effects: list[str] = Field(default_factory=list)
    geographic_scope: GeographicScope = Field(default_factory=GeographicScope)
    assessment_rationale: str | None = None
    confidence: str = "MEDIUM"


class PartAssessment(BaseModel):
    model_config = {"extra": "forbid"}
    part_id: str
    part_name: str
    criticality: str = "UNKNOWN"
    redundancy: str = "UNKNOWN"
    alternative_supplier_ids: list[str] = Field(default_factory=list)


class SupplierPropagation(BaseModel):
    model_config = {"extra": "forbid"}
    supplier_node_id: str
    supplier_name: str
    minimum_graph_depth_to_oem: int = 1
    tier_label: str = "Tier-1"
    supplied_parts: list[PartAssessment] = Field(default_factory=list)
    affected_oem_products: list[OEMProductRef] = Field(default_factory=list)
    affected_oem_sites: list[OEMProductRef] = Field(default_factory=list)
    path_count_to_oem_products: int = 0
    guaranteed_reach: int = 0
    total_reach: int = 0
    is_single_point_of_failure: bool = False


class PropagationScope(BaseModel):
    model_config = {"extra": "forbid"}
    product_scope: str = "UNKNOWN"
    affected_product_count: int = 0
    affected_site_count: int = 0
    critical_part_count: int = 0
    sole_source_part_count: int = 0
    max_guaranteed_reach: int = 0


class SupplyChainAssessment(BaseModel):
    model_config = {"extra": "forbid"}
    incident_id: str
    affected_suppliers: list[SupplierPropagation] = Field(default_factory=list)
    propagation_scope: PropagationScope = Field(default_factory=PropagationScope)
    oem_exposure: RiskLevel
    exposure_rationale: str
    confidence: str = "HIGH"

class FinalRiskAssessment(BaseModel):
    model_config = {"extra": "forbid"}
    incident_id: str
    incident_severity: RiskLevel
    oem_exposure: RiskLevel
    overall_risk: RiskLevel
    risk_score_numeric: int
    risk_rationale: str
    key_risk_drivers: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)
    confidence: str = "HIGH"


class UnifiedPipelineResult(BaseModel):
    article_id: str
    pipeline_status: PipelineStatus
    canonical_incident: CanonicalIncident | None = None
    incident_assessment: IncidentAssessment | None = None
    supply_chain_assessment: SupplyChainAssessment | None = None
    final_risk_assessment: FinalRiskAssessment | None = None
    report: str | None = None
    agent_02: Agent02Output | None = None
    filter_reason: str | None = None
    error: str | None = None


class BaselineOutput(BaseModel):
    model_config = {"extra": "forbid"}
    article_id: str
    pipeline_status: PipelineStatus
    cyber_classification: CyberClassification
    cyber_reason: str | None = None
    relevant_suppliers: list[RelevantSupplier] = Field(default_factory=list)
    filter_reason: str | None = None
    merge_decision: MergeDecision = MergeDecision.CREATE_NEW
    merge_into_incident_id: str | None = None
    canonical_summary: str | None = None
    incident_severity: RiskLevel | None = None
    main_effect: str | None = None
    oem_exposure: RiskLevel | None = None
    overall_risk: RiskLevel | None = None
    risk_rationale: str | None = None
    key_risk_drivers: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)
    report: str | None = None
    supplier_node_ids: list[str] = Field(default_factory=list)
