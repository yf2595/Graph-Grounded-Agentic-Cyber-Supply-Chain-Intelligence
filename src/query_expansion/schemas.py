"""Pydantic schemas for Agent 4 LLM I/O and corpus/evaluation records."""
from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Corpus generation (LLM output schemas)
# ---------------------------------------------------------------------------


class GeneratedReport(BaseModel):
    """A single generated report body (supplement or noise)."""

    model_config = {"extra": "forbid"}
    headline: str
    body: str


class SupplementPlanItem(BaseModel):
    model_config = {"extra": "forbid"}
    evidence_types: list[str] = Field(default_factory=list)
    new_facts: list[str] = Field(default_factory=list)
    expected_agent5_effect: str  # increase | decrease | no_ordinal_change
    rationale: str


# ---------------------------------------------------------------------------
# Agent 4 - Stage A: gap analysis
# ---------------------------------------------------------------------------


class Gap(BaseModel):
    model_config = {"extra": "forbid"}
    gap_type: str
    reason: str
    priority: int = 1


class GapAnalysis(BaseModel):
    model_config = {"extra": "forbid"}
    gaps: list[Gap] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 4 - Stage B: query generation
# ---------------------------------------------------------------------------


class Query(BaseModel):
    model_config = {"extra": "forbid"}
    query_id: str
    gap_types: list[str] = Field(default_factory=list)
    query: str


class QuerySet(BaseModel):
    model_config = {"extra": "forbid"}
    queries: list[Query] = Field(default_factory=list)


class SingleQuery(BaseModel):
    """Used for one-off regeneration of a rejected query."""

    model_config = {"extra": "forbid"}
    query: str


# ---------------------------------------------------------------------------
# Agent 4 - Stage D: relevance + incremental value
# ---------------------------------------------------------------------------


class SameIncidentDecision(BaseModel):
    model_config = {"extra": "forbid"}
    same_incident: bool
    confidence: float = 0.0
    reason: str = ""


class NewEvidenceItem(BaseModel):
    model_config = {"extra": "forbid"}
    type: str
    fact: str


class IncrementalDecision(BaseModel):
    model_config = {"extra": "forbid"}
    adds_new_information: bool
    new_evidence: list[NewEvidenceItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Enriched reference labeling (Section 19)
# ---------------------------------------------------------------------------


class EnrichedReferenceLabel(BaseModel):
    model_config = {"extra": "forbid"}
    original_reference_risk: str
    enriched_reference_risk: str
    expected_direction: str  # increase | decrease | same
    reference_rationale: str
