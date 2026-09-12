"""Agent 05 — incident severity assessment (no OEM risk)."""
from __future__ import annotations

from src.schemas.models import CanonicalIncident, IncidentAssessment, RiskLevel
from src.services.llm_service import LLMService

SYSTEM = """You are Agent 05, Incident Risk Assessment.

Assess the cyber incident itself (severity, timeframe, main operational effect, geography).
Do NOT calculate OEM supply-chain risk.
Do NOT invent suppliers, parts, or OEM dependencies.

Severity guide:
- VERY_HIGH/HIGH: confirmed production shutdown / major plant halt
- MEDIUM: meaningful ops disruption without full halt clarity
- LOW: limited/contained IT impact, brief delays
- VERY_LOW: data-theft claim or IT-only with plants running / no ops impact

main_effect examples: PRODUCTION_SHUTDOWN, PARTIAL_SLOWDOWN, DATA_EXFILTRATION_ONLY, IT_DISRUPTION, UNKNOWN
"""


def run_agent_05(llm: LLMService, canonical: CanonicalIncident) -> IncidentAssessment:
    user = (
        f"incident_id: {canonical.incident_id}\n"
        f"summary: {canonical.canonical_summary}\n"
        f"suppliers: {canonical.supplier_names}\n"
        f"source_articles: {canonical.source_article_ids}\n"
    )
    out = llm.structured(SYSTEM, user, IncidentAssessment)
    return out.model_copy(update={"incident_id": canonical.incident_id})
