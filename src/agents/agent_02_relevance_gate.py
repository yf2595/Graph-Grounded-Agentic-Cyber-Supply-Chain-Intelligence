"""Agent 02 — cyber + OEM supply-chain relevance gate."""
from __future__ import annotations

from src.schemas.models import (
    Agent01Output,
    Agent02CyberLLM,
    Agent02Output,
    CyberClassification,
    PipelineStatus,
)
from src.services.entity_resolution import EntityResolutionService
from src.services.llm_service import LLMService

SYSTEM = """You are Agent 02 cyber classifier for an OEM supply-chain monitoring system.

Classify whether the incident is cyber-related versus non-cyber or uncertain.

CYBER examples: ransomware, malware, phishing/BEC, data breach or exfiltration claim,
OT/IT intrusion, DDoS against industrial/IT systems, confirmed cyber attack on a company.

NOT_CYBER examples: strike, fire, flood, earthquake, product recall, bankruptcy/restructuring,
ordinary financial loss, non-cyber outage with no attack claim.

UNCERTAIN: weak or ambiguous wording where cyber is possible but not supported by clear signals.

Rules:
- Use the article headline/body and the structured extraction.
- Do not invent facts, organizations, or cyber claims.
- Prefer NOT_CYBER when the text clearly attributes the event to non-cyber causes.
- Prefer UNCERTAIN only when evidence is genuinely mixed or incomplete.
Return structured output only."""


def run_agent_02(
    llm: LLMService,
    extracted: Agent01Output,
    resolver: EntityResolutionService,
    body: str | None = None,
) -> Agent02Output:
    # Prefer richer context for cyber classification
    body_snip = (body or "")[:3000]
    user = (
        f"article_id: {extracted.article_id}\n"
        f"headline: {extracted.headline}\n"
        f"body: {body_snip}\n"
        f"summary: {extracted.incident_summary}\n"
        f"cyber_signals: {[c.model_dump() for c in extracted.cyber_signals]}\n"
        f"effects: {[e.model_dump() for e in extracted.stated_operational_effects]}\n"
        f"entities: {[e.model_dump() for e in extracted.entities]}\n"
    )
    cyber = llm.structured(SYSTEM, user, Agent02CyberLLM)

    surfaces = [
        e.surface_name
        for e in extracted.entities
        if e.entity_type.upper().startswith("ORG") or e.entity_type == "ORGANIZATION"
    ]
    if not surfaces:
        surfaces = [e.surface_name for e in extracted.entities]

    # Ensure resolver can call LLM disambiguation when needed
    prev_llm = getattr(resolver, "llm", None)
    resolver.llm = llm
    article_context = f"{extracted.headline}\n{body_snip}\n{extracted.incident_summary}"
    try:
        suppliers = resolver.resolve_surfaces(
            surfaces,
            article_id=extracted.article_id,
            article_context=article_context,
        )
    finally:
        resolver.llm = prev_llm

    if cyber.cyber_classification == CyberClassification.NOT_CYBER:
        return Agent02Output(
            article_id=extracted.article_id,
            pipeline_status=PipelineStatus.FILTERED_OUT,
            cyber_classification=cyber.cyber_classification,
            cyber_reason=cyber.cyber_reason,
            relevant_suppliers=[],
            filter_reason="NOT_CYBER",
            incident_summary=extracted.incident_summary,
        )

    if cyber.cyber_classification == CyberClassification.UNCERTAIN and not extracted.cyber_signals:
        return Agent02Output(
            article_id=extracted.article_id,
            pipeline_status=PipelineStatus.FILTERED_OUT,
            cyber_classification=cyber.cyber_classification,
            cyber_reason=cyber.cyber_reason,
            relevant_suppliers=suppliers,
            filter_reason="NOT_CYBER",
            incident_summary=extracted.incident_summary,
        )

    if not suppliers:
        return Agent02Output(
            article_id=extracted.article_id,
            pipeline_status=PipelineStatus.FILTERED_OUT,
            cyber_classification=CyberClassification.CYBER
            if cyber.cyber_classification != CyberClassification.NOT_CYBER
            else cyber.cyber_classification,
            cyber_reason=cyber.cyber_reason,
            relevant_suppliers=[],
            filter_reason="NO_RELEVANT_SUPPLIER",
            incident_summary=extracted.incident_summary,
        )

    return Agent02Output(
        article_id=extracted.article_id,
        pipeline_status=PipelineStatus.PASSED,
        cyber_classification=CyberClassification.CYBER,
        cyber_reason=cyber.cyber_reason,
        relevant_suppliers=suppliers,
        filter_reason=None,
        incident_summary=extracted.incident_summary,
    )
