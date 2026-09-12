"""Agent 08 — structured report generator with claim provenance.

Hard factual fields are copied deterministically from Agents 3/5/6/7 + graph.
LLM only fills Executive Summary / Why This Matters / Monitoring Priorities.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.schemas.models import (
    CanonicalIncident,
    FinalRiskAssessment,
    IncidentAssessment,
    SupplyChainAssessment,
)
from src.services.llm_service import LLMService

SYSTEM = """You are Agent 08, Report Generator.
Write ONLY the three prose sections requested.
Do NOT invent suppliers, parts, products, graph facts, risk labels, or dates.
Do NOT contradict the provided structured fields.
Keep Monitoring Priorities non-destructive (monitor/verify/escalate — no remediation orders).
"""


class ProseSections(BaseModel):
    model_config = {"extra": "forbid"}
    executive_summary: str
    why_this_matters: str
    monitoring_priorities: list[str] = Field(default_factory=list)


class ReportClaim(BaseModel):
    model_config = {"extra": "forbid"}
    claim_id: str
    text: str
    claim_type: str  # GRAPH_FACT | INCIDENT_FACT | RISK_FACT | PROSE
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    section: str = "structured"


class StructuredReport(BaseModel):
    model_config = {"extra": "forbid"}
    report_markdown: str
    claims: list[ReportClaim] = Field(default_factory=list)
    deterministic_fields: dict[str, Any] = Field(default_factory=dict)
    required_field_coverage: dict[str, bool] = Field(default_factory=dict)


def _build_deterministic_fields(
    incident: CanonicalIncident,
    incident_assessment: IncidentAssessment,
    supply_chain_assessment: SupplyChainAssessment,
    final_risk: FinalRiskAssessment,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    graph_facts: list[str] = []
    for sp in supply_chain_assessment.affected_suppliers or []:
        for part in sp.supplied_parts or []:
            parts.append(
                {
                    "supplier_node_id": sp.supplier_node_id,
                    "part_id": part.part_id,
                    "part_name": part.part_name,
                    "criticality": part.criticality,
                    "redundancy": part.redundancy,
                }
            )
            graph_facts.append(
                f"GRAPH:{sp.supplier_node_id}->PART:{part.part_id}:{part.criticality}:{part.redundancy}"
            )
        for prod in sp.affected_oem_products or []:
            products.append(
                {
                    "supplier_node_id": sp.supplier_node_id,
                    "product_id": prod.product_id,
                    "product_name": prod.product_name,
                }
            )
            graph_facts.append(f"GRAPH:{sp.supplier_node_id}->PRODUCT:{prod.product_id}")

    return {
        "supplier_names": list(incident.supplier_names or []),
        "supplier_node_ids": list(incident.supplier_node_ids or []),
        "event_type": incident.event_type,
        "incident_id": incident.incident_id,
        "incident_status": incident.pipeline_status,
        "risk_label": getattr(final_risk.overall_risk, "value", str(final_risk.overall_risk)),
        "risk_confidence": final_risk.confidence,
        "incident_severity": getattr(
            incident_assessment.severity, "value", str(incident_assessment.severity)
        ),
        "main_effect": incident_assessment.main_effect,
        "oem_exposure": getattr(
            supply_chain_assessment.oem_exposure, "value", str(supply_chain_assessment.oem_exposure)
        ),
        "supplied_parts": parts,
        "affected_products": products,
        "source_article_ids": list(incident.source_article_ids or []),
        "key_risk_drivers": list(final_risk.key_risk_drivers or []),
        "mitigating_factors": list(final_risk.mitigating_factors or []),
        "graph_fact_ids": graph_facts,
        "propagation_scope": supply_chain_assessment.propagation_scope.model_dump(),
    }


def build_structured_claims(fields: dict[str, Any]) -> list[ReportClaim]:
    claims: list[ReportClaim] = []
    n = 0

    def add(text: str, ctype: str, evid: list[str], section: str = "structured") -> None:
        nonlocal n
        n += 1
        claims.append(
            ReportClaim(
                claim_id=f"C{n:03d}",
                text=text,
                claim_type=ctype,
                supporting_evidence_ids=evid,
                section=section,
            )
        )

    names = fields.get("supplier_names") or []
    ids = fields.get("supplier_node_ids") or []
    for name, sid in zip(names, ids):
        add(
            f"Supplier involved: {name} ({sid}).",
            "INCIDENT_FACT",
            [f"INCIDENT:supplier:{sid}", f"ARTICLE:source"],
        )
    if not names and ids:
        for sid in ids:
            add(f"Supplier node involved: {sid}.", "INCIDENT_FACT", [f"INCIDENT:supplier:{sid}"])

    if fields.get("event_type"):
        add(
            f"Event type: {fields['event_type']}.",
            "INCIDENT_FACT",
            ["INCIDENT:event_type"],
        )
    if fields.get("main_effect"):
        add(
            f"Main operational effect: {fields['main_effect']}.",
            "INCIDENT_FACT",
            ["AGENT5:main_effect", "ARTICLE:source"],
        )
    if fields.get("risk_label"):
        add(
            f"Overall OEM risk is {fields['risk_label']}.",
            "RISK_FACT",
            ["AGENT7:overall_risk"],
        )
    if fields.get("incident_severity"):
        add(
            f"Incident severity is {fields['incident_severity']}.",
            "RISK_FACT",
            ["AGENT5:severity"],
        )
    if fields.get("oem_exposure"):
        add(
            f"OEM exposure is {fields['oem_exposure']}.",
            "RISK_FACT",
            ["AGENT6:oem_exposure"],
        )

    for part in fields.get("supplied_parts") or []:
        add(
            (
                f"Supplier {part.get('supplier_node_id')} supplies part "
                f"{part.get('part_name')} ({part.get('part_id')}) with "
                f"criticality={part.get('criticality')} and redundancy={part.get('redundancy')}."
            ),
            "GRAPH_FACT",
            [
                f"GRAPH:{part.get('supplier_node_id')}->PART:{part.get('part_id')}:"
                f"{part.get('criticality')}:{part.get('redundancy')}"
            ],
        )
    for prod in fields.get("affected_products") or []:
        add(
            (
                f"Supplier {prod.get('supplier_node_id')} reaches OEM product "
                f"{prod.get('product_name')} ({prod.get('product_id')})."
            ),
            "GRAPH_FACT",
            [f"GRAPH:{prod.get('supplier_node_id')}->PRODUCT:{prod.get('product_id')}"],
        )

    scope = fields.get("propagation_scope") or {}
    if scope.get("affected_product_count") is not None:
        add(
            f"Affected product count (graph): {scope.get('affected_product_count')}.",
            "GRAPH_FACT",
            ["AGENT6:propagation_scope"],
        )
    if scope.get("sole_source_part_count"):
        add(
            f"Sole-source part count: {scope.get('sole_source_part_count')}.",
            "GRAPH_FACT",
            ["AGENT6:propagation_scope"],
        )

    for d in fields.get("key_risk_drivers") or []:
        add(f"Key risk driver: {d}.", "RISK_FACT", ["AGENT7:key_risk_drivers"])
    for m in fields.get("mitigating_factors") or []:
        add(f"Mitigating factor: {m}.", "RISK_FACT", ["AGENT7:mitigating_factors"])

    sources = fields.get("source_article_ids") or []
    if sources:
        add(
            f"Source article IDs: {', '.join(sources)}.",
            "INCIDENT_FACT",
            [f"ARTICLE:{a}" for a in sources],
            section="sources",
        )
    return claims


def required_field_coverage(fields: dict[str, Any]) -> dict[str, bool]:
    scope = fields.get("propagation_scope") or {}
    return {
        "supplier": bool(fields.get("supplier_names") or fields.get("supplier_node_ids")),
        "event": bool(fields.get("event_type") or fields.get("main_effect")),
        "risk": bool(fields.get("risk_label")),
        "main_effect": bool(fields.get("main_effect")),
        "critical_dependency": bool(
            any(
                (p.get("criticality") or "").upper() in {"CRITICAL", "HIGH", "VERY_HIGH"}
                for p in (fields.get("supplied_parts") or [])
            )
            or (scope.get("critical_part_count") or 0) > 0
        ),
        "redundancy": bool(
            any((p.get("redundancy") or "").upper() not in {"", "UNKNOWN"} for p in (fields.get("supplied_parts") or []))
            or scope.get("sole_source_part_count") is not None
        ),
        "affected_product_scope": bool(
            fields.get("affected_products") or (scope.get("affected_product_count") or 0) > 0
        ),
        "sources": bool(fields.get("source_article_ids")),
    }


def _render_markdown(
    fields: dict[str, Any],
    prose: ProseSections,
) -> str:
    parts_lines = []
    for p in fields.get("supplied_parts") or []:
        parts_lines.append(
            f"- {p.get('part_name')} ({p.get('part_id')}): criticality={p.get('criticality')}, "
            f"redundancy={p.get('redundancy')}"
        )
    prod_lines = []
    for p in fields.get("affected_products") or []:
        prod_lines.append(f"- {p.get('product_name')} ({p.get('product_id')})")
    mon = "\n".join(f"- {x}" for x in (prose.monitoring_priorities or [])) or "- Monitor supplier status updates."
    drivers = "\n".join(f"- {d}" for d in (fields.get("key_risk_drivers") or [])) or "- (none listed)"
    mitig = "\n".join(f"- {m}" for m in (fields.get("mitigating_factors") or [])) or "- (none listed)"
    suppliers = ", ".join(fields.get("supplier_names") or fields.get("supplier_node_ids") or ["unknown"])
    sources = ", ".join(fields.get("source_article_ids") or [])
    return (
        "# Cyber Supply-Chain Risk Alert\n\n"
        f"## Executive Summary\n{prose.executive_summary.strip()}\n\n"
        f"## Overall OEM Risk\n- Risk Level: {fields.get('risk_label')}\n"
        f"- Confidence: {fields.get('risk_confidence')}\n"
        f"- Incident severity: {fields.get('incident_severity')}\n"
        f"- OEM exposure: {fields.get('oem_exposure')}\n\n"
        f"## Incident\n- Incident ID: {fields.get('incident_id')}\n"
        f"- Status: {fields.get('incident_status')}\n"
        f"- Suppliers: {suppliers}\n"
        f"- Event type: {fields.get('event_type') or 'n/a'}\n"
        f"- Main effect: {fields.get('main_effect')}\n\n"
        f"## Why This Matters to the OEM\n{prose.why_this_matters.strip()}\n\n"
        f"## Supply-Chain Exposure\n"
        f"- Affected products:\n{chr(10).join(prod_lines) if prod_lines else '- (none listed)'}\n"
        f"- Supplied parts:\n{chr(10).join(parts_lines) if parts_lines else '- (none listed)'}\n\n"
        f"## Key Risk Drivers\n{drivers}\n\n"
        f"## Mitigating Factors\n{mitig}\n\n"
        f"## Recommended Monitoring Priorities\n{mon}\n\n"
        f"## Sources\n- {sources}\n"
    )


def run_agent_08(
    llm: LLMService,
    incident: CanonicalIncident,
    incident_assessment: IncidentAssessment,
    supply_chain_assessment: SupplyChainAssessment,
    final_risk: FinalRiskAssessment,
) -> StructuredReport | str:
    """Generate structured report. Returns StructuredReport; str kept for legacy callers."""
    fields = _build_deterministic_fields(
        incident, incident_assessment, supply_chain_assessment, final_risk
    )
    claims = build_structured_claims(fields)
    coverage = required_field_coverage(fields)

    user = (
        "Fill ONLY these prose fields given the structured facts below.\n"
        "Do not add new suppliers/parts/products/risks.\n\n"
        f"structured_fields={fields}\n"
    )
    try:
        prose = llm.structured(SYSTEM, user, ProseSections)
    except Exception:
        prose = ProseSections(
            executive_summary=(
                f"{', '.join(fields.get('supplier_names') or ['Supplier'])}: "
                f"{fields.get('main_effect') or 'cyber incident'}. "
                f"Overall OEM risk: {fields.get('risk_label')}."
            ),
            why_this_matters=final_risk.risk_rationale or "OEM exposure depends on graph reach of affected suppliers.",
            monitoring_priorities=[
                "Monitor supplier operational status",
                "Verify sole-source part exposure",
                "Track OEM product impact updates",
            ],
        )

    # Provenance for prose: cite upstream agent rationale IDs (not new facts)
    prose_claims: list[ReportClaim] = []
    base = len(claims)
    if prose.executive_summary.strip():
        prose_claims.append(
            ReportClaim(
                claim_id=f"C{base+1:03d}",
                text=prose.executive_summary.strip(),
                claim_type="PROSE",
                supporting_evidence_ids=["AGENT5:main_effect", "AGENT7:overall_risk", "ARTICLE:source"],
                section="executive_summary",
            )
        )
    if prose.why_this_matters.strip():
        prose_claims.append(
            ReportClaim(
                claim_id=f"C{base+len(prose_claims)+1:03d}",
                text=prose.why_this_matters.strip(),
                claim_type="PROSE",
                supporting_evidence_ids=["AGENT7:risk_rationale", "AGENT6:exposure_rationale"],
                section="why_this_matters",
            )
        )
    for i, prio in enumerate(prose.monitoring_priorities or []):
        prose_claims.append(
            ReportClaim(
                claim_id=f"C{base+len(prose_claims)+1:03d}",
                text=prio,
                claim_type="PROSE",
                supporting_evidence_ids=["AGENT7:key_risk_drivers"],
                section="monitoring_priorities",
            )
        )

    all_claims = claims + prose_claims
    md = _render_markdown(fields, prose)
    return StructuredReport(
        report_markdown=md,
        claims=all_claims,
        deterministic_fields=fields,
        required_field_coverage=coverage,
    )
