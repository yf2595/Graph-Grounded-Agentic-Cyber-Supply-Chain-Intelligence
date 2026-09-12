"""Extract claim objects from B2 baseline structured output + report."""
from __future__ import annotations

import re
from typing import Any

from src.agents.agent_08_report_generator import ReportClaim, required_field_coverage


def claims_from_baseline(result: dict[str, Any]) -> tuple[list[ReportClaim], dict[str, Any], dict[str, bool]]:
    fields: dict[str, Any] = {
        "supplier_names": [s.get("canonical_name") for s in (result.get("relevant_suppliers") or []) if s.get("canonical_name")],
        "supplier_node_ids": list(result.get("supplier_node_ids") or []),
        "event_type": None,
        "incident_id": result.get("merge_into_incident_id"),
        "incident_status": result.get("pipeline_status"),
        "risk_label": result.get("overall_risk"),
        "risk_confidence": None,
        "incident_severity": result.get("incident_severity"),
        "main_effect": result.get("main_effect"),
        "oem_exposure": result.get("oem_exposure"),
        "supplied_parts": [],
        "affected_products": [],
        "source_article_ids": [result.get("article_id")] if result.get("article_id") else [],
        "key_risk_drivers": list(result.get("key_risk_drivers") or []),
        "mitigating_factors": list(result.get("mitigating_factors") or []),
        "graph_fact_ids": [],
        "propagation_scope": {},
    }
    claims: list[ReportClaim] = []
    n = 0

    def add(text: str, ctype: str, evid: list[str], section: str = "structured") -> None:
        nonlocal n
        n += 1
        claims.append(
            ReportClaim(
                claim_id=f"B{n:03d}",
                text=text,
                claim_type=ctype,
                supporting_evidence_ids=evid,
                section=section,
            )
        )

    for s in result.get("relevant_suppliers") or []:
        add(
            f"Supplier involved: {s.get('canonical_name')} ({s.get('supplier_node_id')}).",
            "INCIDENT_FACT",
            [f"INCIDENT:supplier:{s.get('supplier_node_id')}", "ARTICLE:source"],
        )
    if result.get("main_effect"):
        add(f"Main operational effect: {result['main_effect']}.", "INCIDENT_FACT", ["ARTICLE:source"])
    if result.get("overall_risk"):
        add(f"Overall OEM risk is {result['overall_risk']}.", "RISK_FACT", ["ARTICLE:source"])
    if result.get("incident_severity"):
        add(f"Incident severity is {result['incident_severity']}.", "RISK_FACT", ["ARTICLE:source"])
    if result.get("oem_exposure"):
        add(f"OEM exposure is {result['oem_exposure']}.", "RISK_FACT", ["ARTICLE:source"])
    for d in result.get("key_risk_drivers") or []:
        add(f"Key risk driver: {d}.", "RISK_FACT", ["ARTICLE:source"])
    for m in result.get("mitigating_factors") or []:
        add(f"Mitigating factor: {m}.", "RISK_FACT", ["ARTICLE:source"])

    # Bullet lines from free-form report as prose claims (potential hallucination surface)
    report = result.get("report") or ""
    for line in report.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        text = re.sub(r"^[-*]\s*", "", line).strip()
        text = re.sub(r"\*\*", "", text)
        if len(text) < 12:
            continue
        # skip lines that merely repeat structured fields already claimed
        add(text, "PROSE", ["ARTICLE:source"], section="report_bullet")

    cov = required_field_coverage(fields)
    # B2 rarely has deterministic part/product lists in BaselineOutput
    return claims, fields, cov
