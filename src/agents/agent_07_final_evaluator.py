"""Agent 07 — deterministic risk fusion + LLM rationale."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.schemas.models import (
    FinalRiskAssessment,
    IncidentAssessment,
    RiskLevel,
    SupplyChainAssessment,
)
from src.services.llm_service import LLMService

SYSTEM = """You write a short OEM risk rationale.
Do NOT change the provided overall_risk, incident_severity, or oem_exposure values.
Explain why those scores fit the evidence in 2-4 sentences.
Also list key_risk_drivers and mitigating_factors as short bullet strings."""


class _RationaleOnly(FinalRiskAssessment):
    pass


def load_fusion_matrix(config_path: str | Path) -> dict[str, str]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return dict(data.get("fusion_matrix") or {})


def fuse_risk(severity: RiskLevel, exposure: RiskLevel, matrix: dict[str, str]) -> RiskLevel:
    key = f"{severity.value}|{exposure.value}"
    val = matrix.get(key)
    if not val:
        # fallback: max-ish
        order = [RiskLevel.VERY_LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        return order[min(len(order) - 1, max(order.index(severity), order.index(exposure)))]
    return RiskLevel(val)


def run_agent_07(
    llm: LLMService,
    incident_assessment: IncidentAssessment,
    supply_chain_assessment: SupplyChainAssessment,
    config_path: str | Path,
) -> FinalRiskAssessment:
    matrix = load_fusion_matrix(config_path)
    overall = fuse_risk(
        incident_assessment.severity,
        supply_chain_assessment.oem_exposure,
        matrix,
    )
    scores = {"VERY_LOW": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "VERY_HIGH": 5}

    # Escalate/mitigate lightly
    drivers: list[str] = []
    mitigations: list[str] = []
    if incident_assessment.main_effect == "PRODUCTION_SHUTDOWN":
        drivers.append("Confirmed or reported production shutdown")
    if supply_chain_assessment.propagation_scope.sole_source_part_count > 0:
        drivers.append("Sole-source parts in OEM graph")
    if any(s.is_single_point_of_failure for s in supply_chain_assessment.affected_suppliers):
        drivers.append("Single point of failure supplier")
    if supply_chain_assessment.oem_exposure in (RiskLevel.VERY_LOW, RiskLevel.LOW):
        mitigations.append("Limited sole-source OEM exposure / redundancy present")
    if incident_assessment.severity == RiskLevel.VERY_LOW:
        mitigations.append("Limited operational impact from incident itself")

    draft = FinalRiskAssessment(
        incident_id=incident_assessment.incident_id,
        incident_severity=incident_assessment.severity,
        oem_exposure=supply_chain_assessment.oem_exposure,
        overall_risk=overall,
        risk_score_numeric=scores[overall.value],
        risk_rationale="",
        key_risk_drivers=drivers,
        mitigating_factors=mitigations,
        confidence="HIGH",
    )

    user = (
        f"Fixed scores (do not change): overall_risk={overall.value}, "
        f"incident_severity={incident_assessment.severity.value}, "
        f"oem_exposure={supply_chain_assessment.oem_exposure.value}\n"
        f"main_effect={incident_assessment.main_effect}\n"
        f"exposure_rationale={supply_chain_assessment.exposure_rationale}\n"
        f"assessment_rationale={incident_assessment.assessment_rationale}\n"
        f"suggested_drivers={drivers}\n"
        f"suggested_mitigations={mitigations}\n"
        f"Return FinalRiskAssessment with the SAME overall_risk/severity/exposure and a written risk_rationale."
    )
    try:
        out = llm.structured(SYSTEM, user, FinalRiskAssessment)
        return out.model_copy(
            update={
                "incident_id": incident_assessment.incident_id,
                "incident_severity": incident_assessment.severity,
                "oem_exposure": supply_chain_assessment.oem_exposure,
                "overall_risk": overall,
                "risk_score_numeric": scores[overall.value],
            }
        )
    except Exception:
        draft.risk_rationale = (
            f"Overall OEM risk {overall.value} from incident severity "
            f"{incident_assessment.severity.value} and OEM exposure "
            f"{supply_chain_assessment.oem_exposure.value}. "
            f"{supply_chain_assessment.exposure_rationale}"
        )
        return draft
