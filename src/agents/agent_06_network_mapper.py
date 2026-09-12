"""Agent 06 — graph-driven OEM propagation / exposure."""
from __future__ import annotations

from src.schemas.models import (
    CanonicalIncident,
    OEMProductRef,
    PartAssessment,
    PropagationScope,
    RiskLevel,
    SupplierPropagation,
    SupplyChainAssessment,
)
from src.services.graph_service import SupplyChainGraphService


def _exposure_from_metrics(props: list[dict]) -> tuple[RiskLevel, str]:
    if not props:
        return RiskLevel.VERY_LOW, "No supplier propagation facts."
    max_g = max(p.get("guaranteed_reach", 0) for p in props)
    max_crit_sole = max(p.get("critical_sole_product_count", 0) for p in props)
    any_spf = any(p.get("is_single_point_of_failure") for p in props)
    max_total = max(p.get("total_reach", 0) for p in props)

    if max_g >= 50 or (max_crit_sole > 0 and max_g >= 30) or (any_spf and max_g >= 30):
        return (
            RiskLevel.VERY_HIGH,
            f"High sole-source reach (guaranteed_reach={max_g}) and/or critical sole-source parts.",
        )
    if max_g >= 30 or max_crit_sole >= 1:
        return RiskLevel.HIGH, f"Sole-source bottleneck exposure (guaranteed_reach={max_g})."
    if max_g >= 10:
        return RiskLevel.MEDIUM, f"Moderate sole-source reach (guaranteed_reach={max_g})."
    if max_g >= 1:
        return RiskLevel.LOW, f"Limited sole-source reach (guaranteed_reach={max_g})."
    if max_total >= 30:
        return (
            RiskLevel.LOW,
            f"Large total reach ({max_total}) but guaranteed_reach=0 (second-sourced).",
        )
    return RiskLevel.VERY_LOW, "Low OEM structural exposure from graph metrics."


def run_agent_06(
    graph: SupplyChainGraphService,
    canonical: CanonicalIncident,
) -> SupplyChainAssessment:
    props_raw = []
    affected: list[SupplierPropagation] = []
    names = canonical.supplier_names or canonical.supplier_node_ids
    for sid, sname in zip(canonical.supplier_node_ids, names):
        raw = graph.build_propagation(sid)
        if not raw:
            continue
        props_raw.append(raw)
        parts = [
            PartAssessment(
                part_id=p["part_id"],
                part_name=p["part_name"],
                criticality=p["criticality"],
                redundancy=p["redundancy"],
                alternative_supplier_ids=p.get("alternative_supplier_ids") or [],
            )
            for p in raw.get("supplied_parts", [])
        ]
        products = [
            OEMProductRef(product_id=p["product_id"], product_name=p["product_name"])
            for p in raw.get("affected_oem_products", [])
        ]
        affected.append(
            SupplierPropagation(
                supplier_node_id=raw["supplier_node_id"],
                supplier_name=raw.get("supplier_name") or sname,
                minimum_graph_depth_to_oem=1,
                tier_label="Tier-1",
                supplied_parts=parts,
                affected_oem_products=products,
                affected_oem_sites=[],
                path_count_to_oem_products=raw.get("path_count_to_oem_products", 0),
                guaranteed_reach=raw.get("guaranteed_reach", 0),
                total_reach=raw.get("total_reach", 0),
                is_single_point_of_failure=bool(raw.get("is_single_point_of_failure")),
            )
        )

    exposure, rationale = _exposure_from_metrics(props_raw)
    scope = PropagationScope(
        product_scope="MULTIPLE_PRODUCTS"
        if sum(a.path_count_to_oem_products for a in affected) > 1
        else "SINGLE_PRODUCT",
        affected_product_count=sum(a.path_count_to_oem_products for a in affected),
        affected_site_count=0,
        critical_part_count=sum(
            len([p for p in a.supplied_parts if p.criticality == "CRITICAL"]) for a in affected
        ),
        sole_source_part_count=sum(
            len([p for p in a.supplied_parts if p.redundancy == "SOLE_SOURCE"]) for a in affected
        ),
        max_guaranteed_reach=max((a.guaranteed_reach for a in affected), default=0),
    )
    return SupplyChainAssessment(
        incident_id=canonical.incident_id,
        affected_suppliers=affected,
        propagation_scope=scope,
        oem_exposure=exposure,
        exposure_rationale=rationale,
        confidence="HIGH",
    )
