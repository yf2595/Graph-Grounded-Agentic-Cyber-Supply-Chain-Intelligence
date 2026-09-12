"""Unit/smoke tests for fusion, graph gate helpers, and pipeline pieces."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.agent_06_network_mapper import _exposure_from_metrics
from src.agents.agent_07_final_evaluator import fuse_risk, load_fusion_matrix
from src.schemas.models import RiskLevel
from src.services.graph_service import SupplyChainGraphService


def _licensed_graph_available() -> bool:
    return (ROOT / "dataset" / "nodes.csv").exists()


def test_fusion_matrix_loads():
    m = load_fusion_matrix(ROOT / "config" / "risk_rules.yaml")
    assert fuse_risk(RiskLevel.HIGH, RiskLevel.HIGH, m) == RiskLevel.VERY_HIGH
    assert fuse_risk(RiskLevel.LOW, RiskLevel.LOW, m) == RiskLevel.LOW
    assert fuse_risk(RiskLevel.VERY_HIGH, RiskLevel.LOW, m) == RiskLevel.HIGH


def test_exposure_from_metrics():
    exp, _ = _exposure_from_metrics([{"guaranteed_reach": 56, "critical_sole_product_count": 0, "is_single_point_of_failure": True, "total_reach": 100}])
    assert exp in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)
    exp2, _ = _exposure_from_metrics([{"guaranteed_reach": 0, "critical_sole_product_count": 0, "is_single_point_of_failure": False, "total_reach": 200}])
    assert exp2 == RiskLevel.LOW


@pytest.mark.skipif(not _licensed_graph_available(), reason="Licensed OEM graph CSVs not present")
def test_graph_resolve_hone():
    g = SupplyChainGraphService(ROOT / "dataset")
    hit = g.resolve_name("H-ONE Co., Ltd.")
    assert hit is not None
    assert hit.supplier_node_id.startswith("ORG_")
    meta = g.get_supplier(hit.supplier_node_id)
    assert meta and meta["guaranteed_reach"] >= 30


@pytest.mark.skipif(not _licensed_graph_available(), reason="Licensed OEM graph CSVs not present")
def test_graph_off_graph_name():
    g = SupplyChainGraphService(ROOT / "dataset")
    assert g.resolve_name("Nordic Precision Castings AB") is None


def test_example_graph_loads():
    g = SupplyChainGraphService(ROOT / "examples" / "graph")
    hit = g.resolve_name("NorthForge Components Ltd.")
    assert hit is not None
    meta = g.get_supplier(hit.supplier_node_id)
    assert meta and meta["guaranteed_reach"] == 2
    assert meta["critical_sole_product_count"] >= 1
    assert g.resolve_name("NorthForge") is not None
    assert g.resolve_name("Acme Aerospace Castings") is None
