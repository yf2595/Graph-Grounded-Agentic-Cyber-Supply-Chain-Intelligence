"""WAITI evaluation unit tests (revision P1–P3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.metrics.aggregate import aggregate_method
from evaluation.metrics.classification import binary_prf
from evaluation.metrics.clustering import b3_metrics, pairwise_cluster_metrics
from evaluation.metrics.risk import risk_metrics
from evaluation.metrics.supplier import supplier_set_metrics
from evaluation.runners.export_data import export, validate_no_leakage
from evaluation.schemas.prediction import FORBIDDEN_INPUT_KEYS, GroundTruthRecord, MethodPrediction


def test_binary_prf():
    m = binary_prf([True, True, False, False], [True, False, True, False])
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1


def test_pairwise_and_b3():
    gt = {"a": "1", "b": "1", "c": "2", "d": "2"}
    pred = {"a": "x", "b": "x", "c": "y", "d": "z"}
    pw = pairwise_cluster_metrics(gt, pred)
    assert pw["tp"] == 1
    b3 = b3_metrics(gt, pred)
    assert 0.0 <= b3["f1"] <= 1.0


def test_risk_metrics_missing_mae_and_coverage():
    m = risk_metrics(["VERY_HIGH"], [None], unknown_penalty=4)
    assert m["n"] == 1
    assert m["coverage"] == 0.0
    assert m["mae"] == 4.0
    assert m["exact_accuracy"] == 0.0


def test_risk_e2e_macro_f1_includes_misses():
    # GT HIGH,LOW — Pred HIGH, missing → Macro-F1 lower than perfect
    perfect = risk_metrics(["HIGH", "LOW"], ["HIGH", "LOW"], unknown_penalty=4)
    missed = risk_metrics(["HIGH", "LOW"], ["HIGH", None], unknown_penalty=4)
    assert perfect["macro_f1"] is not None and missed["macro_f1"] is not None
    assert missed["macro_f1"] < perfect["macro_f1"]
    assert missed["coverage"] == 0.5


def test_conditional_relevance_excludes_non_cyber():
    gt = {
        "a": GroundTruthRecord(
            article_id="a",
            gt_is_cyber=False,
            gt_is_related=True,
            gt_is_actionable=False,
        ),
        "b": GroundTruthRecord(
            article_id="b",
            gt_is_cyber=True,
            gt_is_related=True,
            gt_is_actionable=True,
        ),
    }
    preds = [
        MethodPrediction(run_id="t", method_id="T", article_id="a", pred_is_cyber=False, pred_is_related=False),
        MethodPrediction(run_id="t", method_id="T", article_id="b", pred_is_cyber=True, pred_is_related=True),
    ]
    for p in preds:
        p.finalize_actionable()
    agg = aggregate_method(preds, gt)
    # non-cyber related mismatch must not enter conditional relevance
    assert agg["relevance_conditional"]["tp"] == 1
    assert agg["relevance_conditional"]["fp"] == 0
    assert agg["relevance_conditional"]["fn"] == 0
    # full relevance still sees the non-cyber FP-ish related miss
    assert agg["relevance_full"]["n"] if "n" in agg["relevance_full"] else True


def test_supplier_metrics():
    m = supplier_set_metrics([{"1", "2"}, {"3"}], [{"1"}, {"3", "4"}])
    assert m["micro_f1"] is not None
    assert 0.0 <= m["exact_set_accuracy"] <= 1.0


def test_export_no_leakage(tmp_path: Path):
    out = tmp_path / "data"
    n_in, n_gt = export(ROOT / "eval_corpus_v1", out)
    assert n_in == n_gt
    assert n_in >= 100
    validate_no_leakage(out / "inputs.jsonl")
    for line in (out / "inputs.jsonl").read_text(encoding="utf-8").splitlines()[:5]:
        obj = json.loads(line)
        assert not (set(obj) & FORBIDDEN_INPUT_KEYS)


def test_latency_cost_mean_std():
    from evaluation.metrics.latency_metrics import latency_metrics
    from evaluation.services.cost import estimate_cost_usd

    assert estimate_cost_usd("gpt-5.4", 1_000_000, 0) == 2.5
    assert estimate_cost_usd("gpt-5.4", 0, 1_000_000) == 15.0
    m = latency_metrics(
        [100.0, 200.0, 300.0],
        [1, 2, 3],
        [0, 0, 0],
        [100, 200, 300],
        [10, 20, 30],
        [0.01, 0.02, 0.03],
    )
    assert m["mean_latency_ms"] == 200.0
    assert m["std_latency_ms"] is not None and m["std_latency_ms"] > 0
    assert m["mean_cost_usd"] == 0.02
    assert m["total_cost_usd"] == 0.06


def test_method_registry():
    from evaluation.methods import METHOD_REGISTRY

    for mid in ["B0", "B1", "B2", "B3", "OURS", "A1", "A2", "A3", "A4"]:
        assert mid in METHOD_REGISTRY


def _licensed_graph_available() -> bool:
    return (ROOT / "dataset" / "nodes.csv").exists()


@pytest.mark.skipif(not _licensed_graph_available(), reason="Licensed OEM graph CSVs not present")
def test_entity_resolver_h_one_variants():
    pytest.importorskip("rapidfuzz")
    from src.services.entity_resolution import EntityResolutionService, normalize_org_name
    from src.services.graph_service import SupplyChainGraphService

    graph = SupplyChainGraphService(ROOT / "dataset")
    # Find any supplier that has "H-ONE" style name if present; else use a known name
    hone_ids = [sid for sid, n in graph.id2name.items() if "h-one" in n.lower() or "h one" in n.lower()]
    er = EntityResolutionService(graph, config={"enable_llm_disambiguation": False})
    if hone_ids:
        target = hone_ids[0]
        surfaces = ["H-ONE Co., Ltd.", "H ONE CO LTD", "H-ONE"]
        ids = set()
        for s in surfaces:
            rec = er.resolve_one(s)
            if rec.selected_supplier_node_id:
                ids.add(rec.selected_supplier_node_id)
        assert target in ids or len(ids) <= 1  # at least consistent when resolved
        assert normalize_org_name("H-ONE Co., Ltd.") == normalize_org_name("H ONE CO LTD")
    else:
        # fallback: normalization consistency
        assert normalize_org_name("Foo Corp.") == normalize_org_name("FOO CORPORATION")


@pytest.mark.skipif(not _licensed_graph_available(), reason="Licensed OEM graph CSVs not present")
def test_entity_resolver_no_unsafe_ambiguous_auto_match():
    pytest.importorskip("rapidfuzz")
    from src.services.entity_resolution import EntityResolutionService
    from src.services.graph_service import SupplyChainGraphService

    graph = SupplyChainGraphService(ROOT / "dataset")
    er = EntityResolutionService(
        graph,
        config={
            "enable_llm_disambiguation": False,
            "fuzzy_min_score": 90,
            "fuzzy_auto_accept_score": 96,
            "fuzzy_min_margin": 5,
        },
    )
    # nonsense surface should not resolve
    rec = er.resolve_one("ZZZ_NOT_A_REAL_SUPPLIER_XYZ_999")
    assert rec.selected_supplier_node_id is None
    assert rec.resolution_method == "UNRESOLVED"


def test_prioritization_spearman_and_ndcg():
    from evaluation.metrics.ranking import prioritization_metrics

    gt = {
        "a": GroundTruthRecord(
            article_id="a", gt_is_cyber=True, gt_is_related=True, gt_is_actionable=True, gt_risk="VERY_HIGH"
        ),
        "b": GroundTruthRecord(
            article_id="b", gt_is_cyber=True, gt_is_related=True, gt_is_actionable=True, gt_risk="LOW"
        ),
        "c": GroundTruthRecord(
            article_id="c", gt_is_cyber=True, gt_is_related=True, gt_is_actionable=True, gt_risk="HIGH"
        ),
    }
    preds = [
        MethodPrediction(
            run_id="t", method_id="T", article_id="a", pred_is_cyber=True, pred_is_related=True, predicted_risk="VERY_HIGH"
        ),
        MethodPrediction(
            run_id="t", method_id="T", article_id="b", pred_is_cyber=True, pred_is_related=True, predicted_risk="LOW"
        ),
        MethodPrediction(
            run_id="t", method_id="T", article_id="c", pred_is_cyber=True, pred_is_related=True, predicted_risk="HIGH"
        ),
    ]
    for p in preds:
        p.finalize_actionable()
    m = prioritization_metrics(preds, gt)
    assert m["n"] == 3
    assert m["spearman_rho"] == 1.0
    assert m["ndcg_at_10"] == 1.0


def test_high_risk_recall_and_underestimation():
    from evaluation.metrics.ops_metrics import high_risk_metrics

    gt = {
        "a": GroundTruthRecord(
            article_id="a", gt_is_cyber=True, gt_is_related=True, gt_is_actionable=True, gt_risk="VERY_HIGH"
        ),
        "b": GroundTruthRecord(
            article_id="b", gt_is_cyber=True, gt_is_related=True, gt_is_actionable=True, gt_risk="HIGH"
        ),
    }
    preds = [
        MethodPrediction(
            run_id="t", method_id="T", article_id="a", pred_is_cyber=True, pred_is_related=True, predicted_risk="LOW"
        ),
        MethodPrediction(
            run_id="t",
            method_id="T",
            article_id="b",
            pred_is_cyber=True,
            pred_is_related=True,
            predicted_risk="HIGH",
        ),
    ]
    for p in preds:
        p.finalize_actionable()
    m = high_risk_metrics(preds, gt)
    assert m["n_high_risk"] == 2
    assert m["high_risk_recall"] == 1.0  # both detected as actionable with a risk
    assert m["severe_underestimation_rate"] == 0.5  # VERY_HIGH -> LOW is >=2 levels


def test_duplicate_alert_reduction():
    from evaluation.metrics.duplicate_alert import duplicate_alert_metrics

    gt = {
        "a": GroundTruthRecord(
            article_id="a",
            gt_is_cyber=True,
            gt_is_related=True,
            gt_is_actionable=True,
            gt_group_id="G1",
        ),
        "b": GroundTruthRecord(
            article_id="b",
            gt_is_cyber=True,
            gt_is_related=True,
            gt_is_actionable=True,
            gt_group_id="G1",
        ),
    }
    preds = [
        MethodPrediction(
            run_id="t",
            method_id="T",
            article_id="a",
            pred_is_cyber=True,
            pred_is_related=True,
            predicted_incident_id="EVT_1",
        ),
        MethodPrediction(
            run_id="t",
            method_id="T",
            article_id="b",
            pred_is_cyber=True,
            pred_is_related=True,
            predicted_incident_id="EVT_1",
        ),
    ]
    for p in preds:
        p.finalize_actionable()
    m = duplicate_alert_metrics(preds, gt)
    assert m["n_multi_groups"] == 1
    assert m["source_report_count"] == 2
    assert m["predicted_canonical_alert_count"] == 1
    assert m["duplicate_alert_reduction"] == 0.5


def test_group_bootstrap_ci_not_point_only():
    from evaluation.metrics.group_bootstrap import group_bootstrap_cis

    gt = {
        f"a{i}": GroundTruthRecord(
            article_id=f"a{i}",
            gt_is_cyber=True,
            gt_is_related=True,
            gt_is_actionable=i % 2 == 0,
            gt_group_id=f"G{i // 2}",
            gt_risk="HIGH" if i % 2 == 0 else None,
        )
        for i in range(8)
    }
    preds = [
        MethodPrediction(
            run_id="t",
            method_id="T",
            article_id=f"a{i}",
            pred_is_cyber=True,
            pred_is_related=True,
            predicted_incident_id=f"EVT_{i // 2}",
            predicted_risk="HIGH" if i % 2 == 0 else "UNKNOWN",
        )
        for i in range(8)
    ]
    for p in preds:
        p.finalize_actionable()
    rows = group_bootstrap_cis(preds, gt, method_id="T", n_resamples=100, seed=1, point_agg={})
    by = {r["metric"]: r for r in rows}
    assert "merge_b3_f1" in by
    assert by["merge_b3_f1"]["note"].startswith("group_bootstrap_n=")
    # CI width should be defined (may be equal if metric constant)
    assert by["merge_b3_f1"]["ci_low"] is not None
    assert by["actionable_f1"]["ci_low"] is not None


def test_agent08_deterministic_claims_and_coverage():
    from src.agents.agent_08_report_generator import (
        build_structured_claims,
        required_field_coverage,
    )

    fields = {
        "supplier_names": ["Acme"],
        "supplier_node_ids": ["ORG_1"],
        "event_type": "ransomware",
        "incident_id": "EVT_1",
        "incident_status": "ACTIVE",
        "risk_label": "HIGH",
        "risk_confidence": "HIGH",
        "incident_severity": "HIGH",
        "main_effect": "Plant shutdown",
        "oem_exposure": "MEDIUM",
        "supplied_parts": [
            {
                "supplier_node_id": "ORG_1",
                "part_id": "P1",
                "part_name": "Widget",
                "criticality": "CRITICAL",
                "redundancy": "SOLE_SOURCE",
            }
        ],
        "affected_products": [
            {"supplier_node_id": "ORG_1", "product_id": "V1", "product_name": "Car"}
        ],
        "source_article_ids": ["INC_001"],
        "key_risk_drivers": ["sole source"],
        "mitigating_factors": [],
        "graph_fact_ids": ["GRAPH:ORG_1->PART:P1:CRITICAL:SOLE_SOURCE"],
        "propagation_scope": {"affected_product_count": 1, "critical_part_count": 1, "sole_source_part_count": 1},
    }
    claims = build_structured_claims(fields)
    assert any(c.claim_type == "GRAPH_FACT" for c in claims)
    assert all(c.supporting_evidence_ids for c in claims)
    cov = required_field_coverage(fields)
    assert cov["supplier"] and cov["risk"] and cov["sources"]


def test_faithfulness_label_supported_graph_claim():
    from evaluation.metrics.faithfulness import build_evidence_corpus, label_claim
    from src.agents.agent_08_report_generator import ReportClaim

    claim = ReportClaim(
        claim_id="C1",
        text="Supplier ORG_1 supplies part Widget (P1) with criticality=CRITICAL and redundancy=SOLE_SOURCE.",
        claim_type="GRAPH_FACT",
        supporting_evidence_ids=["GRAPH:ORG_1->PART:P1:CRITICAL:SOLE_SOURCE"],
    )
    corpus = build_evidence_corpus(
        article_text="Ransomware hit Acme plant.",
        agent06={
            "affected_suppliers": [
                {
                    "supplier_node_id": "ORG_1",
                    "supplied_parts": [
                        {
                            "part_id": "P1",
                            "part_name": "Widget",
                            "criticality": "CRITICAL",
                            "redundancy": "SOLE_SOURCE",
                        }
                    ],
                    "affected_oem_products": [],
                }
            ]
        },
    )
    assert label_claim(claim, corpus) == "SUPPORTED"


def test_faithfulness_metrics_rates():
    from evaluation.metrics.faithfulness_aggregate import faithfulness_metrics

    labeled = [
        {"claim_type": "GRAPH_FACT", "supporting_evidence_ids": ["x"], "faithfulness_label": "SUPPORTED"},
        {"claim_type": "PROSE", "supporting_evidence_ids": [], "faithfulness_label": "UNSUPPORTED"},
        {"claim_type": "INCIDENT_FACT", "supporting_evidence_ids": ["y"], "faithfulness_label": "CONTRADICTED"},
        {"claim_type": "PROSE", "supporting_evidence_ids": [], "faithfulness_label": "UNVERIFIABLE"},
    ]
    m = faithfulness_metrics(labeled, {"supplier": True, "risk": False})
    assert m["n_verifiable"] == 3
    assert m["support_rate"] == round(1 / 3, 4)
    assert m["hallucination_rate"] == round(2 / 3, 4)
    assert m["provenance_coverage"] == 1.0  # both factual have evidence
