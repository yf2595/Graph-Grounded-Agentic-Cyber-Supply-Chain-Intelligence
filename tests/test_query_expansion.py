"""Unit/integration tests for Agent 4 (privacy-aware query expansion)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.query_expansion import config as C
from src.query_expansion import corpus_validation, selection
from src.query_expansion.agent4 import (
    ALLOWED_INPUT_KEYS,
    FORBIDDEN_INPUT_KEYS,
    build_public_input,
    run_agent4,
)
from src.query_expansion.enrichment import build_enriched_incident
from src.query_expansion.privacy_validator import validate_query
from src.query_expansion.schemas import (
    Gap,
    GapAnalysis,
    IncrementalDecision,
    NewEvidenceItem,
    Query,
    QuerySet,
    SameIncidentDecision,
    SingleQuery,
)
from src.query_expansion.search_backend import SearchResult
from src.schemas.models import CanonicalIncident


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, *, gap=None, queries=None, same_fn=None, incr_fn=None, regen="acme cyberattack recovery"):
        self.gap = gap or GapAnalysis(gaps=[Gap(gap_type="recovery_status", reason="r", priority=1)])
        self.queries = queries or QuerySet(
            queries=[Query(query_id=f"q{i}", gap_types=["recovery_status"], query=f"acme cyber query {i}") for i in range(1, 6)]
        )
        self.same_fn = same_fn or (lambda user: SameIncidentDecision(same_incident=True, confidence=0.9, reason="ok"))
        self.incr_fn = incr_fn or (
            lambda user: IncrementalDecision(
                adds_new_information=True, new_evidence=[NewEvidenceItem(type="recovery_status", fact="Operations resumed")]
            )
        )
        self.regen = regen

    def structured(self, system, user, schema):
        name = schema.__name__
        if name == "GapAnalysis":
            return self.gap
        if name == "QuerySet":
            return self.queries
        if name == "SingleQuery":
            return SingleQuery(query=self.regen)
        if name == "SameIncidentDecision":
            return self.same_fn(user)
        if name == "IncrementalDecision":
            return self.incr_fn(user)
        raise AssertionError(f"unexpected schema {name}")


class FakeBackend:
    def __init__(self, doc: SearchResult):
        self.doc = doc

    def search(self, query, top_k=1):
        return [self.doc]


def _canonical():
    return CanonicalIncident(
        incident_id="INC_TEST",
        source_article_ids=["A1"],
        canonical_summary="Ransomware disrupted Acme Corp manufacturing systems.",
        event_type="cyber",
        supplier_names=["Acme Corp"],
        supplier_node_ids=["ORG_TESTABCDEF01"],
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_selection_deterministic_and_unique():
    a = selection.select_incidents(selection.load_candidates())
    b = selection.select_incidents(selection.load_candidates())
    assert [c.incident_id for c in a] == [c.incident_id for c in b]
    assert len(a) == C.N_TARGET_INCIDENTS
    ids = [c.incident_id for c in a]
    assert len(set(ids)) == len(ids)
    group_keys = [c.group_key for c in a]
    assert len(set(group_keys)) == len(group_keys)  # no two share a ground-truth group


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def test_agent4_input_has_no_graph_fields():
    payload = build_public_input(_canonical())
    assert set(payload).issubset(ALLOWED_INPUT_KEYS)
    assert not (set(payload) & FORBIDDEN_INPUT_KEYS)
    assert "supplier_node_ids" not in payload


def test_privacy_validator_blocks_injected_graph_term():
    public = "organization(s): Acme\nevent_type: cyber\nsummary: ransomware at acme"
    clean, viol = validate_query("acme sole-source bottleneck exposure", public)
    assert not clean
    assert "bottleneck" in viol or "sole-source" in viol


def test_privacy_validator_allows_public_terms():
    public = "summary: acme ransomware production halt recovery"
    clean, viol = validate_query("acme ransomware recovery timeline", public)
    assert clean and viol == []


def test_injected_graph_metadata_cannot_leak_via_generated_query():
    # A query that leaks is regenerated once; if the regen is clean it is used.
    llm = FakeLLM(
        queries=QuerySet(
            queries=[
                Query(query_id="q1", gap_types=["x"], query="acme sole-source bottleneck downstream reach"),
                *[Query(query_id=f"q{i}", gap_types=["x"], query=f"acme clean {i}") for i in range(2, 6)],
            ]
        ),
        regen="acme cyberattack recovery status",
    )
    backend = FakeBackend(SearchResult(doc_id="QE_INC_TEST_SUPP_1", score=0.9, headline="h", body="b"))
    _, result = run_agent4(llm, _canonical(), backend)
    q1 = next(q for q in result.queries if q["query_id"] == "q1")
    assert q1["privacy_clean"] is True
    assert "sole-source" not in q1["query"]


def test_graph_object_not_reachable_from_agent4():
    import inspect

    import src.query_expansion.agent4 as a4

    src = inspect.getsource(a4)
    assert "SupplyChainGraphService" not in src
    assert "graph_service" not in src


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_exactly_five_queries_top1_and_dedup():
    doc = SearchResult(doc_id="QE_INC_TEST_SUPP_1", score=0.9, headline="Acme recovery", body="Operations resumed")
    _, result = run_agent4(FakeLLM(), _canonical(), FakeBackend(doc))
    assert len(result.queries) == 5
    assert len(result.retrievals) == 5  # one top-1 per query
    assert all(r["retrieved_rank"] == 1 for r in result.retrievals)
    # same doc from all queries -> deduped to a single candidate decision
    assert len(result.candidate_decisions) == 1


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------


def test_unrelated_candidate_rejected():
    llm = FakeLLM(same_fn=lambda user: SameIncidentDecision(same_incident=False, confidence=0.1, reason="diff"))
    doc = SearchResult(doc_id="QE_NOISE_001", score=0.5, headline="Aero", body="Unrelated")
    enriched, result = run_agent4(llm, _canonical(), FakeBackend(doc))
    assert result.accepted == []
    assert result.changed is False
    assert enriched.canonical_summary == _canonical().canonical_summary


def test_same_incident_non_incremental_rejected():
    llm = FakeLLM(incr_fn=lambda user: IncrementalDecision(adds_new_information=False, new_evidence=[]))
    doc = SearchResult(doc_id="QE_INC_TEST_SUPP_1", score=0.9, headline="dup", body="same facts")
    _, result = run_agent4(llm, _canonical(), FakeBackend(doc))
    assert result.accepted == []
    assert result.changed is False


def test_related_incremental_candidate_accepted():
    doc = SearchResult(doc_id="QE_INC_TEST_SUPP_1", score=0.9, headline="Acme recovery", body="Operations resumed")
    enriched, result = run_agent4(FakeLLM(), _canonical(), FakeBackend(doc))
    assert len(result.accepted) == 1
    assert result.changed is True
    assert "Supplemental evidence" in enriched.canonical_summary


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def test_enrichment_no_accepted_semantically_unchanged():
    can = _canonical()
    enriched, changed = build_enriched_incident(can, [])
    assert changed is False
    assert enriched.canonical_summary == can.canonical_summary
    assert enriched is not can  # copy, not the same object


def test_enrichment_appends_not_substitutes_with_provenance():
    can = _canonical()
    accepted = [{"doc_id": "QE_INC_TEST_SUPP_1", "new_evidence": [{"type": "recovery_status", "fact": "Operations resumed at plant"}]}]
    enriched, changed = build_enriched_incident(can, accepted)
    assert changed is True
    assert can.canonical_summary in enriched.canonical_summary  # original preserved
    assert "Operations resumed at plant" in enriched.canonical_summary
    assert "QE_INC_TEST_SUPP_1" in enriched.canonical_summary  # provenance


# ---------------------------------------------------------------------------
# Corpus counts / validation
# ---------------------------------------------------------------------------


_ADJ = ["overnight", "gradual", "regional", "prolonged", "partial", "confirmed", "escalating", "contained"]
_NOUN = ["restoration", "forensics", "shipment", "shift", "encryption", "outage", "backlog", "audit"]


def _synthetic_selected():
    """Twenty distinct incidents for corpus-count tests (no frozen artifacts)."""
    selected = []
    for i in range(1, C.N_TARGET_INCIDENTS + 1):
        selected.append(
            selection.IncidentCandidate(
                incident_id=f"INC_{i:03d}",
                group_key=f"G_{i:03d}",
                group_id=f"G_{i:03d}",
                supplier_names=[f"Supplier {i} Ltd."],
            )
        )
    return selected


def _synthetic_corpus():
    selected = _synthetic_selected()
    documents, gt = [], []
    for idx, c in enumerate(selected):
        for i in (1, 2):
            did = f"QE_{c.incident_id}_SUPP_{i}"
            adj = _ADJ[(idx * 2 + i) % len(_ADJ)]
            noun = _NOUN[(idx * 3 + i) % len(_NOUN)]
            body = (
                f"Follow-up coverage of the {c.primary_supplier()} matter describes {adj} "
                f"{noun} developments and detailed operational specifics unique to variant {i} "
                f"of case {c.incident_id}, referencing named plants and staffing figures {idx}{i}."
            )
            documents.append(
                {"doc_id": did, "headline": f"{c.primary_supplier()} {adj} {noun} update {i}", "body": body,
                 "combined_search_text": f"{c.primary_supplier()} {adj} {noun} update {i}\n\n{body}"}
            )
            gt.append({"doc_id": did, "target_incident_id": c.incident_id, "target_group_id": c.group_id,
                       "is_relevant_to_target": True, "is_noise": False, "evidence_types": [], "new_facts": [],
                       "expected_agent5_effect": "increase"})
    for i in range(1, C.N_NOISE + 1):
        did = f"QE_NOISE_{i:03d}"
        adj = _ADJ[i % len(_ADJ)]
        noun = _NOUN[i % len(_NOUN)]
        body = (
            f"An aviation avionics manufacturer reported a {adj} ransomware {noun} affecting "
            f"missile guidance assembly line {i}, with satellite telemetry crews handling "
            f"the aerospace defense contractor recovery effort under case reference {i}{i}."
        )
        documents.append(
            {"doc_id": did, "headline": f"Aerospace vendor {adj} {noun} breach {i}", "body": body,
             "combined_search_text": f"Aerospace vendor {adj} {noun} breach {i}\n\n{body}"}
        )
        gt.append({"doc_id": did, "is_noise": True, "is_relevant_to_target": False, "target_incident_id": None})
    return documents, gt


def test_corpus_counts_and_validation_pass():
    documents, gt = _synthetic_corpus()
    result = corpus_validation.validate(documents, gt)
    checks = {c["name"]: c["ok"] for c in result["checks"]}
    assert checks["exactly 40 supplemental reports"]
    assert checks["exactly 20 noise reports"]
    assert checks["exactly 60 total documents"]
    assert checks["every target incident has exactly 2 supplements"]
    assert checks["no duplicate doc IDs"]
    assert checks["all metadata stored separately from searchable text"]
    assert result["all_ok"]


def test_corpus_detects_leakage():
    documents, gt = _synthetic_corpus()
    documents[0]["combined_search_text"] += " this supplier is sole-source and a bottleneck node"
    result = corpus_validation.validate(documents, gt)
    checks = {c["name"]: c["ok"] for c in result["checks"]}
    assert not checks["no graph-only OEM attributes leaked into generated text"]
    assert not result["all_ok"]


# ---------------------------------------------------------------------------
# Pipeline regression
# ---------------------------------------------------------------------------


def test_pipeline_config_query_expansion_disabled_by_default():
    from src.orchestration.incident_pipeline import PipelineConfig

    assert PipelineConfig().query_expansion_enabled is False
