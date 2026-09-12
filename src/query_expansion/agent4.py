"""Agent 4 - Privacy-Aware Query Expansion (Stages A-E).

Runs after Agent 3 and before Agent 5. Receives ONLY the public canonical
incident and a SearchBackend; it has no access to the OEM supply-chain graph.
All LLM stages run at GPT-5.4 reasoning=low (matching Agents 5-8).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.query_expansion import config as C
from src.query_expansion import relevance_validator as rv
from src.query_expansion.enrichment import build_enriched_incident
from src.query_expansion.privacy_validator import validate_query
from src.query_expansion.schemas import GapAnalysis, Query, QuerySet, SingleQuery
from src.query_expansion.search_backend import SearchBackend
from src.schemas.models import CanonicalIncident

# Fields Agent 4 is allowed to receive. Anything else is a privacy violation.
ALLOWED_INPUT_KEYS = {
    "incident_id",
    "headline_summary",
    "affected_organizations",
    "event_type",
    "source_evidence",
}
# Graph-derived fields that must never enter the Agent-4 payload.
FORBIDDEN_INPUT_KEYS = {
    "oem_exposure",
    "supplier_criticality",
    "products_supplied",
    "vehicle_programs",
    "sole_source",
    "redundancy",
    "bottleneck",
    "downstream_reach",
    "graph_derived_risk",
    "final_risk_label",
    "ground_truth",
    "supplier_node_ids",
    "guaranteed_reach",
    "total_reach",
}

GAP_SYSTEM = """You are Agent 4, a privacy-aware evidence-gap analyst for automotive cyber \
supply-chain incidents. Given ONLY public incident evidence, identify what important \
information about the cyber incident itself is missing, uncertain, or weakly supported.

Rules:
- Identify only gaps relevant to understanding the cyber incident (severity, operations,
  recovery, technical facts).
- Prioritize gaps that could change a supplier-consequence assessment.
- Do NOT ask about OEM-specific exposure, supplier criticality, sole-source status,
  redundancy, bottleneck status, downstream reach, or any private supply-chain graph facts.
- Do NOT infer graph facts.
- Return at most 5 gaps, each with gap_type, reason, priority (1=highest).

Allowed gap_type values: {gap_types}."""

GAP_USER = """Public incident:
{incident_text}

Return the prioritized evidence gaps."""

QUERY_SYSTEM = """You are Agent 4 generating search queries to recover missing incident \
evidence from a document corpus. Generate EXACTLY 5 queries.

Each query must:
- target one or more identified gaps;
- stay grounded in known public incident facts (organization, event type, dates);
- be specific enough to retrieve the same incident;
- vary wording and information need (do not paraphrase the same query five times);
- contain NO OEM-graph-only attributes (no sole-source, redundancy, bottleneck,
  criticality, downstream reach, Honda dependency, vehicle programs, supplied part or
  node identifiers)."""

QUERY_USER = """Public incident:
{incident_text}

Identified gaps:
{gaps}

Return exactly 5 queries. Each: query_id (q1..q5), gap_types, query."""

REGEN_SYSTEM = """You are Agent 4. Your previous query leaked private OEM supply-chain graph \
information. Rewrite it using ONLY public incident facts, with no sole-source, redundancy, \
bottleneck, criticality, downstream reach, Honda dependency, vehicle program, or node/part \
identifier terms."""

REGEN_USER = """Public incident:
{incident_text}

Rejected query (leaked: {violations}): {query}

Rewrite a single clean query targeting the same information need."""


@dataclass
class Agent4Result:
    incident_id: str
    gaps: list[dict] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)
    retrievals: list[dict] = field(default_factory=list)
    candidate_decisions: list[dict] = field(default_factory=list)
    accepted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    privacy_leaks: list[dict] = field(default_factory=list)
    changed: bool = False

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "gaps": self.gaps,
            "queries": self.queries,
            "retrievals": self.retrievals,
            "candidate_decisions": self.candidate_decisions,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "privacy_leaks": self.privacy_leaks,
            "changed": self.changed,
        }


def build_public_input(canonical: CanonicalIncident) -> dict:
    """Public-only sanitizer. Returns a payload with allowed keys only."""
    payload = {
        "incident_id": canonical.incident_id,
        "headline_summary": canonical.canonical_summary,
        "affected_organizations": list(canonical.supplier_names),
        "event_type": canonical.event_type or "cyber",
        "source_evidence": list(canonical.source_article_ids),
    }
    # Hard guarantee: no forbidden keys leak in.
    assert not (set(payload) & FORBIDDEN_INPUT_KEYS), "Agent 4 payload contains forbidden keys"
    assert set(payload).issubset(ALLOWED_INPUT_KEYS), "Agent 4 payload has unexpected keys"
    return payload


def _incident_text(payload: dict) -> str:
    return (
        f"organization(s): {', '.join(payload['affected_organizations'])}\n"
        f"event_type: {payload['event_type']}\n"
        f"summary: {payload['headline_summary']}"
    )


def analyze_gaps(llm, incident_text: str) -> GapAnalysis:
    system = GAP_SYSTEM.format(gap_types=", ".join(C.GAP_TYPES))
    ga = llm.structured(system, GAP_USER.format(incident_text=incident_text), GapAnalysis)
    return GapAnalysis(gaps=ga.gaps[:5])


def generate_queries(llm, incident_text: str, gaps: GapAnalysis) -> list[Query]:
    gaps_text = "\n".join(f"- {g.gap_type}: {g.reason}" for g in gaps.gaps) or "- (none identified)"
    qs = llm.structured(
        QUERY_SYSTEM, QUERY_USER.format(incident_text=incident_text, gaps=gaps_text), QuerySet
    )
    queries = qs.queries[: C.QUERIES_PER_INCIDENT]
    # Normalize query ids.
    for i, q in enumerate(queries, start=1):
        q.query_id = f"q{i}"
    return queries


def run_agent4(
    llm,
    canonical: CanonicalIncident,
    backend: SearchBackend,
) -> tuple[CanonicalIncident, Agent4Result]:
    result = Agent4Result(incident_id=canonical.incident_id)
    payload = build_public_input(canonical)
    incident_text = _incident_text(payload)
    public_text = incident_text  # allowed-facts reference for the privacy validator

    # Stage A: gap analysis
    gaps = analyze_gaps(llm, incident_text)
    result.gaps = [g.model_dump() for g in gaps.gaps]

    # Stage B: exactly 5 queries
    queries = generate_queries(llm, incident_text, gaps)

    # Stage C: privacy validation + top-1 retrieval per query
    seen_docs: dict[str, dict] = {}
    for q in queries:
        clean, violations = validate_query(q.query, public_text)
        final_query = q.query
        dropped = False
        if not clean:
            # regenerate once
            regen = llm.structured(
                REGEN_SYSTEM,
                REGEN_USER.format(
                    incident_text=incident_text, violations=violations, query=q.query
                ),
                SingleQuery,
            )
            clean2, violations2 = validate_query(regen.query, public_text)
            if clean2:
                final_query = regen.query
                result.privacy_leaks.append(
                    {"query_id": q.query_id, "stage": "recovered", "violations": violations}
                )
            else:
                dropped = True
                result.privacy_leaks.append(
                    {"query_id": q.query_id, "stage": "dropped", "violations": violations2}
                )

        qrec = {
            "query_id": q.query_id,
            "gap_types": q.gap_types,
            "query": final_query,
            "privacy_clean": clean or (not dropped),
            "dropped": dropped,
        }
        result.queries.append(qrec)
        if dropped:
            continue

        hits = backend.search(final_query, top_k=C.RETRIEVAL_TOP_K)
        for h in hits:
            result.retrievals.append(
                {
                    "query_id": q.query_id,
                    "query": final_query,
                    "retrieved_doc_id": h.doc_id,
                    "retrieval_score": h.score,
                    "retrieved_rank": h.rank,
                }
            )
            if h.doc_id not in seen_docs:
                seen_docs[h.doc_id] = {"doc_id": h.doc_id, "headline": h.headline, "body": h.body}

    # Stage D: relevance + incremental value on unique candidates
    for doc_id, doc in seen_docs.items():
        same = rv.check_same_incident(llm, incident_text, doc["headline"], doc["body"])
        decision = {
            "doc_id": doc_id,
            "same_incident": same.same_incident,
            "same_confidence": same.confidence,
            "same_reason": same.reason,
            "adds_new_information": False,
            "new_evidence": [],
            "accepted": False,
        }
        if same.same_incident:
            incr = rv.check_incremental(llm, incident_text, doc["headline"], doc["body"])
            decision["adds_new_information"] = incr.adds_new_information
            decision["new_evidence"] = [e.model_dump() for e in incr.new_evidence]
            if incr.adds_new_information and incr.new_evidence:
                decision["accepted"] = True
        result.candidate_decisions.append(decision)
        if decision["accepted"]:
            result.accepted.append({"doc_id": doc_id, "new_evidence": decision["new_evidence"]})
        else:
            result.rejected.append({"doc_id": doc_id, "reason": _reject_reason(decision)})

    # Stage E: build enriched incident (or return original unchanged)
    enriched, changed = build_enriched_incident(canonical, result.accepted)
    result.changed = changed
    return enriched, result


def _reject_reason(decision: dict) -> str:
    if not decision["same_incident"]:
        return "unrelated"
    if not decision["adds_new_information"]:
        return "same_incident_no_new_information"
    return "no_new_evidence_items"
