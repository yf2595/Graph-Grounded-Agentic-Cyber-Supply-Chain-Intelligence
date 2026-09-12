"""Agent 4 - Stage E: build the enriched incident (non-destructive).

The original canonical summary is preserved; accepted supplemental evidence is
appended in a clearly delimited block with provenance so Agent 5 consumes the
same CanonicalIncident schema unchanged.
"""
from __future__ import annotations

from src.schemas.models import CanonicalIncident

ENRICHMENT_HEADER = "Supplemental evidence (retrieved via privacy-aware query expansion):"


def build_enriched_incident(
    original: CanonicalIncident, accepted: list[dict]
) -> tuple[CanonicalIncident, bool]:
    """Return (enriched_incident, changed).

    ``accepted`` items: {"doc_id", "new_evidence": [{"type","fact"}, ...]}.
    If nothing is accepted, returns a semantically identical incident (changed=False).
    """
    facts: list[str] = []
    for cand in accepted:
        doc_id = cand.get("doc_id", "?")
        for ev in cand.get("new_evidence", []):
            etype = ev.get("type", "evidence")
            fact = ev.get("fact", "").strip()
            if fact:
                facts.append(f"- [{etype}] {fact} (source: {doc_id})")

    if not facts:
        # Return a copy so callers never mutate the original object.
        return original.model_copy(deep=True), False

    appended = original.canonical_summary.rstrip() + "\n\n" + ENRICHMENT_HEADER + "\n" + "\n".join(facts)
    enriched = original.model_copy(update={"canonical_summary": appended})
    return enriched, True
