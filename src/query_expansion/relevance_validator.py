"""Agent 4 - Stage D: same-incident relevance + incremental-value checks.

Two independent LLM checks per candidate (Section 15). Ground-truth metadata is
never shown to these checks.
"""
from __future__ import annotations

from src.query_expansion.schemas import IncrementalDecision, SameIncidentDecision

SAME_INCIDENT_SYSTEM = """You judge whether a retrieved report plausibly describes the \
SAME underlying real-world cyber incident as a reference incident.

Consider: affected organization, event/attack type, temporal compatibility, \
facility/operation compatibility, and the technical and operational narrative. A report \
about a different organization, a different attack, or an incompatible timeframe is NOT the \
same incident. Only strong, specific alignment counts as the same incident."""

SAME_INCIDENT_USER = """Reference incident (public):
{incident_text}

Candidate retrieved report:
headline: {headline}
body: {body}

Is the candidate the same underlying incident? Give same_incident (bool), confidence \
(0.0-1.0), and a short reason."""

INCREMENTAL_SYSTEM = """You judge whether a candidate report adds NEW, useful information \
beyond what the reference incident already states. Paraphrases or restatements of existing \
facts do NOT count as new information. Only genuinely new material facts count (e.g. a newly \
named facility, a recovery/containment status, an attack vector, a confirmed data/system \
impact, a duration)."""

INCREMENTAL_USER = """Reference incident (public):
{incident_text}

Candidate retrieved report (already judged to be the same incident):
headline: {headline}
body: {body}

Does the candidate add new information not present in the reference? Give \
adds_new_information (bool) and a list of new_evidence items (type + fact) for each genuinely \
new fact."""


def check_same_incident(llm, incident_text: str, headline: str, body: str) -> SameIncidentDecision:
    return llm.structured(
        SAME_INCIDENT_SYSTEM,
        SAME_INCIDENT_USER.format(incident_text=incident_text, headline=headline, body=body),
        SameIncidentDecision,
    )


def check_incremental(llm, incident_text: str, headline: str, body: str) -> IncrementalDecision:
    return llm.structured(
        INCREMENTAL_SYSTEM,
        INCREMENTAL_USER.format(incident_text=incident_text, headline=headline, body=body),
        IncrementalDecision,
    )
