"""Deterministic query privacy/leakage validator (Section 13).

Rejects any query containing graph-derived (OEM-private) information that is not
already present in the public incident evidence. Target leakage rate: 0%.
"""
from __future__ import annotations

import re

# OEM-graph-only concepts that must never originate from Agent 4 queries.
BLOCKED_CONCEPTS = [
    "sole source",
    "sole-source",
    "single source",
    "single-source",
    "redundancy",
    "redundant supplier",
    "bottleneck",
    "criticality",
    "critical part",
    "downstream reach",
    "guaranteed reach",
    "total reach",
    "supplier tier",
    "tier-1",
    "tier 1",
    "oem dependency",
    "dependency path",
    "honda relationship",
    "honda supplier",
    "supplies honda",
    "vehicle program",
    "part id",
    "node id",
]

_ORG_ID = re.compile(r"org_[0-9a-f]{6,}", re.IGNORECASE)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def validate_query(query: str, public_text: str) -> tuple[bool, list[str]]:
    """Return (is_clean, violations).

    A concept is a violation only if it appears in the query but not in the
    public incident text (i.e. it was not already public).
    """
    q = _norm(query)
    public = _norm(public_text)
    violations: list[str] = []
    for concept in BLOCKED_CONCEPTS:
        if concept in q and concept not in public:
            violations.append(concept)
    if _ORG_ID.search(query) and not _ORG_ID.search(public_text or ""):
        violations.append("org_node_id")
    return (len(violations) == 0, violations)
