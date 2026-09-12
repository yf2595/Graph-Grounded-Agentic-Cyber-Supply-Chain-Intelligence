"""Phase 6 - independent enriched reference labels for the 20 cases (Section 19).

Uses GPT-5.4 reasoning=medium as an independent-effort labeler (weak independence:
same family, different effort/role than the reasoning=low evaluation pipeline).
Given the original incident + BOTH true supplemental reports + frozen structural
graph facts, it produces an updated consequence/risk reference. GPT-5.4 pipeline
outputs are never shown to the labeler.
"""
from __future__ import annotations

import csv
import json

from src.query_expansion import config as C
from src.query_expansion.schemas import EnrichedReferenceLabel
from src.query_expansion.selection import load_selected

SYSTEM = """You are an independent risk-reference labeler for automotive cyber \
supply-chain incidents. Using a fixed 5-level ordinal scale \
(VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH), assess the OEM supply-chain consequence/risk.

You are given: the original incident, two additional true reports about the SAME incident \
that add new evidence, and frozen structural supply-chain facts. Decide whether the new \
evidence changes the correct risk relative to the original label.

- enriched_reference_risk: the correct risk given ALL evidence (original + supplements).
- expected_direction: increase | decrease | same (relative to the original label).
- Base your judgment only on the evidence and structural facts. Do not invent facts.
"""

USER = """Original incident:
headline: {headline}
body: {body}

Established original reference risk (frozen ground truth): {original_risk}

Frozen structural supply-chain facts:
{graph_facts}

Additional true report 1:
headline: {s1_head}
body: {s1_body}

Additional true report 2:
headline: {s2_head}
body: {s2_body}

Provide original_reference_risk (echo the established original risk), \
enriched_reference_risk, expected_direction, and reference_rationale."""


def _incidents_metrics() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with C.INCIDENTS_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["incident_id"]] = row
    return rows


def _graph_facts(row: dict) -> str:
    keys = [
        "guaranteed_reach",
        "total_reach",
        "sole_source_product_count",
        "is_bottleneck",
        "is_sole_source_critical_part",
        "is_multi_supplier",
        "impact_level",
    ]
    return "\n".join(f"- {k}: {row.get(k, '')}" for k in keys)


def _supplements_by_target() -> dict[str, list[dict]]:
    docs = {
        json.loads(l)["doc_id"]: json.loads(l)
        for l in C.DOCUMENTS_JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    gts = [
        json.loads(l)
        for l in C.GROUND_TRUTH_OUT_JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    out: dict[str, list[dict]] = {}
    for g in gts:
        if g.get("is_noise") or not g.get("target_incident_id"):
            continue
        doc = docs[g["doc_id"]]
        out.setdefault(g["target_incident_id"], []).append(doc)
    return out


def run() -> list[dict]:
    C.ensure_artifacts_dir()
    selected = load_selected()
    metrics = _incidents_metrics()
    supps = _supplements_by_target()
    llm = C.make_llm(C.REFERENCE_MODEL, C.REFERENCE_REASONING_EFFORT)

    rows: list[dict] = []
    for inc in selected:
        pair = supps.get(inc.incident_id, [])
        if len(pair) < 2:
            raise RuntimeError(f"Expected 2 supplements for {inc.incident_id}, found {len(pair)}")
        s1, s2 = pair[0], pair[1]
        user = USER.format(
            headline=inc.headline,
            body=inc.body,
            original_risk=inc.gt_risk or "UNKNOWN",
            graph_facts=_graph_facts(metrics.get(inc.incident_id, {})),
            s1_head=s1["headline"],
            s1_body=s1["body"],
            s2_head=s2["headline"],
            s2_body=s2["body"],
        )
        label = llm.structured(SYSTEM, user, EnrichedReferenceLabel)
        rows.append(
            {
                "incident_id": inc.incident_id,
                "original_reference_risk": inc.gt_risk or "UNKNOWN",  # authoritative frozen label
                "enriched_reference_risk": label.enriched_reference_risk,
                "expected_direction": label.expected_direction,
                "reference_rationale": label.reference_rationale,
                "labeler_model": C.REFERENCE_MODEL,
                "labeler_reasoning_effort": C.REFERENCE_REASONING_EFFORT,
            }
        )
        print(f"  {inc.incident_id}: {inc.gt_risk} -> {label.enriched_reference_risk} ({label.expected_direction})")

    _write = C.ENRICHED_REFERENCE_LABELS_JSONL
    with _write.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} enriched reference labels -> {_write}")
    return rows


if __name__ == "__main__":
    run()
