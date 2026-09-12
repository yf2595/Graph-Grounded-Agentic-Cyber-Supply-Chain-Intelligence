"""Phase 2 - generate two supplemental reports per selected incident.

Each supplement describes the SAME underlying cyber incident but adds new
material facts absent from the original canonical incident. Generation uses
GPT-5.4 reasoning=medium. Hidden ground-truth metadata is stored separately
from the searchable document text.

Expected-effect distribution across the 20 incidents (Section 4.4):
    8 increase, 8 decrease, 4 no_ordinal_change.
"""
from __future__ import annotations

import json
import random

from src.query_expansion import config as C
from src.query_expansion.schemas import GeneratedReport, SupplementPlanItem
from src.query_expansion.selection import IncidentCandidate, load_selected

# Report roles (Section 4.3).
ROLE_1 = "operational"   # facility / production impact / disruption scope / geography
ROLE_2 = "recovery_technical"  # recovery / containment / duration / attack vector / attribution

PLAN_SYSTEM = """You are a controlled-evaluation dataset designer for automotive cyber \
supply-chain incidents. You plan what NEW evidence two follow-up news reports about the \
SAME incident should add, without revealing hidden labels or any private OEM supply-chain \
graph attributes (never mention Honda dependency, sole-source status, redundancy, \
bottleneck status, supplier criticality, downstream reach, vehicle programs, or supplied \
part identities unless already explicitly public in the original text).

You will produce a plan for exactly two supplemental reports:
- Report 1 (operational): adds operational evidence such as affected facility, production
  impact, affected business operation, disruption scope, service interruption, geographic
  spread.
- Report 2 (recovery_technical): adds recovery status, expected restoration, containment
  status, incident duration, technical attribution, attack vector, or confirmed
  system/data impact.

The two reports MUST add different information. Both must remain factually and temporally
compatible with the original incident and mutually consistent.

The overall enrichment must be consistent with this expected downstream effect on an
incident-consequence severity assessment: {effect}
  - increase: the new evidence makes the incident look more severe / broader.
  - decrease: the new evidence shows recovery/containment that de-escalates concern.
  - no_ordinal_change: the new evidence adds specificity/completeness but should NOT
    change the ordinal severity.

Only use evidence-gap categories from this taxonomy: {gap_types}."""

PLAN_USER = """Original canonical incident (public):
headline: {headline}
body: {body}
affected organization(s): {suppliers}
expected downstream effect for this case: {effect}

Return a plan with two items (report_1 first = operational, report_2 = recovery_technical).
For each item give: evidence_types (from the taxonomy), new_facts (1-3 concrete new facts
this report will introduce), expected_agent5_effect (must equal "{effect}"), and a short
rationale. The two items must introduce different facts."""

GEN_SYSTEM = """You write a single realistic synthetic news report about an automotive \
cyber supply-chain incident, for controlled research evaluation.

Rules:
- Describe the SAME underlying incident and the SAME affected organization as the original.
- Introduce the planned NEW facts; do not merely paraphrase the original.
- Use a different reporting perspective and wording from the original.
- Stay temporally and factually compatible with the original.
- Do NOT reveal any hidden labels, expected effects, or evaluation metadata.
- Do NOT include private OEM supply-chain graph attributes (Honda dependency, sole-source,
  redundancy, bottleneck, criticality, downstream reach, vehicle programs, supplied parts)
  unless already explicitly public in the original text.
- Do NOT include disclaimers, notes to the reader, or bracketed editorial text.
- Write 90-180 words of body text plus a concise headline."""

GEN_USER = """Original canonical incident (public):
headline: {headline}
body: {body}
affected organization(s): {suppliers}

This report's role: {role}
New facts to introduce (weave in naturally): {new_facts}
Evidence types this report covers: {evidence_types}

Write the headline and body for this supplemental report."""


def assign_effects(
    selected: list[IncidentCandidate], seed: int = C.GENERATION_SEED
) -> dict[str, str]:
    """Assign 8 increase / 8 decrease / 4 no_ordinal_change deterministically."""
    n = len(selected)
    n_inc = round(n * 0.4)
    n_dec = round(n * 0.4)
    n_same = n - n_inc - n_dec
    labels = ["increase"] * n_inc + ["decrease"] * n_dec + ["no_ordinal_change"] * n_same
    rng = random.Random(seed)
    order = list(selected)
    rng.shuffle(order)
    return {c.incident_id: labels[i] for i, c in enumerate(order)}


def _plan_supplements(llm, inc: IncidentCandidate, effect: str) -> list[SupplementPlanItem]:
    from pydantic import BaseModel

    class TwoItemPlan(BaseModel):
        model_config = {"extra": "forbid"}
        report_1: SupplementPlanItem
        report_2: SupplementPlanItem

    system = PLAN_SYSTEM.format(effect=effect, gap_types=", ".join(C.GAP_TYPES))
    user = PLAN_USER.format(
        headline=inc.headline,
        body=inc.body,
        suppliers=", ".join(inc.supplier_names),
        effect=effect,
    )
    plan = llm.structured(system, user, TwoItemPlan)
    # Force the case-level effect onto both items for consistent metadata.
    item1 = plan.report_1.model_copy(update={"expected_agent5_effect": effect})
    item2 = plan.report_2.model_copy(update={"expected_agent5_effect": effect})
    return [item1, item2]


def _generate_report(llm, inc: IncidentCandidate, item: SupplementPlanItem, role: str) -> GeneratedReport:
    user = GEN_USER.format(
        headline=inc.headline,
        body=inc.body,
        suppliers=", ".join(inc.supplier_names),
        role=role,
        new_facts="; ".join(item.new_facts),
        evidence_types=", ".join(item.evidence_types),
    )
    return llm.structured(GEN_SYSTEM, user, GeneratedReport)


def _record(inc: IncidentCandidate, idx: int, report: GeneratedReport, item: SupplementPlanItem) -> dict:
    doc_id = f"QE_{inc.incident_id}_SUPP_{idx}"
    return {
        "document": {
            "doc_id": doc_id,
            "headline": report.headline.strip(),
            "body": report.body.strip(),
        },
        "metadata": {
            "doc_id": doc_id,
            "target_incident_id": inc.incident_id,
            "target_group_id": inc.group_id,
            "is_relevant_to_target": True,
            "is_noise": False,
            "role": ROLE_1 if idx == 1 else ROLE_2,
            "evidence_types": item.evidence_types,
            "new_facts": item.new_facts,
            "expected_agent5_effect": item.expected_agent5_effect,
            "rationale": item.rationale,
            "generation_model": C.GEN_MODEL,
            "generation_reasoning_effort": C.GEN_REASONING_EFFORT,
            "generation_seed": C.GENERATION_SEED,
        },
    }


def run(force: bool = False) -> list[dict]:
    C.ensure_artifacts_dir()
    if C.SUPPLEMENTS_RAW_JSONL.exists() and not force:
        return [json.loads(l) for l in C.SUPPLEMENTS_RAW_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

    selected = load_selected()
    effects = assign_effects(selected)
    llm = C.make_llm(C.GEN_MODEL, C.GEN_REASONING_EFFORT)

    records: list[dict] = []
    for inc in selected:
        effect = effects[inc.incident_id]
        items = _plan_supplements(llm, inc, effect)
        roles = [ROLE_1, ROLE_2]
        for idx, (item, role) in enumerate(zip(items, roles), start=1):
            report = _generate_report(llm, inc, item, role)
            records.append(_record(inc, idx, report, item))
            print(f"  generated {inc.incident_id} SUPP_{idx} ({effect}, {role})")

    with C.SUPPLEMENTS_RAW_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} supplements -> {C.SUPPLEMENTS_RAW_JSONL}")
    return records


if __name__ == "__main__":
    run(force=True)
