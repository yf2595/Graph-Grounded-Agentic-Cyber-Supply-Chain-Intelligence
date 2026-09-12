"""Auto-label report claims for faithfulness (WAITI P4)."""
from __future__ import annotations

import re
from typing import Any

from src.agents.agent_08_report_generator import ReportClaim

LABELS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "UNVERIFIABLE")

_CONTRA_PATTERNS = [
    (re.compile(r"\bhalt(ed|ing)?\b", re.I), re.compile(r"\bno\s+(production\s+)?impact\b", re.I)),
    (re.compile(r"\bransomware\b", re.I), re.compile(r"\bnot\s+cyber\b", re.I)),
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]{3,}", _norm(s))}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def build_evidence_corpus(
    *,
    article_text: str,
    result: dict[str, Any] | None = None,
    agent05: dict[str, Any] | None = None,
    agent06: dict[str, Any] | None = None,
    agent07: dict[str, Any] | None = None,
    deterministic_fields: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Map evidence IDs / buckets to text snippets."""
    corpus: dict[str, str] = {"ARTICLE:source": article_text or ""}
    if result:
        ci = result.get("canonical_incident") or {}
        corpus["INCIDENT:summary"] = str(ci.get("canonical_summary") or result.get("canonical_summary") or "")
        corpus["INCIDENT:event_type"] = str(ci.get("event_type") or "")
        for sid in ci.get("supplier_node_ids") or result.get("supplier_node_ids") or []:
            corpus[f"INCIDENT:supplier:{sid}"] = str(sid)
        for a in ci.get("source_article_ids") or []:
            corpus[f"ARTICLE:{a}"] = article_text or ""
    if agent05:
        corpus["AGENT5:main_effect"] = str(agent05.get("main_effect") or "")
        corpus["AGENT5:severity"] = str(agent05.get("severity") or "")
        corpus["AGENT5:rationale"] = str(agent05.get("assessment_rationale") or "")
    if agent06:
        corpus["AGENT6:oem_exposure"] = str(agent06.get("oem_exposure") or "")
        corpus["AGENT6:exposure_rationale"] = str(agent06.get("exposure_rationale") or "")
        corpus["AGENT6:propagation_scope"] = str(agent06.get("propagation_scope") or "")
        for sp in agent06.get("affected_suppliers") or []:
            sid = sp.get("supplier_node_id")
            for part in sp.get("supplied_parts") or []:
                eid = (
                    f"GRAPH:{sid}->PART:{part.get('part_id')}:"
                    f"{part.get('criticality')}:{part.get('redundancy')}"
                )
                corpus[eid] = (
                    f"{part.get('part_name')} {part.get('part_id')} "
                    f"{part.get('criticality')} {part.get('redundancy')}"
                )
            for prod in sp.get("affected_oem_products") or []:
                eid = f"GRAPH:{sid}->PRODUCT:{prod.get('product_id')}"
                corpus[eid] = f"{prod.get('product_name')} {prod.get('product_id')}"
    if agent07:
        corpus["AGENT7:overall_risk"] = str(agent07.get("overall_risk") or "")
        corpus["AGENT7:risk_rationale"] = str(agent07.get("risk_rationale") or "")
        corpus["AGENT7:key_risk_drivers"] = " ".join(agent07.get("key_risk_drivers") or [])
        corpus["AGENT7:mitigating_factors"] = " ".join(agent07.get("mitigating_factors") or [])
    if deterministic_fields:
        for gid in deterministic_fields.get("graph_fact_ids") or []:
            corpus.setdefault(gid, gid)
    return corpus


def label_claim(claim: ReportClaim | dict, corpus: dict[str, str]) -> str:
    if isinstance(claim, dict):
        text = claim.get("text") or ""
        evid = claim.get("supporting_evidence_ids") or []
        ctype = claim.get("claim_type") or ""
    else:
        text = claim.text
        evid = claim.supporting_evidence_ids
        ctype = claim.claim_type

    snippets = []
    for eid in evid:
        if eid in corpus and corpus[eid]:
            snippets.append(corpus[eid])
        # prefix match for ARTICLE:INC_xxx
        for k, v in corpus.items():
            if k.startswith(eid) or eid.startswith(k):
                snippets.append(v)
    joined = "\n".join(snippets) if snippets else "\n".join(corpus.values())

    nt, nj = _norm(text), _norm(joined)
    if not nt:
        return "UNVERIFIABLE"

    # Direct support: key tokens from claim appear in cited evidence
    ov = _overlap(text, joined)
    # Structured facts with evidence IDs that exist in corpus → supported if overlap or ID present
    if ctype in {"GRAPH_FACT", "INCIDENT_FACT", "RISK_FACT"} and evid:
        present = any(e in corpus or any(k.startswith(e) or e in k for k in corpus) for e in evid)
        if present and ov >= 0.25:
            return "SUPPORTED"
        if present and any(tok in nj for tok in _tokens(text) if len(tok) > 4):
            return "SUPPORTED"
        if present and ctype == "GRAPH_FACT":
            # graph claim constructed from IDs — treat as supported if evidence id exists
            if any(e in corpus for e in evid):
                return "SUPPORTED"
        if not present and ctype == "GRAPH_FACT":
            return "UNSUPPORTED"

    if ov >= 0.35:
        return "SUPPORTED"
    if ov >= 0.2:
        return "SUPPORTED"

    # Contradiction heuristics vs article
    article = corpus.get("ARTICLE:source") or ""
    for a_pat, b_pat in _CONTRA_PATTERNS:
        if a_pat.search(text) and b_pat.search(article):
            return "CONTRADICTED"
        if b_pat.search(text) and a_pat.search(article):
            return "CONTRADICTED"

    # Prose without enough evidence
    if ctype == "PROSE":
        if ov < 0.12:
            return "UNVERIFIABLE"
        return "UNSUPPORTED"

    if not evid:
        return "UNSUPPORTED"
    if ov < 0.1:
        return "UNSUPPORTED"
    return "UNVERIFIABLE"


def label_claims(claims: list[ReportClaim] | list[dict], corpus: dict[str, str]) -> list[dict]:
    out = []
    for c in claims:
        if isinstance(c, ReportClaim):
            row = c.model_dump()
        else:
            row = dict(c)
        row["faithfulness_label"] = label_claim(c, corpus)
        out.append(row)
    return out
