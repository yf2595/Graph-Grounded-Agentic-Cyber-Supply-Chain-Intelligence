"""Agent 03 — incident merger with embedding candidates and veto gates."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.schemas.models import (
    Agent02Output,
    Agent03LLMDecision,
    CanonicalIncident,
    MergeDecision,
)
from src.services.embedding_service import EmbeddingService
from src.services.incident_repository import IncidentRepository
from src.services.llm_service import LLMService

DEFAULT_MERGE_CFG = {
    "max_candidates": 8,
    "enable_embedding_candidates": True,
    "embedding_min_for_candidate": 0.82,
    "embedding_veto_merge_below": 0.78,
    "embedding_text_only_merge_min": 0.90,
    "embedding_skip_llm_below": 0.72,
}

SYSTEM = """You are Agent 03, the Incident Merger for OEM cyber supply-chain monitoring.

Decide if the NEW report describes the SAME real-world cyber incident as one candidate,
or a DISTINCT incident that must remain separate (CREATE_NEW).

MERGE_EXISTING only when evidence strongly supports the same event:
- same primary supplier (or clear org identity), AND
- same attack/event type, AND
- compatible timeframe/location/operational story (not merely the same company name).

CREATE_NEW (preferred when uncertain) when:
- same supplier but different attack waves, dates, sites, or effects
- only weak topical similarity (both ransomware) without shared incident specifics
- candidates are about the same company but clearly different campaigns/outages
- you are not highly confident they are the same event

If MERGE_EXISTING, set merge_into_incident_id to exactly one candidate incident_id.
If CREATE_NEW, merge_into_incident_id must be null.
Be conservative: false merges are worse than false splits."""


def load_merge_config(path: str | Path | None = None) -> dict:
    cfg = dict(DEFAULT_MERGE_CFG)
    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg.update(raw.get("merge") or raw)
    return cfg


def run_agent_03(
    llm: LLMService,
    relevance: Agent02Output,
    repo: IncidentRepository,
    embeddings: EmbeddingService | None = None,
    merge_config: dict | None = None,
    merge_config_path: str | Path | None = None,
) -> CanonicalIncident:
    cfg = {**DEFAULT_MERGE_CFG, **(merge_config or {})}
    if merge_config_path:
        cfg = load_merge_config(merge_config_path)

    supplier_ids = [s.supplier_node_id for s in relevance.relevant_suppliers]
    supplier_names = [s.canonical_name for s in relevance.relevant_suppliers]
    new_summary = relevance.incident_summary or ""
    want = set(supplier_ids)

    candidates = _collect_candidates(
        repo=repo,
        supplier_ids=supplier_ids,
        new_summary=new_summary,
        embeddings=embeddings,
        cfg=cfg,
    )

    if not candidates:
        return _create_new(
            repo,
            relevance.article_id,
            new_summary,
            supplier_ids,
            supplier_names,
            "No merge candidates (no supplier overlap / embedding matches).",
        )

    # If no supplier overlap and best embedding sim is weak, skip LLM
    best_sim = max(float(c.get("embedding_similarity") or 0.0) for c in candidates)
    any_overlap = any(c.get("overlap_supplier_ids") for c in candidates)
    if (
        not any_overlap
        and best_sim < float(cfg["embedding_skip_llm_below"])
    ):
        return _create_new(
            repo,
            relevance.article_id,
            new_summary,
            supplier_ids,
            supplier_names,
            f"Skipped merge LLM: no supplier overlap and best embedding sim={best_sim:.3f}.",
        )

    user = (
        f"NEW article_id: {relevance.article_id}\n"
        f"summary: {new_summary}\n"
        f"suppliers: {supplier_names} ({supplier_ids})\n\n"
        f"CANDIDATES (overlap_supplier_ids + embedding_similarity):\n{candidates}\n"
    )
    decision = llm.structured(SYSTEM, user, Agent03LLMDecision)

    if (
        decision.decision == MergeDecision.MERGE_EXISTING
        and decision.merge_into_incident_id
        and repo.get_event(decision.merge_into_incident_id)
    ):
        target_id = decision.merge_into_incident_id
        target = next((c for c in candidates if c["incident_id"] == target_id), None)
        sim = float((target or {}).get("embedding_similarity") or 0.0)
        overlap = list((target or {}).get("overlap_supplier_ids") or [])
        # If target missing from candidate list, recompute overlap/sim
        if target is None:
            ev = repo.get_event(target_id) or {}
            overlap = sorted(want & set(ev.get("supplier_node_ids") or []))
            if embeddings and (ev.get("canonical_summary") or new_summary):
                sim = embeddings.cosine(
                    embeddings.embed(new_summary),
                    embeddings.embed(ev.get("canonical_summary") or ""),
                )

        veto_below = float(cfg["embedding_veto_merge_below"])
        text_only_min = float(cfg["embedding_text_only_merge_min"])

        if sim < veto_below:
            return _create_new(
                repo,
                relevance.article_id,
                new_summary,
                supplier_ids,
                supplier_names,
                (
                    f"Vetoed MERGE into {target_id}: embedding_similarity={sim:.3f} "
                    f"< {veto_below}. LLM rationale: {decision.rationale}"
                ),
            )
        if not overlap and sim < text_only_min:
            return _create_new(
                repo,
                relevance.article_id,
                new_summary,
                supplier_ids,
                supplier_names,
                (
                    f"Vetoed text-only MERGE into {target_id}: sim={sim:.3f} "
                    f"< {text_only_min}. LLM rationale: {decision.rationale}"
                ),
            )

        ev = repo.get_event(target_id)
        assert ev is not None
        aids = list(dict.fromkeys(ev.get("source_article_ids", []) + [relevance.article_id]))
        summary = ev.get("canonical_summary") or ""
        if new_summary and len(new_summary) > len(summary):
            summary = new_summary
        sids = list(dict.fromkeys(ev.get("supplier_node_ids", []) + supplier_ids))
        snames = list(dict.fromkeys(ev.get("supplier_names", []) + supplier_names))
        can = CanonicalIncident(
            incident_id=ev["incident_id"],
            source_article_ids=aids,
            canonical_summary=summary,
            event_type=ev.get("event_type") or "cyber",
            supplier_node_ids=sids,
            supplier_names=snames,
            merge_decision=MergeDecision.MERGE_EXISTING,
            merge_rationale=(
                f"{decision.rationale} | sim={sim:.3f} overlap={overlap}"
            ),
        )
        repo.upsert_event(can.model_dump(mode="json"))
        return can

    return _create_new(
        repo,
        relevance.article_id,
        new_summary,
        supplier_ids,
        supplier_names,
        decision.rationale,
    )


def _collect_candidates(
    *,
    repo: IncidentRepository,
    supplier_ids: list[str],
    new_summary: str,
    embeddings: EmbeddingService | None,
    cfg: dict,
) -> list[dict[str, Any]]:
    want = set(supplier_ids)
    limit = int(cfg["max_candidates"])
    by_id: dict[str, dict[str, Any]] = {}

    for ev in repo.find_merge_candidates(supplier_ids, limit=limit * 2):
        eid = ev["incident_id"]
        by_id[eid] = {
            "incident_id": eid,
            "canonical_summary": ev.get("canonical_summary"),
            "source_article_ids": ev.get("source_article_ids", []),
            "supplier_node_ids": ev.get("supplier_node_ids", []),
            "supplier_names": ev.get("supplier_names", []),
            "overlap_supplier_ids": ev.get("overlap_supplier_ids") or [],
            "embedding_similarity": 0.0,
            "retrieval": "supplier_overlap",
        }

    if embeddings and new_summary.strip():
        q = embeddings.embed(new_summary)
        min_sim = float(cfg["embedding_min_for_candidate"])
        enable_extra = bool(cfg.get("enable_embedding_candidates", True))
        for ev in repo.all_events():
            eid = ev["incident_id"]
            summary = ev.get("canonical_summary") or ""
            sim = embeddings.cosine(q, embeddings.embed(summary)) if summary else 0.0
            overlap = sorted(want & set(ev.get("supplier_node_ids") or []))
            if eid in by_id:
                by_id[eid]["embedding_similarity"] = sim
                continue
            if enable_extra and sim >= min_sim:
                by_id[eid] = {
                    "incident_id": eid,
                    "canonical_summary": summary,
                    "source_article_ids": ev.get("source_article_ids", []),
                    "supplier_node_ids": ev.get("supplier_node_ids", []),
                    "supplier_names": ev.get("supplier_names", []),
                    "overlap_supplier_ids": overlap,
                    "embedding_similarity": sim,
                    "retrieval": "embedding",
                }

    ranked = sorted(
        by_id.values(),
        key=lambda c: (
            -len(c.get("overlap_supplier_ids") or []),
            -float(c.get("embedding_similarity") or 0.0),
        ),
    )
    return ranked[:limit]


def _create_new(
    repo: IncidentRepository,
    article_id: str,
    summary: str,
    supplier_ids: list[str],
    supplier_names: list[str],
    rationale: str,
) -> CanonicalIncident:
    eid = repo.next_event_id()
    can = CanonicalIncident(
        incident_id=eid,
        source_article_ids=[article_id],
        canonical_summary=summary,
        event_type="cyber",
        supplier_node_ids=supplier_ids,
        supplier_names=supplier_names,
        merge_decision=MergeDecision.CREATE_NEW,
        merge_rationale=rationale,
    )
    repo.upsert_event(can.model_dump(mode="json"))
    return can
