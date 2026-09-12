"""Shared merge-candidate retrieval (never uses group_id)."""
from __future__ import annotations

from dataclasses import dataclass

from evaluation.services.embedding_service import EmbeddingService
from src.services.incident_repository import IncidentRepository


@dataclass
class CanonicalIncidentCandidate:
    incident_id: str
    canonical_summary: str
    source_article_ids: list[str]
    supplier_node_ids: list[str]
    supplier_names: list[str]
    overlap_supplier_ids: list[str]
    embedding_similarity: float = 0.0


class MergeCandidateService:
    def __init__(
        self,
        repo: IncidentRepository,
        embeddings: EmbeddingService | None = None,
        max_candidates: int = 10,
    ):
        self.repo = repo
        self.embeddings = embeddings or EmbeddingService()
        self.max_candidates = max_candidates

    def get_candidates(
        self,
        article_id: str,
        supplier_node_ids: list[str],
        text: str,
        max_candidates: int | None = None,
    ) -> list[CanonicalIncidentCandidate]:
        limit = max_candidates or self.max_candidates
        base = self.repo.find_merge_candidates(supplier_node_ids, limit=limit * 2)
        if not base:
            # also consider all events with embedding only if suppliers empty? skip
            return []

        query_vec = self.embeddings.embed(text)
        out: list[CanonicalIncidentCandidate] = []
        for ev in base:
            summary = ev.get("canonical_summary") or ""
            sim = self.embeddings.cosine(query_vec, self.embeddings.embed(summary)) if summary else 0.0
            out.append(
                CanonicalIncidentCandidate(
                    incident_id=ev["incident_id"],
                    canonical_summary=summary,
                    source_article_ids=ev.get("source_article_ids") or [],
                    supplier_node_ids=ev.get("supplier_node_ids") or [],
                    supplier_names=ev.get("supplier_names") or [],
                    overlap_supplier_ids=ev.get("overlap_supplier_ids") or [],
                    embedding_similarity=sim,
                )
            )
        out.sort(key=lambda c: (-len(c.overlap_supplier_ids), -c.embedding_similarity))
        return out[:limit]
