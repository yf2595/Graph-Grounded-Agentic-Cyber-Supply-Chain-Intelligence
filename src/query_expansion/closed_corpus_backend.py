"""Closed-corpus retrieval backend over the 60-document local index."""
from __future__ import annotations

import json

from src.query_expansion import config as C
from src.query_expansion.search_backend import SearchResult
from src.services.embedding_service import EmbeddingService


class ClosedCorpusSearchBackend:
    """Cosine retrieval over the frozen 60-document index.

    Query embeddings use the same OpenAI text-embedding-3-small model that built
    the index, so query/document vectors live in the same semantic space.
    """

    def __init__(self, index_dir=None):
        index_dir = index_dir or C.INDEX_DIR
        self.vectors: dict[str, list[float]] = json.loads(
            (index_dir / "vectors.json").read_text(encoding="utf-8")
        )
        self.meta: dict[str, dict] = json.loads(
            (index_dir / "doc_meta.json").read_text(encoding="utf-8")
        )
        self.emb = EmbeddingService(mode=C.EMBEDDING_MODE, model=C.EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 1) -> list[SearchResult]:
        qv = self.emb.embed(query)
        scored = [
            (doc_id, EmbeddingService.cosine(qv, vec))
            for doc_id, vec in self.vectors.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[SearchResult] = []
        for rank, (doc_id, score) in enumerate(scored[:top_k], start=1):
            m = self.meta.get(doc_id, {})
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=float(score),
                    headline=m.get("headline", ""),
                    body=m.get("body", ""),
                    rank=rank,
                )
            )
        return results
