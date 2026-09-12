"""Phase 3 - build the closed 60-document retrieval index.

Uses the project's standard semantic embedding (OpenAI text-embedding-3-small),
the same model used elsewhere for merge candidates and GraphRAG. Only the
searchable text (headline + body) is indexed as a whole document; hidden
ground-truth fields are never embedded.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from src.query_expansion import config as C
from src.services.embedding_service import EmbeddingService


def _load_documents() -> list[dict]:
    return [
        json.loads(l)
        for l in C.DOCUMENTS_JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def build() -> dict:
    C.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    documents = _load_documents()
    emb = EmbeddingService(mode=C.EMBEDDING_MODE, model=C.EMBEDDING_MODEL)

    vectors: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}
    for d in documents:
        vectors[d["doc_id"]] = emb.embed(d["combined_search_text"])
        meta[d["doc_id"]] = {"headline": d["headline"], "body": d["body"]}

    (C.INDEX_DIR / "vectors.json").write_text(
        json.dumps(vectors), encoding="utf-8"
    )
    (C.INDEX_DIR / "doc_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )

    corpus_hash = hashlib.sha256(
        "".join(
            sorted(d["doc_id"] + d["combined_search_text"] for d in documents)
        ).encode("utf-8")
    ).hexdigest()[:16]

    dim = len(next(iter(vectors.values()))) if vectors else 0
    manifest = {
        "embedding_model": f"{C.EMBEDDING_MODEL} (OpenAI, WAITI EmbeddingService mode={C.EMBEDDING_MODE})",
        "embedding_revision": C.EMBEDDING_MODEL,
        "dimensionality": dim,
        "normalization": "none (cosine computed at query time)",
        "index_type": "in-memory cosine (brute force, N=60)",
        "similarity": "cosine",
        "granularity": "whole-document (headline + body)",
        "top_k_per_query": C.RETRIEVAL_TOP_K,
        "n_documents": len(documents),
        "build_time": datetime.now(timezone.utc).isoformat(),
        "corpus_hash": corpus_hash,
    }
    C.INDEX_MANIFEST_JSON.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Indexed {len(documents)} documents -> {C.INDEX_DIR}")
    return manifest


if __name__ == "__main__":
    build()
