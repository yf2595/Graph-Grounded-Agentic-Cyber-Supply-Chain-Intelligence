"""Embedding service for merge candidates and GraphRAG.

Supports OpenAI embeddings (hosted runs) and a deterministic local hasher
for vLLM experiments that must not call api.openai.com.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Literal

from openai import OpenAI


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _local_hash_embed(text: str, dim: int = 384) -> list[float]:
    """Deterministic bag-of-tokens embedding (no external API).

    Good enough for merge-candidate / GraphRAG ranking in local evals without
    OpenAI. Not interchangeable with text-embedding-3-small scientifically —
    document that local runs use this when WAITI_LOCAL_EMBEDDINGS=1.
    """
    vec = [0.0] * dim
    toks = _tokenize(text)
    if not toks:
        return vec
    for tok in toks:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        vec[idx] += sign
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class EmbeddingService:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        mode: Literal["auto", "openai", "local"] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        local_dim: int = 384,
    ):
        env_mode = (os.environ.get("WAITI_LOCAL_EMBEDDINGS") or "").strip().lower()
        if mode is None or mode == "auto":
            if env_mode in ("1", "true", "yes", "local"):
                mode = "local"
            else:
                mode = "openai"
        self.mode = mode
        self.model = model
        self.local_dim = local_dim
        self._cache: dict[str, list[float]] = {}
        self._client: OpenAI | None = None
        if self.mode == "openai":
            kwargs: dict = {}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key is not None:
                kwargs["api_key"] = api_key
            self._client = OpenAI(**kwargs) if kwargs else OpenAI()

    def embed(self, text: str) -> list[float]:
        key = text.strip()[:4000]
        if key in self._cache:
            return self._cache[key]
        if self.mode == "local":
            vec = _local_hash_embed(key, dim=self.local_dim)
        else:
            assert self._client is not None
            resp = self._client.embeddings.create(model=self.model, input=key)
            vec = list(resp.data[0].embedding)
        self._cache[key] = vec
        return vec

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
