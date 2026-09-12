"""Generic search abstraction so Agent 4 is independent of the test corpus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SearchResult:
    doc_id: str
    score: float
    headline: str
    body: str
    rank: int = 1


@runtime_checkable
class SearchBackend(Protocol):
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        ...


class TavilySearchBackend:
    """Adapter boundary for a future public-Web backend.

    Intentionally NOT called in the controlled closed-corpus experiment so the
    paper claim about evaluating query planning independently of Web
    availability stays accurate.
    """

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "TavilySearchBackend is a stub; the closed-corpus experiment does not call Tavily."
        )

    def search(self, query: str, top_k: int) -> list[SearchResult]:  # pragma: no cover - stub
        raise NotImplementedError
