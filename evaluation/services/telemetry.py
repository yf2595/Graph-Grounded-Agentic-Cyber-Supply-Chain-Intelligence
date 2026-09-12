"""Telemetry counters for eval methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Telemetry:
    llm_calls: int = 0
    graph_queries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    started_at: str = ""
    completed_at: str = ""

    def reset(self) -> None:
        self.llm_calls = 0
        self.graph_queries = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.started_at = ""
        self.completed_at = ""

    def start(self) -> None:
        self.reset()
        self.started_at = datetime.now(timezone.utc).isoformat()

    def stop(self) -> float:
        self.completed_at = datetime.now(timezone.utc).isoformat()
        try:
            t0 = datetime.fromisoformat(self.started_at)
            t1 = datetime.fromisoformat(self.completed_at)
            return (t1 - t0).total_seconds() * 1000.0
        except Exception:
            return 0.0

    def absorb_llm(self, usage) -> None:
        """Copy counters from LLMService.usage."""
        self.llm_calls += int(getattr(usage, "llm_calls", 0) or 0)
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
