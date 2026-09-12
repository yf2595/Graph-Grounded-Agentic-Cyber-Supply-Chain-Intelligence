"""Structured per-step logging for pipeline / method runs."""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class StepLogger:
    """Append-only JSONL logger for agent/method steps (thread-safe)."""

    def __init__(self, log_path: str | Path | None = None):
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()
        self.steps: list[dict[str, Any]] = []

    def bind(self, log_path: str | Path) -> "StepLogger":
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, event: dict[str, Any]) -> None:
        row = dict(event)
        row.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with self._lock:
            self.steps.append(row)
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    @contextmanager
    def step(
        self,
        *,
        article_id: str,
        step: str,
        method_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Context manager that records start/end, latency, and optional extras."""
        meta: dict[str, Any] = dict(extra or {})
        t0 = time.perf_counter()
        started = datetime.now(timezone.utc).isoformat()
        status = "ok"
        error: str | None = None
        try:
            yield meta
        except Exception as e:
            status = "error"
            error = str(e)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            payload = {
                "article_id": article_id,
                "method_id": method_id,
                "step": step,
                "status": status,
                "error": error,
                "started_at": started,
                "latency_ms": round(elapsed_ms, 2),
                "llm_calls": meta.get("llm_calls"),
                "input_tokens": meta.get("input_tokens"),
                "output_tokens": meta.get("output_tokens"),
                "graph_queries": meta.get("graph_queries"),
                "summary": meta.get("summary"),
            }
            self.write(payload)
