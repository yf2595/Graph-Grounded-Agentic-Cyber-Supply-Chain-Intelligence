"""Persist stage outputs and canonical incidents for a run (thread-safe)."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class IncidentRepository:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.articles_dir = self.run_dir / "articles"
        self.events_dir = self.run_dir / "events"
        self.articles_dir.mkdir(exist_ok=True)
        self.events_dir.mkdir(exist_ok=True)
        self._article_to_event: dict[str, str] = {}
        self._events: dict[str, dict] = {}
        self._event_counter = 0
        self._lock = threading.RLock()
        self._load_existing()

    def _load_existing(self) -> None:
        idx = self.run_dir / "index.json"
        if idx.exists():
            data = json.loads(idx.read_text(encoding="utf-8"))
            self._article_to_event = data.get("article_to_event", {})
            self._event_counter = data.get("event_counter", 0)
        for p in self.events_dir.glob("EVT_*.json"):
            ev = json.loads(p.read_text(encoding="utf-8"))
            self._events[ev["incident_id"]] = ev

    def _save_index(self) -> None:
        (self.run_dir / "index.json").write_text(
            json.dumps(
                {
                    "article_to_event": self._article_to_event,
                    "event_counter": self._event_counter,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def has_article(self, article_id: str) -> bool:
        return (self.articles_dir / article_id / "result.json").exists()

    def save_stage(self, article_id: str, stage: str, payload: Any) -> None:
        d = self.articles_dir / article_id
        d.mkdir(parents=True, exist_ok=True)
        if hasattr(payload, "model_dump"):
            data = payload.model_dump(mode="json")
        else:
            data = payload
        with self._lock:
            (d / f"{stage}.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    def save_result(self, article_id: str, payload: Any) -> None:
        self.save_stage(article_id, "result", payload)

    def next_event_id(self) -> str:
        with self._lock:
            self._event_counter += 1
            eid = f"EVT_{self._event_counter:05d}"
            self._save_index()
            return eid

    def get_event(self, incident_id: str) -> dict | None:
        with self._lock:
            return self._events.get(incident_id)

    def upsert_event(self, event: dict) -> None:
        with self._lock:
            iid = event["incident_id"]
            self._events[iid] = event
            # Keep counter ahead of any manually assigned EVT_##### ids
            if isinstance(iid, str) and iid.startswith("EVT_"):
                try:
                    n = int(iid.split("_", 1)[1])
                    if n > self._event_counter:
                        self._event_counter = n
                except ValueError:
                    pass
            (self.events_dir / f"{iid}.json").write_text(
                json.dumps(event, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            for aid in event.get("source_article_ids", []):
                self._article_to_event[aid] = iid
            self._save_index()

    def link_article(self, article_id: str, incident_id: str) -> None:
        with self._lock:
            self._article_to_event[article_id] = incident_id
            self._save_index()

    def event_for_article(self, article_id: str) -> str | None:
        with self._lock:
            return self._article_to_event.get(article_id)

    def find_merge_candidates(
        self,
        supplier_node_ids: list[str],
        limit: int = 8,
    ) -> list[dict]:
        """Return existing events that share at least one supplier."""
        want = set(supplier_node_ids)
        with self._lock:
            cands = []
            for ev in self._events.values():
                overlap = want & set(ev.get("supplier_node_ids") or [])
                if overlap:
                    cands.append(
                        {
                            "incident_id": ev["incident_id"],
                            "canonical_summary": ev.get("canonical_summary"),
                            "source_article_ids": ev.get("source_article_ids", []),
                            "supplier_node_ids": ev.get("supplier_node_ids", []),
                            "supplier_names": ev.get("supplier_names", []),
                            "overlap_supplier_ids": sorted(overlap),
                        }
                    )
            return cands[:limit]

    def all_events(self) -> list[dict]:
        with self._lock:
            return list(self._events.values())

    def all_results(self) -> list[dict]:
        out = []
        for p in sorted(self.articles_dir.glob("*/result.json")):
            out.append(json.loads(p.read_text(encoding="utf-8")))
        return out
