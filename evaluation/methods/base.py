"""Evaluation method protocol / base."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.schemas.prediction import MethodPrediction, ModelInputRecord
from evaluation.services.cost import estimate_cost_usd
from evaluation.services.embedding_service import EmbeddingService
from evaluation.services.merge_candidate_service import MergeCandidateService
from evaluation.services.telemetry import Telemetry
from src.services.entity_resolution import EntityResolutionService
from src.services.graph_service import SupplyChainGraphService
from src.services.incident_repository import IncidentRepository


@dataclass
class EvaluationContext:
    run_id: str
    root: Path
    graph: SupplyChainGraphService
    resolver: EntityResolutionService
    embeddings: EmbeddingService
    risk_rules_path: Path
    cyber_rules: dict[str, Any]
    llm_model: str = "gpt-5.4"
    temperature: float = 0.0
    method_config: dict[str, Any] = field(default_factory=dict)
    pricing: dict[str, Any] = field(default_factory=dict)


class EvaluationMethod(ABC):
    method_id: str

    def __init__(self, ctx: EvaluationContext):
        self.ctx = ctx
        self.repo = IncidentRepository(ctx.root / "results" / ctx.run_id / "methods" / self.method_id)
        self.merge_svc = MergeCandidateService(self.repo, ctx.embeddings)
        self.telemetry = Telemetry()

    def reset_state(self) -> None:
        # Fresh repo directory for a clean run is created by using a new run_id;
        # for resume, keep existing. Explicit reset clears in-memory maps only.
        self.repo._article_to_event = {}
        self.repo._events = {}
        self.repo._event_counter = 0
        self.telemetry = Telemetry()

    @abstractmethod
    def process_record(self, record: ModelInputRecord) -> MethodPrediction:
        ...

    def _base_pred(self, record: ModelInputRecord) -> MethodPrediction:
        return MethodPrediction(
            run_id=self.ctx.run_id,
            method_id=self.method_id,
            article_id=record.article_id,
        )

    def _finalize_telemetry(self, pred: MethodPrediction, model: str | None = None) -> MethodPrediction:
        pred.latency_ms = self.telemetry.stop()
        pred.started_at = self.telemetry.started_at
        pred.completed_at = self.telemetry.completed_at
        pred.llm_calls = self.telemetry.llm_calls
        pred.graph_queries = self.telemetry.graph_queries
        pred.input_tokens = self.telemetry.input_tokens
        pred.output_tokens = self.telemetry.output_tokens
        pred.cost_usd = estimate_cost_usd(
            model or self.ctx.llm_model,
            pred.input_tokens,
            pred.output_tokens,
            self.ctx.pricing or None,
        )
        return pred
