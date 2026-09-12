"""OURS + ablations A1-A4 wrapping IncidentPipeline."""
from __future__ import annotations

import json

from evaluation.methods.base import EvaluationContext, EvaluationMethod
from evaluation.methods.normalize import from_pipeline
from evaluation.schemas.prediction import MethodPrediction, ModelInputRecord
from evaluation.services.cost import estimate_cost_usd
from src.orchestration.incident_pipeline import IncidentPipeline, PipelineConfig
from src.schemas.models import RawIncident


def _pipeline_cfg_from_method(method_id: str, method_config: dict) -> PipelineConfig:
    cfg = PipelineConfig(
        enable_agent_03=method_config.get("enable_agent_03", True),
        enable_agent_06=method_config.get("enable_agent_06", True),
        early_gate=method_config.get("early_gate", True),
        unified_risk=method_config.get("unified_risk", False),
        include_report=bool(method_config.get("include_report", False)),
    )
    if method_id == "A1":
        cfg.enable_agent_06 = False
    elif method_id == "A2":
        cfg.enable_agent_03 = False
    elif method_id == "A3":
        cfg.unified_risk = True
    elif method_id == "A4":
        cfg.early_gate = False
    return cfg


class OursMethod(EvaluationMethod):
    method_id = "OURS"

    def __init__(self, ctx: EvaluationContext, method_id: str = "OURS"):
        self.method_id = method_id
        super().__init__(ctx)
        effort = ctx.method_config.get("reasoning_effort", "low")
        model = ctx.method_config.get("model", ctx.llm_model)
        self.model_name = model
        pcfg = _pipeline_cfg_from_method(method_id, ctx.method_config)
        base_url = ctx.method_config.get("base_url") or ctx.method_config.get("llm_base_url")
        api_key = ctx.method_config.get("api_key") or ctx.method_config.get("llm_api_key")
        self.pipeline = IncidentPipeline(
            dataset_dir=ctx.root / "dataset",
            run_dir=self.repo.run_dir,
            config_path=ctx.risk_rules_path,
            model=model,
            reasoning_effort=effort,
            pipeline_config=pcfg,
            llm_base_url=base_url,
            llm_api_key=api_key if base_url else None,
            max_tokens=ctx.method_config.get("max_output_tokens") or ctx.method_config.get("max_tokens"),
            disable_thinking=bool(ctx.method_config.get("disable_thinking", False)),
            embedding_mode="local" if base_url or ctx.method_config.get("_local_llm") else None,
            use_reasoning=ctx.method_config.get("use_reasoning"),
            reasoning_via_chat_template=bool(ctx.method_config.get("reasoning_via_chat_template", False)),
        )
        self.pipeline.repo = self.repo

    def process_record(self, record: ModelInputRecord) -> MethodPrediction:
        self.telemetry.start()
        pred = self._base_pred(record)
        try:
            raw = RawIncident(article_id=record.article_id, headline=record.headline, body=record.body)
            # Fresh diagnostics buffer per article
            if hasattr(self.pipeline.resolver, "clear_diagnostics"):
                self.pipeline.resolver.clear_diagnostics()
            result = self.pipeline.process(raw, force=True)
            usage = self.pipeline.last_usage or self.pipeline.llm.usage
            self.telemetry.absorb_llm(usage)
            if result.supply_chain_assessment is not None:
                self.telemetry.graph_queries += 1
            pred = from_pipeline(pred, result)
            # Persist entity-resolution diagnostics
            diag_path = self.repo.run_dir / "entity_resolution_diagnostics.jsonl"
            if hasattr(self.pipeline.resolver, "diagnostics"):
                with diag_path.open("a", encoding="utf-8") as f:
                    for rec in self.pipeline.resolver.diagnostics:
                        row = rec.to_dict() if hasattr(rec, "to_dict") else rec
                        if isinstance(row, dict):
                            row["method_id"] = self.method_id
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            pred.status = "FAILED"
            pred.error_type = str(e)
        pred.latency_ms = self.telemetry.stop()
        pred.started_at = self.telemetry.started_at
        pred.completed_at = self.telemetry.completed_at
        pred.llm_calls = self.telemetry.llm_calls
        pred.graph_queries = self.telemetry.graph_queries
        pred.input_tokens = self.telemetry.input_tokens
        pred.output_tokens = self.telemetry.output_tokens
        pred.cost_usd = estimate_cost_usd(
            self.model_name,
            pred.input_tokens,
            pred.output_tokens,
            self.ctx.pricing or None,
        )
        return pred


def make_ablation(method_id: str):
    class _Ablation(OursMethod):
        def __init__(self, ctx: EvaluationContext):
            super().__init__(ctx, method_id=method_id)

    _Ablation.method_id = method_id
    return _Ablation


A1Method = make_ablation("A1")
A2Method = make_ablation("A2")
A3Method = make_ablation("A3")
A4Method = make_ablation("A4")
