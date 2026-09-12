"""B2 — Single LLM + graph tools (wraps src.baseline)."""
from __future__ import annotations

from evaluation.methods.base import EvaluationMethod
from evaluation.methods.normalize import from_baseline
from evaluation.schemas.prediction import MethodPrediction, ModelInputRecord
from src.baseline.runner import BaselineRunner
from src.schemas.models import RawIncident


class B2Method(EvaluationMethod):
    method_id = "B2"

    def __init__(self, ctx):
        super().__init__(ctx)
        effort = ctx.method_config.get("reasoning_effort", "medium")
        model = ctx.method_config.get("model", ctx.llm_model)
        base_url = ctx.method_config.get("base_url") or ctx.method_config.get("llm_base_url")
        api_key = ctx.method_config.get("api_key") or ctx.method_config.get("llm_api_key")
        self.runner = BaselineRunner(
            dataset_dir=str(ctx.root / "dataset"),
            run_dir=str(self.repo.run_dir),
            model=model,
            reasoning_effort=effort,
            base_url=base_url,
            api_key=api_key if base_url else None,
            max_tool_rounds=int(ctx.method_config.get("max_tool_rounds", 8)),
            max_tokens=ctx.method_config.get("max_output_tokens") or ctx.method_config.get("max_tokens"),
            disable_thinking=bool(ctx.method_config.get("disable_thinking", False)),
            tool_result_chars=int(ctx.method_config.get("tool_result_chars", 12000)),
            article_body_chars=ctx.method_config.get("article_body_chars"),
            temperature=float(
                ctx.method_config.get("temperature", ctx.temperature if ctx.temperature is not None else 0.0)
            ),
            use_reasoning=ctx.method_config.get("use_reasoning"),
            reasoning_via_chat_template=bool(ctx.method_config.get("reasoning_via_chat_template", False)),
        )
        # share same repo instance
        self.runner.repo = self.repo

    def process_record(self, record: ModelInputRecord) -> MethodPrediction:
        self.telemetry.start()
        pred = self._base_pred(record)
        try:
            raw = RawIncident(article_id=record.article_id, headline=record.headline, body=record.body)
            out = self.runner.process(raw, force=True)
            self.telemetry.absorb_llm(self.runner.usage)
            self.telemetry.graph_queries += 3
            pred = from_baseline(pred, out)
            if not pred.predicted_incident_id and pred.pred_is_actionable:
                eid = self.repo.event_for_article(record.article_id) or self.repo.next_event_id()
                pred.predicted_incident_id = eid
        except Exception as e:
            pred.status = "FAILED"
            pred.error_type = str(e)
        return self._finalize_telemetry(pred, model=self.runner.model_name)
