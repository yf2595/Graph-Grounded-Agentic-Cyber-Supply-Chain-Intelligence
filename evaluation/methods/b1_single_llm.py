"""B1 — Single LLM, no graph tools."""
from __future__ import annotations

from pydantic import BaseModel, Field

from evaluation.methods.base import EvaluationMethod
from evaluation.schemas.prediction import MethodPrediction, ModelInputRecord, RiskLevel
from src.services.llm_service import LLMService


class B1Out(BaseModel):
    model_config = {"extra": "forbid"}
    pred_is_cyber: bool
    pred_is_related: bool
    predicted_supplier_names: list[str] = Field(default_factory=list)
    # Must NOT invent graph node IDs — leave empty in B1
    merge_action: str = "CREATE_NEW"
    predicted_incident_severity: RiskLevel = "UNKNOWN"
    predicted_oem_exposure: RiskLevel = "UNKNOWN"
    predicted_risk: RiskLevel = "UNKNOWN"
    predicted_main_effect: str | None = None
    risk_rationale: str | None = None


SYSTEM = """You are a single-LLM OEM cyber supply-chain analyst WITHOUT graph access.
Classify cyber vs not, guess if organizations might be automotive suppliers (related),
assess severity and overall OEM risk qualitatively.
Do NOT invent supplier_node_ids or claim graph facts.
If unsure an org is in the OEM graph, set pred_is_related=false.
Return structured output only."""


class B1Method(EvaluationMethod):
    method_id = "B1"

    def __init__(self, ctx):
        super().__init__(ctx)
        effort = ctx.method_config.get("reasoning_effort", "low")
        base_url = ctx.method_config.get("base_url") or ctx.method_config.get("llm_base_url")
        api_key = ctx.method_config.get("api_key") or ctx.method_config.get("llm_api_key")
        self.llm = LLMService(
            model=ctx.method_config.get("model", ctx.llm_model),
            reasoning_effort=effort,
            temperature=ctx.temperature,
            base_url=base_url,
            api_key=api_key if base_url else None,
            max_tokens=ctx.method_config.get("max_output_tokens") or ctx.method_config.get("max_tokens"),
            disable_thinking=bool(ctx.method_config.get("disable_thinking", False)),
            use_reasoning=ctx.method_config.get("use_reasoning"),
            reasoning_via_chat_template=bool(ctx.method_config.get("reasoning_via_chat_template", False)),
        )

    def process_record(self, record: ModelInputRecord) -> MethodPrediction:
        self.telemetry.start()
        pred = self._base_pred(record)
        try:
            user = f"article_id={record.article_id}\nheadline={record.headline}\nbody={record.body}"
            self.llm.reset_usage()
            out = self.llm.structured(SYSTEM, user, B1Out)
            self.telemetry.absorb_llm(self.llm.usage)
            pred.pred_is_cyber = out.pred_is_cyber
            pred.pred_is_related = out.pred_is_related
            pred.predicted_supplier_names = out.predicted_supplier_names
            pred.predicted_supplier_node_ids = []  # no graph
            pred.merge_action = out.merge_action
            pred.predicted_incident_severity = out.predicted_incident_severity
            pred.predicted_oem_exposure = out.predicted_oem_exposure
            pred.predicted_risk = out.predicted_risk
            pred.predicted_main_effect = out.predicted_main_effect
            pred.risk_rationale = out.risk_rationale
            pred.finalize_actionable()
            if not pred.pred_is_actionable:
                pred.status = "FILTERED_OUT"
            else:
                eid = self.repo.next_event_id()
                self.repo.upsert_event(
                    {
                        "incident_id": eid,
                        "source_article_ids": [record.article_id],
                        "canonical_summary": record.headline,
                        "supplier_node_ids": [],
                        "supplier_names": pred.predicted_supplier_names,
                    }
                )
                pred.predicted_incident_id = eid
                pred.status = "COMPLETED"
        except Exception as e:
            pred.status = "FAILED"
            pred.error_type = str(e)
        return self._finalize_telemetry(pred, model=self.llm.model_name)
