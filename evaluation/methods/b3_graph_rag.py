"""B3 — Single-pass GraphRAG (retrieve then one LLM call)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from evaluation.methods.base import EvaluationMethod
from evaluation.schemas.prediction import MethodPrediction, ModelInputRecord, RiskLevel
from src.services.llm_service import LLMService


class B3Out(BaseModel):
    model_config = {"extra": "forbid"}
    pred_is_cyber: bool
    pred_is_related: bool
    predicted_supplier_names: list[str] = Field(default_factory=list)
    predicted_supplier_node_ids: list[str] = Field(default_factory=list)
    merge_action: str = "CREATE_NEW"
    merge_into_incident_id: str | None = None
    predicted_incident_severity: RiskLevel = "UNKNOWN"
    predicted_oem_exposure: RiskLevel = "UNKNOWN"
    predicted_risk: RiskLevel = "UNKNOWN"
    predicted_main_effect: str | None = None
    risk_rationale: str | None = None


SYSTEM = """You are a GraphRAG OEM cyber supply-chain analyst.
You receive the article PLUS retrieved OEM graph snippets.
Use only retrieved node IDs; do not invent IDs not present in the context.
Decide cyber, related, merge vs create, severity, exposure, overall risk.
Return structured output."""


class B3Method(EvaluationMethod):
    method_id = "B3"

    def __init__(self, ctx):
        super().__init__(ctx)
        effort = ctx.method_config.get("reasoning_effort", "low")
        self.top_k = int(ctx.method_config.get("top_k", 8))
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
        self._index = self._build_index()

    def _build_index(self) -> list[tuple[str, str, list[float]]]:
        """Index supplier name + short profile strings."""
        items = []
        # Sample up to 400 suppliers by walking metrics keys
        ids = list(self.ctx.graph._metrics.keys())[:400]
        for sid in ids:
            meta = self.ctx.graph.get_supplier(sid) or {}
            name = meta.get("supplier_name") or sid
            text = (
                f"supplier_node_id={sid}; name={name}; "
                f"guaranteed_reach={meta.get('guaranteed_reach')}; "
                f"total_reach={meta.get('total_reach')}; "
                f"sole={meta.get('sole_source_product_count')}"
            )
            items.append((sid, text, None))  # embed lazily
        return items  # type: ignore[return-value]

    def _retrieve(self, query: str) -> list[str]:
        qv = self.ctx.embeddings.embed(query)
        self.telemetry.graph_queries += 1
        scored = []
        # embed on the fly with cache; limit compute
        for sid, text, _ in self._index[:200]:
            sim = self.ctx.embeddings.cosine(qv, self.ctx.embeddings.embed(text))
            scored.append((sim, text))
        scored.sort(reverse=True)
        return [t for _, t in scored[: self.top_k]]

    def process_record(self, record: ModelInputRecord) -> MethodPrediction:
        self.telemetry.start()
        pred = self._base_pred(record)
        try:
            article = f"{record.headline}\n{record.body}"
            snippets = self._retrieve(article)
            cands = self.merge_svc.get_candidates(record.article_id, [], article)
            cand_txt = [
                {
                    "incident_id": c.incident_id,
                    "summary": c.canonical_summary,
                    "suppliers": c.supplier_node_ids,
                    "sim": c.embedding_similarity,
                }
                for c in cands[:5]
            ]
            user = (
                f"article_id={record.article_id}\nARTICLE:\n{article}\n\n"
                f"GRAPH_SNIPPETS:\n" + "\n".join(snippets) + "\n\n"
                f"MERGE_CANDIDATES:\n{cand_txt}\n"
            )
            self.llm.reset_usage()
            out = self.llm.structured(SYSTEM, user, B3Out)
            self.telemetry.absorb_llm(self.llm.usage)
            # validate node ids against graph
            valid_ids = []
            for sid in out.predicted_supplier_node_ids:
                if self.ctx.graph.get_supplier(sid):
                    valid_ids.append(sid)
                    self.telemetry.graph_queries += 1
            pred.pred_is_cyber = out.pred_is_cyber
            pred.predicted_supplier_node_ids = valid_ids
            pred.predicted_supplier_names = out.predicted_supplier_names
            pred.pred_is_related = len(valid_ids) > 0
            pred.merge_action = out.merge_action
            pred.merged_into_incident_id = out.merge_into_incident_id
            pred.predicted_incident_severity = out.predicted_incident_severity
            pred.predicted_oem_exposure = out.predicted_oem_exposure
            pred.predicted_risk = out.predicted_risk
            pred.predicted_main_effect = out.predicted_main_effect
            pred.risk_rationale = out.risk_rationale
            pred.finalize_actionable()

            if not pred.pred_is_actionable:
                pred.status = "FILTERED_OUT"
            else:
                if (
                    out.merge_action == "MERGE_EXISTING"
                    and out.merge_into_incident_id
                    and self.repo.get_event(out.merge_into_incident_id)
                ):
                    ev = self.repo.get_event(out.merge_into_incident_id)
                    assert ev
                    ev = dict(ev)
                    ev["source_article_ids"] = list(
                        dict.fromkeys(ev.get("source_article_ids", []) + [record.article_id])
                    )
                    self.repo.upsert_event(ev)
                    pred.predicted_incident_id = out.merge_into_incident_id
                else:
                    eid = self.repo.next_event_id()
                    self.repo.upsert_event(
                        {
                            "incident_id": eid,
                            "source_article_ids": [record.article_id],
                            "canonical_summary": record.headline,
                            "supplier_node_ids": valid_ids,
                            "supplier_names": pred.predicted_supplier_names,
                        }
                    )
                    pred.predicted_incident_id = eid
                    pred.merge_action = "CREATE_NEW"
                pred.status = "COMPLETED"
        except Exception as e:
            pred.status = "FAILED"
            pred.error_type = str(e)
        return self._finalize_telemetry(pred, model=self.llm.model_name)
