"""Single-LLM baseline with graph tools (gpt-5.4, reasoning_effort=medium)."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.schemas.models import (
    BaselineOutput,
    CyberClassification,
    MergeDecision,
    PipelineStatus,
    RawIncident,
    RelevantSupplier,
    RiskLevel,
)
from src.services.graph_service import SupplyChainGraphService
from src.services.incident_repository import IncidentRepository
from src.services.llm_service import UsageStats, _extract_usage, uses_reasoning_api
from src.services.step_logger import StepLogger


def build_tools(graph: SupplyChainGraphService, repo: IncidentRepository):
    @tool
    def resolve_supplier(name: str) -> dict:
        """Resolve an organization name to an OEM supply-chain supplier node if present."""
        hit = graph.resolve_name(name)
        if not hit:
            return {"matched": False, "name": name}
        return {
            "matched": True,
            "supplier_node_id": hit.supplier_node_id,
            "canonical_name": hit.canonical_name,
            "match_method": hit.match_method,
            "match_confidence": hit.match_confidence,
        }

    @tool
    def get_supplier_profile(supplier_node_id: str) -> dict:
        """Get graph metrics and redundancy for a supplier node id."""
        meta = graph.get_supplier(supplier_node_id) or {}
        red = graph.get_supplier_redundancy(supplier_node_id)
        return {**meta, **red}

    @tool
    def get_supplier_parts(supplier_node_id: str) -> list:
        """List supplied parts with criticality and redundancy from the OEM graph."""
        return graph.get_supplied_parts(supplier_node_id, limit=20)

    @tool
    def get_oem_products(supplier_node_id: str) -> list:
        """List OEM vehicle variants/products reached by this supplier."""
        return graph.get_affected_oem_products(supplier_node_id, limit=30)

    @tool
    def list_merge_candidates(supplier_node_ids: list[str]) -> list:
        """List existing canonical incidents that share supplier ids (for merge decisions)."""
        return repo.find_merge_candidates(supplier_node_ids)

    return [
        resolve_supplier,
        get_supplier_profile,
        get_supplier_parts,
        get_oem_products,
        list_merge_candidates,
    ]


SYSTEM = """You are a single-agent OEM cyber supply-chain analyst.

Tasks (do all of them):
1) Classify cyber vs not cyber.
2) Resolve mentioned organizations to OEM graph suppliers using tools.
3) If not cyber OR no relevant supplier → pipeline_status=FILTERED_OUT with filter_reason.
4) If relevant: decide CREATE_NEW vs MERGE_EXISTING using list_merge_candidates.
5) Assess incident severity and main operational effect.
6) Use graph tools for OEM exposure (do not invent graph facts).
7) Produce overall_risk in {VERY_LOW,LOW,MEDIUM,HIGH,VERY_HIGH} combining severity + exposure.
8) Write a short markdown report.

Use tools before asserting supplier node IDs or part/product facts.
When finished, respond with the BaselineOutput structured fields only (via final structured response).
"""


class BaselineRunner:
    def __init__(
        self,
        dataset_dir: str,
        run_dir: str,
        model: str = "gpt-5.4",
        reasoning_effort: str = "medium",
        max_tool_rounds: int = 8,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        tool_result_chars: int = 12000,
        article_body_chars: int | None = None,
        temperature: float = 0.0,
        use_reasoning: bool | None = None,
        reasoning_via_chat_template: bool = False,
    ):
        self.graph = SupplyChainGraphService(dataset_dir)
        self.repo = IncidentRepository(run_dir)
        self.max_tool_rounds = max_tool_rounds
        self.tool_result_chars = int(tool_result_chars)
        self.article_body_chars = article_body_chars
        # Structured BaselineOutput JSON often needs >1024 tokens on smaller Qwen models
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.structured_max_tokens = max(self.max_tokens or 0, 2048) or 2048
        kwargs: dict = {"model": model, "temperature": float(temperature)}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if base_url:
            kwargs["base_url"] = base_url
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url and disable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        elif base_url and reasoning_via_chat_template and reasoning_effort:
            # gpt-oss (harmony) chat template: keep reasoning on at the requested effort
            kwargs["extra_body"] = {"chat_template_kwargs": {"reasoning_effort": reasoning_effort}}
        use_local = bool(base_url)
        use_reason = (not use_local) and uses_reasoning_api(model, use_reasoning=use_reasoning)
        try:
            if use_local or not use_reason:
                self.llm = ChatOpenAI(**kwargs)
            else:
                self.llm = ChatOpenAI(
                    **kwargs,
                    reasoning_effort=reasoning_effort,
                    use_responses_api=True,
                )
        except TypeError:
            try:
                if use_local or not use_reason:
                    self.llm = ChatOpenAI(**kwargs)
                else:
                    self.llm = ChatOpenAI(**kwargs, reasoning_effort=reasoning_effort)
            except TypeError:
                self.llm = ChatOpenAI(**kwargs)
        self.tools = build_tools(self.graph, self.repo)
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_tools = self.llm.bind_tools(self.tools)
        # Prefer a higher completion budget for the final structured JSON call
        self.llm_structured = self.llm.bind(max_tokens=self.structured_max_tokens)
        self.structured = self.llm_structured.with_structured_output(BaselineOutput)
        self.usage = UsageStats()
        self.model_name = model

    def _invoke_structured_final(self, messages: list[Any], meta: dict) -> BaselineOutput:
        """Invoke structured output; retry once with a tighter prompt on length-limit truncations."""
        def _run(msgs: list[Any]) -> BaselineOutput:
            try:
                structured_raw = self.llm_structured.with_structured_output(
                    BaselineOutput, include_raw=True
                )
                raw_out = structured_raw.invoke(msgs)
                if isinstance(raw_out, dict):
                    inn, out_tok = _extract_usage(raw_out.get("raw"))
                    self.usage.add(input_tokens=inn, output_tokens=out_tok, note="structured_final")
                    meta["input_tokens"] = inn
                    meta["output_tokens"] = out_tok
                    parsed = raw_out.get("parsed")
                    if parsed is None:
                        raise ValueError("structured_final returned no parsed object")
                    return parsed
                self.usage.add(note="structured_final")
                return raw_out
            except TypeError:
                out = self.structured.invoke(msgs)
                self.usage.add(note="structured_final")
                return out

        meta["llm_calls"] = 1
        try:
            return _run(messages)
        except Exception as e:
            err = str(e).lower()
            if "length limit" not in err and "max_tokens" not in err and "could not parse" not in err:
                raise
            # One recovery attempt: ask for compact JSON only
            retry_msgs = list(messages) + [
                HumanMessage(
                    content=(
                        "Previous response was truncated. Reply with compact BaselineOutput JSON only: "
                        "short strings, empty lists when unknown, risk_rationale <= 40 words."
                    )
                )
            ]
            meta["llm_calls"] = 2
            meta["summary"] = {"structured_retry": "length_limit"}
            return _run(retry_msgs)

    def process(self, raw: RawIncident, force: bool = False) -> BaselineOutput:
        if self.repo.has_article(raw.article_id) and not force:
            data = json.loads(
                (self.repo.articles_dir / raw.article_id / "result.json").read_text(encoding="utf-8")
            )
            return BaselineOutput.model_validate(data)

        self.usage.reset()
        log = StepLogger(self.repo.articles_dir / raw.article_id / "steps.jsonl")
        if force and log.log_path and log.log_path.exists():
            log.log_path.write_text("", encoding="utf-8")

        body = raw.body or ""
        if self.article_body_chars is not None and len(body) > self.article_body_chars:
            body = body[: self.article_body_chars] + "\n...[truncated]..."

        messages: list[Any] = [
            SystemMessage(content=SYSTEM),
            HumanMessage(
                content=(
                    f"article_id={raw.article_id}\nheadline={raw.headline}\nbody={body}\n"
                    "Analyze fully using tools as needed, then produce BaselineOutput. "
                    "Keep tool use minimal (resolve suppliers, then decide)."
                )
            ),
        ]

        try:
            for round_i in range(self.max_tool_rounds):
                with log.step(
                    article_id=raw.article_id, step=f"tool_round_{round_i+1}", method_id="B2"
                ) as meta:
                    ai = self.llm_tools.invoke(messages)
                    inn, out_tok = _extract_usage(ai)
                    self.usage.add(input_tokens=inn, output_tokens=out_tok, note=f"tool_round_{round_i+1}")
                    messages.append(ai)
                    tool_calls = getattr(ai, "tool_calls", None) or []
                    meta["llm_calls"] = 1
                    meta["input_tokens"] = inn
                    meta["output_tokens"] = out_tok
                    meta["summary"] = {"n_tool_calls": len(tool_calls)}
                    if not tool_calls:
                        break
                    for tc in tool_calls:
                        name = tc["name"]
                        args = tc.get("args") or {}
                        tool = self.tool_map[name]
                        try:
                            result = tool.invoke(args)
                        except Exception as e:
                            result = {"error": str(e)}
                        messages.append(
                            ToolMessage(
                                content=json.dumps(result, ensure_ascii=False)[: self.tool_result_chars],
                                tool_call_id=tc["id"],
                            )
                        )

            with log.step(article_id=raw.article_id, step="structured_final", method_id="B2") as meta:
                messages.append(
                    HumanMessage(
                        content=(
                            "Now return the final BaselineOutput JSON fields only. "
                            "Be concise (short rationale). "
                            f"article_id must be {raw.article_id}."
                        )
                    )
                )
                out = self._invoke_structured_final(messages, meta)
                if not isinstance(out, BaselineOutput):
                    out = BaselineOutput.model_validate(out)
                out = out.model_copy(update={"article_id": raw.article_id})
            # Sync merge repo lightly for subsequent baseline articles
            if out.pipeline_status == PipelineStatus.COMPLETED or (
                out.cyber_classification == CyberClassification.CYBER and out.supplier_node_ids
            ):
                if out.merge_decision == MergeDecision.MERGE_EXISTING and out.merge_into_incident_id:
                    ev = self.repo.get_event(out.merge_into_incident_id)
                    if ev:
                        aids = list(dict.fromkeys(ev.get("source_article_ids", []) + [raw.article_id]))
                        ev = dict(ev)
                        ev["source_article_ids"] = aids
                        self.repo.upsert_event(ev)
                elif out.pipeline_status != PipelineStatus.FILTERED_OUT:
                    eid = out.merge_into_incident_id or self.repo.next_event_id()
                    self.repo.upsert_event(
                        {
                            "incident_id": eid,
                            "source_article_ids": [raw.article_id],
                            "canonical_summary": out.canonical_summary or raw.headline,
                            "supplier_node_ids": out.supplier_node_ids
                            or [s.supplier_node_id for s in out.relevant_suppliers],
                            "supplier_names": [s.canonical_name for s in out.relevant_suppliers],
                        }
                    )

            self.repo.save_result(raw.article_id, out)
            log.write(
                {
                    "article_id": raw.article_id,
                    "step": "baseline_complete",
                    "status": getattr(getattr(out, "pipeline_status", None), "value", None),
                    "llm_calls": self.usage.llm_calls,
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                }
            )
            return out
        except Exception as e:
            out = BaselineOutput(
                article_id=raw.article_id,
                pipeline_status=PipelineStatus.FAILED,
                cyber_classification=CyberClassification.UNCERTAIN,
                filter_reason=str(e),
            )
            self.repo.save_result(raw.article_id, out)
            log.write(
                {
                    "article_id": raw.article_id,
                    "step": "baseline_complete",
                    "status": "FAILED",
                    "error": str(e),
                    "llm_calls": self.usage.llm_calls,
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                }
            )
            return out
