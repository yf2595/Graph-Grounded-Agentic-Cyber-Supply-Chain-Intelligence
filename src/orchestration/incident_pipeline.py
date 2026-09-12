"""Sequential multi-agent incident pipeline with ablation flags and step logging."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from src.agents import (
    run_agent_01,
    run_agent_02,
    run_agent_03,
    run_agent_05,
    run_agent_06,
    run_agent_07,
    run_agent_08,
)
from src.schemas.models import (
    CanonicalIncident,
    FinalRiskAssessment,
    IncidentAssessment,
    MergeDecision,
    PipelineStatus,
    PropagationScope,
    RawIncident,
    RiskLevel,
    SupplyChainAssessment,
    UnifiedPipelineResult,
)
from src.services.entity_resolution import EntityResolutionService
from src.services.embedding_service import EmbeddingService
from src.services.graph_service import SupplyChainGraphService
from src.services.incident_repository import IncidentRepository
from src.services.llm_service import LLMService
from src.services.step_logger import StepLogger


@dataclass
class PipelineConfig:
    enable_agent_03: bool = True
    enable_agent_06: bool = True
    early_gate: bool = True
    unified_risk: bool = False
    include_report: bool = True
    # Optional Agent 4 (privacy-aware query expansion). Disabled by default so the
    # main 129-report experiment behaves exactly as before.
    query_expansion_enabled: bool = False
    # Force pre-vetted incidents past the Agent-2 relevance gate. Used ONLY by the
    # controlled query-expansion experiment, where the 20 selected incidents are
    # known cyber + supplier-relevant and must all reach Agent 4. Never enabled for
    # the 129-report benchmark.
    bypass_relevance_gate: bool = False


class UnifiedRiskLLM(BaseModel):
    model_config = {"extra": "forbid"}
    overall_risk: RiskLevel
    incident_severity: RiskLevel
    oem_exposure: RiskLevel
    main_effect: str = "UNKNOWN"
    risk_rationale: str = ""
    key_risk_drivers: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)


class IncidentPipeline:
    def __init__(
        self,
        dataset_dir: str | Path,
        run_dir: str | Path,
        config_path: str | Path,
        model: str = "gpt-5.4",
        reasoning_effort: str = "low",
        pipeline_config: PipelineConfig | None = None,
        step_logger: StepLogger | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        embedding_mode: str | None = None,
        use_reasoning: bool | None = None,
        reasoning_via_chat_template: bool = False,
        search_backend=None,
        agent4_llm: "LLMService | None" = None,
        forced_suppliers: "dict[str, list[dict]] | None" = None,
    ):
        self.graph = SupplyChainGraphService(dataset_dir)
        er_cfg = Path(__file__).resolve().parents[2] / "config" / "entity_resolution.yaml"
        self.llm = LLMService(
            model=model,
            reasoning_effort=reasoning_effort,
            temperature=0.0,
            base_url=llm_base_url,
            api_key=llm_api_key,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
            use_reasoning=use_reasoning,
            reasoning_via_chat_template=reasoning_via_chat_template,
        )
        self.resolver = EntityResolutionService(
            self.graph,
            config_path=er_cfg if er_cfg.exists() else None,
            llm=self.llm,
        )
        self.repo = IncidentRepository(run_dir)
        self.config_path = Path(config_path)
        self.pcfg = pipeline_config or PipelineConfig()
        self.step_logger = step_logger or StepLogger()
        self.last_usage = None
        # Agent 4 (query expansion) plumbing. Only used when enabled.
        self.search_backend = search_backend
        self.agent4_llm = agent4_llm
        self.forced_suppliers = forced_suppliers or {}
        self.last_agent4_result = None
        merge_cfg = Path(__file__).resolve().parents[2] / "config" / "merge_rules.yaml"
        self.merge_config_path = merge_cfg if merge_cfg.exists() else None
        emb_kwargs: dict = {}
        if embedding_mode:
            emb_kwargs["mode"] = embedding_mode  # type: ignore[assignment]
        elif llm_base_url:
            # Local vLLM runs must not call OpenAI embeddings
            emb_kwargs["mode"] = "local"
        self.embeddings = EmbeddingService(**emb_kwargs)
    def _article_log(self, article_id: str) -> StepLogger:
        path = self.repo.articles_dir / article_id / "steps.jsonl"
        return StepLogger(path)

    def process(self, raw: RawIncident, force: bool = False) -> UnifiedPipelineResult:
        if self.repo.has_article(raw.article_id) and not force:
            from json import loads

            p = self.repo.articles_dir / raw.article_id / "result.json"
            return UnifiedPipelineResult.model_validate(loads(p.read_text(encoding="utf-8")))

        log = self._article_log(raw.article_id)
        # fresh file for this force-run
        if log.log_path and log.log_path.exists() and force:
            log.log_path.write_text("", encoding="utf-8")
        self.llm.reset_usage()
        if self.agent4_llm is not None and self.agent4_llm is not self.llm:
            self.agent4_llm.reset_usage()
        self.last_agent4_result = None
        method_id = "pipeline"

        try:
            with log.step(article_id=raw.article_id, step="agent_01", method_id=method_id) as meta:
                u0 = (self.llm.usage.llm_calls, self.llm.usage.input_tokens, self.llm.usage.output_tokens)
                extracted = run_agent_01(self.llm, raw)
                self.repo.save_stage(raw.article_id, "agent_01", extracted)
                meta["llm_calls"] = self.llm.usage.llm_calls - u0[0]
                meta["input_tokens"] = self.llm.usage.input_tokens - u0[1]
                meta["output_tokens"] = self.llm.usage.output_tokens - u0[2]
                meta["summary"] = {
                    "event_type": getattr(extracted, "event_type", None),
                    "org_count": len(getattr(extracted, "mentioned_organizations", None) or []),
                }

            with log.step(article_id=raw.article_id, step="agent_02", method_id=method_id) as meta:
                u0 = (self.llm.usage.llm_calls, self.llm.usage.input_tokens, self.llm.usage.output_tokens)
                relevance = run_agent_02(self.llm, extracted, self.resolver, body=raw.body)
                self.repo.save_stage(raw.article_id, "agent_02", relevance)
                meta["llm_calls"] = self.llm.usage.llm_calls - u0[0]
                meta["input_tokens"] = self.llm.usage.input_tokens - u0[1]
                meta["output_tokens"] = self.llm.usage.output_tokens - u0[2]
                meta["summary"] = {
                    "cyber": getattr(getattr(relevance, "cyber_classification", None), "value", None),
                    "status": getattr(getattr(relevance, "pipeline_status", None), "value", None),
                    "n_suppliers": len(relevance.relevant_suppliers or []),
                    "filter_reason": relevance.filter_reason,
                }

            # Controlled query-expansion experiment: force pre-vetted incidents past
            # the relevance gate using their frozen supplier resolutions, so all
            # selected incidents reach Agent 4 (identical for A4-OFF and A4-ON).
            if self.pcfg.bypass_relevance_gate and raw.article_id in self.forced_suppliers:
                from src.schemas.models import Agent02Output, CyberClassification, RelevantSupplier

                forced = [RelevantSupplier(**s) for s in self.forced_suppliers[raw.article_id]]
                relevance = Agent02Output(
                    article_id=raw.article_id,
                    pipeline_status=PipelineStatus.PASSED,
                    cyber_classification=CyberClassification.CYBER,
                    cyber_reason=relevance.cyber_reason,
                    relevant_suppliers=forced,
                    filter_reason=None,
                    incident_summary=relevance.incident_summary or extracted.incident_summary,
                )
                self.repo.save_stage(raw.article_id, "agent_02", relevance)

            if self.pcfg.early_gate and relevance.pipeline_status != PipelineStatus.PASSED:
                result = UnifiedPipelineResult(
                    article_id=raw.article_id,
                    pipeline_status=PipelineStatus.FILTERED_OUT,
                    agent_02=relevance,
                    filter_reason=relevance.filter_reason,
                )
                self.repo.save_result(raw.article_id, result)
                log.write(
                    {
                        "article_id": raw.article_id,
                        "step": "pipeline_complete",
                        "status": "FILTERED_OUT",
                        "filter_reason": relevance.filter_reason,
                        "llm_calls": self.llm.usage.llm_calls,
                        "input_tokens": self.llm.usage.input_tokens,
                        "output_tokens": self.llm.usage.output_tokens,
                    }
                )
                self.last_usage = self.llm.usage
                return result

            if relevance.pipeline_status != PipelineStatus.PASSED and not relevance.relevant_suppliers:
                pass

            with log.step(article_id=raw.article_id, step="agent_03", method_id=method_id) as meta:
                u0 = (self.llm.usage.llm_calls, self.llm.usage.input_tokens, self.llm.usage.output_tokens)
                if self.pcfg.enable_agent_03 and relevance.relevant_suppliers:
                    canonical = run_agent_03(
                        self.llm,
                        relevance,
                        self.repo,
                        embeddings=self.embeddings,
                        merge_config_path=self.merge_config_path,
                    )
                else:
                    supplier_ids = [s.supplier_node_id for s in relevance.relevant_suppliers]
                    supplier_names = [s.canonical_name for s in relevance.relevant_suppliers]
                    eid = self.repo.next_event_id()
                    canonical = CanonicalIncident(
                        incident_id=eid,
                        source_article_ids=[raw.article_id],
                        canonical_summary=relevance.incident_summary or extracted.incident_summary,
                        event_type="cyber",
                        supplier_node_ids=supplier_ids,
                        supplier_names=supplier_names,
                        merge_decision=MergeDecision.CREATE_NEW,
                        merge_rationale="Merger disabled or no suppliers.",
                    )
                    self.repo.upsert_event(canonical.model_dump(mode="json"))
                self.repo.save_stage(raw.article_id, "agent_03", canonical)
                meta["llm_calls"] = self.llm.usage.llm_calls - u0[0]
                meta["input_tokens"] = self.llm.usage.input_tokens - u0[1]
                meta["output_tokens"] = self.llm.usage.output_tokens - u0[2]
                meta["summary"] = {
                    "incident_id": canonical.incident_id,
                    "merge": getattr(getattr(canonical, "merge_decision", None), "value", None),
                }

            # Agent 4 - Privacy-Aware Query Expansion (optional; between Agent 3 and Agent 5).
            if self.pcfg.query_expansion_enabled:
                with log.step(article_id=raw.article_id, step="agent_04", method_id=method_id) as meta:
                    from src.query_expansion.agent4 import run_agent4

                    if self.search_backend is None:
                        raise RuntimeError("query_expansion_enabled but no search_backend provided")
                    a4_llm = self.agent4_llm or self.llm
                    u0 = (a4_llm.usage.llm_calls, a4_llm.usage.input_tokens, a4_llm.usage.output_tokens)
                    enriched, a4_result = run_agent4(a4_llm, canonical, self.search_backend)
                    canonical = enriched
                    self.last_agent4_result = a4_result
                    self.repo.save_stage(raw.article_id, "agent_04", a4_result.to_dict())
                    meta["llm_calls"] = a4_llm.usage.llm_calls - u0[0]
                    meta["input_tokens"] = a4_llm.usage.input_tokens - u0[1]
                    meta["output_tokens"] = a4_llm.usage.output_tokens - u0[2]
                    meta["summary"] = {
                        "n_queries": len(a4_result.queries),
                        "n_accepted": len(a4_result.accepted),
                        "changed": a4_result.changed,
                    }

            if self.pcfg.unified_risk:
                with log.step(article_id=raw.article_id, step="agent_05_07_unified", method_id=method_id) as meta:
                    u0 = (self.llm.usage.llm_calls, self.llm.usage.input_tokens, self.llm.usage.output_tokens)
                    user = (
                        f"incident_id={canonical.incident_id}\n"
                        f"summary={canonical.canonical_summary}\n"
                        f"suppliers={canonical.supplier_names}\n"
                        "Assess overall OEM risk in one step (severity + exposure fused). "
                        "Do not invent graph node facts."
                    )
                    uni = self.llm.structured(
                        "You produce a unified OEM cyber supply-chain risk assessment.",
                        user,
                        UnifiedRiskLLM,
                    )
                    assessment = IncidentAssessment(
                        incident_id=canonical.incident_id,
                        severity=uni.incident_severity,
                        main_effect=uni.main_effect,
                        assessment_rationale=uni.risk_rationale,
                    )
                    network = SupplyChainAssessment(
                        incident_id=canonical.incident_id,
                        affected_suppliers=[],
                        propagation_scope=PropagationScope(),
                        oem_exposure=uni.oem_exposure,
                        exposure_rationale="Unified risk path (ablation A3); no separate graph propagation.",
                        confidence="MEDIUM",
                    )
                    scores = {"VERY_LOW": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "VERY_HIGH": 5}
                    final_risk = FinalRiskAssessment(
                        incident_id=canonical.incident_id,
                        incident_severity=uni.incident_severity,
                        oem_exposure=uni.oem_exposure,
                        overall_risk=uni.overall_risk,
                        risk_score_numeric=scores.get(uni.overall_risk.value, 3),
                        risk_rationale=uni.risk_rationale,
                        key_risk_drivers=uni.key_risk_drivers,
                        mitigating_factors=uni.mitigating_factors,
                    )
                    self.repo.save_stage(raw.article_id, "agent_05", assessment)
                    self.repo.save_stage(raw.article_id, "agent_06", network)
                    self.repo.save_stage(raw.article_id, "agent_07", final_risk)
                    meta["llm_calls"] = self.llm.usage.llm_calls - u0[0]
                    meta["input_tokens"] = self.llm.usage.input_tokens - u0[1]
                    meta["output_tokens"] = self.llm.usage.output_tokens - u0[2]
                    meta["summary"] = {"overall_risk": uni.overall_risk.value}
            else:
                with log.step(article_id=raw.article_id, step="agent_05", method_id=method_id) as meta:
                    u0 = (self.llm.usage.llm_calls, self.llm.usage.input_tokens, self.llm.usage.output_tokens)
                    assessment = run_agent_05(self.llm, canonical)
                    self.repo.save_stage(raw.article_id, "agent_05", assessment)
                    meta["llm_calls"] = self.llm.usage.llm_calls - u0[0]
                    meta["input_tokens"] = self.llm.usage.input_tokens - u0[1]
                    meta["output_tokens"] = self.llm.usage.output_tokens - u0[2]
                    meta["summary"] = {"severity": getattr(getattr(assessment, "severity", None), "value", None)}

                with log.step(article_id=raw.article_id, step="agent_06", method_id=method_id) as meta:
                    if self.pcfg.enable_agent_06:
                        network = run_agent_06(self.graph, canonical)
                        meta["graph_queries"] = 1
                    else:
                        network = SupplyChainAssessment(
                            incident_id=canonical.incident_id,
                            affected_suppliers=[],
                            propagation_scope=PropagationScope(),
                            oem_exposure=RiskLevel.MEDIUM,
                            exposure_rationale="Graph disabled (ablation A1); exposure set to MEDIUM placeholder.",
                            confidence="LOW",
                        )
                        meta["graph_queries"] = 0
                    self.repo.save_stage(raw.article_id, "agent_06", network)
                    meta["summary"] = {
                        "oem_exposure": getattr(getattr(network, "oem_exposure", None), "value", None),
                        "n_affected": len(network.affected_suppliers or []),
                    }

                with log.step(article_id=raw.article_id, step="agent_07", method_id=method_id) as meta:
                    u0 = (self.llm.usage.llm_calls, self.llm.usage.input_tokens, self.llm.usage.output_tokens)
                    final_risk = run_agent_07(self.llm, assessment, network, self.config_path)
                    self.repo.save_stage(raw.article_id, "agent_07", final_risk)
                    meta["llm_calls"] = self.llm.usage.llm_calls - u0[0]
                    meta["input_tokens"] = self.llm.usage.input_tokens - u0[1]
                    meta["output_tokens"] = self.llm.usage.output_tokens - u0[2]
                    meta["summary"] = {
                        "overall_risk": getattr(getattr(final_risk, "overall_risk", None), "value", None)
                    }

            report = None
            report_payload: dict | None = None
            if self.pcfg.include_report:
                with log.step(article_id=raw.article_id, step="agent_08", method_id=method_id) as meta:
                    u0 = (self.llm.usage.llm_calls, self.llm.usage.input_tokens, self.llm.usage.output_tokens)
                    report_out = run_agent_08(self.llm, canonical, assessment, network, final_risk)
                    if hasattr(report_out, "report_markdown"):
                        report = report_out.report_markdown
                        report_payload = report_out.model_dump()
                    else:
                        report = str(report_out)
                        report_payload = {"report_markdown": report}
                    self.repo.save_stage(raw.article_id, "agent_08", report_payload)
                    meta["llm_calls"] = self.llm.usage.llm_calls - u0[0]
                    meta["input_tokens"] = self.llm.usage.input_tokens - u0[1]
                    meta["output_tokens"] = self.llm.usage.output_tokens - u0[2]

            status = PipelineStatus.COMPLETED
            result = UnifiedPipelineResult(
                article_id=raw.article_id,
                pipeline_status=status,
                canonical_incident=canonical,
                incident_assessment=assessment,
                supply_chain_assessment=network,
                final_risk_assessment=final_risk,
                report=report,
                agent_02=relevance,
            )
            self.repo.save_result(raw.article_id, result)
            log.write(
                {
                    "article_id": raw.article_id,
                    "step": "pipeline_complete",
                    "status": "COMPLETED",
                    "llm_calls": self.llm.usage.llm_calls,
                    "input_tokens": self.llm.usage.input_tokens,
                    "output_tokens": self.llm.usage.output_tokens,
                }
            )
            self.last_usage = self.llm.usage
            return result
        except Exception as e:
            result = UnifiedPipelineResult(
                article_id=raw.article_id,
                pipeline_status=PipelineStatus.FAILED,
                error=str(e),
            )
            self.repo.save_result(raw.article_id, result)
            log.write(
                {
                    "article_id": raw.article_id,
                    "step": "pipeline_complete",
                    "status": "FAILED",
                    "error": str(e),
                    "llm_calls": self.llm.usage.llm_calls,
                    "input_tokens": self.llm.usage.input_tokens,
                    "output_tokens": self.llm.usage.output_tokens,
                }
            )
            self.last_usage = self.llm.usage
            return result
