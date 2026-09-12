"""B0 — Deterministic rules + embeddings + graph."""
from __future__ import annotations

import re

from evaluation.methods.base import EvaluationContext, EvaluationMethod
from evaluation.schemas.prediction import MethodPrediction, ModelInputRecord
from src.agents.agent_06_network_mapper import run_agent_06
from src.agents.agent_07_final_evaluator import fuse_risk, load_fusion_matrix
from src.schemas.models import CanonicalIncident, IncidentAssessment, MergeDecision, RiskLevel


class B0Method(EvaluationMethod):
    method_id = "B0"

    def process_record(self, record: ModelInputRecord) -> MethodPrediction:
        self.telemetry.start()
        pred = self._base_pred(record)
        try:
            text = f"{record.headline}\n{record.body}"
            rules = self.ctx.cyber_rules
            pred.pred_is_cyber = self._is_cyber(text, rules)
            self.telemetry.graph_queries += 1

            # crude org extraction: lines/phrases with Ltd/Inc/Corp or Title Case sequences
            surfaces = self._extract_orgs(text, rules)
            hits = self.ctx.resolver.resolve_surfaces(surfaces)
            self.telemetry.graph_queries += len(surfaces) + 1
            pred.predicted_supplier_names = [h.canonical_name for h in hits]
            pred.predicted_supplier_node_ids = [h.supplier_node_id for h in hits]
            pred.pred_is_related = len(hits) > 0
            pred.finalize_actionable()

            if not pred.pred_is_actionable:
                pred.status = "FILTERED_OUT"
                pred.latency_ms = self.telemetry.stop()
                pred.started_at = self.telemetry.started_at
                pred.completed_at = self.telemetry.completed_at
                pred.graph_queries = self.telemetry.graph_queries
                return pred

            # merge
            cands = self.merge_svc.get_candidates(
                record.article_id,
                pred.predicted_supplier_node_ids,
                text,
            )
            thr = float(
                self.ctx.method_config.get(
                    "embedding_similarity_threshold",
                    rules.get("merge", {}).get("embedding_similarity_threshold", 0.82),
                )
            )
            merged = None
            for c in cands:
                if c.overlap_supplier_ids and c.embedding_similarity >= thr:
                    merged = c
                    break

            if merged:
                ev = self.repo.get_event(merged.incident_id) or {
                    "incident_id": merged.incident_id,
                    "source_article_ids": list(merged.source_article_ids),
                    "canonical_summary": merged.canonical_summary,
                    "supplier_node_ids": list(merged.supplier_node_ids),
                    "supplier_names": list(merged.supplier_names),
                }
                aids = list(dict.fromkeys(ev.get("source_article_ids", []) + [record.article_id]))
                ev["source_article_ids"] = aids
                self.repo.upsert_event(ev)
                pred.predicted_incident_id = merged.incident_id
                pred.merge_action = "MERGE_EXISTING"
                pred.merged_into_incident_id = merged.incident_id
            else:
                eid = self.repo.next_event_id()
                self.repo.upsert_event(
                    {
                        "incident_id": eid,
                        "source_article_ids": [record.article_id],
                        "canonical_summary": text[:500],
                        "supplier_node_ids": pred.predicted_supplier_node_ids,
                        "supplier_names": pred.predicted_supplier_names,
                    }
                )
                pred.predicted_incident_id = eid
                pred.merge_action = "CREATE_NEW"

            # severity heuristic
            sev = self._severity(text, rules)
            pred.predicted_incident_severity = sev
            pred.predicted_main_effect = (
                "PRODUCTION_SHUTDOWN" if sev in ("HIGH", "VERY_HIGH") else "IT_DISRUPTION"
            )

            can = CanonicalIncident(
                incident_id=pred.predicted_incident_id or "EVT_TMP",
                source_article_ids=[record.article_id],
                canonical_summary=text[:500],
                supplier_node_ids=pred.predicted_supplier_node_ids,
                supplier_names=pred.predicted_supplier_names,
                merge_decision=MergeDecision.CREATE_NEW,
            )
            network = run_agent_06(self.ctx.graph, can)
            self.telemetry.graph_queries += 2
            pred.predicted_oem_exposure = network.oem_exposure.value  # type: ignore[assignment]
            matrix = load_fusion_matrix(self.ctx.risk_rules_path)
            overall = fuse_risk(RiskLevel(sev), network.oem_exposure, matrix)
            pred.predicted_risk = overall.value  # type: ignore[assignment]
            pred.risk_rationale = f"B0 deterministic fusion: severity={sev}, exposure={network.oem_exposure.value}"
            pred.status = "COMPLETED"
        except Exception as e:
            pred.status = "FAILED"
            pred.error_type = str(e)
        return self._finalize_telemetry(pred)

    @staticmethod
    def _is_cyber(text: str, rules: dict) -> bool:
        low = text.lower()
        for neg in rules.get("negative_context") or []:
            if neg.lower() in low:
                return False
        for term in rules.get("positive_terms") or []:
            if term.lower() in low:
                return True
        return False

    @staticmethod
    def _extract_orgs(text: str, rules: dict) -> list[str]:
        found: list[str] = []
        for m in re.finditer(r'"([^"]{3,80})"', text):
            found.append(m.group(1))
        suffixes = rules.get("org_suffixes") or ["Ltd.", "Inc.", "Corp.", "Corporation", "GmbH", "LLC"]
        for suf in suffixes:
            for m in re.finditer(rf"([A-Z][\w\-&.]*(?:\s+[A-Z][\w\-&.]*){{0,6}}\s+{re.escape(suf)})", text):
                found.append(m.group(1).strip())
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9&\-]{2,40})\b", text):
            found.append(m.group(1))
        # multi-word Title Case
        for m in re.finditer(r"\b((?:[A-Z][\w\-&.]+(?:\s+|\s*&\s*)){1,5}[A-Z][\w\-&.]+)\b", text):
            found.append(m.group(1).strip())
        out = []
        seen = set()
        for f in found:
            if f.lower() not in seen and len(f) >= 3:
                seen.add(f.lower())
                out.append(f)
        return out[:25]

    @staticmethod
    def _severity(text: str, rules: dict) -> str:
        low = text.lower()
        sk = rules.get("severity_keywords") or {}
        for term in sk.get("severe") or []:
            if term in low:
                return "HIGH"
        for term in sk.get("none") or []:
            if term in low:
                return "VERY_LOW"
        for term in sk.get("minor") or []:
            if term in low:
                return "LOW"
        return "MEDIUM"
