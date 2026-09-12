"""Generic multi-stage OEM supplier entity resolution (WAITI P3)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.schemas.models import RelevantSupplier
from src.services.graph_service import SupplyChainGraphService
from src.services.llm_service import LLMService

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover
    fuzz = None  # type: ignore
    process = None  # type: ignore

from pydantic import BaseModel, Field

LEGAL_SUFFIXES = [
    "co., ltd.",
    "co ltd",
    "co.,ltd.",
    "company limited",
    "corp.",
    "corporation",
    "incorporated",
    "limited",
    "ltd.",
    "ltd",
    "llc",
    "plc",
    "gmbh",
    "inc.",
    "inc",
    "s.p.a.",
    "s.a.",
    "k.k.",
    "ag",
    "co",
    "company",
]


DEFAULT_CFG = {
    "fuzzy_top_k": 5,
    "fuzzy_min_score": 90,
    "fuzzy_auto_accept_score": 96,
    "fuzzy_min_margin": 5,
    "enable_llm_disambiguation": True,
}


def load_er_config(path: str | Path | None = None) -> dict:
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return {**DEFAULT_CFG, **(data.get("entity_resolution") or data)}
    return dict(DEFAULT_CFG)


def normalize_org_name(name: str) -> str:
    """WAITI §12.1 normalization (NFKC, casefold, punct, suffixes)."""
    s = unicodedata.normalize("NFKC", name or "")
    s = s.casefold()
    s = s.replace("&", " and ")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
    s = re.sub(r"[-_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # strip legal suffixes from the end repeatedly
    changed = True
    while changed and s:
        changed = False
        for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
            if s.endswith(" " + suf) or s == suf:
                s = s[: -len(suf)].strip() if s != suf else ""
                changed = True
                break
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class CandidateMatch:
    supplier_node_id: str
    canonical_name: str
    match_score: float
    match_method: str


@dataclass
class ResolutionRecord:
    article_id: str | None
    surface_name: str
    normalized_name: str
    candidate_matches: list[CandidateMatch] = field(default_factory=list)
    selected_supplier_node_id: str | None = None
    resolution_method: str = "UNRESOLVED"
    resolution_confidence: float = 0.0
    reason: str = ""
    error_category: str = "NO_ALIAS_MATCH"

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "surface_name": self.surface_name,
            "normalized_name": self.normalized_name,
            "candidate_matches": [
                {
                    "supplier_node_id": c.supplier_node_id,
                    "canonical_name": c.canonical_name,
                    "match_score": c.match_score,
                    "match_method": c.match_method,
                }
                for c in self.candidate_matches
            ],
            "selected_supplier_node_id": self.selected_supplier_node_id,
            "resolution_method": self.resolution_method,
            "resolution_confidence": self.resolution_confidence,
            "reason": self.reason,
            "error_category": self.error_category,
        }


class DisambiguationLLM(BaseModel):
    model_config = {"extra": "forbid"}
    selected_supplier_node_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: str = ""


DISAMBIG_SYSTEM = """You are Agent 02 organization disambiguator for an OEM supply-chain graph.

Task: given an article snippet and a surface organization name, choose which candidate
supplier (if any) is the organization being mentioned.

Rules:
- Select ONLY from the provided candidate supplier_node_id list, or return null.
- Do NOT invent IDs.
- Do NOT infer cyber risk, OEM exposure, or relevance beyond identity matching.
- If unsure or none of the candidates match the mentioned organization, return null.
Return structured output only."""


class EntityResolutionService:
    def __init__(
        self,
        graph: SupplyChainGraphService,
        config: dict | None = None,
        config_path: str | Path | None = None,
        llm: LLMService | None = None,
    ):
        self.graph = graph
        self.cfg = {**DEFAULT_CFG, **(config or {})}
        if config_path:
            self.cfg = load_er_config(config_path)
        self.llm = llm
        self.diagnostics: list[ResolutionRecord] = []
        self._build_indexes()

    def _build_indexes(self) -> None:
        self.canonical_exact: dict[str, str] = {}  # lower raw -> id
        self.alias_exact: dict[str, list[str]] = {}  # lower raw -> list of ids
        self.norm_to_ids: dict[str, list[str]] = {}  # normalized -> ids
        self.id_to_name: dict[str, str] = dict(self.graph.id2name)
        self.norm_choices: list[tuple[str, str]] = []  # (normalized_name, sid)

        for sid, name in self.graph.id2name.items():
            self.canonical_exact[name.strip().lower()] = sid
            nrm = normalize_org_name(name)
            if nrm:
                self.norm_to_ids.setdefault(nrm, [])
                if sid not in self.norm_to_ids[nrm]:
                    self.norm_to_ids[nrm].append(sid)
                self.norm_choices.append((nrm, sid))

        for alias, sid in self.graph.alias2id.items():
            key = alias.strip().lower()
            self.alias_exact.setdefault(key, [])
            if sid not in self.alias_exact[key]:
                self.alias_exact[key].append(sid)
            nrm = normalize_org_name(alias)
            if nrm:
                self.norm_to_ids.setdefault(nrm, [])
                if sid not in self.norm_to_ids[nrm]:
                    self.norm_to_ids[nrm].append(sid)

        # unique choice strings for fuzzy
        seen = set()
        uniq = []
        for nrm, sid in self.norm_choices:
            key = (nrm, sid)
            if key not in seen:
                seen.add(key)
                uniq.append((nrm, sid))
        self.norm_choices = uniq

    def clear_diagnostics(self) -> None:
        self.diagnostics.clear()

    def resolve_surfaces(
        self,
        surfaces: list[str],
        *,
        article_id: str | None = None,
        article_context: str | None = None,
    ) -> list[RelevantSupplier]:
        hits: list[RelevantSupplier] = []
        seen_ids: set[str] = set()
        for surface in surfaces:
            if not surface or not str(surface).strip():
                rec = ResolutionRecord(
                    article_id=article_id,
                    surface_name=surface or "",
                    normalized_name="",
                    error_category="NO_ENTITY_EXTRACTED",
                    reason="empty surface",
                )
                self.diagnostics.append(rec)
                continue
            rec = self.resolve_one(surface, article_id=article_id, article_context=article_context)
            self.diagnostics.append(rec)
            if rec.selected_supplier_node_id and rec.selected_supplier_node_id not in seen_ids:
                seen_ids.add(rec.selected_supplier_node_id)
                hits.append(
                    RelevantSupplier(
                        surface_name=surface,
                        canonical_name=self.id_to_name.get(
                            rec.selected_supplier_node_id, rec.selected_supplier_node_id
                        ),
                        supplier_node_id=rec.selected_supplier_node_id,
                        relationship_to_oem="direct_supplier",
                        match_method=rec.resolution_method,
                        match_confidence=rec.resolution_confidence,
                    )
                )
        return hits

    def resolve_one(
        self,
        surface: str,
        *,
        article_id: str | None = None,
        article_context: str | None = None,
    ) -> ResolutionRecord:
        raw = surface.strip()
        nrm = normalize_org_name(raw)
        rec = ResolutionRecord(
            article_id=article_id,
            surface_name=raw,
            normalized_name=nrm,
        )

        # 12.2 Exact canonical
        sid = self.canonical_exact.get(raw.lower())
        if sid:
            return self._accept(rec, sid, "EXACT_CANONICAL", 1.0, "exact canonical name")

        # 12.3 Exact alias
        alias_ids = self.alias_exact.get(raw.lower(), [])
        if len(alias_ids) == 1:
            return self._accept(rec, alias_ids[0], "EXACT_ALIAS", 0.98, "exact alias")
        if len(alias_ids) > 1:
            cands = [
                CandidateMatch(s, self.id_to_name.get(s, s), 0.95, "EXACT_ALIAS") for s in alias_ids[:5]
            ]
            rec.candidate_matches = cands
            return self._disambiguate_or_unresolved(rec, cands, article_context)

        # 12.4 Normalized exact
        if nrm and nrm in self.norm_to_ids:
            ids = self.norm_to_ids[nrm]
            if len(ids) == 1:
                return self._accept(rec, ids[0], "NORMALIZED", 0.95, "normalized exact match")
            cands = [
                CandidateMatch(s, self.id_to_name.get(s, s), 0.9, "NORMALIZED") for s in ids[:5]
            ]
            rec.candidate_matches = cands
            return self._disambiguate_or_unresolved(rec, cands, article_context)

        # 12.5 Fuzzy
        cands = self._fuzzy_candidates(nrm or raw)
        rec.candidate_matches = cands
        if not cands:
            rec.error_category = "FUZZY_BELOW_THRESHOLD"
            rec.reason = "no fuzzy candidate above min score"
            return rec

        top = cands[0]
        margin = top.match_score - (cands[1].match_score if len(cands) > 1 else 100.0)
        auto = float(self.cfg["fuzzy_auto_accept_score"])
        min_margin = float(self.cfg["fuzzy_min_margin"])
        min_score = float(self.cfg["fuzzy_min_score"])

        # Unique candidate above min score is not ambiguous → accept
        if len(cands) == 1 and top.match_score >= min_score:
            return self._accept(
                rec,
                top.supplier_node_id,
                "FUZZY",
                top.match_score / 100.0,
                f"unique fuzzy candidate score={top.match_score}",
            )

        if top.match_score >= auto and margin >= min_margin:
            return self._accept(
                rec,
                top.supplier_node_id,
                "FUZZY",
                top.match_score / 100.0,
                f"fuzzy auto-accept score={top.match_score} margin={margin:.1f}",
            )

        if len(cands) >= 2:
            rec.error_category = "AMBIGUOUS_CANDIDATES"
            return self._disambiguate_or_unresolved(rec, cands, article_context)

        rec.error_category = "FUZZY_BELOW_THRESHOLD"
        rec.reason = "fuzzy below threshold"
        return rec

    def _fuzzy_candidates(self, query: str) -> list[CandidateMatch]:
        if not query or process is None or fuzz is None:
            return []
        top_k = int(self.cfg["fuzzy_top_k"])
        min_score = float(self.cfg["fuzzy_min_score"])
        choices = [n for n, _ in self.norm_choices]
        if not choices:
            return []
        results = process.extract(query, choices, scorer=fuzz.WRatio, limit=top_k * 3)
        out: list[CandidateMatch] = []
        seen: set[str] = set()
        for match_str, score, idx in results:
            if score < min_score:
                continue
            sid = self.norm_choices[idx][1]
            if sid in seen:
                continue
            seen.add(sid)
            out.append(
                CandidateMatch(
                    supplier_node_id=sid,
                    canonical_name=self.id_to_name.get(sid, sid),
                    match_score=float(score),
                    match_method="FUZZY",
                )
            )
            if len(out) >= top_k:
                break
        return out

    def _disambiguate_or_unresolved(
        self,
        rec: ResolutionRecord,
        cands: list[CandidateMatch],
        article_context: str | None,
    ) -> ResolutionRecord:
        enable = bool(self.cfg.get("enable_llm_disambiguation", True))
        if not enable or self.llm is None or not (2 <= len(cands) <= int(self.cfg["fuzzy_top_k"])):
            rec.resolution_method = "UNRESOLVED"
            rec.error_category = "AMBIGUOUS_CANDIDATES"
            rec.reason = "ambiguous candidates; LLM disambiguation unavailable or disabled"
            return rec
        return self._llm_disambiguate(rec, cands, article_context)

    def _llm_disambiguate(
        self,
        rec: ResolutionRecord,
        cands: list[CandidateMatch],
        article_context: str | None,
    ) -> ResolutionRecord:
        assert self.llm is not None
        cand_lines = []
        allowed = set()
        for c in cands:
            allowed.add(c.supplier_node_id)
            meta = self.graph.get_supplier(c.supplier_node_id) or {}
            cand_lines.append(
                {
                    "supplier_node_id": c.supplier_node_id,
                    "canonical_name": c.canonical_name,
                    "match_score": c.match_score,
                    "guaranteed_reach": meta.get("guaranteed_reach"),
                    "total_reach": meta.get("total_reach"),
                }
            )
        user = (
            f"surface_organization_name: {rec.surface_name}\n"
            f"normalized_name: {rec.normalized_name}\n"
            f"article_context:\n{(article_context or '')[:2500]}\n\n"
            f"candidates:\n{cand_lines}\n"
            "Select the matching supplier_node_id or null."
        )
        try:
            out = self.llm.structured(DISAMBIG_SYSTEM, user, DisambiguationLLM)
        except Exception as e:
            rec.resolution_method = "UNRESOLVED"
            rec.error_category = "AMBIGUOUS_CANDIDATES"
            rec.reason = f"LLM disambiguation failed: {e}"
            return rec

        sid = out.selected_supplier_node_id
        if sid and sid in allowed:
            return self._accept(
                rec,
                sid,
                "LLM_DISAMBIG",
                float(out.confidence or 0.7),
                out.reason or "llm disambiguation",
            )
        rec.resolution_method = "UNRESOLVED"
        rec.error_category = "AMBIGUOUS_CANDIDATES"
        rec.reason = out.reason or "llm returned null / invalid id"
        return rec

    @staticmethod
    def _accept(
        rec: ResolutionRecord,
        sid: str,
        method: str,
        confidence: float,
        reason: str,
    ) -> ResolutionRecord:
        rec.selected_supplier_node_id = sid
        rec.resolution_method = method
        rec.resolution_confidence = float(confidence)
        rec.reason = reason
        rec.error_category = "RESOLVED_CORRECT"
        return rec
