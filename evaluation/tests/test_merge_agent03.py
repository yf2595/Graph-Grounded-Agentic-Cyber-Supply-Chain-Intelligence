"""Tests for Agent 03 merge guardrails."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.agent_03_incident_merger import _collect_candidates, load_merge_config, run_agent_03
from src.schemas.models import Agent02Output, CyberClassification, PipelineStatus, RelevantSupplier
from src.services.incident_repository import IncidentRepository


class FakeEmb:
    def __init__(self, sims: dict[str, float]):
        self.sims = sims

    def embed(self, text: str):
        # return a 1-d proxy vector keyed by text identity
        return [hash(text.strip()[:80]) % 1000 / 1000.0, 1.0]

    def cosine(self, a, b):
        # caller embeds summary then compares — we intercept via monkey by using text in embed cache
        return 0.0


class FakeEmb2:
    """Embedding service where cosine depends on summary strings stored at embed time."""

    def __init__(self, pair_sim: dict[tuple[str, str], float], default: float = 0.5):
        self.pair_sim = pair_sim
        self.default = default
        self._last = ""

    def embed(self, text: str):
        self._last = text
        return [text]

    def cosine(self, a, b):
        ta = a[0] if isinstance(a, list) else a
        tb = b[0] if isinstance(b, list) else b
        return self.pair_sim.get((ta, tb), self.pair_sim.get((tb, ta), self.default))


def test_load_merge_config():
    cfg = load_merge_config(ROOT / "config" / "merge_rules.yaml")
    assert cfg["embedding_veto_merge_below"] == 0.78
    assert cfg["enable_embedding_candidates"] is True


def test_collect_candidates_supplier_and_embedding(tmp_path: Path):
    repo = IncidentRepository(tmp_path)
    repo.upsert_event(
        {
            "incident_id": "EVT_00001",
            "source_article_ids": ["A"],
            "canonical_summary": "ransomware at supplier alpha plant",
            "supplier_node_ids": ["S1"],
            "supplier_names": ["Alpha"],
        }
    )
    repo.upsert_event(
        {
            "incident_id": "EVT_00002",
            "source_article_ids": ["B"],
            "canonical_summary": "totally different fire at warehouse",
            "supplier_node_ids": ["S9"],
            "supplier_names": ["Other"],
        }
    )
    emb = FakeEmb2(
        {
            ("new ransomware alpha", "ransomware at supplier alpha plant"): 0.91,
            ("new ransomware alpha", "totally different fire at warehouse"): 0.20,
        },
        default=0.2,
    )
    cands = _collect_candidates(
        repo=repo,
        supplier_ids=["S1"],
        new_summary="new ransomware alpha",
        embeddings=emb,
        cfg=load_merge_config(ROOT / "config" / "merge_rules.yaml"),
    )
    ids = [c["incident_id"] for c in cands]
    assert "EVT_00001" in ids
    assert cands[0]["incident_id"] == "EVT_00001"
    assert cands[0]["embedding_similarity"] >= 0.9


def test_veto_low_similarity_merge(tmp_path: Path, monkeypatch):
    repo = IncidentRepository(tmp_path)
    repo.upsert_event(
        {
            "incident_id": "EVT_00001",
            "source_article_ids": ["A"],
            "canonical_summary": "denso ransomware wave one january",
            "supplier_node_ids": ["DENSO"],
            "supplier_names": ["Denso"],
        }
    )
    relevance = Agent02Output(
        article_id="B",
        pipeline_status=PipelineStatus.PASSED,
        cyber_classification=CyberClassification.CYBER,
        relevant_suppliers=[
            RelevantSupplier(
                surface_name="Denso",
                canonical_name="Denso",
                supplier_node_id="DENSO",
            )
        ],
        incident_summary="denso different ransomware wave two in mexico later",
    )

    class FakeLLM:
        def structured(self, system, user, schema):
            return schema(
                decision="MERGE_EXISTING",
                merge_into_incident_id="EVT_00001",
                rationale="same company ransomware",
            )

    emb = FakeEmb2(
        {
            (
                "denso different ransomware wave two in mexico later",
                "denso ransomware wave one january",
            ): 0.55
        },
        default=0.55,
    )
    out = run_agent_03(
        FakeLLM(),  # type: ignore[arg-type]
        relevance,
        repo,
        embeddings=emb,
        merge_config={
            "enable_embedding_candidates": True,
            "embedding_min_for_candidate": 0.5,
            "embedding_veto_merge_below": 0.78,
            "embedding_text_only_merge_min": 0.90,
            "embedding_skip_llm_below": 0.4,
            "max_candidates": 8,
        },
    )
    assert out.merge_decision.value == "CREATE_NEW"
    assert "Vetoed MERGE" in (out.merge_rationale or "")
    assert out.incident_id != "EVT_00001"
