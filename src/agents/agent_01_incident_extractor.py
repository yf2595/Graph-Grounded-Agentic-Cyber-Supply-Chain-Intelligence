"""Agent 01 — incident extractor."""
from __future__ import annotations

from src.schemas.models import Agent01Output, RawIncident
from src.services.llm_service import LLMService

SYSTEM = """You are Agent 01, the News Collector / Incident Extractor.

Extract structured facts from the article only.
Do NOT decide OEM supply-chain relevance.
Do NOT query or invent graph relationships.
Do NOT assign final OEM risk.
If unknown, use empty lists.
Return structured output matching the schema."""


def run_agent_01(llm: LLMService, raw: RawIncident) -> Agent01Output:
    user = (
        f"article_id: {raw.article_id}\n"
        f"headline: {raw.headline}\n"
        f"body: {raw.body}\n\n"
        "Extract summary, organizations/entities, cyber signals, and stated operational effects."
    )
    out = llm.structured(SYSTEM, user, Agent01Output)
    # Force article_id / headline from input
    return out.model_copy(update={"article_id": raw.article_id, "headline": raw.headline})
