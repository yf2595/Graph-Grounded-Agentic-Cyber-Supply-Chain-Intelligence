"""Phase 2 - generate 20 unrelated aerospace/defense cyber noise reports.

Noise reports use realistic cybersecurity terminology (ransomware, production
disruption, plant, systems offline, recovery, ...) so retrieval is non-trivial,
but they are unrelated to Honda or any selected supplier/incident. Any report
that mentions a banned entity is regenerated. Generation uses GPT-5.4 medium.
"""
from __future__ import annotations

import json
import random
import re

from src.query_expansion import config as C
from src.query_expansion.schemas import GeneratedReport
from src.query_expansion.selection import load_selected

DOMAINS = [
    "aircraft manufacturer",
    "aerospace component vendor",
    "avionics provider",
    "satellite / space supplier",
    "missile / defense manufacturing",
    "airport / aviation operational technology",
    "aerospace engineering contractor",
]

# Generic corporate/industry words that are NOT distinctive supplier identifiers.
_STOPWORDS = {
    "co", "ltd", "inc", "corp", "corporation", "company", "group", "limited",
    "the", "and", "of", "automotive", "auto", "systems", "system", "electric",
    "electronics", "industries", "industry", "manufacturing", "technologies",
    "technology", "parts", "components", "component", "usa", "america",
    "american", "japan", "thai", "thailand", "china", "india", "plant",
    "formerly", "public", "international", "motor", "motors", "holdings",
    "engineering", "products", "solutions", "global",
}

GEN_SYSTEM = """You write a single realistic synthetic news report about a cybersecurity \
incident affecting an organization in the AEROSPACE / AVIATION / SPACE / MISSILE / \
DEFENSE-INDUSTRIAL domain, for controlled research evaluation.

Rules:
- Use a fictional but realistic organization name in the given domain.
- The incident must be UNRELATED to the automotive industry and to Honda.
- Do NOT mention Honda or any automotive parts supplier.
- Use realistic cybersecurity terminology (e.g. ransomware, production disruption,
  supplier, manufacturing, plant, systems offline, recovery, cyberattack, operational
  impact) so the text is not a trivial negative.
- Do NOT include disclaimers, notes to the reader, or bracketed editorial text.
- Write 90-180 words of body text plus a concise headline."""

GEN_USER = """Domain for this report: {domain}
Variation seed: {seed}

Write the headline and body for one unrelated aerospace/defense cyber incident report. \
Invent a plausible organization name in this domain (must not be an automotive company \
and must not be Honda)."""


def _banned_terms(selected) -> set[str]:
    banned: set[str] = {"honda"}
    for inc in selected:
        banned.add(inc.incident_id.lower())
        for name in inc.supplier_names:
            low = name.lower().strip()
            if low:
                banned.add(low)
            for tok in re.findall(r"[a-z0-9]+", low):
                if len(tok) > 3 and tok not in _STOPWORDS:
                    banned.add(tok)
    return banned


def _overlaps(text: str, banned: set[str]) -> list[str]:
    low = text.lower()
    hits = []
    for term in banned:
        # word-ish boundary match to avoid accidental substrings
        if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", low):
            hits.append(term)
    return hits


def run(force: bool = False, max_attempts: int = 6) -> list[dict]:
    C.ensure_artifacts_dir()
    if C.NOISE_RAW_JSONL.exists() and not force:
        return [json.loads(l) for l in C.NOISE_RAW_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

    selected = load_selected()
    banned = _banned_terms(selected)
    llm = C.make_llm(C.GEN_MODEL, C.GEN_REASONING_EFFORT)
    rng = random.Random(C.GENERATION_SEED)

    records: list[dict] = []
    for i in range(1, C.N_NOISE + 1):
        domain = DOMAINS[(i - 1) % len(DOMAINS)]
        doc_id = f"QE_NOISE_{i:03d}"
        report: GeneratedReport | None = None
        for attempt in range(max_attempts):
            seed_val = rng.randint(0, 1_000_000)
            candidate = llm.structured(
                GEN_SYSTEM, GEN_USER.format(domain=domain, seed=seed_val), GeneratedReport
            )
            hits = _overlaps(candidate.headline + " " + candidate.body, banned)
            if not hits:
                report = candidate
                break
            print(f"  {doc_id} attempt {attempt + 1} overlap {hits}; regenerating")
        if report is None:
            raise RuntimeError(f"Could not generate clean noise report {doc_id} after {max_attempts} attempts")
        records.append(
            {
                "document": {
                    "doc_id": doc_id,
                    "headline": report.headline.strip(),
                    "body": report.body.strip(),
                },
                "metadata": {
                    "doc_id": doc_id,
                    "is_noise": True,
                    "is_relevant_to_target": False,
                    "target_incident_id": None,
                    "target_group_id": None,
                    "linked_target_incident_id": None,
                    "domain": domain,
                    "generation_model": C.GEN_MODEL,
                    "generation_reasoning_effort": C.GEN_REASONING_EFFORT,
                    "generation_seed": C.GENERATION_SEED,
                },
            }
        )
        print(f"  generated {doc_id} ({domain})")

    with C.NOISE_RAW_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} noise reports -> {C.NOISE_RAW_JSONL}")
    return records


if __name__ == "__main__":
    run(force=True)
