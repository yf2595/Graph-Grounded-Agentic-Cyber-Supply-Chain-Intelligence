"""Agent 4 (Query Expansion) closed-corpus experiment runner.

Executes the full experiment end to end:

  Phase 1  select 20 distinct actionable incidents
  Phase 2  generate 40 supplements + 20 noise (GPT-5.4 medium), validate corpus
  Phase 3  build the closed local retrieval index
  Phase 5  paired downstream run: A4-OFF vs A4-ON on the 20 incidents (GPT-5.4)
  Phase 6  enriched reference labels (GPT-5.4 medium)
  Phase 7  evaluate + write reports

The frozen 129-report run is never touched. Requires OPENAI_API_KEY.

Usage:
  python -m evaluation.runners.run_query_expansion_experiment --phase all
  python -m evaluation.runners.run_query_expansion_experiment --phase corpus
  python -m evaluation.runners.run_query_expansion_experiment --phase paired
  python -m evaluation.runners.run_query_expansion_experiment --phase evaluate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.query_expansion import config as C
from src.query_expansion import corpus_validation, index_builder, noise_generation
from src.query_expansion import selection, supplement_generation
from src.query_expansion.closed_corpus_backend import ClosedCorpusSearchBackend
from src.orchestration.incident_pipeline import IncidentPipeline, PipelineConfig
from src.schemas.models import RawIncident


def require_openai_api_key() -> None:
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_TOKEN")):
        raise SystemExit(
            "Missing OpenAI API credentials. Set OPENAI_API_KEY in the environment."
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _raw_incident(inc) -> RawIncident:
    return RawIncident(article_id=inc.incident_id, headline=inc.headline, body=inc.body)


def _forced_suppliers(selected) -> dict:
    """Frozen supplier resolutions per incident, used to force the pre-vetted
    selection past the Agent-2 relevance gate so all 20 reach Agent 4."""
    forced: dict[str, list[dict]] = {}
    for inc in selected:
        rows = []
        node_ids = inc.supplier_node_ids or []
        names = inc.supplier_names or []
        for i, sid in enumerate(node_ids):
            name = names[i] if i < len(names) else sid
            rows.append(
                {
                    "surface_name": name,
                    "canonical_name": name,
                    "supplier_node_id": sid,
                    "relationship_to_oem": "supplier",
                    "match_method": "frozen_selection",
                    "match_confidence": 1.0,
                }
            )
        if rows:
            forced[inc.incident_id] = rows
    return forced


# ---------------------------------------------------------------------------
# Phase 1-3: corpus
# ---------------------------------------------------------------------------


def phase_select() -> None:
    sel = selection.run()
    print(f"Phase 1: selected {len(sel)} incidents")


def phase_corpus() -> None:
    require_openai_api_key()
    supplement_generation.run()
    noise_generation.run()
    result = corpus_validation.run(strict=True)
    print(f"Phase 2: corpus validation all_ok={result['all_ok']}")
    index_builder.build()
    print("Phase 3: index built")


# ---------------------------------------------------------------------------
# Phase 5: paired downstream run
# ---------------------------------------------------------------------------


def _make_pipeline(condition: str, backend=None, forced=None):
    run_dir = C.RESULTS_RUN_DIR / ("A4_ON" if condition == "on" else "A4_OFF")
    pcfg = PipelineConfig(
        include_report=True,
        query_expansion_enabled=(condition == "on"),
        # Both conditions bypass the gate identically so the paired comparison runs
        # over the same 20 incidents (all reach Agent 4).
        bypass_relevance_gate=True,
    )
    agent4_llm = None
    if condition == "on":
        agent4_llm = C.make_llm(C.AGENT4_MODEL, C.AGENT4_REASONING_EFFORT)
    pipe = IncidentPipeline(
        dataset_dir=C.GRAPH_DIR,
        run_dir=run_dir,
        config_path=ROOT / "config" / "risk_rules.yaml",
        model=C.DOWNSTREAM_MODEL,
        reasoning_effort=C.DOWNSTREAM_REASONING_EFFORT,
        pipeline_config=pcfg,
        max_tokens=C.DOWNSTREAM_MAX_OUTPUT_TOKENS,
        search_backend=backend,
        agent4_llm=agent4_llm,
        forced_suppliers=forced,
    )
    return pipe


def phase_reindex() -> None:
    """Rebuild ONLY the retrieval index over the frozen 60 documents using the
    configured semantic embedding. Does not regenerate the corpus."""
    require_openai_api_key()
    if not C.DOCUMENTS_JSONL.exists():
        raise SystemExit(f"Frozen corpus missing: {C.DOCUMENTS_JSONL}")
    manifest = index_builder.build()
    print(f"Reindex: {manifest['n_documents']} docs with {manifest['embedding_model']}")


def phase_paired() -> None:
    require_openai_api_key()
    selected = selection.load_selected()
    forced = _forced_suppliers(selected)
    backend = ClosedCorpusSearchBackend()

    off_pipe = _make_pipeline("off", forced=forced)
    on_pipe = _make_pipeline("on", backend=backend, forced=forced)

    a5_before, a5_after = [], []
    fin_before, fin_after = [], []
    enriched_rows, queries_rows, retrieval_rows, relevance_rows, privacy_rows = [], [], [], [], []

    for inc in selected:
        raw = _raw_incident(inc)

        # A4 OFF
        res_off = off_pipe.process(raw, force=True)
        ia_off = res_off.incident_assessment
        fr_off = res_off.final_risk_assessment
        a5_before.append({"incident_id": inc.incident_id, "assessment": ia_off.model_dump() if ia_off else None})
        fin_before.append(
            {
                "incident_id": inc.incident_id,
                "final_risk": fr_off.model_dump() if fr_off else None,
                "report": res_off.report,
            }
        )

        # A4 ON
        res_on = on_pipe.process(raw, force=True)
        ia_on = res_on.incident_assessment
        fr_on = res_on.final_risk_assessment
        a5_after.append({"incident_id": inc.incident_id, "assessment": ia_on.model_dump() if ia_on else None})
        fin_after.append(
            {
                "incident_id": inc.incident_id,
                "final_risk": fr_on.model_dump() if fr_on else None,
                "report": res_on.report,
            }
        )

        a4 = on_pipe.last_agent4_result
        can_on = res_on.canonical_incident
        enriched_rows.append(
            {
                "incident_id": inc.incident_id,
                "changed": bool(a4.changed) if a4 else False,
                "original_summary": inc.body,
                "enriched_summary": can_on.canonical_summary if can_on else None,
                "accepted": a4.accepted if a4 else [],
            }
        )
        if a4:
            for q in a4.queries:
                queries_rows.append({"incident_id": inc.incident_id, **q})
            for r in a4.retrievals:
                retrieval_rows.append({"incident_id": inc.incident_id, **r})
            for d in a4.candidate_decisions:
                relevance_rows.append({"incident_id": inc.incident_id, **d})
            for p in a4.privacy_leaks:
                privacy_rows.append({"incident_id": inc.incident_id, **p})

        print(
            f"  {inc.incident_id}: OFF risk={_risk(fr_off)} -> ON risk={_risk(fr_on)} "
            f"(a4 changed={a4.changed if a4 else False}, accepted={len(a4.accepted) if a4 else 0})"
        )

    _write_jsonl(C.AGENT5_BEFORE_JSONL, a5_before)
    _write_jsonl(C.AGENT5_AFTER_JSONL, a5_after)
    _write_jsonl(C.FINAL_BEFORE_JSONL, fin_before)
    _write_jsonl(C.FINAL_AFTER_JSONL, fin_after)
    _write_jsonl(C.ENRICHED_INCIDENTS_JSONL, enriched_rows)
    _write_jsonl(C.QUERIES_JSONL, queries_rows)
    _write_jsonl(C.RETRIEVAL_RAW_JSONL, retrieval_rows)
    _write_jsonl(C.RELEVANCE_CHECKS_JSONL, relevance_rows)
    _write_jsonl(C.PRIVACY_LEAKS_JSONL, privacy_rows)
    print(f"Phase 5: paired run complete for {len(selected)} incidents")


def _risk(fr) -> str:
    if fr is None:
        return "UNKNOWN"
    return getattr(getattr(fr, "overall_risk", None), "value", "UNKNOWN")


# ---------------------------------------------------------------------------
# Phase 6-7: reference labels + evaluation
# ---------------------------------------------------------------------------


def phase_reference() -> None:
    require_openai_api_key()
    from src.query_expansion import reference_labels

    reference_labels.run()
    print("Phase 6: enriched reference labels written")


def phase_evaluate() -> None:
    from src.query_expansion import evaluation_downstream, evaluation_retrieval

    retrieval = evaluation_retrieval.run()
    downstream = evaluation_downstream.run()
    print("Phase 7: evaluation complete")
    print(json.dumps({"retrieval": retrieval, "downstream": downstream}, indent=2)[:2000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "select", "corpus", "reindex", "paired", "reference", "evaluate", "rerun"],
    )
    args = parser.parse_args()

    # "rerun" = corrected final run over the FROZEN corpus: re-embed the index with
    # the semantic model, run the paired A4-OFF/A4-ON comparison, and evaluate.
    # It never regenerates supplements/noise or the enriched reference labels.
    if args.phase in ("all", "select"):
        phase_select()
    if args.phase in ("all", "corpus"):
        phase_corpus()
    if args.phase in ("all", "reindex", "rerun"):
        phase_reindex()
    if args.phase in ("all", "paired", "rerun"):
        phase_paired()
    if args.phase in ("all", "reference"):
        phase_reference()
    if args.phase in ("all", "evaluate", "rerun"):
        phase_evaluate()


if __name__ == "__main__":
    main()
