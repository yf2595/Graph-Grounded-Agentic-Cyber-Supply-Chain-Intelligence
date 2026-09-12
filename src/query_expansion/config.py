"""Shared paths and constants for the query-expansion (Agent 4) experiment."""
from __future__ import annotations

from pathlib import Path

# Repo root = .../SC_Eval
ROOT = Path(__file__).resolve().parents[2]

# Data sources (read-only)
INCIDENTS_CSV = ROOT / "eval_corpus_v1" / "incidents.csv"
ARTICLES_DIR = ROOT / "eval_corpus_v1" / "articles"
GROUND_TRUTH_JSONL = ROOT / "evaluation" / "data" / "ground_truth.jsonl"
GRAPH_DIR = ROOT / "dataset"

# Experiment artifacts (written)
ARTIFACTS_DIR = ROOT / "artifacts" / "query_expansion"
INDEX_DIR = ARTIFACTS_DIR / "index"
GENERATED_DIR = ARTIFACTS_DIR / "_generated"
SUPPLEMENTS_RAW_JSONL = GENERATED_DIR / "supplements.jsonl"
NOISE_RAW_JSONL = GENERATED_DIR / "noise.jsonl"

SELECTED_INCIDENTS_JSON = ARTIFACTS_DIR / "selected_incidents.json"
SELECTION_REPORT_MD = ARTIFACTS_DIR / "selection_report.md"
DOCUMENTS_JSONL = ARTIFACTS_DIR / "documents.jsonl"
GROUND_TRUTH_OUT_JSONL = ARTIFACTS_DIR / "ground_truth.jsonl"
CORPUS_MANIFEST_JSON = ARTIFACTS_DIR / "corpus_manifest.json"
INDEX_MANIFEST_JSON = ARTIFACTS_DIR / "index_manifest.json"
QUERIES_JSONL = ARTIFACTS_DIR / "queries.jsonl"
PRIVACY_LEAKS_JSONL = ARTIFACTS_DIR / "privacy_leaks.jsonl"
RETRIEVAL_RAW_JSONL = ARTIFACTS_DIR / "retrieval_raw.jsonl"
RELEVANCE_CHECKS_JSONL = ARTIFACTS_DIR / "relevance_checks.jsonl"
ENRICHED_INCIDENTS_JSONL = ARTIFACTS_DIR / "enriched_incidents.jsonl"
AGENT5_BEFORE_JSONL = ARTIFACTS_DIR / "agent5_before.jsonl"
AGENT5_AFTER_JSONL = ARTIFACTS_DIR / "agent5_after.jsonl"
FINAL_BEFORE_JSONL = ARTIFACTS_DIR / "final_before.jsonl"
FINAL_AFTER_JSONL = ARTIFACTS_DIR / "final_after.jsonl"
ENRICHED_REFERENCE_LABELS_JSONL = ARTIFACTS_DIR / "enriched_reference_labels.jsonl"
PER_INCIDENT_RESULTS_CSV = ARTIFACTS_DIR / "per_incident_results.csv"
AGGREGATE_METRICS_JSON = ARTIFACTS_DIR / "aggregate_metrics.json"
EXPERIMENT_REPORT_MD = ARTIFACTS_DIR / "experiment_report.md"
FAILURE_ANALYSIS_MD = ARTIFACTS_DIR / "failure_analysis.md"
REPRODUCIBILITY_MANIFEST_JSON = ARTIFACTS_DIR / "reproducibility_manifest.json"

# Paired downstream run directory (never touches frozen runs)
RESULTS_RUN_DIR = ROOT / "results" / "waiti_qe_gpt54"

# Experiment constants
SELECTION_SEED = 42
GENERATION_SEED = 42
N_TARGET_INCIDENTS = 20
N_SUPPLEMENTS_PER_INCIDENT = 2
N_NOISE = 20
N_TOTAL_DOCS = 60  # 40 supplements + 20 noise
QUERIES_PER_INCIDENT = 5
RETRIEVAL_TOP_K = 1
RELEVANT_DOCS_PER_INCIDENT = 2

# Model configuration
# Corpus generation + reference labels remain frozen from the original run
# (reasoning_effort=medium); we do NOT regenerate them in the corrected run.
GEN_MODEL = "gpt-5.4"
GEN_REASONING_EFFORT = "medium"
# Agent 4's own LLM calls now run at reasoning_effort=low, matching Agents 5-8
# and the paper's experimental description (corrected run).
AGENT4_MODEL = "gpt-5.4"
AGENT4_REASONING_EFFORT = "low"
REFERENCE_MODEL = "gpt-5.4"
REFERENCE_REASONING_EFFORT = "medium"
# Downstream pipeline (Agents 5/6/7/8) matches the frozen waiti_final_gpt54 run.
DOWNSTREAM_MODEL = "gpt-5.4"
DOWNSTREAM_REASONING_EFFORT = "low"
DOWNSTREAM_TEMPERATURE = 0.0
DOWNSTREAM_MAX_OUTPUT_TOKENS = 2048

# Retrieval embedding. The project has no local *semantic* embedding model; the
# standard/default semantic embedding used elsewhere (merge candidates, GraphRAG)
# is OpenAI text-embedding-3-small. The corrected run uses it for both indexing
# and query embedding (replaces the weak 256/384-dim deterministic hash).
EMBEDDING_MODE = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"

RISK_ORDER = ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]

# Evidence-gap taxonomy (Section 4.2). OEM-graph attributes are intentionally absent.
GAP_TYPES = [
    "disruption_scope",
    "affected_facilities",
    "affected_operations",
    "production_or_service_impact",
    "operational_duration",
    "recovery_status",
    "recovery_timeline",
    "containment_status",
    "attack_vector",
    "malware_or_actor_attribution",
    "technical_scope",
    "data_or_system_impact",
    "geographic_scope",
    "supplier_public_response",
]

EXPECTED_EFFECTS = ["increase", "decrease", "no_ordinal_change"]


def ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def make_llm(model: str, reasoning_effort: str):
    """Construct an LLMService for OpenAI GPT-5.4 (reasoning API)."""
    from src.services.llm_service import LLMService

    return LLMService(model=model, reasoning_effort=reasoning_effort, temperature=0.0)
