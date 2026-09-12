"""Faithfulness aggregate metrics (WAITI P4 §18)."""
from __future__ import annotations


def faithfulness_metrics(labeled_claims: list[dict], required_coverage: dict[str, bool] | None = None) -> dict:
    labels = [c.get("faithfulness_label") for c in labeled_claims]
    n = len(labels)
    supported = sum(1 for x in labels if x == "SUPPORTED")
    contradicted = sum(1 for x in labels if x == "CONTRADICTED")
    unsupported = sum(1 for x in labels if x == "UNSUPPORTED")
    unverifiable = sum(1 for x in labels if x == "UNVERIFIABLE")
    verifiable = supported + contradicted + unsupported  # exclude unverifiable
    factual = [c for c in labeled_claims if c.get("claim_type") != "PROSE"]
    with_evid = sum(1 for c in factual if c.get("supporting_evidence_ids"))

    cov = required_coverage or {}
    cov_vals = list(cov.values()) if cov else []
    return {
        "n_claims": n,
        "n_factual_claims": len(factual),
        "n_supported": supported,
        "n_contradicted": contradicted,
        "n_unsupported": unsupported,
        "n_unverifiable": unverifiable,
        "n_verifiable": verifiable,
        "support_rate": round(supported / verifiable, 4) if verifiable else None,
        "hallucination_rate": round((unsupported + contradicted) / verifiable, 4) if verifiable else None,
        "contradiction_rate": round(contradicted / verifiable, 4) if verifiable else None,
        "provenance_coverage": round(with_evid / len(factual), 4) if factual else None,
        "required_field_coverage": round(sum(1 for v in cov_vals if v) / len(cov_vals), 4) if cov_vals else None,
        **{f"field_{k}": bool(v) for k, v in cov.items()},
    }
