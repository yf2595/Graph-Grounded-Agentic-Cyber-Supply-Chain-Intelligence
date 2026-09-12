"""Evaluation Part 1-3 - query and retrieval quality (Sections 20-22)."""
from __future__ import annotations

import json

from src.query_expansion import config as C


def _load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _truth_maps() -> tuple[dict[str, set[str]], set[str], dict[str, str]]:
    """Return (target -> set of its 2 supplement doc_ids, noise doc ids, doc_id -> target)."""
    gts = _load_jsonl(C.GROUND_TRUTH_OUT_JSONL)
    target_supps: dict[str, set[str]] = {}
    noise: set[str] = set()
    doc_target: dict[str, str] = {}
    for g in gts:
        if g.get("is_noise"):
            noise.add(g["doc_id"])
        else:
            t = g["target_incident_id"]
            target_supps.setdefault(t, set()).add(g["doc_id"])
            doc_target[g["doc_id"]] = t
    return target_supps, noise, doc_target


def _candidates_by_incident(retrieval_rows: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for r in retrieval_rows:
        out.setdefault(r["incident_id"], set()).add(r["retrieved_doc_id"])
    return out


def compute() -> dict:
    target_supps, noise, doc_target = _truth_maps()
    retrieval_rows = _load_jsonl(C.RETRIEVAL_RAW_JSONL)
    queries_rows = _load_jsonl(C.QUERIES_JSONL)
    relevance_rows = _load_jsonl(C.RELEVANCE_CHECKS_JSONL)

    incidents = sorted(target_supps)
    n_inc = len(incidents)
    cand_by_inc = _candidates_by_incident(retrieval_rows)

    # ---- Raw retrieval (Section 20) ----
    per_incident: dict[str, dict] = {}
    retrieved_true = 0
    any_hit = 0
    both_hit = 0
    recall_sum = 0.0
    total_unique_candidates = 0
    true_candidates = 0
    for inc in incidents:
        true_docs = target_supps.get(inc, set())
        cands = cand_by_inc.get(inc, set())
        hit_docs = cands & true_docs
        retrieved_true += len(hit_docs)
        recall_i = len(hit_docs) / max(1, len(true_docs))
        recall_sum += recall_i
        if hit_docs:
            any_hit += 1
        if len(hit_docs) == len(true_docs) and true_docs:
            both_hit += 1
        total_unique_candidates += len(cands)
        true_candidates += sum(1 for d in cands if doc_target.get(d) == inc)
        per_incident[inc] = {
            "retrieved_true_docs": len(hit_docs),
            "n_unique_candidates": len(cands),
            "retrieved_noise_docs": len(cands & noise),
            "recall_i": round(recall_i, 4),
        }

    relevant_doc_recall = retrieved_true / (n_inc * C.RELEVANT_DOCS_PER_INCIDENT) if n_inc else 0.0
    mean_recall = recall_sum / n_inc if n_inc else 0.0
    candidate_precision = true_candidates / total_unique_candidates if total_unique_candidates else 0.0

    # ---- Relevance filter (Section 21) ----
    accepted = [d for d in relevance_rows if d.get("accepted")]
    n_accepted = len(accepted)
    accepted_true = 0
    accepted_noise = 0
    accepted_wrong_target = 0
    for d in accepted:
        did = d["doc_id"]
        if did in noise:
            accepted_noise += 1
        elif doc_target.get(did) == d["incident_id"]:
            accepted_true += 1
        else:
            accepted_wrong_target += 1

    # retrieved true supplements available to accept
    retrieved_true_docs_set = set()
    for inc in incidents:
        retrieved_true_docs_set |= (cand_by_inc.get(inc, set()) & target_supps.get(inc, set()))
    acceptance_precision = accepted_true / n_accepted if n_accepted else 0.0
    acceptance_recall = accepted_true / len(retrieved_true_docs_set) if retrieved_true_docs_set else 0.0
    false_acceptance_rate = (accepted_noise + accepted_wrong_target) / n_accepted if n_accepted else 0.0

    # Non-incremental related rejection accuracy: of same-incident candidates that were
    # rejected, how many were correctly rejected (not a true supplement for this target).
    same_incident_rejected = [
        d for d in relevance_rows if d.get("same_incident") and not d.get("accepted")
    ]
    correct_rejections = sum(
        1 for d in same_incident_rejected if doc_target.get(d["doc_id"]) != d["incident_id"]
    )
    non_incr_rejection_accuracy = (
        correct_rejections / len(same_incident_rejected) if same_incident_rejected else 1.0
    )

    # ---- Query quality (Section 22) ----
    n_queries = len(queries_rows)
    query_success = 0
    # map (incident, query_id) -> retrieved doc
    retr_by_q = {(r["incident_id"], r["query_id"]): r["retrieved_doc_id"] for r in retrieval_rows}
    for q in queries_rows:
        did = retr_by_q.get((q["incident_id"], q["query_id"]))
        if did and did in target_supps.get(q["incident_id"], set()):
            query_success += 1
    query_success_rate = query_success / n_queries if n_queries else 0.0

    # privacy leakage: final queries that are not privacy-clean
    leaked_final = sum(1 for q in queries_rows if not q.get("privacy_clean", True))
    query_privacy_leakage_rate = leaked_final / n_queries if n_queries else 0.0

    # gap diversity per incident
    gap_div: dict[str, int] = {}
    for q in queries_rows:
        gap_div.setdefault(q["incident_id"], set()).update(q.get("gap_types") or [])  # type: ignore
    gap_diversity = {k: len(v) for k, v in gap_div.items()}  # type: ignore
    mean_gap_diversity = sum(gap_diversity.values()) / len(gap_diversity) if gap_diversity else 0.0

    aggregate = {
        "n_incidents": n_inc,
        "relevant_doc_recall": round(relevant_doc_recall, 4),
        "retrieved_true_docs": retrieved_true,
        "relevant_docs_total": n_inc * C.RELEVANT_DOCS_PER_INCIDENT,
        "mean_per_incident_recall": round(mean_recall, 4),
        "any_hit_rate": f"{any_hit}/{n_inc}",
        "both_hit_rate": f"{both_hit}/{n_inc}",
        "raw_candidate_precision": round(candidate_precision, 4),
        "acceptance_precision": round(acceptance_precision, 4),
        "acceptance_recall": round(acceptance_recall, 4),
        "false_acceptance_rate": round(false_acceptance_rate, 4),
        "noise_acceptance_count": accepted_noise,
        "accepted_wrong_target_count": accepted_wrong_target,
        "non_incremental_related_rejection_accuracy": round(non_incr_rejection_accuracy, 4),
        "n_accepted": n_accepted,
        "query_success_rate": round(query_success_rate, 4),
        "query_successes": query_success,
        "n_queries": n_queries,
        "query_privacy_leakage_rate": round(query_privacy_leakage_rate, 4),
        "query_privacy_leaks": leaked_final,
        "mean_gap_diversity": round(mean_gap_diversity, 4),
    }
    return {"aggregate": aggregate, "per_incident": per_incident, "gap_diversity": gap_diversity}


def run() -> dict:
    res = compute()
    print("Retrieval metrics:")
    for k, v in res["aggregate"].items():
        print(f"  {k}: {v}")
    return res


if __name__ == "__main__":
    run()
