"""Evaluation Part 4-5 + reporting - Agent 5 and final pipeline benefit.

Consumes the paired A4-OFF/A4-ON artifacts and writes the master analysis table,
aggregate metrics, the paper-facing report, a failure analysis, and the
reproducibility manifest (Sections 23-29, 33).
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone

from src.query_expansion import config as C
from src.query_expansion import evaluation_retrieval as er

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "were", "was", "has",
    "have", "had", "into", "after", "their", "there", "which", "while", "been",
    "about", "would", "could", "these", "those", "some", "than", "then",
    "incident", "company", "report", "reports", "reported", "cyber", "cyberattack",
}


def _load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _by_incident(rows: list[dict], key: str = "incident_id") -> dict[str, dict]:
    return {r[key]: r for r in rows}


def _risk_index(risk: str | None) -> int | None:
    if not risk or risk == "UNKNOWN":
        return None
    try:
        return C.RISK_ORDER.index(risk)
    except ValueError:
        return None


def _sign(delta: int | None) -> int | None:
    if delta is None:
        return None
    return (delta > 0) - (delta < 0)


def _effect_to_sign(effect: str) -> int:
    return {"increase": 1, "decrease": -1, "no_ordinal_change": 0, "same": 0}.get(effect, 0)


def _salient(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 5 and t not in _STOP}


def _fact_covered(fact: str, target_text: str) -> bool:
    toks = _salient(fact)
    if not toks:
        return False
    present = _salient(target_text)
    hit = len(toks & present)
    return hit / len(toks) >= 0.5


def _assessment_text(assessment: dict | None) -> str:
    if not assessment:
        return ""
    parts = [
        assessment.get("main_effect", ""),
        assessment.get("assessment_rationale", "") or "",
        " ".join(assessment.get("secondary_effects", []) or []),
    ]
    return " ".join(str(p) for p in parts)


def compute() -> dict:
    retrieval = er.compute()

    a5_before = _by_incident(_load_jsonl(C.AGENT5_BEFORE_JSONL))
    a5_after = _by_incident(_load_jsonl(C.AGENT5_AFTER_JSONL))
    fin_before = _by_incident(_load_jsonl(C.FINAL_BEFORE_JSONL))
    fin_after = _by_incident(_load_jsonl(C.FINAL_AFTER_JSONL))
    enriched = _by_incident(_load_jsonl(C.ENRICHED_INCIDENTS_JSONL))
    ref = _by_incident(_load_jsonl(C.ENRICHED_REFERENCE_LABELS_JSONL))

    # case-level expected effect from supplement ground truth
    effect_by_inc: dict[str, str] = {}
    for g in _load_jsonl(C.GROUND_TRUTH_OUT_JSONL):
        if not g.get("is_noise") and g.get("target_incident_id"):
            effect_by_inc.setdefault(g["target_incident_id"], g.get("expected_agent5_effect", "no_ordinal_change"))

    incidents = sorted(a5_after) or sorted(effect_by_inc)

    rows: list[dict] = []
    cov_before_sum = cov_after_sum = cov_n = 0.0
    cons_unchanged = cons_up = cons_down = 0
    direction_agree = 0
    before_correct = after_correct = 0
    mae_before = mae_after = 0.0
    mae_n = 0
    correct_direction_final = 0
    stable_when_should = 0
    should_change = should_stay = 0

    for inc in incidents:
        ia_b = (a5_before.get(inc) or {}).get("assessment")
        ia_a = (a5_after.get(inc) or {}).get("assessment")
        sev_b = ia_b.get("severity") if ia_b else None
        sev_a = ia_a.get("severity") if ia_a else None
        fr_b = (fin_before.get(inc) or {}).get("final_risk") or {}
        fr_a = (fin_after.get(inc) or {}).get("final_risk") or {}
        risk_b = fr_b.get("overall_risk")
        risk_a = fr_a.get("overall_risk")
        r = ref.get(inc, {})
        ref_enriched = r.get("enriched_reference_risk")
        ref_original = r.get("original_reference_risk")
        expected_effect = effect_by_inc.get(inc, "no_ordinal_change")

        # 23.1 coverage
        accepted = (enriched.get(inc) or {}).get("accepted") or []
        facts = [ev.get("fact", "") for a in accepted for ev in a.get("new_evidence", [])]
        if facts:
            txt_b = _assessment_text(ia_b)
            txt_a = _assessment_text(ia_a)
            cov_b = sum(1 for f in facts if _fact_covered(f, txt_b)) / len(facts)
            cov_a = sum(1 for f in facts if _fact_covered(f, txt_a)) / len(facts)
            cov_before_sum += cov_b
            cov_after_sum += cov_a
            cov_n += 1
        else:
            cov_b = cov_a = 0.0

        # 23.2 consequence change
        di_sev = None
        if _risk_index(sev_b) is not None and _risk_index(sev_a) is not None:
            di_sev = _risk_index(sev_a) - _risk_index(sev_b)
            if di_sev == 0:
                cons_unchanged += 1
            elif di_sev > 0:
                cons_up += 1
            else:
                cons_down += 1

        # 23.3 expected-direction agreement
        observed_sign = _sign(di_sev)
        if observed_sign is not None and observed_sign == _effect_to_sign(expected_effect):
            direction_agree += 1

        # 24.1 final risk accuracy vs enriched reference
        if ref_enriched:
            if risk_b == ref_enriched:
                before_correct += 1
            if risk_a == ref_enriched:
                after_correct += 1
            bi, ai, ri = _risk_index(risk_b), _risk_index(risk_a), _risk_index(ref_enriched)
            if ri is not None:
                if bi is not None:
                    mae_before += abs(bi - ri)
                if ai is not None:
                    mae_after += abs(ai - ri)
                mae_n += 1

        # 24.2 correct-direction final change
        if ref_enriched and ref_original:
            ref_delta = _sign(
                (_risk_index(ref_enriched) or 0) - (_risk_index(ref_original) or 0)
            )
            fin_delta = _sign(
                (_risk_index(risk_a) if _risk_index(risk_a) is not None else 0)
                - (_risk_index(risk_b) if _risk_index(risk_b) is not None else 0)
            )
            if ref_delta != 0:
                should_change += 1
                if fin_delta == ref_delta:
                    correct_direction_final += 1
            else:
                should_stay += 1
                if risk_a == risk_b:
                    stable_when_should += 1

        pinc = retrieval["per_incident"].get(inc, {})
        rows.append(
            {
                "incident_id": inc,
                "retrieved_true_docs": pinc.get("retrieved_true_docs", 0),
                "retrieved_noise_docs": pinc.get("retrieved_noise_docs", 0),
                "n_unique_candidates": pinc.get("n_unique_candidates", 0),
                "accepted_true_docs": len(
                    [a for a in accepted if a.get("doc_id", "").startswith(f"QE_{inc}_SUPP")]
                ),
                "n_accepted": len(accepted),
                "query_successes": 0,  # filled in _write_master_table
                "agent5_severity_before": sev_b,
                "agent5_severity_after": sev_a,
                "coverage_before": round(cov_b, 3),
                "coverage_after": round(cov_a, 3),
                "agent5_direction_expected": expected_effect,
                "agent5_direction_observed": {1: "increase", -1: "decrease", 0: "no_ordinal_change"}.get(
                    observed_sign, "unknown"
                ),
                "final_risk_before": risk_b,
                "final_risk_after": risk_a,
                "reference_risk_original": ref_original,
                "reference_risk_enriched": ref_enriched,
                "a4_changed": (enriched.get(inc) or {}).get("changed", False),
            }
        )

    n = len(incidents)
    aggregate = {
        "n_incidents": n,
        "agent5_evidence_coverage_before": round(cov_before_sum / cov_n, 4) if cov_n else 0.0,
        "agent5_evidence_coverage_after": round(cov_after_sum / cov_n, 4) if cov_n else 0.0,
        "agent5_evidence_coverage_delta": round((cov_after_sum - cov_before_sum) / cov_n, 4) if cov_n else 0.0,
        "consequence_unchanged": cons_unchanged,
        "consequence_increased": cons_up,
        "consequence_decreased": cons_down,
        "direction_agreement": f"{direction_agree}/{n}",
        "final_risk_accuracy_before": round(before_correct / n, 4) if n else 0.0,
        "final_risk_accuracy_after": round(after_correct / n, 4) if n else 0.0,
        "final_risk_accuracy_delta": round((after_correct - before_correct) / n, 4) if n else 0.0,
        "final_risk_mae_before": round(mae_before / mae_n, 4) if mae_n else None,
        "final_risk_mae_after": round(mae_after / mae_n, 4) if mae_n else None,
        "correct_direction_final_changes": f"{correct_direction_final}/{should_change}" if should_change else "0/0",
        "stable_when_should_not_change": f"{stable_when_should}/{should_stay}" if should_stay else "0/0",
    }
    return {"aggregate": aggregate, "rows": rows, "retrieval": retrieval}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(C.ROOT), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _write_master_table(rows: list[dict], query_success_by_inc: dict[str, int]) -> None:
    for r in rows:
        r["query_successes"] = query_success_by_inc.get(r["incident_id"], 0)
    fields = list(rows[0].keys()) if rows else ["incident_id"]
    with C.PER_INCIDENT_RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _query_success_by_incident() -> dict[str, int]:
    retrieval_rows = _load_jsonl(C.RETRIEVAL_RAW_JSONL)
    gts = _load_jsonl(C.GROUND_TRUTH_OUT_JSONL)
    target_supps: dict[str, set[str]] = {}
    for g in gts:
        if not g.get("is_noise") and g.get("target_incident_id"):
            target_supps.setdefault(g["target_incident_id"], set()).add(g["doc_id"])
    out: dict[str, int] = {}
    for r in retrieval_rows:
        inc = r["incident_id"]
        if r["retrieved_doc_id"] in target_supps.get(inc, set()):
            out[inc] = out.get(inc, 0) + 1
    return out


def _write_report(res: dict) -> None:
    agg = res["aggregate"]
    ragg = res["retrieval"]["aggregate"]
    lines = [
        "# Query-Expansion (Agent 4) Closed-Corpus Experiment",
        "",
        "Controlled closed-corpus enrichment experiment on 20 actionable incidents "
        "(40 relevant supplements + 20 aerospace/defense noise = 60 indexed documents).",
        "This is NOT a Web-search benchmark and is separate from the 129-report main results.",
        "",
        "## Paper-facing metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Any-hit incident retrieval | {ragg['any_hit_rate']} |",
        f"| Both supplemental reports retrieved | {ragg['both_hit_rate']} |",
        f"| Relevant-document recall | {ragg['retrieved_true_docs']}/{ragg['relevant_docs_total']} "
        f"({ragg['relevant_doc_recall']}) |",
        f"| Accepted-document precision | {ragg['acceptance_precision']} |",
        f"| Query privacy leakage | {ragg['query_privacy_leaks']}/{ragg['n_queries']} |",
        f"| Agent-5 evidence coverage | {agg['agent5_evidence_coverage_before']} -> "
        f"{agg['agent5_evidence_coverage_after']} |",
        f"| Agent-5 direction agreement | {agg['direction_agreement']} |",
        f"| Final enriched-risk accuracy | {agg['final_risk_accuracy_before']} -> "
        f"{agg['final_risk_accuracy_after']} |",
        "",
        "## Retrieval and query quality",
        "",
        f"- Mean per-incident recall: {ragg['mean_per_incident_recall']}",
        f"- Raw candidate precision: {ragg['raw_candidate_precision']}",
        f"- Query success rate: {ragg['query_successes']}/{ragg['n_queries']} "
        f"({ragg['query_success_rate']})",
        f"- Acceptance recall: {ragg['acceptance_recall']}",
        f"- False acceptance rate: {ragg['false_acceptance_rate']} "
        f"(noise accepted: {ragg['noise_acceptance_count']})",
        f"- Mean gap diversity: {ragg['mean_gap_diversity']}",
        "",
        "## Downstream benefit",
        "",
        f"- Consequence change: {agg['consequence_increased']} up / "
        f"{agg['consequence_decreased']} down / {agg['consequence_unchanged']} unchanged",
        f"- Final risk MAE (vs enriched reference): {agg['final_risk_mae_before']} -> "
        f"{agg['final_risk_mae_after']}",
        f"- Correct-direction final changes: {agg['correct_direction_final_changes']}",
        f"- Stable when reference unchanged: {agg['stable_when_should_not_change']}",
        "",
        "Note: N=20; report paired counts and exact proportions. QWK is not reported as a "
        "headline result. Enriched reference labels use GPT-5.4 (reasoning=medium) as an "
        "independent-effort labeler (weak independence).",
        "",
    ]
    C.EXPERIMENT_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failure_analysis(res: dict) -> None:
    rows = res["rows"]
    relevance = _load_jsonl(C.RELEVANCE_CHECKS_JSONL)
    gts = {
        g["doc_id"]: g
        for g in _load_jsonl(C.GROUND_TRUTH_OUT_JSONL)
    }

    def is_true_supp(inc, did):
        g = gts.get(did, {})
        return (not g.get("is_noise")) and g.get("target_incident_id") == inc

    lines = ["# Query-Expansion Failure Analysis", ""]

    # 1 relevant supplement never retrieved
    lines.append("## 1. Relevant supplement never retrieved")
    for r in rows:
        if r["retrieved_true_docs"] < 2:
            lines.append(f"- {r['incident_id']}: retrieved {r['retrieved_true_docs']}/2 true supplements")

    # 2 correct supplement retrieved but rejected
    lines.append("\n## 2. Correct supplement retrieved but rejected")
    for d in relevance:
        if is_true_supp(d["incident_id"], d["doc_id"]) and not d.get("accepted"):
            lines.append(f"- {d['incident_id']} {d['doc_id']}: reason same_incident={d.get('same_incident')}, "
                         f"adds_new={d.get('adds_new_information')}")

    # 3 noise accepted
    lines.append("\n## 3. Noise document accepted")
    for d in relevance:
        if gts.get(d["doc_id"], {}).get("is_noise") and d.get("accepted"):
            lines.append(f"- {d['incident_id']} accepted noise {d['doc_id']}")

    # 5/6 evidence accepted but Agent 5 ignores / wrong direction
    lines.append("\n## 5-6. Evidence accepted but Agent 5 coverage low or wrong direction")
    for r in rows:
        if r["n_accepted"] > 0 and r["coverage_after"] < 0.5:
            lines.append(f"- {r['incident_id']}: coverage_after={r['coverage_after']} despite {r['n_accepted']} accepted")
        if r["agent5_direction_observed"] != r["agent5_direction_expected"]:
            lines.append(
                f"- {r['incident_id']}: expected {r['agent5_direction_expected']}, "
                f"observed {r['agent5_direction_observed']}"
            )

    # 7 Agent5 improves but final risk does not / 8 unsupported claims (best-effort)
    lines.append("\n## 7. Agent 5 changed but final risk unchanged")
    for r in rows:
        sev_changed = r["agent5_severity_before"] != r["agent5_severity_after"]
        risk_changed = r["final_risk_before"] != r["final_risk_after"]
        if sev_changed and not risk_changed:
            lines.append(f"- {r['incident_id']}: severity {r['agent5_severity_before']}->"
                         f"{r['agent5_severity_after']} but risk stayed {r['final_risk_before']}")

    C.FAILURE_ANALYSIS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(res: dict) -> None:
    index_manifest = {}
    if C.INDEX_MANIFEST_JSON.exists():
        index_manifest = json.loads(C.INDEX_MANIFEST_JSON.read_text(encoding="utf-8"))
    selected = json.loads(C.SELECTED_INCIDENTS_JSON.read_text(encoding="utf-8"))
    manifest = {
        "run_id": "waiti_qe_gpt54",
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_incident_ids": [c["incident_id"] for c in selected["incidents"]],
        "selection_seed": C.SELECTION_SEED,
        "generation_model": C.GEN_MODEL,
        "generation_reasoning_effort": C.GEN_REASONING_EFFORT,
        "generation_seed": C.GENERATION_SEED,
        "agent4_model": C.AGENT4_MODEL,
        "agent4_reasoning_effort": C.AGENT4_REASONING_EFFORT,
        "downstream_model": C.DOWNSTREAM_MODEL,
        "downstream_reasoning_effort": C.DOWNSTREAM_REASONING_EFFORT,
        "downstream_temperature": C.DOWNSTREAM_TEMPERATURE,
        "downstream_max_output_tokens": C.DOWNSTREAM_MAX_OUTPUT_TOKENS,
        "reference_model": C.REFERENCE_MODEL,
        "reference_reasoning_effort": C.REFERENCE_REASONING_EFFORT,
        "embedding_model": index_manifest.get("embedding_model"),
        "embedding_dim": index_manifest.get("dimensionality"),
        "retrieval_top_k": C.RETRIEVAL_TOP_K,
        "queries_per_incident": C.QUERIES_PER_INCIDENT,
        "corpus_hash": index_manifest.get("corpus_hash"),
        "compared_baseline": "results/waiti_final_gpt54 (A4-OFF replicates its frozen config)",
    }
    C.REPRODUCIBILITY_MANIFEST_JSON.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run() -> dict:
    res = compute()
    C.ensure_artifacts_dir()
    qsucc = _query_success_by_incident()
    _write_master_table(res["rows"], qsucc)
    C.AGGREGATE_METRICS_JSON.write_text(
        json.dumps(
            {"retrieval": res["retrieval"]["aggregate"], "downstream": res["aggregate"]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(res)
    _write_failure_analysis(res)
    _write_manifest(res)
    print("Downstream metrics:")
    for k, v in res["aggregate"].items():
        print(f"  {k}: {v}")
    return res["aggregate"]


if __name__ == "__main__":
    run()
