"""Freeze WAITI paper tables from chosen run IDs (P8).

Default fairness policy (plan):
  - Baselines B0–B3: results/waiti_metrics_v2 (full1 stack, revised metrics)
  - OURS + A1/A3/A4: results/waiti_merge_v2 (Agent2 ER + merger v2)
  - A2: waiti_metrics_v2 (not re-run under merge_v2)
  - Faithfulness: waiti_faithfulness_v1
  - Local: placeholder until GPU/vLLM runs complete
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(x, digits: int = 3) -> str:
    if x is None or x == "":
        return "—"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _by_method(rows: list[dict]) -> dict[str, dict]:
    return {r["method_id"]: r for r in rows if r.get("method_id")}


def compose_main_rows(
    metrics_v2: dict[str, dict],
    merge_v2: dict[str, dict],
) -> list[tuple[str, dict, str]]:
    """Return (method, row, source_label)."""
    out = []
    for mid in ["B0", "B1", "B2", "B3"]:
        if mid in metrics_v2:
            out.append((mid, metrics_v2[mid], "metrics_v2 / full1 ER+merge"))
    if "OURS" in merge_v2:
        out.append(("OURS", merge_v2["OURS"], "merge_v2 (Agent2 ER + merger v2)"))
    return out


def write_main(
    path: Path,
    rows: list[tuple[str, dict, str]],
    *,
    version_note: str | None = None,
) -> None:
    note = version_note or (
        "**Version note:** Baselines B0–B3 are from `waiti_metrics_v2` (predictions from `waiti_full1`, "
        "re-aggregated with revised metrics). OURS is from `waiti_merge_v2` (improved Agent-2 entity "
        "resolution + merger v2). ER/merge stacks therefore differ; do not interpret baseline–OURS "
        "gaps as pure architecture deltas on an identical merger."
    )
    lines = [
        "# Main results",
        "",
        "**Corpus:** 129 reports (`eval_corpus_v1`).",
        "",
        note,
        "",
        "| Method | Cyber F1 | Cond. Rel F1 | Supplier μF1 | Actionable F1 | Merge Pair F1 | Merge B³ F1 | "
        "Risk Macro-F1 | Risk MAE | Risk QWK (det.) | Coverage | Group Macro-F1 | Spearman ρ | nDCG@10 | "
        "High-risk recall | Dup-alert ↓ | Median lat (ms) | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mid, r, src in rows:
        lines.append(
            "| {m} | {c} | {cr} | {s} | {a} | {mp} | {mb} | {rm} | {mae} | {qwk} | {cov} | {gm} | "
            "{sp} | {nd} | {hr} | {dup} | {lat} | {src} |".format(
                m=mid,
                c=_f(r.get("cyber_f1")),
                cr=_f(r.get("relevance_conditional_f1")),
                s=_f(r.get("supplier_micro_f1")),
                a=_f(r.get("actionable_f1")),
                mp=_f(r.get("merge_pairwise_f1")),
                mb=_f(r.get("merge_b3_f1")),
                rm=_f(r.get("risk_e2e_macro_f1")),
                mae=_f(r.get("risk_e2e_mae")),
                qwk=_f(r.get("risk_qwk_detected")),
                cov=_f(r.get("risk_coverage")),
                gm=_f(r.get("risk_group_macro_f1_e2e")),
                sp=_f(r.get("spearman_rho")),
                nd=_f(r.get("ndcg_at_10")),
                hr=_f(r.get("high_risk_recall")),
                dup=_f(r.get("duplicate_alert_reduction")),
                lat=_f(r.get("median_latency_ms"), 1),
                src=src,
            )
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation(
    path: Path,
    merge_v2: dict[str, dict],
    metrics_v2: dict[str, dict],
    *,
    caption: str | None = None,
    source_label: str | None = None,
    notes: str | None = None,
) -> None:
    lines = [
        "# Ablation results",
        "",
        caption
        or (
            "OURS / A1 / A3 / A4 from `waiti_merge_v2`. A2 (no merger) from `waiti_metrics_v2` "
            "(not re-run under merger v2)."
        ),
        "",
        "| Method | Actionable F1 | Supplier μF1 | Merge B³ F1 | Risk Macro-F1 | Risk QWK | Coverage | "
        "Group Macro-F1 | Spearman ρ | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mid in ["OURS", "A1", "A2", "A3", "A4"]:
        if mid in merge_v2:
            r = merge_v2[mid]
            src = source_label or "merge_v2"
        elif mid in metrics_v2:
            r = metrics_v2[mid]
            src = source_label or "metrics_v2"
        else:
            continue
        lines.append(
            "| {m} | {a} | {s} | {mb} | {rm} | {qwk} | {cov} | {gm} | {sp} | {src} |".format(
                m=mid,
                a=_f(r.get("actionable_f1")),
                s=_f(r.get("supplier_micro_f1")),
                mb=_f(r.get("merge_b3_f1")),
                rm=_f(r.get("risk_e2e_macro_f1")),
                qwk=_f(r.get("risk_qwk_detected")),
                cov=_f(r.get("risk_coverage")),
                gm=_f(r.get("risk_group_macro_f1_e2e")),
                sp=_f(r.get("spearman_rho")),
                src=src,
            )
        )
    lines.append("")
    lines.append(
        notes
        or (
            "Directional notes (merge_v2): A1 (no graph) and A3 (unified risk) reduce risk QWK vs OURS; "
            "actionable F1 stays near OURS."
        )
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trustworthiness(
    path: Path,
    faith_rows: list[dict],
    merge_v2: dict[str, dict],
    *,
    run_id: str | None = None,
) -> None:
    faith = _by_method(faith_rows)
    rid = run_id or "waiti_faithfulness_v1"
    ours_src = run_id or "waiti_merge_v2"
    b2_src = run_id or "waiti_full1"
    lines = [
        "# Trustworthiness / report faithfulness",
        "",
        "Auto-labeled claim faithfulness on gold-actionable articles "
        f"(`{rid}`). OURS reports regenerated with structured Agent 8 "
        f"(deterministic fields + LLM prose only) from `{ours_src}` stages; "
        f"B2 claims extracted from `{b2_src}` baseline reports.",
        "",
        "| Method | Articles | Claims | Support rate | Hallucination rate | Contradiction rate | "
        "Provenance cov. | Req. field cov. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mid in ["OURS", "B2"]:
        r = faith.get(mid) or {}
        lines.append(
            "| {m} | {n} | {nc} | {sr} | {hr} | {cr} | {pc} | {rfc} |".format(
                m=mid,
                n=_f(r.get("n_articles"), 0),
                nc=_f(r.get("n_claims"), 0),
                sr=_f(r.get("support_rate")),
                hr=_f(r.get("hallucination_rate")),
                cr=_f(r.get("contradiction_rate")),
                pc=_f(r.get("provenance_coverage")),
                rfc=_f(r.get("mean_required_field_coverage") or r.get("required_field_coverage")),
            )
        )
    ours = merge_v2.get("OURS") or {}
    lines.extend(
        [
            "",
            f"## Ops / ranking (OURS, `{ours_src}`)",
            "",
            f"- Spearman ρ: {_f(ours.get('spearman_rho'))}",
            f"- nDCG@10 / @20: {_f(ours.get('ndcg_at_10'))} / {_f(ours.get('ndcg_at_20'))}",
            f"- High-risk recall: {_f(ours.get('high_risk_recall'))}",
            f"- Very-high-risk recall: {_f(ours.get('very_high_risk_recall'))}",
            f"- Severe under-estimation rate: {_f(ours.get('severe_underestimation_rate'))}",
            f"- Duplicate-alert reduction: {_f(ours.get('duplicate_alert_reduction'))}",
            "",
            f"Human audit template: `results/{rid}/human_report_audit_template.csv` "
            "(target ≥30 stratified reports; Cohen's κ if dual-annotated).",
            "",
            "Paper caveat: Agent-2 ER improvements were informed by error analysis on this 129-report "
            "corpus (no blind holdout).",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_local(path: Path, merge_v2: dict[str, dict], local_dirs: list[Path]) -> None:
    ours = merge_v2.get("OURS") or {}
    lines = [
        "# Local model portability",
        "",
        "Serving configs: `evaluation/config/local_models.yaml`. "
        "Method IDs: `OURS_QWEN3_14B`, `OURS_GEMMA3_12B`, optional `B2_QWEN3_14B`.",
        "",
        "GPT-5.4 OURS reference (`waiti_merge_v2`) vs local runs (fill when vLLM serves):",
        "",
        "| Stack | Supplier μF1 | Actionable F1 | Merge B³ F1 | Risk Group Macro-F1 | "
        "Median lat (ms) | Peak VRAM | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        "| OURS + GPT-5.4 | {s} | {a} | {mb} | {gm} | {lat} | n/a (API) | complete |".format(
            s=_f(ours.get("supplier_micro_f1")),
            a=_f(ours.get("actionable_f1")),
            mb=_f(ours.get("merge_b3_f1")),
            gm=_f(ours.get("risk_group_macro_f1_e2e")),
            lat=_f(ours.get("median_latency_ms"), 1),
        ),
    ]
    for d in local_dirs:
        status = "pending"
        if (d / "STATUS.md").exists():
            status = "endpoint_unavailable (config ready)"
        agg = _by_method(_read_csv(d / "aggregate_metrics.csv"))
        # pick first method row if any
        row = next(iter(agg.values()), {})
        rec = {}
        if (d / "serving_record.json").exists():
            rec = json.loads((d / "serving_record.json").read_text(encoding="utf-8"))
        name = rec.get("model_name") or d.name
        lines.append(
            "| OURS + {name} | {s} | {a} | {mb} | {gm} | {lat} | — | {st} |".format(
                name=name,
                s=_f(row.get("supplier_micro_f1")),
                a=_f(row.get("actionable_f1")),
                mb=_f(row.get("merge_b3_f1")),
                gm=_f(row.get("risk_group_macro_f1_e2e")),
                lat=_f(row.get("median_latency_ms"), 1),
                st=status,
            )
        )
    lines.extend(
        [
            "",
            "Start serving: `bash scripts/serve_local_vllm.sh qwen3_14b` "
            "(or `gemma3_12b`), then "
            "`python -m evaluation.runners.run_local_models --run-id waiti_local_qwen3_14b`.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/paper_tables")
    parser.add_argument("--metrics-v2", default="results/waiti_metrics_v2")
    parser.add_argument("--merge-v2", default="results/waiti_merge_v2")
    parser.add_argument("--faithfulness", default="results/waiti_faithfulness_v1")
    args = parser.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    m2 = _by_method(_read_csv(ROOT / args.metrics_v2 / "aggregate_metrics.csv"))
    mv = _by_method(_read_csv(ROOT / args.merge_v2 / "aggregate_metrics.csv"))
    faith = _read_csv(ROOT / args.faithfulness / "faithfulness_metrics.csv")

    write_main(out / "main_results.md", compose_main_rows(m2, mv))
    write_ablation(out / "ablation_results.md", mv, m2)
    write_trustworthiness(out / "trustworthiness_results.md", faith, mv)
    local_dirs = [
        ROOT / "results" / "waiti_local_qwen3_14b",
        ROOT / "results" / "waiti_local_gemma3_12b",
        ROOT / "results" / "waiti_local_qwen3_14b_b2",
    ]
    write_local(out / "local_model_results.md", mv, local_dirs)

    meta = {
        "baselines_run": args.metrics_v2,
        "ours_run": args.merge_v2,
        "faithfulness_run": args.faithfulness,
        "policy": "OURS/A1/A3/A4 from merge_v2; B0-B3 and A2 from metrics_v2; versions labeled in captions",
    }
    (out / "freeze_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote paper tables to {out}")


if __name__ == "__main__":
    main()
