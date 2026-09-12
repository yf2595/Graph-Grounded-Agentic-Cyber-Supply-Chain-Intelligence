"""Build markdown tables from aggregate metrics (WAITI revision)."""
from __future__ import annotations

import json
from pathlib import Path


def _f(x) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_markdown_tables(run_dir: Path, aggregates: dict) -> None:
    lines = ["# WAITI Evaluation Report", ""]

    lines.append("## Main comparison")
    lines.append("")
    lines.append(
        "| Method | Cyber F1 | Cond. Rel F1 | Supplier F1 | Actionable F1 | Merge Pair F1 | Merge B3 F1 | "
        "Risk Macro-F1 E2E | Risk MAE | Risk QWK det | Coverage | Group Macro-F1 | Median lat (ms) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mid, agg in aggregates.items():
        lat = agg.get("latency") or {}
        rg = agg.get("risk_group") or {}
        re2 = agg.get("risk_e2e") or {}
        lines.append(
            "| {m} | {c} | {r} | {s} | {a} | {mp} | {mb} | {rm} | {mae} | {qwk} | {cov} | {gm} | {lat} |".format(
                m=mid,
                c=_f((agg.get("cyber") or {}).get("f1")),
                r=_f((agg.get("relevance_conditional") or agg.get("relevance") or {}).get("f1")),
                s=_f((agg.get("supplier_actionable") or {}).get("micro_f1")),
                a=_f((agg.get("actionable") or {}).get("f1")),
                mp=_f((agg.get("merge_pairwise") or {}).get("f1")),
                mb=_f((agg.get("merge_b3") or {}).get("f1")),
                rm=_f(re2.get("macro_f1")),
                mae=_f(re2.get("mae")),
                qwk=_f(re2.get("risk_qwk_detected") or re2.get("quadratic_weighted_kappa")),
                cov=_f(re2.get("coverage")),
                gm=_f(rg.get("risk_group_macro_f1_e2e")),
                lat=_f(lat.get("median_latency_ms")),
            )
        )

    lines.append("")
    lines.append("## Relevance diagnostics")
    lines.append("")
    lines.append("| Method | Cond. Rel P | Cond. Rel R | Cond. Rel F1 | Full Rel P | Full Rel R | Full Rel F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mid, agg in aggregates.items():
        rc = agg.get("relevance_conditional") or {}
        rf = agg.get("relevance_full") or {}
        lines.append(
            f"| {mid} | {_f(rc.get('precision'))} | {_f(rc.get('recall'))} | {_f(rc.get('f1'))} | "
            f"{_f(rf.get('precision'))} | {_f(rf.get('recall'))} | {_f(rf.get('f1'))} |"
        )

    lines.append("")
    lines.append("## Ablation view (OURS + A1-A4)")
    lines.append("")
    lines.append("| Method | Actionable F1 | Risk Macro-F1 | Risk QWK | Coverage | Merge B3 F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mid in ["OURS", "A1", "A2", "A3", "A4"]:
        if mid not in aggregates:
            continue
        agg = aggregates[mid]
        re2 = agg.get("risk_e2e") or {}
        lines.append(
            f"| {mid} | {_f((agg.get('actionable') or {}).get('f1'))} | "
            f"{_f(re2.get('macro_f1'))} | "
            f"{_f(re2.get('risk_qwk_detected') or re2.get('quadratic_weighted_kappa'))} | "
            f"{_f(re2.get('coverage'))} | "
            f"{_f((agg.get('merge_b3') or {}).get('f1'))} |"
        )

    lines.append("")
    lines.append("## Detection diagnostics")
    lines.append("")
    lines.append("| Method | Cyber P | Cyber R | Cond Rel P | Cond Rel R | Actionable P | Actionable R |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mid, agg in aggregates.items():
        c = agg.get("cyber") or {}
        r = agg.get("relevance_conditional") or agg.get("relevance") or {}
        a = agg.get("actionable") or {}
        lines.append(
            f"| {mid} | {_f(c.get('precision'))} | {_f(c.get('recall'))} | "
            f"{_f(r.get('precision'))} | {_f(r.get('recall'))} | "
            f"{_f(a.get('precision'))} | {_f(a.get('recall'))} |"
        )

    lines.append("")
    lines.append("## Latency & cost")
    lines.append("")
    lines.append(
        "| Method | Mean lat (ms) | Std lat | Median | P95 | Mean cost ($) | Std cost | Total cost ($) | Mean tokens in/out |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mid, agg in aggregates.items():
        lat = agg.get("latency") or {}
        tin = lat.get("mean_input_tokens")
        tout = lat.get("mean_output_tokens")
        tok = f"{_f(tin)}/{_f(tout)}"
        lines.append(
            "| {m} | {mean} | {std} | {med} | {p95} | {mc} | {sc} | {tc} | {tok} |".format(
                m=mid,
                mean=_f(lat.get("mean_latency_ms")),
                std=_f(lat.get("std_latency_ms")),
                med=_f(lat.get("median_latency_ms")),
                p95=_f(lat.get("p95_latency_ms")),
                mc=_f(lat.get("mean_cost_usd")),
                sc=_f(lat.get("std_cost_usd")),
                tc=_f(lat.get("total_cost_usd")),
                tok=tok,
            )
        )

    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # strip bulky group_rows for raw dump
    slim = {}
    for mid, agg in aggregates.items():
        slim[mid] = {k: v for k, v in agg.items() if k != "group_rows"}
    (run_dir / "aggregates_raw.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
