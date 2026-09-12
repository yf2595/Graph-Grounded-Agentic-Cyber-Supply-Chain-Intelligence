"""Faithfulness evaluation runner for OURS Agent-8 reports and B2 claims.

Reads saved per-article outputs from existing runs and writes metrics under
results/<run-id>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics.b2_claims import claims_from_baseline
from evaluation.metrics.faithfulness import build_evidence_corpus, label_claims
from evaluation.metrics.faithfulness_aggregate import faithfulness_metrics
from evaluation.schemas.prediction import GroundTruthRecord
from src.agents.agent_08_report_generator import run_agent_08
from src.schemas.models import (
    CanonicalIncident,
    FinalRiskAssessment,
    IncidentAssessment,
    SupplyChainAssessment,
)
from src.services.llm_service import LLMService


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def actionable_article_ids(gt: dict[str, GroundTruthRecord], limit: int = 0) -> list[str]:
    ids = [a for a, g in gt.items() if g.gt_is_actionable]
    ids.sort()
    return ids[:limit] if limit else ids


def eval_ours_article(
    llm: LLMService,
    article_dir: Path,
    article_text: str,
) -> dict | None:
    result = load_json(article_dir / "result.json")
    if not result or result.get("pipeline_status") != "COMPLETED":
        return None
    ci = result.get("canonical_incident")
    a5 = load_json(article_dir / "agent_05.json") or result.get("incident_assessment")
    a6 = load_json(article_dir / "agent_06.json") or result.get("supply_chain_assessment")
    a7 = load_json(article_dir / "agent_07.json") or result.get("final_risk_assessment")
    if not (ci and a5 and a6 and a7):
        return None

    incident = CanonicalIncident.model_validate(ci)
    assessment = IncidentAssessment.model_validate(a5)
    network = SupplyChainAssessment.model_validate(a6)
    final_risk = FinalRiskAssessment.model_validate(a7)

    report = run_agent_08(llm, incident, assessment, network, final_risk)
    if hasattr(report, "model_dump"):
        payload = report.model_dump()
    else:
        payload = {"report_markdown": str(report), "claims": [], "deterministic_fields": {}, "required_field_coverage": {}}

    (article_dir / "agent_08.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    corpus = build_evidence_corpus(
        article_text=article_text,
        result=result,
        agent05=a5,
        agent06=a6,
        agent07=a7,
        deterministic_fields=payload.get("deterministic_fields"),
    )
    labeled = label_claims(payload.get("claims") or [], corpus)
    metrics = faithfulness_metrics(labeled, payload.get("required_field_coverage"))
    return {
        "method_id": "OURS",
        "article_id": article_dir.name,
        "claims": labeled,
        "metrics": metrics,
        "report_markdown": payload.get("report_markdown"),
    }


def eval_b2_article(article_dir: Path, article_text: str) -> dict | None:
    result = load_json(article_dir / "result.json")
    if not result:
        return None
    status = result.get("pipeline_status")
    if status not in {"COMPLETED", "completed"} and result.get("overall_risk") is None:
        return None
    claims, fields, cov = claims_from_baseline(result)
    corpus = build_evidence_corpus(article_text=article_text, result=result)
    # Enrich corpus with baseline free-text fields
    corpus["AGENT7:risk_rationale"] = str(result.get("risk_rationale") or "")
    corpus["AGENT7:overall_risk"] = str(result.get("overall_risk") or "")
    corpus["AGENT5:main_effect"] = str(result.get("main_effect") or "")
    labeled = label_claims(claims, corpus)
    metrics = faithfulness_metrics(labeled, cov)
    return {
        "method_id": "B2",
        "article_id": article_dir.name,
        "claims": labeled,
        "metrics": metrics,
        "report_markdown": result.get("report"),
    }


def write_human_audit_template(path: Path, rows: list[dict]) -> None:
    """CSV template for §20 human audit (pre-filled auto labels)."""
    out_rows = []
    for r in rows:
        for c in r.get("claims") or []:
            out_rows.append(
                {
                    "method_id": r["method_id"],
                    "article_id": r["article_id"],
                    "claim_id": c.get("claim_id"),
                    "claim_type": c.get("claim_type"),
                    "claim_text": c.get("text"),
                    "supporting_evidence_ids": "|".join(c.get("supporting_evidence_ids") or []),
                    "auto_label": c.get("faithfulness_label"),
                    "human_label": "",
                    "annotator_id": "",
                    "notes": "",
                }
            )
    fields = [
        "method_id",
        "article_id",
        "claim_id",
        "claim_type",
        "claim_text",
        "supporting_evidence_ids",
        "auto_label",
        "human_label",
        "annotator_id",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)


def aggregate_method(rows: list[dict]) -> dict:
    all_claims = []
    cov_counts: dict[str, list[bool]] = {}
    for r in rows:
        all_claims.extend(r.get("claims") or [])
        m = r.get("metrics") or {}
        for k, v in m.items():
            if k.startswith("field_"):
                cov_counts.setdefault(k, []).append(bool(v))
    # average required coverage from per-article
    req = {}
    for k, vals in cov_counts.items():
        req[k.replace("field_", "")] = sum(vals) / len(vals) if vals else 0.0
    # rebuild coverage bools as majority for aggregate display
    cov_bool = {k: v >= 0.5 for k, v in req.items()}
    base = faithfulness_metrics(all_claims, cov_bool)
    base["n_articles"] = len(rows)
    base["mean_required_field_coverage"] = (
        round(sum(req.values()) / len(req), 4) if req else None
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="waiti_faithfulness")
    parser.add_argument("--ours-source", default="waiti_final_gpt41")
    parser.add_argument("--b2-source", default="waiti_final_gpt41")
    parser.add_argument("--limit", type=int, default=0, help="Limit actionable articles (0=all)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip OURS Agent8 LLM; use deterministic-only fallback")
    parser.add_argument("--methods", default="OURS,B2")
    parser.add_argument(
        "--config",
        default="evaluation/config/evaluation.yaml",
        help="Eval config (use evaluation_gpt41.yaml for GPT-4.1 finalize)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override LLM model for Agent-8 regeneration (e.g. gpt-4.1)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8")) or {}
    inputs = {x["article_id"]: x for x in load_jsonl(ROOT / cfg["dataset"]["input_path"])}
    gt = {
        x["article_id"]: GroundTruthRecord.model_validate(x)
        for x in load_jsonl(ROOT / cfg["dataset"]["ground_truth_path"])
    }
    article_ids = actionable_article_ids(gt, limit=args.limit)

    out_dir = ROOT / "results" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    llm = None
    if "OURS" in methods and not args.skip_llm:
        model = args.model or (cfg.get("llm") or {}).get("model", "gpt-5.4")
        use_reasoning = (cfg.get("llm") or {}).get("use_reasoning")
        llm = LLMService(model=model, reasoning_effort="low", use_reasoning=use_reasoning)
        print(f"Faithfulness LLM: {model} (use_reasoning={use_reasoning})")

    all_rows: list[dict] = []
    claim_path = out_dir / "faithfulness_claims.jsonl"
    with claim_path.open("w", encoding="utf-8") as cf:
        if "OURS" in methods:
            ours_root = ROOT / "results" / args.ours_source / "methods" / "OURS" / "articles"
            # also write regenerated reports under faithfulness run
            ours_out = out_dir / "methods" / "OURS" / "articles"
            ours_out.mkdir(parents=True, exist_ok=True)
            for aid in article_ids:
                src = ours_root / aid
                if not src.exists():
                    continue
                text = f"{inputs.get(aid, {}).get('headline', '')}\n{inputs.get(aid, {}).get('body', '')}"
                if args.skip_llm or llm is None:
                    # deterministic-only path: build claims without prose LLM
                    from src.agents.agent_08_report_generator import (
                        ProseSections,
                        _build_deterministic_fields,
                        _render_markdown,
                        build_structured_claims,
                        required_field_coverage,
                    )

                    result = load_json(src / "result.json")
                    if not result or result.get("pipeline_status") != "COMPLETED":
                        continue
                    incident = CanonicalIncident.model_validate(result["canonical_incident"])
                    a5 = load_json(src / "agent_05.json") or result["incident_assessment"]
                    a6 = load_json(src / "agent_06.json") or result["supply_chain_assessment"]
                    a7 = load_json(src / "agent_07.json") or result["final_risk_assessment"]
                    fields = _build_deterministic_fields(
                        incident,
                        IncidentAssessment.model_validate(a5),
                        SupplyChainAssessment.model_validate(a6),
                        FinalRiskAssessment.model_validate(a7),
                    )
                    prose = ProseSections(
                        executive_summary=f"{fields.get('main_effect')} Risk={fields.get('risk_label')}.",
                        why_this_matters=str(a7.get("risk_rationale") or ""),
                        monitoring_priorities=["Monitor supplier status"],
                    )
                    claims = build_structured_claims(fields)
                    payload = {
                        "report_markdown": _render_markdown(fields, prose),
                        "claims": [c.model_dump() for c in claims],
                        "deterministic_fields": fields,
                        "required_field_coverage": required_field_coverage(fields),
                    }
                    corpus = build_evidence_corpus(
                        article_text=text,
                        result=result,
                        agent05=a5,
                        agent06=a6,
                        agent07=a7,
                        deterministic_fields=fields,
                    )
                    labeled = label_claims(payload["claims"], corpus)
                    row = {
                        "method_id": "OURS",
                        "article_id": aid,
                        "claims": labeled,
                        "metrics": faithfulness_metrics(labeled, payload["required_field_coverage"]),
                        "report_markdown": payload["report_markdown"],
                    }
                else:
                    # copy stages into faithfulness run dir then generate
                    dst = ours_out / aid
                    dst.mkdir(parents=True, exist_ok=True)
                    for name in ("result.json", "agent_05.json", "agent_06.json", "agent_07.json"):
                        p = src / name
                        if p.exists():
                            (dst / name).write_bytes(p.read_bytes())
                    row = eval_ours_article(llm, dst, text)
                if not row:
                    continue
                all_rows.append(row)
                cf.write(json.dumps({"method_id": "OURS", "article_id": aid, "claims": row["claims"]}, ensure_ascii=False) + "\n")
                print(f"OURS {aid} claims={row['metrics'].get('n_claims')} support={row['metrics'].get('support_rate')}")

        if "B2" in methods:
            b2_root = ROOT / "results" / args.b2_source / "methods" / "B2" / "articles"
            for aid in article_ids:
                src = b2_root / aid
                if not src.exists():
                    continue
                text = f"{inputs.get(aid, {}).get('headline', '')}\n{inputs.get(aid, {}).get('body', '')}"
                row = eval_b2_article(src, text)
                if not row:
                    continue
                all_rows.append(row)
                cf.write(json.dumps({"method_id": "B2", "article_id": aid, "claims": row["claims"]}, ensure_ascii=False) + "\n")
                print(f"B2 {aid} claims={row['metrics'].get('n_claims')} support={row['metrics'].get('support_rate')}")

    # Per-article metrics CSV
    art_rows = [{"method_id": r["method_id"], "article_id": r["article_id"], **r["metrics"]} for r in all_rows]
    if art_rows:
        fields = list(art_rows[0].keys())
        with (out_dir / "faithfulness_per_article.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(art_rows)

    # Method aggregate
    agg_rows = []
    for mid in methods:
        subset = [r for r in all_rows if r["method_id"] == mid]
        if not subset:
            continue
        agg_rows.append({"method_id": mid, **aggregate_method(subset)})
    if agg_rows:
        with (out_dir / "faithfulness_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(agg_rows)
        (out_dir / "faithfulness_metrics.json").write_text(json.dumps(agg_rows, indent=2) + "\n", encoding="utf-8")

    write_human_audit_template(out_dir / "human_report_audit_template.csv", all_rows)

    # Stratified sample note for human audit (≥30)
    sample_note = {
        "target_manual_reports": 30,
        "stratify": ["risk levels", "single vs merged", "supplier strata", "correct vs incorrect risk"],
        "auto_labeled_articles": len({(r["method_id"], r["article_id"]) for r in all_rows}),
        "template": "human_report_audit_template.csv",
    }
    (out_dir / "human_audit_plan.json").write_text(json.dumps(sample_note, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Faithfulness results in {out_dir}")


if __name__ == "__main__":
    main()
