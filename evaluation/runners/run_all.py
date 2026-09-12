"""Run all WAITI evaluation methods."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.methods import METHOD_REGISTRY
from evaluation.methods.base import EvaluationContext
from evaluation.reports.build_tables import write_markdown_tables
from evaluation.runners.export_data import export, validate_no_leakage
from evaluation.runners.metrics_io import write_aggregates
from evaluation.schemas.prediction import GroundTruthRecord, MethodPrediction, ModelInputRecord
from evaluation.services.embedding_service import EmbeddingService
from src.services.entity_resolution import EntityResolutionService
from src.services.graph_service import SupplyChainGraphService

# Methods whose merge decisions depend on prior articles in the same run.
MERGE_DEPENDENT_METHODS = {
    "B0",
    "B2",
    "B3",
    "OURS",
    "A1",
    "A3",
    "A4",
    "OURS_QWEN3_14B",
    "OURS_GEMMA3_12B",
    "B2_QWEN3_14B",
}


def require_openai_api_key() -> None:
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_TOKEN")):
        raise SystemExit(
            "Missing OpenAI API credentials. Set OPENAI_API_KEY (or OPENAI_API_TOKEN) in the environment. "
            "Do not hard-code tokens into source, config, or results."
        )


def git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def graph_freeze_stats(graph_dir: Path) -> dict:
    import csv as _csv

    def _n(name: str) -> int:
        p = graph_dir / name
        if not p.exists():
            return -1
        with p.open(encoding="utf-8", newline="") as f:
            return sum(1 for _ in _csv.DictReader(f))

    return {
        "nodes": _n("nodes.csv"),
        "edges": _n("edges.csv"),
        "aliases": _n("aliases.csv"),
        "nodes_sha": file_sha256(graph_dir / "nodes.csv") if (graph_dir / "nodes.csv").exists() else None,
        "edges_sha": file_sha256(graph_dir / "edges.csv") if (graph_dir / "edges.csv").exists() else None,
        "aliases_sha": file_sha256(graph_dir / "aliases.csv") if (graph_dir / "aliases.csv").exists() else None,
    }


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_context(
    run_id: str,
    cfg: dict,
    method_id: str,
    *,
    model_override: str | None = None,
) -> EvaluationContext:
    method_cfg_path = ROOT / "evaluation" / "config" / "method_configs" / f"{method_id}.yaml"
    method_cfg = load_yaml(method_cfg_path) if method_cfg_path.exists() else {}
    method_cfg = dict(method_cfg)
    llm_cfg = dict(cfg.get("llm") or {})
    if model_override:
        method_cfg["model"] = model_override
        llm_cfg["model"] = model_override
    # Propagate use_reasoning from eval llm block into method config when set
    if "use_reasoning" in llm_cfg and "use_reasoning" not in method_cfg:
        method_cfg["use_reasoning"] = llm_cfg.get("use_reasoning")
    if model_override and method_cfg.get("use_reasoning") is None:
        # Non-reasoning override models (gpt-4.1) must not inherit gpt-5.4 reasoning_effort path
        from src.services.llm_service import uses_reasoning_api

        method_cfg["use_reasoning"] = uses_reasoning_api(model_override)
    cyber = load_yaml(ROOT / cfg["paths"]["cyber_rules"])
    graph = SupplyChainGraphService(ROOT / cfg["dataset"]["graph_dir"])
    er_path = ROOT / (cfg.get("entity_resolution") or {}).get("config_path", "config/entity_resolution.yaml")
    resolver = EntityResolutionService(graph, config_path=er_path if er_path.exists() else None)
    return EvaluationContext(
        run_id=run_id,
        root=ROOT,
        graph=graph,
        resolver=resolver,
        embeddings=EmbeddingService(),
        risk_rules_path=ROOT / cfg["paths"]["risk_rules"],
        cyber_rules=cyber,
        llm_model=llm_cfg.get("model", "gpt-5.4"),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        method_config=method_cfg,
        pricing=cfg.get("pricing") or {},
    )


def save_predictions(run_dir: Path, method_id: str, preds: list[MethodPrediction]) -> None:
    mdir = run_dir / "methods" / method_id
    mdir.mkdir(parents=True, exist_ok=True)
    path = mdir / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")


def load_method_preds(run_dir: Path, method_id: str) -> list[MethodPrediction]:
    path = run_dir / "methods" / method_id / "predictions.jsonl"
    if not path.exists():
        return []
    return [
        MethodPrediction.model_validate(json.loads(l))
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def write_per_record_csv(run_dir: Path, all_preds: list[MethodPrediction], gt: dict[str, GroundTruthRecord]) -> None:
    rows = []
    for p in all_preds:
        g = gt[p.article_id]
        risk_err = None
        if g.gt_risk and p.predicted_risk not in (None, "UNKNOWN"):
            order = ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
            try:
                risk_err = abs(order.index(g.gt_risk) - order.index(p.predicted_risk))
            except ValueError:
                risk_err = None
        rows.append(
            {
                "run_id": p.run_id,
                "repeat_id": p.repeat_id,
                "method_id": p.method_id,
                "article_id": p.article_id,
                "gt_is_cyber": g.gt_is_cyber,
                "pred_is_cyber": p.pred_is_cyber,
                "cyber_correct": (p.pred_is_cyber == g.gt_is_cyber) if p.pred_is_cyber is not None else False,
                "gt_is_related": g.gt_is_related,
                "pred_is_related": p.pred_is_related,
                "relevance_correct": (p.pred_is_related == g.gt_is_related) if p.pred_is_related is not None else False,
                "gt_is_actionable": g.gt_is_actionable,
                "pred_is_actionable": p.pred_is_actionable,
                "actionable_correct": (p.pred_is_actionable == g.gt_is_actionable)
                if p.pred_is_actionable is not None
                else False,
                "gt_group_id": g.gt_group_id or "",
                "predicted_incident_id": p.predicted_incident_id or "",
                "gt_risk": g.gt_risk or "",
                "predicted_risk": p.predicted_risk,
                "risk_exact_correct": (g.gt_risk == p.predicted_risk) if g.gt_risk else "",
                "risk_absolute_error": risk_err if risk_err is not None else "",
                "predicted_supplier_names": "|".join(p.predicted_supplier_names),
                "predicted_supplier_node_ids": "|".join(p.predicted_supplier_node_ids),
                "latency_ms": p.latency_ms,
                "llm_calls": p.llm_calls,
                "graph_queries": p.graph_queries,
                "input_tokens": p.input_tokens,
                "output_tokens": p.output_tokens,
                "cost_usd": getattr(p, "cost_usd", 0.0),
                "status": p.status,
                "error_type": p.error_type or "",
            }
        )
    path = run_dir / "per_record_predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["article_id"])
        w.writeheader()
        w.writerows(rows)
    with (run_dir / "per_record_predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in all_preds:
            g = gt[p.article_id]
            f.write(
                json.dumps({"prediction": p.model_dump(), "ground_truth": g.model_dump()}, ensure_ascii=False)
                + "\n"
            )


def write_comparison(run_dir: Path, methods: list[str], gt: dict[str, GroundTruthRecord]) -> None:
    by_method = {m: {p.article_id: p for p in load_method_preds(run_dir, m)} for m in methods}
    path = run_dir / "per_incident_comparison.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for aid, g in gt.items():
            obj = {
                "article_id": aid,
                "ground_truth": g.model_dump(),
                "predictions": {
                    m: by_method[m][aid].model_dump()
                    for m in methods
                    if aid in by_method[m]
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _effective_article_workers(method_id: str, requested: int, unsafe_parallel_merge: bool) -> int:
    requested = max(1, int(requested))
    if requested == 1:
        return 1
    if method_id in MERGE_DEPENDENT_METHODS and not unsafe_parallel_merge:
        print(
            f"  note: {method_id} is merge-dependent; using article_workers=1 "
            f"(pass --unsafe-parallel-merge to override)"
        )
        return 1
    return requested


def run_method_articles(
    run_id: str,
    cfg: dict,
    mid: str,
    inputs: list[ModelInputRecord],
    run_dir: Path,
    article_workers: int,
    unsafe_parallel_merge: bool,
) -> list[MethodPrediction]:
    ctx = build_context(run_id, cfg, mid, model_override=cfg.get("_model_override"))
    method = METHOD_REGISTRY[mid](ctx)
    existing = {p.article_id: p for p in load_method_preds(run_dir, mid)}
    workers = _effective_article_workers(mid, article_workers, unsafe_parallel_merge)
    print(f"=== {mid} ({len(inputs)} articles, workers={workers}) ===")

    order = [r.article_id for r in inputs]
    by_id: dict[str, MethodPrediction] = {}
    pending: list[ModelInputRecord] = []
    for rec in inputs:
        prev = existing.get(rec.article_id)
        if prev is not None and prev.status != "FAILED":
            by_id[rec.article_id] = prev
            print(f"  [resume] {rec.article_id}")
        else:
            pending.append(rec)

    save_lock = threading.Lock()

    def _checkpoint() -> None:
        preds_ordered = [by_id[aid] for aid in order if aid in by_id]
        with save_lock:
            save_predictions(run_dir, mid, preds_ordered)

    def _one(rec: ModelInputRecord) -> MethodPrediction:
        pred = method.process_record(rec)
        by_id[rec.article_id] = pred
        _checkpoint()
        print(
            f"  {mid} {rec.article_id} -> {pred.status} "
            f"cyber={pred.pred_is_cyber} rel={pred.pred_is_related} "
            f"risk={pred.predicted_risk} lat={pred.latency_ms:.0f}ms "
            f"cost=${pred.cost_usd:.6f}"
        )
        return pred

    if workers == 1 or not pending:
        for rec in pending:
            _one(rec)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_one, rec): rec.article_id for rec in pending}
            for fut in as_completed(futs):
                fut.result()

    return [by_id[aid] for aid in order if aid in by_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="evaluation/config/evaluation.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--methods", default="", help="Comma-separated subset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--source-run",
        default="",
        help="Copy method predictions from an existing run into this run_id (source is never modified).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel article workers within each method (overrides config)",
    )
    parser.add_argument(
        "--method-workers",
        type=int,
        default=None,
        help="Parallel methods (each method has isolated state)",
    )
    parser.add_argument(
        "--unsafe-parallel-merge",
        action="store_true",
        help="Allow parallel articles for merge-dependent methods (merge metrics may be non-deterministic)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override LLM model for all methods (e.g. gpt-4.1). Does not edit shared method YAMLs.",
    )
    args = parser.parse_args()

    cfg = load_yaml(ROOT / args.config)
    cfg = dict(cfg)
    if args.model:
        llm = dict(cfg.get("llm") or {})
        llm["model"] = args.model
        # Non-reasoning chat models: do not use Responses API / reasoning_effort
        from src.services.llm_service import uses_reasoning_api

        llm["use_reasoning"] = uses_reasoning_api(args.model, use_reasoning=llm.get("use_reasoning"))
        cfg["llm"] = llm
        cfg["_model_override"] = args.model
        print(f"Model override: {args.model} (use_reasoning={llm['use_reasoning']})")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or list(cfg.get("methods") or [])
    non_llm = {"B0"}
    needs_llm = (not args.aggregate_only) and any(m not in non_llm for m in methods)
    if needs_llm:
        require_openai_api_key()
    par = cfg.get("parallel") or {}
    article_workers = args.workers if args.workers is not None else int(par.get("article_workers", 1))
    method_workers = args.method_workers if args.method_workers is not None else int(par.get("method_workers", 1))
    unsafe_merge = bool(args.unsafe_parallel_merge or par.get("unsafe_parallel_merge", False))

    FROZEN_FINAL_RUNS = {"waiti_final_gpt54", "waiti_final_gpt41"}
    if args.run_id in FROZEN_FINAL_RUNS:
        # Final paper runs: never mix old predictions; keep merge stateful & deterministic
        if getattr(args, "source_run", ""):
            raise SystemExit(f"{args.run_id} refuses --source-run (must generate fresh predictions).")
        if unsafe_merge:
            raise SystemExit(f"{args.run_id} refuses --unsafe-parallel-merge.")
        article_workers = 1
        method_workers = 1
        print(f"{args.run_id} freeze: article_workers=1, method_workers=1, unsafe_parallel_merge=false")

    if not args.skip_export:
        export(ROOT / cfg["dataset"]["corpus_dir"], ROOT / "evaluation" / "data")
    validate_no_leakage(ROOT / cfg["dataset"]["input_path"])

    inputs = [ModelInputRecord.model_validate(x) for x in load_jsonl(ROOT / cfg["dataset"]["input_path"])]
    gts = [GroundTruthRecord.model_validate(x) for x in load_jsonl(ROOT / cfg["dataset"]["ground_truth_path"])]
    gt_by_id = {g.article_id: g for g in gts}
    if args.limit:
        inputs = inputs[: args.limit]

    run_dir = ROOT / cfg["paths"]["results_dir"] / args.run_id
    if args.run_id == "waiti_full1":
        raise SystemExit("Refusing to write into waiti_full1 (frozen). Use a new --run-id.")
    run_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "source_run", ""):
        src = ROOT / cfg["paths"]["results_dir"] / args.source_run
        if not src.exists():
            raise SystemExit(f"source run not found: {src}")
        if src.resolve() == run_dir.resolve():
            raise SystemExit("--source-run must differ from --run-id")
        import shutil

        for mid in methods:
            sp = src / "methods" / mid / "predictions.jsonl"
            if not sp.exists():
                print(f"  warn: missing {sp}")
                continue
            dp = run_dir / "methods" / mid
            dp.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dp / "predictions.jsonl")
            print(f"  copied {mid} predictions from {args.source_run}")

    if not args.aggregate_only:
        for mid in methods:
            if mid not in METHOD_REGISTRY:
                raise SystemExit(f"Unknown method {mid}")

        method_workers = max(1, method_workers)
        if method_workers == 1:
            for mid in methods:
                run_method_articles(
                    args.run_id, cfg, mid, inputs, run_dir, article_workers, unsafe_merge
                )
        else:
            print(f"Running {len(methods)} methods with method_workers={method_workers}")
            with ThreadPoolExecutor(max_workers=method_workers) as ex:
                futs = {
                    ex.submit(
                        run_method_articles,
                        args.run_id,
                        cfg,
                        mid,
                        inputs,
                        run_dir,
                        article_workers,
                        unsafe_merge,
                    ): mid
                    for mid in methods
                }
                for fut in as_completed(futs):
                    mid = futs[fut]
                    fut.result()
                    print(f"=== finished {mid} ===")

    all_preds: list[MethodPrediction] = []
    if args.aggregate_only and not (args.methods or "").strip():
        # Only aggregate methods that actually have predictions on disk
        methods = [
            m
            for m in methods
            if (run_dir / "methods" / m / "predictions.jsonl").exists()
        ]
        if not methods:
            raise SystemExit(f"No method predictions found under {run_dir / 'methods'}")
    for mid in methods:
        all_preds.extend(load_method_preds(run_dir, mid))

    write_per_record_csv(run_dir, all_preds, gt_by_id)
    write_comparison(run_dir, methods, gt_by_id)
    aggregates = write_aggregates(run_dir, methods, gt_by_id, cfg)
    write_markdown_tables(run_dir, aggregates)

    # Concatenate ER diagnostics to run root when present
    er_out = run_dir / "entity_resolution_diagnostics.jsonl"
    with er_out.open("w", encoding="utf-8") as out:
        for mid in methods:
            p = run_dir / "methods" / mid / "entity_resolution_diagnostics.jsonl"
            if p.exists():
                out.write(p.read_text(encoding="utf-8"))

    graph_stats = graph_freeze_stats(ROOT / cfg["dataset"]["graph_dir"])
    commit = git_commit_hash()
    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "methods": methods,
        "n_articles": len(inputs),
        "article_order": [r.article_id for r in inputs],
        "article_workers": article_workers,
        "method_workers": method_workers,
        "unsafe_parallel_merge": unsafe_merge,
        "llm": cfg.get("llm"),
        "pricing": cfg.get("pricing"),
        "source_run": getattr(args, "source_run", None) or None,
        "entity_resolution_version": "staged_v1_final",
        "merger_version": "merge_v2_final",
        "bootstrap_resamples": (cfg.get("bootstrap") or {}).get("n_resamples"),
        "bootstrap_seed": (cfg.get("bootstrap") or {}).get("seed", 42),
        "bootstrap_unit": (cfg.get("bootstrap") or {}).get("unit", "group_id"),
        "risk_metric_policy": {
            "unknown_mae_penalty": (cfg.get("risk") or {}).get("unknown_penalty", 4),
            "qwk_population": "detected_only",
            "macro_f1_population": "all_gt_actionable",
        },
        "dataset_inputs_sha": file_sha256(ROOT / cfg["dataset"]["input_path"]),
        "dataset_gt_sha": file_sha256(ROOT / cfg["dataset"]["ground_truth_path"]),
        "graph_freeze": graph_stats,
        "expected_graph": {"nodes": 4661, "edges": 26035, "aliases": 8886},
        "config": cfg,
        "freeze_note": (
            "Do not modify prompts, thresholds, aliases, graph rules, ER, merger, or dataset "
            "after this run starts. Paper tables must use this run only."
            if args.run_id in ("waiti_final_gpt54", "waiti_final_gpt41")
            else None
        ),
        "model_override": getattr(args, "model", None) or None,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Results in {run_dir}")
    if args.run_id == "waiti_final_gpt54":
        print(
            "Next: python -m evaluation.runners.finalize_final_gpt54 "
            "(faithfulness + paper markdown tables)"
        )
    elif args.run_id == "waiti_final_gpt41":
        print(
            "Next: python -m evaluation.runners.finalize_final_gpt54 "
            "--run-id waiti_final_gpt41 --config evaluation/config/evaluation_gpt41.yaml --model gpt-4.1"
        )


if __name__ == "__main__":
    main()
