"""Run WAITI methods against a local vLLM endpoint (no OpenAI / hosted API fallback).

Fails if the local server is unavailable. Does not compute API cost.
Checkpoints predictions continuously.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.methods import METHOD_REGISTRY
from evaluation.runners.run_all import (
    file_sha256,
    git_commit_hash,
    graph_freeze_stats,
    load_jsonl,
    load_method_preds,
    load_yaml,
    save_predictions,
    write_comparison,
    write_per_record_csv,
)
from evaluation.runners.run_all import build_context as _build_context_base
from evaluation.schemas.prediction import GroundTruthRecord, ModelInputRecord

FORBIDDEN_HOST_FRAGMENTS = (
    "api.openai.com",
    "openai.azure.com",
    "anthropic.com",
    "googleapis.com",
    "openrouter.ai",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_local_cfg() -> dict:
    path = ROOT / "evaluation/config/local_models.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_model(cfg: dict, alias: str) -> dict:
    models = cfg.get("models") or {}
    legacy = cfg.get("legacy_aliases") or {}
    key = legacy.get(alias, alias)
    if key not in models:
        raise SystemExit(f"Unknown model alias '{alias}'. Known: {sorted(models)}")
    m = dict(models[key])
    m["_alias"] = key
    return m


def assert_local_endpoint(endpoint: str) -> None:
    low = (endpoint or "").lower()
    for frag in FORBIDDEN_HOST_FRAGMENTS:
        if frag in low:
            raise SystemExit(
                f"Refusing non-local endpoint '{endpoint}' (matched '{frag}'). "
                "Local experiments must not call hosted APIs."
            )
    if not low.startswith("http://127.0.0.1") and not low.startswith("http://localhost"):
        # Allow localhost-style endpoints; still block hosted APIs above
        if "openai" in low or "azure" in low:
            raise SystemExit(f"Refusing suspicious endpoint: {endpoint}")


def probe_vllm(endpoint: str, timeout: float = 5.0) -> dict:
    url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "url": url, "models": data}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}


def wait_for_vllm(endpoint: str, max_wait_s: int = 600, interval_s: float = 5.0) -> dict:
    deadline = time.time() + max_wait_s
    last = {"ok": False, "error": "not_probed"}
    while time.time() < deadline:
        last = probe_vllm(endpoint)
        if last.get("ok"):
            return last
        time.sleep(interval_s)
    return last


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def collect_software_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "vllm": _pkg_version("vllm"),
        "langchain_openai": _pkg_version("langchain_openai"),
    }


def nvidia_smi_query() -> dict:
    """Best-effort GPU snapshot; returns empty dict if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        ).strip()
    except Exception as e:
        return {"available": False, "error": str(e)}
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            rows.append(
                {
                    "name": parts[0],
                    "memory_total_mib": _float(parts[1]),
                    "memory_used_mib": _float(parts[2]),
                    "utilization_gpu_pct": _float(parts[3]),
                    "utilization_memory_pct": _float(parts[4]),
                    "driver_version": parts[5],
                }
            )
    return {"available": True, "gpus": rows, "raw": out}


def _float(x: str) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def cuda_version() -> str | None:
    try:
        out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True, timeout=10)
        for line in out.splitlines():
            if "release" in line.lower():
                return line.strip()
    except Exception:
        pass
    try:
        import torch

        return getattr(torch.version, "cuda", None)
    except Exception:
        return None


class HardwareSampler:
    """Background nvidia-smi sampler for peak VRAM."""

    def __init__(self, interval_s: float = 10.0):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []
        self.peak_memory_used_mib: float | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            snap = nvidia_smi_query()
            ts = _now()
            row = {"ts": ts, **snap}
            self.samples.append(row)
            if snap.get("available") and snap.get("gpus"):
                used = max((g.get("memory_used_mib") or 0.0) for g in snap["gpus"])
                if self.peak_memory_used_mib is None or used > self.peak_memory_used_mib:
                    self.peak_memory_used_mib = used
            self._stop.wait(self.interval_s)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return {
            "n_samples": len(self.samples),
            "peak_memory_used_mib": self.peak_memory_used_mib,
            "last": self.samples[-1] if self.samples else None,
        }


def write_invalid(run_dir: Path, reasons: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN_INVALID").write_text("\n".join(reasons) + "\n", encoding="utf-8")
    (run_dir / "STATUS.md").write_text(
        "# RUN_INVALID\n\n" + "\n".join(f"- {r}" for r in reasons) + "\n",
        encoding="utf-8",
    )


def local_method_overlay(model: dict, endpoint: str, api_key: str) -> dict:
    return {
        "model": model.get("served_model_name") or model.get("hf_repo"),
        "temperature": float(model.get("temperature", 0.0)),
        "reasoning_effort": str(model.get("reasoning_effort", "low")),
        # gpt-oss keeps reasoning on and receives effort via chat_template_kwargs
        "reasoning_via_chat_template": bool(model.get("reasoning_via_chat_template", False)),
        "base_url": endpoint,
        "api_key": api_key,
        "include_report": False,
        "max_output_tokens": int(model.get("max_output_tokens", 2048)),
        "disable_thinking": bool(model.get("disable_thinking", True)),
        "max_tool_rounds": int(model.get("max_tool_rounds", 4)),
        "tool_result_chars": int(model.get("tool_result_chars", 2000)),
        "article_body_chars": int(model.get("article_body_chars", 6000)),
        # Disable cost-oriented pricing for local
        "_local_llm": True,
        "_model_alias": model.get("_alias"),
        "_hf_repo": model.get("hf_repo"),
        "_revision": model.get("revision"),
    }


def build_local_context(run_id: str, eval_cfg: dict, method_id: str, overlay: dict):
    from evaluation.services.embedding_service import EmbeddingService

    ctx = _build_context_base(run_id, eval_cfg, method_id)
    merged = dict(ctx.method_config or {})
    merged.update(overlay)
    # Force local model identity into context
    ctx.method_config = merged
    ctx.llm_model = overlay["model"]
    ctx.temperature = float(overlay.get("temperature", 0.0))
    ctx.pricing = {}  # no API cost for local
    # Never call OpenAI embeddings during local/vLLM runs
    ctx.embeddings = EmbeddingService(mode="local")
    return ctx


def run_method_local(
    run_id: str,
    eval_cfg: dict,
    mid: str,
    inputs: list[ModelInputRecord],
    run_dir: Path,
    overlay: dict,
) -> list:
    """Like run_method_articles but with forced local overlay and workers=1."""
    ctx = build_local_context(run_id, eval_cfg, mid, overlay)
    if mid not in METHOD_REGISTRY:
        raise SystemExit(f"Unknown method {mid}")
    method = METHOD_REGISTRY[mid](ctx)
    # Ensure method sees overlay (OURS/B2 read ctx.method_config in __init__ — already set)
    existing = {p.article_id: p for p in load_method_preds(run_dir, mid)}
    print(f"=== {mid} LOCAL ({len(inputs)} articles, workers=1) model={overlay['model']} ===")
    order = [r.article_id for r in inputs]
    by_id = {}
    pending = []
    for rec in inputs:
        prev = existing.get(rec.article_id)
        if prev is not None and prev.status != "FAILED":
            by_id[rec.article_id] = prev
            print(f"  [resume] {rec.article_id}")
        else:
            pending.append(rec)

    def _checkpoint() -> None:
        preds_ordered = [by_id[aid] for aid in order if aid in by_id]
        save_predictions(run_dir, mid, preds_ordered)
        # also mirror to raw/
        raw = run_dir / "raw" / f"{mid}.jsonl"
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "methods" / mid / "predictions.jsonl", raw)

    for rec in pending:
        # Refuse if overlay somehow lost local base_url
        bu = (getattr(method, "ctx", ctx).method_config or {}).get("base_url") or overlay.get("base_url")
        assert_local_endpoint(bu or "")
        pred = method.process_record(rec)
        # Zero out cost for local
        pred.cost_usd = 0.0
        by_id[rec.article_id] = pred
        _checkpoint()
        print(
            f"  {mid} {rec.article_id} -> {pred.status} "
            f"cyber={pred.pred_is_cyber} rel={pred.pred_is_related} "
            f"risk={pred.predicted_risk} lat={pred.latency_ms:.0f}ms"
        )
    # Always refresh methods/ + raw/ mirrors (covers full-resume paths)
    if by_id:
        _checkpoint()
    return [by_id[aid] for aid in order if aid in by_id]


def export_raw_and_telemetry(
    run_dir: Path,
    methods: list[str],
    model: dict,
    hw_summary: dict,
    software: dict,
    probe: dict,
    started: str,
    ended: str,
) -> None:
    tel = run_dir / "telemetry"
    tel.mkdir(parents=True, exist_ok=True)
    hw = {
        "model_alias": model.get("_alias"),
        "hf_repo": model.get("hf_repo"),
        "revision": model.get("revision"),
        "dtype": model.get("dtype"),
        "quantization": model.get("quantization"),
        "started_at": started,
        "completed_at": ended,
        "software": software,
        "cuda": cuda_version(),
        "nvidia_smi_start": hw_summary.get("start"),
        "nvidia_smi_end": hw_summary.get("end"),
        "peak_memory_used_mib": hw_summary.get("peak_memory_used_mib"),
        "n_samples": hw_summary.get("n_samples"),
        "vllm_probe": {"ok": probe.get("ok"), "url": probe.get("url")},
    }
    (tel / "hardware_metrics.json").write_text(json.dumps(hw, indent=2) + "\n", encoding="utf-8")
    # CSV summary
    with (tel / "hardware_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_alias",
                "gpu_name",
                "memory_total_mib",
                "peak_memory_used_mib",
                "driver_version",
                "cuda",
                "dtype",
                "quantization",
            ],
        )
        w.writeheader()
        gpus = ((hw_summary.get("end") or {}).get("gpus") or (hw_summary.get("start") or {}).get("gpus") or [{}])
        g0 = gpus[0] if gpus else {}
        w.writerow(
            {
                "model_alias": model.get("_alias"),
                "gpu_name": g0.get("name"),
                "memory_total_mib": g0.get("memory_total_mib"),
                "peak_memory_used_mib": hw_summary.get("peak_memory_used_mib"),
                "driver_version": g0.get("driver_version"),
                "cuda": cuda_version(),
                "dtype": model.get("dtype"),
                "quantization": model.get("quantization"),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="WAITI local vLLM experiment (no OpenAI fallback)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-alias", required=True, help="qwen14b | qwen7b | qwen3b")
    parser.add_argument("--methods", default="", help="Comma-separated; default from local_models.yaml experiment")
    parser.add_argument("--endpoint", default="", help="Override vLLM OpenAI-compatible base URL")
    parser.add_argument("--config", default="evaluation/config/evaluation.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--wait-vllm-seconds", type=int, default=0, help="If >0, poll until healthy")
    parser.add_argument("--aggregate", action="store_true", help="Also write aggregate metrics")
    args = parser.parse_args()

    local_cfg = load_local_cfg()
    model = resolve_model(local_cfg, args.model_alias)
    endpoint = args.endpoint or local_cfg.get("default_endpoint") or "http://127.0.0.1:8000/v1"
    api_key = local_cfg.get("api_key") or "EMPTY"
    assert_local_endpoint(endpoint)

    # Resolve methods from experiment block if not provided
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        for exp in local_cfg.get("experiments") or []:
            if exp.get("run_id") == args.run_id or exp.get("model_alias") == model["_alias"]:
                methods = list(exp.get("methods") or [])
                break
    if not methods:
        raise SystemExit("No methods specified and none found in local_models.yaml experiments")

    eval_cfg = load_yaml(ROOT / args.config)
    # Force local-ish llm block in cfg for manifests (does not hit OpenAI)
    eval_cfg = dict(eval_cfg)
    eval_cfg["llm"] = {
        "model": model.get("served_model_name"),
        "temperature": float(model.get("temperature", 0.0)),
        "max_output_tokens": int(model.get("max_output_tokens", 2048)),
        "base_url": endpoint,
        "provider": "vllm_local",
    }
    eval_cfg["pricing"] = {}

    run_dir = ROOT / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "telemetry").mkdir(parents=True, exist_ok=True)

    # Clear prior completion markers for this attempt
    for marker in ("RUN_COMPLETE", "RUN_INVALID"):
        p = run_dir / marker
        if p.exists():
            p.unlink()

    if not args.skip_export:
        from evaluation.runners.export_data import export, validate_no_leakage

        export(ROOT / eval_cfg["dataset"]["corpus_dir"], ROOT / "evaluation" / "data")
        validate_no_leakage(ROOT / eval_cfg["dataset"]["input_path"])
    else:
        from evaluation.runners.export_data import validate_no_leakage

        validate_no_leakage(ROOT / eval_cfg["dataset"]["input_path"])

    inputs = [ModelInputRecord.model_validate(x) for x in load_jsonl(ROOT / eval_cfg["dataset"]["input_path"])]
    gts = [GroundTruthRecord.model_validate(x) for x in load_jsonl(ROOT / eval_cfg["dataset"]["ground_truth_path"])]
    gt_by_id = {g.article_id: g for g in gts}
    if args.limit:
        inputs = inputs[: args.limit]

    # Probe vLLM — NO OpenAI fallback
    if args.wait_vllm_seconds > 0:
        probe = wait_for_vllm(endpoint, max_wait_s=args.wait_vllm_seconds)
    else:
        probe = probe_vllm(endpoint)
    if not probe.get("ok"):
        reasons = [
            f"Local vLLM endpoint unhealthy: {endpoint}",
            f"probe_error: {probe.get('error')}",
            "No OpenAI/API fallback is permitted for local experiments.",
        ]
        write_invalid(run_dir, reasons)
        raise SystemExit("; ".join(reasons))

    overlay = local_method_overlay(model, endpoint, api_key)
    # Ensure env does not silently redirect to OpenAI: set dummy key for client only
    os.environ.setdefault("OPENAI_API_KEY", api_key)

    started = _now()
    software = collect_software_versions()
    hw_start = nvidia_smi_query()
    sampler = HardwareSampler(interval_s=15.0)
    sampler.start()

    all_preds = []
    try:
        for mid in methods:
            preds = run_method_local(args.run_id, eval_cfg, mid, inputs, run_dir, overlay)
            all_preds.extend(preds)
    except Exception as e:
        sampler.stop()
        write_invalid(run_dir, [f"method_execution_failed: {e}"])
        raise

    hw_sample = sampler.stop()
    ended = _now()
    hw_end = nvidia_smi_query()
    hw_summary = {
        "start": hw_start,
        "end": hw_end,
        "peak_memory_used_mib": hw_sample.get("peak_memory_used_mib"),
        "n_samples": hw_sample.get("n_samples"),
    }
    export_raw_and_telemetry(run_dir, methods, model, hw_summary, software, probe, started, ended)

    write_per_record_csv(run_dir, all_preds, gt_by_id)
    write_comparison(run_dir, methods, gt_by_id)

    # Concatenate ER diagnostics if present
    er_out = run_dir / "entity_resolution_diagnostics.jsonl"
    with er_out.open("w", encoding="utf-8") as out:
        for mid in methods:
            p = run_dir / "methods" / mid / "entity_resolution_diagnostics.jsonl"
            if p.exists():
                out.write(p.read_text(encoding="utf-8"))

    if args.aggregate:
        from evaluation.runners.metrics_io import write_aggregates
        from evaluation.reports.build_tables import write_markdown_tables

        aggregates = write_aggregates(run_dir, methods, gt_by_id, eval_cfg)
        write_markdown_tables(run_dir, aggregates)

    graph_stats = graph_freeze_stats(ROOT / eval_cfg["dataset"]["graph_dir"])
    manifest = {
        "run_id": args.run_id,
        "created_at": started,
        "completed_at": ended,
        "git_commit": git_commit_hash(),
        "model_alias": model.get("_alias"),
        "model_repository": model.get("hf_repo"),
        "model_revision": model.get("revision"),
        "served_model_name": model.get("served_model_name"),
        "dtype": model.get("dtype"),
        "quantization": model.get("quantization"),
        "max_context_length": model.get("max_model_len"),
        "temperature": model.get("temperature"),
        "max_output_tokens": model.get("max_output_tokens"),
        "endpoint": endpoint,
        "methods": methods,
        "n_articles": len(inputs),
        "article_order": [r.article_id for r in inputs],
        "article_workers": 1,
        "unsafe_parallel_merge": False,
        "no_openai_fallback": True,
        "software": software,
        "cuda": cuda_version(),
        "hardware": hw_summary,
        "dataset_inputs_sha": file_sha256(ROOT / eval_cfg["dataset"]["input_path"]),
        "dataset_gt_sha": file_sha256(ROOT / eval_cfg["dataset"]["ground_truth_path"]),
        "graph_freeze": graph_stats,
        "expected_graph": {"nodes": 4661, "edges": 26035, "aliases": 8886},
        "vllm_probe": {"ok": probe.get("ok"), "url": probe.get("url")},
        "reference_gpt54_run": (local_cfg.get("reference") or {}).get("gpt54_run", "waiti_final_gpt54"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (run_dir / "STATUS.md").write_text(
        f"# {args.run_id}\n\n"
        f"- model_alias: `{model.get('_alias')}`\n"
        f"- hf_repo: `{model.get('hf_repo')}`\n"
        f"- methods: {', '.join(methods)}\n"
        f"- n_articles: {len(inputs)}\n"
        f"- endpoint: `{endpoint}`\n"
        f"- peak_vram_mib: {hw_summary.get('peak_memory_used_mib')}\n"
        f"- status: complete\n",
        encoding="utf-8",
    )
    print(f"Done local run. Results in {run_dir}")
    print("Optional: python -m evaluation.reports.validate_cluster_run --run-id", args.run_id)


if __name__ == "__main__":
    main()
