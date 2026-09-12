"""Lightweight validation for a completed local LLM run.

Does not compute full metrics. Writes RUN_COMPLETE or RUN_INVALID.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORBIDDEN_HOST_FRAGMENTS = (
    "api.openai.com",
    "openai.azure.com",
    "anthropic.com",
    "googleapis.com",
    "openrouter.ai",
)

EXPECTED_N = 129


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _experiment_methods(local_cfg: dict, run_id: str, model_alias: str | None) -> list[str]:
    for exp in local_cfg.get("experiments") or []:
        if exp.get("run_id") == run_id:
            return list(exp.get("methods") or [])
        if model_alias and exp.get("model_alias") == model_alias:
            return list(exp.get("methods") or [])
    return []


def _parse_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"{path.name}:line {i}: {e}")
    return rows, errors


def _scan_for_openai(text: str) -> list[str]:
    hits = []
    low = text.lower()
    for frag in FORBIDDEN_HOST_FRAGMENTS:
        if frag in low:
            hits.append(frag)
    return hits


def validate_run(run_id: str, model_alias: str | None = None) -> list[str]:
    reasons: list[str] = []
    run_dir = ROOT / "results" / run_id
    if not run_dir.is_dir():
        return [f"missing run directory: {run_dir}"]

    local_cfg = _load_yaml(ROOT / "evaluation/config/local_models.yaml")
    methods = _experiment_methods(local_cfg, run_id, model_alias)
    if not methods:
        reasons.append(f"no experiment methods found in local_models.yaml for run_id={run_id}")

    # Manifest
    man_path = run_dir / "run_manifest.json"
    if not man_path.exists():
        reasons.append("missing run_manifest.json")
        manifest: dict = {}
    else:
        try:
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            reasons.append(f"run_manifest.json unparseable: {e}")
            manifest = {}

    alias = model_alias or manifest.get("model_alias")
    model_cfg = None
    if alias:
        models = local_cfg.get("models") or {}
        legacy = local_cfg.get("legacy_aliases") or {}
        key = legacy.get(alias, alias)
        model_cfg = models.get(key)
        if not model_cfg:
            reasons.append(f"model alias '{alias}' not in local_models.yaml")
        elif manifest:
            if manifest.get("model_repository") and manifest["model_repository"] != model_cfg.get("hf_repo"):
                reasons.append(
                    f"manifest model_repository mismatch: {manifest.get('model_repository')} "
                    f"vs config {model_cfg.get('hf_repo')}"
                )
            if manifest.get("served_model_name") and manifest["served_model_name"] != model_cfg.get(
                "served_model_name"
            ):
                reasons.append(
                    f"manifest served_model_name mismatch: {manifest.get('served_model_name')} "
                    f"vs config {model_cfg.get('served_model_name')}"
                )
            endpoint = str(manifest.get("endpoint") or "")
            for frag in _scan_for_openai(endpoint):
                reasons.append(f"manifest endpoint contains forbidden host: {frag}")
            if endpoint and "127.0.0.1" not in endpoint and "localhost" not in endpoint.lower():
                reasons.append(f"manifest endpoint is not local: {endpoint}")

    # Expected article IDs
    inputs_path = ROOT / "evaluation/data/inputs.jsonl"
    expected_ids: list[str] = []
    if inputs_path.exists():
        rows, errs = _parse_jsonl(inputs_path)
        reasons.extend(errs)
        expected_ids = [r.get("article_id") for r in rows if r.get("article_id")]
        if len(expected_ids) != EXPECTED_N:
            reasons.append(f"inputs.jsonl has {len(expected_ids)} articles; expected {EXPECTED_N}")
    else:
        reasons.append("missing evaluation/data/inputs.jsonl")

    # Method folders + predictions
    for mid in methods:
        mdir = run_dir / "methods" / mid
        pred_path = mdir / "predictions.jsonl"
        raw_path = run_dir / "raw" / f"{mid}.jsonl"
        if not mdir.is_dir():
            reasons.append(f"missing method folder: methods/{mid}")
            continue
        if not pred_path.exists():
            reasons.append(f"missing predictions: methods/{mid}/predictions.jsonl")
            continue
        preds, errs = _parse_jsonl(pred_path)
        reasons.extend(errs)
        got_ids = [p.get("article_id") for p in preds if p.get("article_id")]
        if expected_ids and set(got_ids) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(got_ids))
            extra = sorted(set(got_ids) - set(expected_ids))
            if missing:
                reasons.append(f"{mid}: missing {len(missing)} article IDs (e.g. {missing[:3]})")
            if extra:
                reasons.append(f"{mid}: unexpected extra article IDs: {extra[:3]}")
        if len(preds) < EXPECTED_N and expected_ids:
            reasons.append(f"{mid}: only {len(preds)} predictions; expected {EXPECTED_N}")
        if not raw_path.exists():
            reasons.append(f"missing raw/{mid}.jsonl")
        else:
            raw_rows, raw_errs = _parse_jsonl(raw_path)
            reasons.extend(raw_errs)
            if len(raw_rows) != len(preds):
                reasons.append(f"{mid}: raw/{mid}.jsonl count {len(raw_rows)} != methods count {len(preds)}")

    # Telemetry
    tel = run_dir / "telemetry"
    if not (tel / "hardware_metrics.json").exists():
        reasons.append("missing telemetry/hardware_metrics.json")
    if not (tel / "hardware_metrics.csv").exists():
        reasons.append("missing telemetry/hardware_metrics.csv")

    # STATUS.md
    if not (run_dir / "STATUS.md").exists():
        reasons.append("missing STATUS.md")

    # Scan logs/configs for OpenAI hosts (best-effort)
    scan_targets = [man_path]
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        for p in list(logs_dir.glob("*.out"))[:5] + list(logs_dir.glob("*.err"))[:5]:
            scan_targets.append(p)
    for p in scan_targets:
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for frag in _scan_for_openai(text):
            # Allow mentions in documentation-style "no openai" messages
            if "no openai" in text.lower() or "refusing" in text.lower():
                continue
            # Only flag if it looks like an actual URL usage
            if f"https://{frag}" in text.lower() or f"http://{frag}" in text.lower():
                reasons.append(f"forbidden host '{frag}' found in {p.relative_to(run_dir)}")

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a completed local LLM run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-alias", default="")
    args = parser.parse_args()

    run_dir = ROOT / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Clear old markers
    for marker in ("RUN_COMPLETE", "RUN_INVALID"):
        p = run_dir / marker
        if p.exists():
            p.unlink()

    reasons = validate_run(args.run_id, args.model_alias or None)
    if reasons:
        (run_dir / "RUN_INVALID").write_text("\n".join(reasons) + "\n", encoding="utf-8")
        status = "# RUN_INVALID\n\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        (run_dir / "STATUS.md").write_text(status, encoding="utf-8")
        print("RUN_INVALID")
        for r in reasons:
            print(f"  - {r}")
        raise SystemExit(1)

    (run_dir / "RUN_COMPLETE").write_text("ok\n", encoding="utf-8")
    # Preserve/refresh STATUS
    status_path = run_dir / "STATUS.md"
    if status_path.exists():
        text = status_path.read_text(encoding="utf-8")
        if "pending cluster validation" in text:
            text = text.replace("pending cluster validation marker", "RUN_COMPLETE")
            status_path.write_text(text, encoding="utf-8")
    else:
        status_path.write_text(f"# {args.run_id}\n\n- status: RUN_COMPLETE\n", encoding="utf-8")
    print("RUN_COMPLETE")


if __name__ == "__main__":
    main()
