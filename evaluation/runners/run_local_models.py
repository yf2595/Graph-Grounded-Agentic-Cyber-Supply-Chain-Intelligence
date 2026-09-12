"""Probe local vLLM and optionally launch WAITI local-model runs (P6)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "evaluation/config/local_models.yaml").read_text(encoding="utf-8")) or {}


def probe_endpoint(base_url: str, timeout: float = 3.0) -> dict:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "models": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def write_serving_record(out_dir: Path, model_key: str, cfg: dict, probe: dict) -> None:
    m = (cfg.get("models") or {}).get(model_key) or {}
    record = {
        "model_key": model_key,
        "model_name": m.get("display_name"),
        "hf_repo": m.get("hf_repo"),
        "revision": m.get("revision"),
        "dtype": m.get("dtype"),
        "quantization": m.get("quantization"),
        "max_context_length": m.get("max_model_len"),
        "temperature": m.get("temperature"),
        "max_output_tokens": m.get("max_output_tokens"),
        "endpoint": cfg.get("default_endpoint"),
        "hardware_target": cfg.get("hardware_target"),
        "probe": probe,
        "status": "ready" if probe.get("ok") else "endpoint_unavailable",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "serving_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--all-experiments", action="store_true", help="Probe/placeholder for all configured experiments")
    args = parser.parse_args()

    cfg = load_cfg()
    endpoint = cfg.get("default_endpoint") or "http://127.0.0.1:8000/v1"
    probe = probe_endpoint(endpoint)
    print("endpoint", endpoint, "ok" if probe.get("ok") else "DOWN", probe.get("error", ""))

    if args.probe_only:
        print(json.dumps(probe, indent=2))
        return

    experiments = list(cfg.get("experiments") or [])
    if args.all_experiments:
        selected = experiments
    else:
        exp = None
        if args.run_id:
            exp = next((e for e in experiments if e.get("run_id") == args.run_id), None)
        elif args.model_key:
            exp = next((e for e in experiments if e.get("model_key") == args.model_key), None)
        if not exp:
            raise SystemExit("Provide --run-id, --model-key, or --all-experiments")
        selected = [exp]

    for exp in selected:
        run_id = exp["run_id"]
        model_key = exp["model_key"]
        methods = exp.get("methods") or ["OURS"]
        out_dir = ROOT / "results" / run_id
        write_serving_record(out_dir, model_key, cfg, probe)

        if not probe.get("ok") and not args.force_run:
            ref = (cfg.get("reference") or {}).get("gpt54_ours_run")
            (out_dir / "STATUS.md").write_text(
                f"# {run_id}\n\n"
                f"Local endpoint `{endpoint}` unavailable.\n\n"
                f"Serve: `bash scripts/serve_local_vllm.sh {model_key}`\n\n"
                f"Then: `python -m evaluation.runners.run_local_models --run-id {run_id}`\n\n"
                f"GPT-5.4 reference run: `{ref}`.\n",
                encoding="utf-8",
            )
            print(f"Placeholder written: {out_dir} (endpoint down)")
            continue

        cmd = [
            sys.executable,
            "-m",
            "evaluation.runners.run_all",
            "--run-id",
            run_id,
            "--methods",
            ",".join(methods),
            "--skip-export",
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
