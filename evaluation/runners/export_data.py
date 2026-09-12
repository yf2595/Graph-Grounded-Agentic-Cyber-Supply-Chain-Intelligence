"""Export leak-safe inputs + ground truth from eval_corpus_v1."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.schemas.prediction import FORBIDDEN_INPUT_KEYS, GroundTruthRecord, ModelInputRecord

RISK_MAP = {
    "very_low": "VERY_LOW",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "very_high": "VERY_HIGH",
}


def export(corpus_dir: Path, out_dir: Path) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((corpus_dir / "incidents.csv").open(encoding="utf-8")))
    inputs_path = out_dir / "inputs.jsonl"
    gt_path = out_dir / "ground_truth.jsonl"

    n_in = n_gt = 0
    with inputs_path.open("w", encoding="utf-8") as fi, gt_path.open("w", encoding="utf-8") as fg:
        for r in rows:
            inp = ModelInputRecord(
                article_id=r["incident_id"],
                headline=r["headline"],
                body=r["body"],
            )
            dumped = inp.model_dump()
            bad = set(dumped) & FORBIDDEN_INPUT_KEYS
            if bad:
                raise RuntimeError(f"leakage keys in input: {bad}")
            fi.write(json.dumps(dumped, ensure_ascii=False) + "\n")
            n_in += 1

            risk_raw = (r.get("risk_bin_draft") or "").strip().lower()
            gt_risk = RISK_MAP.get(risk_raw) if risk_raw else None
            # Only actionable rows have meaningful risk gold
            if r.get("relevant") != "true":
                gt_risk = None

            gt = GroundTruthRecord(
                article_id=r["incident_id"],
                gt_is_cyber=r.get("is_cyber") == "true",
                gt_is_related=r.get("in_supply_chain") == "true",
                gt_is_actionable=r.get("relevant") == "true",
                gt_risk=gt_risk,  # type: ignore[arg-type]
                gt_group_id=(r.get("group_id") or None) or None,
                gt_supplier_node_ids=[x for x in (r.get("supplier_node_ids") or "").split("|") if x],
            )
            fg.write(json.dumps(gt.model_dump(), ensure_ascii=False) + "\n")
            n_gt += 1

    return n_in, n_gt


def validate_no_leakage(inputs_path: Path) -> None:
    for line in inputs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        leaked = set(obj) & FORBIDDEN_INPUT_KEYS
        if leaked:
            raise AssertionError(f"Forbidden keys in inputs: {leaked}")
        for k in ("article_id", "headline", "body"):
            if k not in obj:
                raise AssertionError(f"Missing required input field {k}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="eval_corpus_v1")
    parser.add_argument("--out", default="evaluation/data")
    args = parser.parse_args()
    n_in, n_gt = export(ROOT / args.corpus, ROOT / args.out)
    validate_no_leakage(ROOT / args.out / "inputs.jsonl")
    print(f"Exported {n_in} inputs and {n_gt} ground-truth rows to {args.out}")


if __name__ == "__main__":
    main()
