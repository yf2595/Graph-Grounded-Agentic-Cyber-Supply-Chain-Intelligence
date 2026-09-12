"""Integrity checks for eval_corpus_v1."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "dataset"

REQUIRED_STRATA = {
    "bottleneck_sole": 14,
    "critical_part_sole": 12,
    "redundant_high_reach": 12,
    "multi_supplier": 10,
    "moderate_reach": 12,
    "low_reach": 10,
    "off_graph_cyber": 15,
    "in_graph_non_cyber": 15,
}

VALID_BINS = {"very_low", "low", "medium", "high", "very_high", ""}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    incidents_path = ROOT / "incidents.csv"
    if not incidents_path.exists():
        fail("incidents.csv missing")

    rows = list(csv.DictReader(incidents_path.open(encoding="utf-8")))
    base = [r for r in rows if int(r["incident_id"].split("_")[1]) <= 100]
    variants = [r for r in rows if int(r["incident_id"].split("_")[1]) > 100]
    if len(base) != 100:
        fail(f"expected 100 base incidents, got {len(base)}")
    if len(set(r["incident_id"] for r in rows)) != len(rows):
        fail("incident_id not unique")
    # Merge subset: 25 groups, sizes {4:1, 3:2, 2:22} → 54 articles, 29 variants
    if variants and len(variants) != 29:
        fail(f"expected 29 merge variants, got {len(variants)}")
    groups = Counter(r["group_id"] for r in rows if r.get("group_id"))
    if groups:
        sizes = Counter(groups.values())
        if sizes.get(4, 0) != 1 or sizes.get(3, 0) != 2 or sizes.get(2, 0) != 22:
            fail(f"bad merge group sizes: {dict(sizes)}")
        if len(groups) != 25:
            fail(f"expected 25 merge groups, got {len(groups)}")

    # Load graph supplier ids
    node_ids = set()
    if (DATASET / "nodes.csv").exists():
        for n in csv.DictReader((DATASET / "nodes.csv").open(encoding="utf-8")):
            if n["node_kind"] == "organization":
                node_ids.add(n["node_id"])

    strata = Counter(r["case_stratum"] for r in base)
    for name, n in REQUIRED_STRATA.items():
        if strata.get(name, 0) != n:
            fail(f"stratum {name}: expected {n}, got {strata.get(name, 0)}")

    n_rel = sum(1 for r in base if r["relevant"] == "true")
    n_irr = sum(1 for r in base if r["relevant"] == "false")
    if n_rel != 70 or n_irr != 30:
        fail(f"relevant split expected 70/30, got {n_rel}/{n_irr}")

    for r in rows:
        is_cyber = r["is_cyber"] == "true"
        in_sc = r["in_supply_chain"] == "true"
        relevant = r["relevant"] == "true"
        if relevant != (is_cyber and in_sc):
            fail(f"{r['incident_id']}: relevant != is_cyber AND in_supply_chain")

        art = ROOT / "articles" / f"{r['incident_id']}.json"
        if not art.exists():
            fail(f"missing article {art.name}")
        payload = json.loads(art.read_text(encoding="utf-8"))
        if payload.get("headline") != r["headline"]:
            fail(f"{r['incident_id']}: article headline mismatch")
        if not payload.get("synthetic"):
            fail(f"{r['incident_id']}: article missing synthetic=true")
        if "SYNTHETIC" not in (payload.get("disclaimer") or ""):
            fail(f"{r['incident_id']}: article missing disclaimer")

        ids_field = [x for x in r["supplier_node_ids"].split("|") if x]
        if in_sc:
            if not ids_field and r["case_stratum"] != "off_graph_cyber":
                # off_graph should not be in_sc
                if not r["supplier_names"]:
                    fail(f"{r['incident_id']}: in_supply_chain but no suppliers")
            for sid in ids_field:
                if node_ids and sid not in node_ids:
                    fail(f"{r['incident_id']}: unknown supplier_node_id {sid}")
        else:
            if ids_field:
                fail(f"{r['incident_id']}: off-graph row has supplier_node_ids")

        if r["case_stratum"] == "off_graph_cyber":
            if is_cyber is False or in_sc:
                fail(f"{r['incident_id']}: off_graph_cyber label inconsistency")
        if r["case_stratum"] == "in_graph_non_cyber":
            if is_cyber or not in_sc:
                fail(f"{r['incident_id']}: in_graph_non_cyber label inconsistency")

        if relevant:
            if r["risk_bin_draft"] not in VALID_BINS - {""}:
                fail(f"{r['incident_id']}: bad risk_bin_draft {r['risk_bin_draft']}")
        else:
            if r["risk_bin_draft"]:
                fail(f"{r['incident_id']}: irrelevant row should have empty risk_bin_draft")

        if r["risk_bin_gold"] and r["risk_bin_gold"] not in VALID_BINS - {""}:
            fail(f"{r['incident_id']}: bad risk_bin_gold {r['risk_bin_gold']}")

        if r.get("is_multi_supplier") == "true" and r["case_stratum"] == "multi_supplier":
            if r["supplier_node_ids"].count("|") < 1:
                fail(f"{r['incident_id']}: multi_supplier needs >=2 suppliers")

    seeds_path = ROOT / "seeds.csv"
    if not seeds_path.exists():
        fail("seeds.csv missing")
    seeds = {s["seed_id"] for s in csv.DictReader(seeds_path.open(encoding="utf-8"))}
    for r in rows:
        if r["seed_id"] not in seeds:
            fail(f"{r['incident_id']}: unknown seed_id {r['seed_id']}")

    if not (ROOT / "review" / "risk_bin_review.md").exists():
        fail("review/risk_bin_review.md missing")
    if not (ROOT / "README.md").exists():
        fail("README.md missing")
    if not (ROOT / "dataset_manifest.json").exists():
        fail("dataset_manifest.json missing")

    n_rewritten_base = sum(1 for r in base if r.get("rewritten") == "true")
    if n_rewritten_base and n_rewritten_base != 100:
        fail(f"partial rewrite: {n_rewritten_base}/100 base have rewritten=true")
    for r in rows:
        if r.get("rewritten") == "true":
            art = json.loads((ROOT / "articles" / f"{r['incident_id']}.json").read_text(encoding="utf-8"))
            if int(r["incident_id"].split("_")[1]) <= 100:
                if not art.get("rewritten"):
                    fail(f"{r['incident_id']}: csv rewritten but article not flagged")
                if not art.get("rewrite_style"):
                    fail(f"{r['incident_id']}: missing rewrite_style")

    print("OK: eval_corpus_v1 validation passed")
    print(f"  incidents={len(rows)} base=100 variants={len(variants)} relevant_base=70")
    print(f"  strata={dict(strata)}")
    draft = Counter(r["risk_bin_draft"] for r in base if r["relevant"] == "true")
    print(f"  risk_bin_draft={dict(draft)}")
    n_gold = sum(1 for r in base if r["risk_bin_gold"])
    print(f"  risk_bin_gold_filled={n_gold}/70")
    if groups:
        print(f"  merge_groups={len(groups)} sizes={dict(Counter(groups.values()))}")


if __name__ == "__main__":
    main()
