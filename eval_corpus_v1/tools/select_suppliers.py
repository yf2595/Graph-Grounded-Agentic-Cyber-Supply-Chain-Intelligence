"""Compute supplier reach metrics and emit stratum candidate tables."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parents[1] / "candidates"

CRITICAL_CATEGORY_PATTERNS = [
    r"brake",
    r"abs\b",
    r"esc\b",
    r"ecu",
    r"airbag",
    r"steering",
    r"powertrain",
    r"engine ass",
    r"engine ecu",
    r"cvt",
    r"transmission",
    r"radar",
    r"lidar",
    r"wiring harness",
    r"turbo",
    r"fuel inject",
    r"seat belt",
]


def is_critical_category(name: str) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in CRITICAL_CATEGORY_PATTERNS)


def load_graph():
    nodes = list(csv.DictReader((DATASET / "nodes.csv").open(encoding="utf-8")))
    edges = list(csv.DictReader((DATASET / "edges.csv").open(encoding="utf-8")))
    id2name = {
        n["node_id"]: n["node_name"]
        for n in nodes
        if n["node_kind"] == "organization"
    }

    prod_sup: dict[str, set[str]] = defaultdict(set)
    sup_prods: dict[str, set[str]] = defaultdict(set)
    prod_vars: dict[str, set[str]] = defaultdict(set)
    prod_cat: dict[str, str] = {}

    for e in edges:
        et = e["edge_type"]
        if et == "supplies_product":
            prod_sup[e["target_id"]].add(e["source_id"])
            sup_prods[e["source_id"]].add(e["target_id"])
        elif et == "product_used_in_variant":
            prod_vars[e["source_id"]].add(e["target_id"])
        elif et == "product_in_category":
            prod_cat[e["source_id"]] = e["target_name"]

    sole_prods = {p for p, s in prod_sup.items() if len(s) == 1}

    rows = []
    for sid, prods in sorted(sup_prods.items()):
        gvars: set[str] = set()
        tvars: set[str] = set()
        sole_count = 0
        crit_sole = 0
        crit_cats: set[str] = set()
        for p in prods:
            tvars |= prod_vars.get(p, set())
            cat = prod_cat.get(p, "")
            if p in sole_prods:
                sole_count += 1
                gvars |= prod_vars.get(p, set())
                if cat and is_critical_category(cat):
                    crit_sole += 1
                    crit_cats.add(cat)
        rows.append(
            {
                "supplier_node_id": sid,
                "supplier_name": id2name.get(sid, sid),
                "guaranteed_reach": len(gvars),
                "total_reach": len(tvars),
                "sole_source_product_count": sole_count,
                "critical_sole_product_count": crit_sole,
                "critical_categories": "|".join(sorted(crit_cats)[:5]),
                "is_bottleneck": len(gvars) >= 30,
                "is_critical_part_sole": crit_sole > 0,
            }
        )
    return rows


def pick_strata(rows: list[dict]) -> dict[str, list[dict]]:
    by_id = {r["supplier_node_id"]: r for r in rows}
    used: set[str] = set()

    def take(pred, n, key=lambda r: (-r["guaranteed_reach"], -r["total_reach"])):
        cands = [r for r in rows if pred(r) and r["supplier_node_id"] not in used]
        cands.sort(key=key)
        picked = cands[:n]
        used.update(p["supplier_node_id"] for p in picked)
        return picked

    bottleneck = take(lambda r: r["guaranteed_reach"] >= 30, 18)
    critical = take(
        lambda r: r["critical_sole_product_count"] > 0 and r["guaranteed_reach"] >= 5,
        16,
        key=lambda r: (-r["critical_sole_product_count"], -r["guaranteed_reach"]),
    )
    redundant = take(
        lambda r: r["guaranteed_reach"] == 0 and r["total_reach"] >= 30,
        16,
        key=lambda r: -r["total_reach"],
    )
    moderate = take(
        lambda r: 10 <= r["guaranteed_reach"] <= 29,
        16,
    )
    low = take(
        lambda r: 1 <= r["guaranteed_reach"] <= 9,
        14,
        key=lambda r: (r["guaranteed_reach"], -r["total_reach"]),
    )

    # Multi-supplier pool: prefer unused + some reused bottlenecks/redundants
    multi_pool = (
        [by_id[i] for i in list(used) if by_id[i]["guaranteed_reach"] >= 20][:8]
        + [r for r in redundant if r["total_reach"] >= 100][:8]
        + [r for r in moderate if r["supplier_node_id"] not in {x["supplier_node_id"] for x in bottleneck}][:6]
    )

    return {
        "bottleneck_sole": bottleneck,
        "critical_part_sole": critical,
        "redundant_high_reach": redundant,
        "moderate_reach": moderate,
        "low_reach": low,
        "multi_supplier_pool": multi_pool,
        "non_cyber_pool": take(
            lambda r: r["total_reach"] >= 5,
            20,
            key=lambda r: -r["total_reach"],
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_graph()
    all_path = OUT / "supplier_metrics.csv"
    with all_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    strata = pick_strata(rows)
    summary = {}
    for name, items in strata.items():
        path = OUT / f"stratum_{name}.csv"
        if not items:
            summary[name] = 0
            continue
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
            w.writeheader()
            w.writerows(items)
        summary[name] = len(items)

    (OUT / "selection_summary.json").write_text(
        json.dumps({"total_suppliers": len(rows), "strata": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote metrics for {len(rows)} suppliers to {all_path}")


if __name__ == "__main__":
    main()
