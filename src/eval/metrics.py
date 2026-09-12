"""Evaluation metrics for pipeline and baseline runs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

RISK_ORDER = ["very_low", "low", "medium", "high", "very_high"]


def _norm_risk(x: str | None) -> str | None:
    if not x:
        return None
    return x.strip().lower().replace(" ", "_")


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def load_gold(corpus_dir: Path) -> list[dict]:
    return list(csv.DictReader((corpus_dir / "incidents.csv").open(encoding="utf-8")))


def load_pipeline_preds(run_dir: Path) -> dict[str, dict]:
    out = {}
    for p in (run_dir / "articles").glob("*/result.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        aid = data.get("article_id") or p.parent.name
        out[aid] = data
    return out


def pred_is_cyber(pred: dict, mode: str) -> bool | None:
    if mode == "pipeline":
        a2 = pred.get("agent_02") or {}
        c = (a2.get("cyber_classification") or "").upper()
        if c == "CYBER":
            return True
        if c == "NOT_CYBER":
            return False
        # filtered NOT_CYBER
        if pred.get("filter_reason") == "NOT_CYBER":
            return False
        if pred.get("pipeline_status") == "COMPLETED":
            return True
        return None
    # baseline
    c = (pred.get("cyber_classification") or "").upper()
    if c == "CYBER":
        return True
    if c == "NOT_CYBER":
        return False
    if pred.get("filter_reason") == "NOT_CYBER":
        return False
    return None


def pred_relevant(pred: dict, mode: str) -> bool:
    if mode == "pipeline":
        return pred.get("pipeline_status") == "COMPLETED" or (
            (pred.get("agent_02") or {}).get("pipeline_status") == "PASSED"
        )
    return pred.get("pipeline_status") in ("COMPLETED", "PASSED") and bool(
        pred.get("supplier_node_ids") or pred.get("relevant_suppliers")
    )


def pred_supplier_ids(pred: dict, mode: str) -> set[str]:
    if mode == "pipeline":
        a2 = pred.get("agent_02") or {}
        ids = {s["supplier_node_id"] for s in a2.get("relevant_suppliers") or []}
        can = pred.get("canonical_incident") or {}
        ids |= set(can.get("supplier_node_ids") or [])
        return ids
    ids = set(pred.get("supplier_node_ids") or [])
    ids |= {s["supplier_node_id"] for s in pred.get("relevant_suppliers") or []}
    return ids


def pred_risk(pred: dict, mode: str) -> str | None:
    if mode == "pipeline":
        fr = pred.get("final_risk_assessment") or {}
        return _norm_risk(fr.get("overall_risk"))
    return _norm_risk(pred.get("overall_risk"))


def pred_event_id(pred: dict, mode: str, run_dir: Path) -> str | None:
    if mode == "pipeline":
        can = pred.get("canonical_incident") or {}
        return can.get("incident_id")
    # baseline: look up repo index
    idx = run_dir / "index.json"
    if idx.exists():
        data = json.loads(idx.read_text(encoding="utf-8"))
        return data.get("article_to_event", {}).get(pred.get("article_id"))
    return pred.get("merge_into_incident_id")


def pairwise_merge_f1(gold_groups: dict[str, set[str]], pred_groups: dict[str, set[str]], active: set[str]) -> dict:
    """Pairwise clustering F1 over articles that appear in gold groups AND were predicted."""
    articles = set()
    for s in gold_groups.values():
        articles |= s
    articles &= active

    def pairs(groups: dict[str, set[str]]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        inv = {}
        for gid, members in groups.items():
            for m in members:
                if m in articles:
                    inv[m] = gid
        arts = sorted(a for a in articles if a in inv)
        for i, a in enumerate(arts):
            for b in arts[i + 1 :]:
                if inv[a] == inv[b]:
                    out.add((a, b))
        return out

    gp = pairs(gold_groups)
    pp = pairs(pred_groups)
    tp = len(gp & pp)
    fp = len(pp - gp)
    fn = len(gp - pp)
    return prf(tp, fp, fn)


def evaluate(corpus_dir: Path, run_dir: Path, mode: str = "pipeline") -> dict:
    gold_rows = load_gold(corpus_dir)
    preds = load_pipeline_preds(run_dir)

    # Cyber
    tp = fp = fn = tn = 0
    for g in gold_rows:
        if g["incident_id"] not in preds:
            continue
        pred = preds[g["incident_id"]]
        gold_c = g["is_cyber"] == "true"
        pc = pred_is_cyber(pred, mode)
        if pc is None:
            continue
        if gold_c and pc:
            tp += 1
        elif not gold_c and pc:
            fp += 1
        elif gold_c and not pc:
            fn += 1
        else:
            tn += 1
    cyber = prf(tp, fp, fn)
    cyber["tn"] = tn

    # Relevance
    tp = fp = fn = 0
    for g in gold_rows:
        if g["incident_id"] not in preds:
            continue
        pred = preds[g["incident_id"]]
        gold_r = g["relevant"] == "true"
        pr = pred_relevant(pred, mode)
        if gold_r and pr:
            tp += 1
        elif not gold_r and pr:
            fp += 1
        elif gold_r and not pr:
            fn += 1
    relevance = prf(tp, fp, fn)

    # Supplier micro P/R (only rows with gold supplier ids or predicted)
    stp = sfp = sfn = 0
    for g in gold_rows:
        if g["incident_id"] not in preds:
            continue
        gold_ids = {x for x in g.get("supplier_node_ids", "").split("|") if x}
        pred_ids = pred_supplier_ids(preds[g["incident_id"]], mode)
        stp += len(gold_ids & pred_ids)
        sfp += len(pred_ids - gold_ids)
        sfn += len(gold_ids - pred_ids)
    suppliers = prf(stp, sfp, sfn)

    # Risk (relevant with draft bin)
    exact = adj = total = 0
    for g in gold_rows:
        if g["relevant"] != "true" or not g.get("risk_bin_draft"):
            continue
        if g["incident_id"] not in preds:
            continue
        if not pred_relevant(preds[g["incident_id"]], mode):
            continue
        gold_b = _norm_risk(g["risk_bin_draft"])
        pred_b = pred_risk(preds[g["incident_id"]], mode)
        if not gold_b or not pred_b:
            continue
        total += 1
        if gold_b == pred_b:
            exact += 1
            adj += 1
        else:
            try:
                if abs(RISK_ORDER.index(gold_b) - RISK_ORDER.index(pred_b)) <= 1:
                    adj += 1
            except ValueError:
                pass
    risk = {
        "n": total,
        "exact_accuracy": round(exact / total, 4) if total else None,
        "adjacent_accuracy": round(adj / total, 4) if total else None,
    }

    # Merge pairwise
    gold_groups: dict[str, set[str]] = defaultdict(set)
    for g in gold_rows:
        if g.get("group_id"):
            gold_groups[g["group_id"]].add(g["incident_id"])

    pred_groups: dict[str, set[str]] = defaultdict(set)
    for g in gold_rows:
        if not g.get("group_id"):
            continue
        if g["incident_id"] not in preds:
            continue
        eid = pred_event_id(preds[g["incident_id"]], mode, run_dir) or f"SOLO_{g['incident_id']}"
        pred_groups[eid].add(g["incident_id"])

    merge = (
        pairwise_merge_f1(dict(gold_groups), dict(pred_groups), set(preds.keys()))
        if gold_groups
        else {}
    )

    return {
        "mode": mode,
        "run_dir": str(run_dir),
        "n_preds": len(preds),
        "cyber": cyber,
        "relevance": relevance,
        "suppliers": suppliers,
        "risk": risk,
        "merge_pairwise": merge,
    }
