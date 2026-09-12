"""Supplier entity-resolution metrics (WAITI P2)."""
from __future__ import annotations


def supplier_set_metrics(
    gold_sets: list[set[str]],
    pred_sets: list[set[str]],
) -> dict:
    """Micro/macro P/R/F1 and exact set accuracy over supplier node ID sets."""
    if not gold_sets:
        return {
            "n": 0,
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "exact_set_accuracy": None,
        }

    tp = fp = fn = 0
    per_p: list[float] = []
    per_r: list[float] = []
    per_f1: list[float] = []
    exact = 0

    for g, p in zip(gold_sets, pred_sets):
        g = set(g or set())
        p = set(p or set())
        inter = g & p
        tp += len(inter)
        fp += len(p - g)
        fn += len(g - p)
        if g == p:
            exact += 1
        # per-example: empty gold and empty pred => perfect
        if not g and not p:
            per_p.append(1.0)
            per_r.append(1.0)
            per_f1.append(1.0)
            continue
        prec = len(inter) / len(p) if p else (1.0 if not g else 0.0)
        rec = len(inter) / len(g) if g else (1.0 if not p else 0.0)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_p.append(prec)
        per_r.append(rec)
        per_f1.append(f1)

    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return {
        "n": len(gold_sets),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(sum(per_p) / len(per_p), 4),
        "macro_recall": round(sum(per_r) / len(per_r), 4),
        "macro_f1": round(sum(per_f1) / len(per_f1), 4),
        "exact_set_accuracy": round(exact / len(gold_sets), 4),
    }
