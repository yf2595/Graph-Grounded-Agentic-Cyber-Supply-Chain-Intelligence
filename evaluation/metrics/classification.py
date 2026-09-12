"""Binary classification metrics."""
from __future__ import annotations


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def binary_prf(y_true: list[bool], y_pred: list[bool | None]) -> dict:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p is None:
            if t:
                fn += 1
            continue
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif t and not p:
            fn += 1
        else:
            tn += 1
    out = prf(tp, fp, fn)
    out["tn"] = tn
    return out
