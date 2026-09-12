"""Risk classification metrics (WAITI revision P1)."""
from __future__ import annotations

from collections import Counter

RISK_ORDER = ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
RISK_SCORE = {r: i + 1 for i, r in enumerate(RISK_ORDER)}
NO_PREDICTION = "__NO_PREDICTION__"


def _norm(r: str | None) -> str | None:
    if not r or r == "UNKNOWN" or r == NO_PREDICTION:
        return None
    v = r.upper().replace(" ", "_")
    return v if v in RISK_SCORE else None


def risk_metrics(
    y_true: list[str | None],
    y_pred: list[str | None],
    unknown_penalty: int = 4,
) -> dict:
    """End-to-end / conditional risk metrics.

    Macro-F1: missing predictions stay in the population as FN for the true label
    (averaged only over RISK_ORDER — NO_PREDICTION is not a label).
    QWK: detected-only (rows with a valid predicted risk).
    Exact/adjacent/MAE: missing predictions count as wrong / penalty.
    """
    pairs: list[tuple[str, str | None]] = []
    for t, p in zip(y_true, y_pred):
        nt = _norm(t)
        if nt is None:
            continue
        np_ = _norm(p)
        pairs.append((nt, np_))

    empty = {
        "n": 0,
        "coverage": 0.0,
        "exact_accuracy": None,
        "adjacent_accuracy": None,
        "mae": None,
        "macro_f1": None,
        "quadratic_weighted_kappa": None,
        "risk_qwk_detected": None,
    }
    if not pairs:
        return empty

    covered = [(t, p) for t, p in pairs if p is not None]
    coverage = len(covered) / len(pairs)

    exact = adj = 0
    abs_errs: list[float] = []
    for t, p in pairs:
        if p is None:
            abs_errs.append(float(unknown_penalty))
            continue
        if t == p:
            exact += 1
            adj += 1
            abs_errs.append(0.0)
        else:
            abs_errs.append(float(abs(RISK_SCORE[t] - RISK_SCORE[p])))
            if abs(RISK_SCORE[t] - RISK_SCORE[p]) <= 1:
                adj += 1

    # Macro-F1 over all pairs: missing pred => FN for true class, not a predicted class
    f1s = []
    for lab in RISK_ORDER:
        tp = sum(1 for t, p in pairs if t == lab and p == lab)
        fp = sum(1 for t, p in pairs if t != lab and p == lab)
        fn = sum(1 for t, p in pairs if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro_f1 = sum(f1s) / len(f1s)

    kappa = _quadratic_weighted_kappa([t for t, p in covered], [p for t, p in covered])

    return {
        "n": len(pairs),
        "coverage": round(coverage, 4),
        "exact_accuracy": round(exact / len(pairs), 4),
        "adjacent_accuracy": round(adj / len(pairs), 4),
        "mae": round(sum(abs_errs) / len(abs_errs), 4),
        "macro_f1": round(macro_f1, 4),
        "quadratic_weighted_kappa": round(kappa, 4) if kappa is not None else None,
        "risk_qwk_detected": round(kappa, 4) if kappa is not None else None,
    }


def _quadratic_weighted_kappa(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true:
        return None
    n = len(RISK_ORDER)
    idx = {r: i for i, r in enumerate(RISK_ORDER)}
    O = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            continue
        O[idx[t]][idx[p]] += 1
    hist_t = Counter(idx[t] for t in y_true if t in idx)
    hist_p = Counter(idx[p] for p in y_pred if p in idx)
    N = sum(sum(row) for row in O)
    if N == 0:
        return None
    E = [[(hist_t[i] * hist_p[j]) / N for j in range(n)] for i in range(n)]
    W = [[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    num = sum(W[i][j] * O[i][j] for i in range(n) for j in range(n))
    den = sum(W[i][j] * E[i][j] for i in range(n) for j in range(n))
    if den == 0:
        return 1.0
    return 1.0 - num / den
