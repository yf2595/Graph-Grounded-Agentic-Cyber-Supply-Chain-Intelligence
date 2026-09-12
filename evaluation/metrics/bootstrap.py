"""Bootstrap CIs and paired deltas."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    estimates = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    alpha = 1 - confidence
    lo = estimates[int(alpha / 2 * (n_resamples - 1))]
    hi = estimates[int((1 - alpha / 2) * (n_resamples - 1))]
    return {
        "mean": round(sum(values) / len(values), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
    }


def bootstrap_metric_by_group(
    article_to_group: dict[str, str],
    article_ids: list[str],
    metric_fn: Callable[[list[str]], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Resample groups (with replacement), recompute metric on articles in sampled groups."""
    groups: dict[str, list[str]] = defaultdict(list)
    for aid in article_ids:
        gid = article_to_group.get(aid) or f"SOLO_{aid}"
        groups[gid].append(aid)
    gids = list(groups.keys())
    if not gids:
        return {"mean": None, "ci_low": None, "ci_high": None}

    rng = random.Random(seed)
    estimates = []
    for _ in range(n_resamples):
        sample_g = [gids[rng.randrange(len(gids))] for _ in range(len(gids))]
        sample_articles = []
        for i, g in enumerate(sample_g):
            # unique temp ids if group repeated
            for a in groups[g]:
                sample_articles.append(f"{a}__{i}")
        # metric_fn must accept remapped ids — for simple metrics we pass original by stripping
        orig = [a.split("__")[0] for a in sample_articles]
        try:
            estimates.append(metric_fn(orig))
        except Exception:
            continue
    if not estimates:
        return {"mean": None, "ci_low": None, "ci_high": None}
    estimates.sort()
    alpha = 1 - confidence
    lo = estimates[int(alpha / 2 * (len(estimates) - 1))]
    hi = estimates[int((1 - alpha / 2) * (len(estimates) - 1))]
    return {
        "mean": round(sum(estimates) / len(estimates), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
    }


def paired_delta_ci(
    values_a: list[float],
    values_b: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for mean(a - b)."""
    assert len(values_a) == len(values_b)
    if not values_a:
        return {"delta": None, "ci_low": None, "ci_high": None}
    deltas = [a - b for a, b in zip(values_a, values_b)]
    ci = bootstrap_ci(deltas, n_resamples=n_resamples, confidence=confidence, seed=seed)
    return {"delta": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
