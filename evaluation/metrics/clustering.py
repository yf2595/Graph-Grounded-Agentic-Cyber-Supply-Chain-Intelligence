"""Pairwise and B³ clustering metrics for merge eval."""
from __future__ import annotations

from collections import defaultdict

from evaluation.metrics.classification import prf


def pairwise_cluster_metrics(
    gt_labels: dict[str, str],
    pred_labels: dict[str, str],
) -> dict:
    """gt_labels/pred_labels: article_id -> cluster_id (only articles in both)."""
    articles = sorted(set(gt_labels) & set(pred_labels))

    def pairs(labels: dict[str, str]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for i, a in enumerate(articles):
            for b in articles[i + 1 :]:
                if labels[a] == labels[b]:
                    out.add((a, b))
        return out

    gp = pairs(gt_labels)
    pp = pairs(pred_labels)
    return prf(len(gp & pp), len(pp - gp), len(gp - pp))


def b3_metrics(gt_labels: dict[str, str], pred_labels: dict[str, str]) -> dict:
    """BCubed precision/recall/F1."""
    articles = sorted(set(gt_labels) & set(pred_labels))
    if not articles:
        return prf(0, 0, 0)

    # precision: for each article, fraction of pred-cluster mates that share gt label
    precs = []
    recalls = []
    pred_groups: dict[str, list[str]] = defaultdict(list)
    gt_groups: dict[str, list[str]] = defaultdict(list)
    for a in articles:
        pred_groups[pred_labels[a]].append(a)
        gt_groups[gt_labels[a]].append(a)

    for a in articles:
        pred_mates = pred_groups[pred_labels[a]]
        gt_mates = gt_groups[gt_labels[a]]
        correct = sum(1 for m in pred_mates if gt_labels[m] == gt_labels[a])
        precs.append(correct / len(pred_mates) if pred_mates else 0.0)
        correct_r = sum(1 for m in gt_mates if pred_labels[m] == pred_labels[a])
        recalls.append(correct_r / len(gt_mates) if gt_mates else 0.0)

    p = sum(precs) / len(precs)
    r = sum(recalls) / len(recalls)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
