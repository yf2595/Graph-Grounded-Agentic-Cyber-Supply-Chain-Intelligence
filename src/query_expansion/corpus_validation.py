"""Phase 2/3 - assemble the 60-document closed corpus and validate it.

Composes the frozen searchable ``documents.jsonl`` and the hidden
``ground_truth.jsonl`` from the generated supplement/noise staging files, then
enforces the Section 7 checklist (counts, uniqueness, no exact/near duplicates,
no OEM-graph leakage, metadata separated from searchable text).
"""
from __future__ import annotations

import hashlib
import json
import re

from src.query_expansion import config as C
from src.query_expansion.selection import load_selected

# Searchable document fields only. Everything else is hidden ground truth.
DOC_FIELDS = ("doc_id", "headline", "body", "combined_search_text")

# Private OEM-graph-only phrases that must never leak into generated text.
# Patterns are word-bounded and context-specific to avoid false positives on
# benign words (e.g. "geographically", "redundant backups").
LEAKAGE_PATTERNS = [
    r"\bsole[- ]?source",
    r"\bsingle[- ]?source",
    r"\bbottleneck",
    r"\bsupplier redundanc",
    r"\bsourcing redundanc",
    r"\bredundant suppl",
    r"\bdownstream reach",
    r"\bguaranteed reach",
    r"\btotal reach",
    r"\bcriticality\b",
    r"\b(?:supply[- ]?chain|oem|dependency)[- ]?graph\b",
    r"\bgraph node\b",
    r"\bnode id\b",
    r"org_[0-9a-f]{6,}",
    r"\bvehicle program",
]


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def build_corpus() -> tuple[list[dict], list[dict]]:
    """Return (documents, ground_truth) from staging files; also freeze to disk."""
    supp = [
        json.loads(l)
        for l in C.SUPPLEMENTS_RAW_JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    noise = [
        json.loads(l)
        for l in C.NOISE_RAW_JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    documents: list[dict] = []
    ground_truth: list[dict] = []
    for rec in supp + noise:
        doc = rec["document"]
        combined = f"{doc['headline']}\n\n{doc['body']}".strip()
        documents.append(
            {
                "doc_id": doc["doc_id"],
                "headline": doc["headline"],
                "body": doc["body"],
                "combined_search_text": combined,
            }
        )
        ground_truth.append(rec["metadata"])

    C.ensure_artifacts_dir()
    with C.DOCUMENTS_JSONL.open("w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with C.GROUND_TRUTH_OUT_JSONL.open("w", encoding="utf-8") as f:
        for g in ground_truth:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    return documents, ground_truth


def _corpus_hash(documents: list[dict]) -> str:
    h = hashlib.sha256()
    for d in sorted(documents, key=lambda x: x["doc_id"]):
        h.update((d["doc_id"] + "\x1f" + d["combined_search_text"]).encode("utf-8"))
    return h.hexdigest()[:16]


def validate(documents: list[dict], ground_truth: list[dict]) -> dict:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    gt_by_id = {g["doc_id"]: g for g in ground_truth}
    supp = [g for g in ground_truth if not g.get("is_noise")]
    noise = [g for g in ground_truth if g.get("is_noise")]

    check("exactly 40 supplemental reports", len(supp) == 40, f"{len(supp)}")
    check("exactly 20 noise reports", len(noise) == 20, f"{len(noise)}")
    check("exactly 60 total documents", len(documents) == 60, f"{len(documents)}")

    # 2 supplements per target, each mapping to exactly one target
    from collections import Counter

    per_target = Counter(g["target_incident_id"] for g in supp)
    try:
        selected_ids = {c.incident_id for c in load_selected()}
    except FileNotFoundError:
        selected_ids = {g["target_incident_id"] for g in supp if g.get("target_incident_id")}
    two_each = all(per_target[t] == 2 for t in selected_ids) and set(per_target) == selected_ids
    check(
        "every target incident has exactly 2 supplements",
        two_each,
        f"targets={len(per_target)} expected={len(selected_ids)}",
    )
    check(
        "every supplement maps to exactly 1 target",
        all(g.get("target_incident_id") for g in supp),
    )
    check(
        "no noise report maps to a target incident",
        all(not g.get("target_incident_id") for g in noise),
    )

    # unique doc ids
    ids = [d["doc_id"] for d in documents]
    check("no duplicate doc IDs", len(ids) == len(set(ids)))

    # exact duplicate text
    norms = [_norm_text(d["combined_search_text"]) for d in documents]
    check("no exact duplicate text", len(norms) == len(set(norms)))

    # near-duplicate detection among supplements (log, do not delete)
    near_dups = _near_duplicates(documents)
    check(
        "no near-duplicate supplemental reports",
        len(near_dups) == 0,
        f"flagged={near_dups}",
    )

    # OEM graph leakage in searchable text
    leaks = []
    for d in documents:
        text = d["combined_search_text"].lower()
        for pat in LEAKAGE_PATTERNS:
            if re.search(pat, text):
                leaks.append((d["doc_id"], pat))
    check("no graph-only OEM attributes leaked into generated text", len(leaks) == 0, f"{leaks}")

    # metadata separated from searchable text
    stray = [d["doc_id"] for d in documents if any(k not in DOC_FIELDS for k in d.keys())]
    check("all metadata stored separately from searchable text", len(stray) == 0, f"{stray}")

    all_ok = all(ok for _, ok, _ in checks)
    return {
        "all_ok": all_ok,
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "near_duplicates": near_dups,
        "leaks": leaks,
        "corpus_hash": _corpus_hash(documents),
    }


def _near_duplicates(documents: list[dict], threshold: float = 0.98) -> list[dict]:
    from src.services.embedding_service import EmbeddingService

    emb = EmbeddingService(mode="local")
    vecs = {d["doc_id"]: emb.embed(d["combined_search_text"]) for d in documents}
    ids = list(vecs)
    flagged: list[dict] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = EmbeddingService.cosine(vecs[ids[i]], vecs[ids[j]])
            if sim >= threshold:
                flagged.append({"a": ids[i], "b": ids[j], "cosine": round(sim, 4)})
    return flagged


def write_manifest(documents: list[dict], ground_truth: list[dict], result: dict) -> None:
    manifest = {
        "corpus": "query_expansion_closed_v1",
        "n_documents": len(documents),
        "n_supplements": sum(1 for g in ground_truth if not g.get("is_noise")),
        "n_noise": sum(1 for g in ground_truth if g.get("is_noise")),
        "n_targets": len({g["target_incident_id"] for g in ground_truth if not g.get("is_noise")}),
        "generation_model": C.GEN_MODEL,
        "generation_reasoning_effort": C.GEN_REASONING_EFFORT,
        "generation_seed": C.GENERATION_SEED,
        "corpus_hash": result["corpus_hash"],
        "validation": {"all_ok": result["all_ok"], "checks": result["checks"]},
    }
    C.CORPUS_MANIFEST_JSON.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run(strict: bool = True) -> dict:
    documents, ground_truth = build_corpus()
    result = validate(documents, ground_truth)
    write_manifest(documents, ground_truth, result)
    print("Corpus validation:")
    for c in result["checks"]:
        print(f"  [{'x' if c['ok'] else ' '}] {c['name']} {c['detail']}")
    if strict and not result["all_ok"]:
        raise RuntimeError("Corpus validation failed; see corpus_manifest.json")
    return result


if __name__ == "__main__":
    run()
