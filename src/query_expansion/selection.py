"""Phase 1 - select 20 diverse, distinct actionable incidents.

Deterministic (seed 42). Deduplicates by ground-truth group so no two selected
cases are alternate reports of the same underlying incident. Maximizes coverage
of case_stratum / seed_id / supplier / risk class.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.query_expansion import config as C


@dataclass
class IncidentCandidate:
    incident_id: str          # canonical article id for the underlying incident
    group_key: str            # group_id if present else incident_id
    group_id: str | None
    member_article_ids: list[str] = field(default_factory=list)
    headline: str = ""
    body: str = ""
    supplier_names: list[str] = field(default_factory=list)
    supplier_node_ids: list[str] = field(default_factory=list)
    case_stratum: str = ""
    seed_id: str = ""
    template_family: str = ""
    impact_level: str = ""
    gt_risk: str | None = None
    is_originally_multi_report: bool = False

    def primary_supplier(self) -> str:
        return self.supplier_names[0] if self.supplier_names else ""


def _load_ground_truth() -> dict[str, dict]:
    gt: dict[str, dict] = {}
    for line in C.GROUND_TRUTH_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            gt[rec["article_id"]] = rec
    return gt


def _load_incidents_csv() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with C.INCIDENTS_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["incident_id"]] = row
    return rows


def _load_article(article_id: str) -> dict:
    path = C.ARTICLES_DIR / f"{article_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _article_num(article_id: str) -> int:
    try:
        return int(article_id.split("_")[-1])
    except ValueError:
        return 10**9


def load_candidates() -> list[IncidentCandidate]:
    """Return distinct actionable underlying incidents (deduplicated by group)."""
    gt = _load_ground_truth()
    inc = _load_incidents_csv()

    # Group actionable articles by their ground-truth group (or themselves).
    groups: dict[str, list[str]] = {}
    for aid, g in gt.items():
        if not g.get("gt_is_actionable"):
            continue
        # Prefer the incidents.csv group_id, fall back to ground-truth group.
        csv_row = inc.get(aid, {})
        group_id = (csv_row.get("group_id") or g.get("gt_group_id") or "").strip() or None
        key = group_id or aid
        groups.setdefault(key, []).append(aid)

    candidates: list[IncidentCandidate] = []
    for key, members in groups.items():
        members_sorted = sorted(members, key=_article_num)
        canonical = members_sorted[0]
        csv_row = inc.get(canonical, {})
        art = _load_article(canonical)
        g = gt.get(canonical, {})
        group_id = (csv_row.get("group_id") or g.get("gt_group_id") or "").strip() or None
        supplier_names = art.get("supplier_names") or (
            csv_row.get("supplier_names", "").split("|") if csv_row.get("supplier_names") else []
        )
        supplier_ids = art.get("supplier_node_ids") or (
            csv_row.get("supplier_node_ids", "").split("|") if csv_row.get("supplier_node_ids") else []
        )
        candidates.append(
            IncidentCandidate(
                incident_id=canonical,
                group_key=key,
                group_id=group_id,
                member_article_ids=members_sorted,
                headline=art.get("headline", "") or csv_row.get("headline", ""),
                body=art.get("body", "") or csv_row.get("body", ""),
                supplier_names=[s for s in supplier_names if s],
                supplier_node_ids=[s for s in supplier_ids if s],
                case_stratum=art.get("case_stratum") or csv_row.get("case_stratum", ""),
                seed_id=art.get("seed_id") or csv_row.get("seed_id", ""),
                template_family=csv_row.get("template_family", ""),
                impact_level=art.get("impact_level") or csv_row.get("impact_level", ""),
                gt_risk=g.get("gt_risk"),
                is_originally_multi_report=len(members_sorted) > 1,
            )
        )

    candidates.sort(key=lambda c: _article_num(c.incident_id))
    return candidates


def select_incidents(
    candidates: list[IncidentCandidate],
    n: int = C.N_TARGET_INCIDENTS,
    seed: int = C.SELECTION_SEED,
) -> list[IncidentCandidate]:
    """Greedy diversity selection maximizing distinct strata/seeds/suppliers/risks."""
    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)

    selected: list[IncidentCandidate] = []
    seen_strata: set[str] = set()
    seen_seeds: set[str] = set()
    seen_suppliers: set[str] = set()
    seen_risks: set[str] = set()
    seen_multi = {True: 0, False: 0}

    def score(c: IncidentCandidate) -> tuple:
        new_stratum = c.case_stratum not in seen_strata
        new_seed = c.seed_id not in seen_seeds
        new_supplier = c.primary_supplier() not in seen_suppliers
        new_risk = (c.gt_risk or "") not in seen_risks
        # Discourage re-using the same supplier heavily.
        supplier_penalty = 0 if new_supplier else -1
        # Encourage including at least some originally-multi-report incidents.
        multi_bonus = 1 if (c.is_originally_multi_report and seen_multi[True] < n // 4) else 0
        return (
            int(new_stratum),
            int(new_seed),
            int(new_risk),
            int(new_supplier) + supplier_penalty,
            multi_bonus,
            rng.random(),
        )

    while pool and len(selected) < n:
        pool.sort(key=score, reverse=True)
        pick = pool.pop(0)
        selected.append(pick)
        seen_strata.add(pick.case_stratum)
        seen_seeds.add(pick.seed_id)
        seen_suppliers.add(pick.primary_supplier())
        seen_risks.add(pick.gt_risk or "")
        seen_multi[pick.is_originally_multi_report] += 1

    selected.sort(key=lambda c: _article_num(c.incident_id))
    return selected


def _distribution(selected: list[IncidentCandidate], attr: str) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(getattr(c, attr) or "(none)" for c in selected))


def write_outputs(selected: list[IncidentCandidate]) -> None:
    C.ensure_artifacts_dir()
    payload = {
        "selection_seed": C.SELECTION_SEED,
        "n_selected": len(selected),
        "filter": "gt_is_actionable AND is_cyber AND in_supply_chain AND relevant; deduped by group",
        "incidents": [asdict(c) for c in selected],
    }
    C.SELECTED_INCIDENTS_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Query-Expansion Target Selection",
        "",
        f"Seed: {C.SELECTION_SEED}. Selected {len(selected)} distinct actionable incidents.",
        "",
        "## Diversity coverage",
        "",
        f"- Distinct case strata: {len(set(c.case_stratum for c in selected))} "
        f"-> {_distribution(selected, 'case_stratum')}",
        f"- Distinct seed incidents: {len(set(c.seed_id for c in selected))}",
        f"- Distinct primary suppliers: {len(set(c.primary_supplier() for c in selected))}",
        f"- Risk classes: {_distribution(selected, 'gt_risk')}",
        f"- Originally multi-report incidents: {sum(1 for c in selected if c.is_originally_multi_report)}",
        "",
        "## Selected incidents",
        "",
        "| Incident | Group | Stratum | Seed | GT risk | Supplier | Multi-report |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in selected:
        lines.append(
            f"| {c.incident_id} | {c.group_id or '-'} | {c.case_stratum} | {c.seed_id} | "
            f"{c.gt_risk or '-'} | {c.primary_supplier()} | {'yes' if c.is_originally_multi_report else 'no'} |"
        )
    C.SELECTION_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> list[IncidentCandidate]:
    candidates = load_candidates()
    selected = select_incidents(candidates)
    if len(selected) != C.N_TARGET_INCIDENTS:
        raise RuntimeError(
            f"Expected {C.N_TARGET_INCIDENTS} selected incidents, got {len(selected)} "
            f"(only {len(candidates)} distinct actionable candidates available)."
        )
    write_outputs(selected)
    return selected


def load_selected() -> list[IncidentCandidate]:
    data = json.loads(C.SELECTED_INCIDENTS_JSON.read_text(encoding="utf-8"))
    return [IncidentCandidate(**c) for c in data["incidents"]]


if __name__ == "__main__":
    sel = run()
    print(f"Selected {len(sel)} incidents -> {C.SELECTED_INCIDENTS_JSON}")
    for c in sel:
        print(f"  {c.incident_id}  {c.case_stratum:20s}  {c.gt_risk:10s}  {c.primary_supplier()}")
