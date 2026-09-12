"""Generate merge-eval variants from 25 relevant seed incidents.

Creates:
  - 1 group of 4 (1 seed + 3 rewrites)
  - 2 groups of 3 (1 seed + 2 rewrites each)
  - 22 groups of 2 (1 seed + 1 rewrite each)
Total: 54 articles in merge set (25 seeds + 29 new INC_101..INC_129)
"""
from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = ROOT / "articles"
INCIDENTS = ROOT / "incidents.csv"
DISCLAIMER = (
    "SYNTHETIC ARTICLE: this text was machine-generated for research evaluation; "
    "the named company is real (or fictional as noted) but the incident described "
    "is fictional and is not a report of anything that company experienced."
)

STYLES = [
    "reuters_wire",
    "security_trade_blog",
    "automotive_trade",
    "local_business_journal",
    "short_ap_brief",
    "newsletter_digest",
]


def load_rows() -> list[dict]:
    return list(csv.DictReader(INCIDENTS.open(encoding="utf-8")))


def pick_seeds(rows: list[dict], n: int = 25) -> list[dict]:
    rel = [r for r in rows if r["relevant"] == "true"]
    by_stratum: dict[str, list[dict]] = {}
    for r in rel:
        by_stratum.setdefault(r["case_stratum"], []).append(r)
    rng = random.Random(42)
    picked: list[dict] = []
    strata = sorted(by_stratum.keys())
    while len(picked) < n:
        for s in strata:
            pool = [x for x in by_stratum[s] if x not in picked]
            if pool and len(picked) < n:
                picked.append(rng.choice(pool))
    return picked[:n]


def rewrite_once(
    client: OpenAI,
    model: str,
    seed: dict,
    style: str,
    variant_index: int,
) -> dict:
    names = [x for x in seed["supplier_names"].split("|") if x]
    prompt = f"""Rewrite this synthetic news article for a merge-evaluation dataset.

GOAL: Same real-world event, same supplier(s), same cyber/event type, same operational impact.
Change structure, wording, outlet voice, and company surface forms (short brand/initials OK).
A reader should still recognize it as the SAME incident about the SAME companies.

STYLE: {style}
VARIANT INDEX: {variant_index}

CANONICAL SUPPLIERS (must remain the same entities): {names}
IMPACT LEVEL (preserve semantics): {seed['impact_level']}
  - severe = production halt / major ops stoppage
  - minor = limited disruption only
  - none = no production stoppage

ORIGINAL HEADLINE:
{seed['headline']}

ORIGINAL BODY:
{seed['body']}

Rules:
- Do NOT change the underlying event facts or suppliers.
- Do NOT invent OEM disruption claims.
- NEVER use numeric database IDs as company names.
- headline ≤ 110 chars; body 70–160 words.

Return ONLY JSON:
{{"headline":"...","body":"...","surface_names":["..."],"style_notes":"..."}}
"""
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.95,
                presence_penalty=0.5,
                frequency_penalty=0.35,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You rewrite news for evaluation datasets. Output strict JSON only. Preserve event identity.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            out = json.loads(resp.choices[0].message.content or "{}")
            headline = (out.get("headline") or "").strip()
            body = (out.get("body") or "").strip()
            if not headline or not body:
                raise ValueError("empty")
            surface = out.get("surface_names") or names
            if isinstance(surface, str):
                surface = [surface]
            return {
                "headline": headline,
                "body": body,
                "surface_names": [str(s) for s in surface],
                "style_notes": out.get("style_notes", ""),
                "rewrite_style": style,
            }
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"rewrite failed for {seed['incident_id']}: {last_err}")


def main() -> None:
    client = OpenAI()
    model = "gpt-4o"
    rows = load_rows()
    rows = [
        r
        for r in rows
        if r.get("is_merge_variant") != "true"
        and not (
            r["incident_id"].startswith("INC_")
            and int(r["incident_id"].split("_")[1]) > 100
        )
    ]
    for r in rows:
        r["group_id"] = ""
        r["is_merge_variant"] = "false"
        r["merge_seed_article_id"] = ""

    seeds = pick_seeds(rows, 25)
    by_id = {r["incident_id"]: r for r in rows}

    plans: list[tuple[dict, int, str]] = []
    for i, seed in enumerate(seeds):
        gid = f"GRP_{i+1:03d}"
        if i == 0:
            n_rewrites = 3
        elif i in (1, 2):
            n_rewrites = 2
        else:
            n_rewrites = 1
        plans.append((seed, n_rewrites, gid))

    for seed, _, gid in plans:
        by_id[seed["incident_id"]]["group_id"] = gid
        by_id[seed["incident_id"]]["is_merge_variant"] = "false"
        by_id[seed["incident_id"]]["merge_seed_article_id"] = seed["incident_id"]

    jobs = []
    next_id = 101
    rng = random.Random(7)
    for seed, n_rewrites, gid in plans:
        for v in range(n_rewrites):
            iid = f"INC_{next_id:03d}"
            next_id += 1
            jobs.append((iid, seed, gid, rng.choice(STYLES), v + 1))

    assert len(jobs) == 29, f"expected 29 rewrites, got {len(jobs)}"

    print(f"Rewriting {len(jobs)} merge variants with {model}...")
    results: dict[str, dict] = {}
    job_meta = {iid: (seed, gid, style) for iid, seed, gid, style, vidx in jobs}

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(rewrite_once, client, model, seed, style, vidx): iid
            for iid, seed, gid, style, vidx in jobs
        }
        done = 0
        for fut in as_completed(futs):
            iid = futs[fut]
            results[iid] = fut.result()
            done += 1
            if done % 5 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}")

    new_rows = []
    for iid, (seed, gid, style) in job_meta.items():
        r = results[iid]
        row = dict(seed)
        row.update(
            {
                "incident_id": iid,
                "headline": r["headline"],
                "body": r["body"],
                "surface_supplier_names": "|".join(r["surface_names"]),
                "rewrite_style": r["rewrite_style"],
                "name_strategy": "merge_variant",
                "style_notes": r["style_notes"],
                "rewritten": "true",
                "group_id": gid,
                "is_merge_variant": "true",
                "merge_seed_article_id": seed["incident_id"],
                "synthetic": "true",
                "disclaimer": DISCLAIMER,
            }
        )
        new_rows.append(row)

        payload = {
            "article_id": iid,
            "incident_id": iid,
            "headline": r["headline"],
            "body": r["body"],
            "supplier_names": seed["supplier_names"].split("|") if seed["supplier_names"] else [],
            "supplier_node_ids": seed["supplier_node_ids"].split("|") if seed["supplier_node_ids"] else [],
            "is_cyber": seed["is_cyber"] == "true",
            "in_supply_chain": seed["in_supply_chain"] == "true",
            "relevant": seed["relevant"] == "true",
            "case_stratum": seed["case_stratum"],
            "impact_level": seed["impact_level"],
            "seed_id": seed["seed_id"],
            "group_id": gid,
            "is_merge_variant": True,
            "merge_seed_article_id": seed["incident_id"],
            "surface_supplier_names": r["surface_names"],
            "canonical_supplier_names": seed["supplier_names"].split("|") if seed["supplier_names"] else [],
            "rewrite_style": r["rewrite_style"],
            "name_strategy": "merge_variant",
            "style_notes": r["style_notes"],
            "rewritten": True,
            "synthetic": True,
            "disclaimer": DISCLAIMER,
        }
        (ARTICLES / f"{iid}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    for seed, _, gid in plans:
        art_path = ARTICLES / f"{seed['incident_id']}.json"
        payload = json.loads(art_path.read_text(encoding="utf-8"))
        payload["group_id"] = gid
        payload["is_merge_variant"] = False
        payload["merge_seed_article_id"] = seed["incident_id"]
        art_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    all_rows = list(by_id.values()) + new_rows
    all_rows.sort(key=lambda r: r["incident_id"])

    extras = ["group_id", "is_merge_variant", "merge_seed_article_id"]
    fieldnames = list(all_rows[0].keys())
    for e in extras:
        if e not in fieldnames:
            fieldnames.append(e)
        for r in all_rows:
            r.setdefault(e, "")

    with INCIDENTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    groups = Counter(r["group_id"] for r in all_rows if r.get("group_id"))
    print(f"Wrote {len(all_rows)} incidents; merge groups={len(groups)}")
    print("group sizes", Counter(groups.values()))
    print("new variants", len(new_rows))


if __name__ == "__main__":
    main()
