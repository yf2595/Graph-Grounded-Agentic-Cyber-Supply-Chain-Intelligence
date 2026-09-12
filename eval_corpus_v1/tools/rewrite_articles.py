"""LLM-rewrite corpus articles for style/tone/name diversity.

Preserves gold labels (node ids, cyber/SC flags, strata, impact semantics).
Writes surface name forms used in the text separately from canonical names.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = ROOT / "articles"
BACKUP = ROOT / "articles_template_v0"
INCIDENTS = ROOT / "incidents.csv"
ALIASES_PATH = ROOT.parent / "dataset" / "aliases.csv"
MANIFEST = ROOT / "dataset_manifest.json"

STYLES = [
    "reuters_wire",
    "automotive_trade",
    "local_business_journal",
    "security_trade_blog",
    "corporate_press_paraphrase",
    "short_ap_brief",
    "investigative_feature_lede",
    "newsletter_digest",
    "japanese_english_wire",
    "supply_chain_risk_note",
]

NAME_STRATEGIES = [
    "full_legal",
    "short_brand",
    "initialism_or_acronym",
    "ticker_or_market_shorthand",
    "city_plant_unit_of_parent",
    "former_name_parenthetical",
    "mixed_first_mention_full_then_short",
]

DISCLAIMER = (
    "SYNTHETIC ARTICLE: this text was machine-generated for research evaluation; "
    "the named company is real (or fictional as noted) but the incident described "
    "is fictional and is not a report of anything that company experienced."
)


def _clean_alias(alias: str) -> bool:
    a = alias.strip()
    if not a or len(a) < 2:
        return False
    # Drop MarkLines/source numeric ids and bare codes
    if a.isdigit():
        return False
    if len(a) >= 8 and a.replace("-", "").isdigit():
        return False
    if a.lower().startswith("org_"):
        return False
    return True


def load_aliases() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not ALIASES_PATH.exists():
        return out
    for row in csv.DictReader(ALIASES_PATH.open(encoding="utf-8")):
        nid = row["node_id"]
        if not nid.startswith("ORG_"):
            continue
        alias = row["alias"]
        if not _clean_alias(alias):
            continue
        out.setdefault(nid, []).append(alias)
    return out


def pick_style(incident_id: str, rng: random.Random) -> str:
    # Deterministic-ish diversity by id, with RNG shuffle across corpus
    idx = int(incident_id.split("_")[1]) - 1
    return STYLES[idx % len(STYLES)] if rng.random() > 0.15 else rng.choice(STYLES)


def pick_name_strategy(incident_id: str, rng: random.Random) -> str:
    idx = int(incident_id.split("_")[1]) - 1
    return NAME_STRATEGIES[(idx * 3) % len(NAME_STRATEGIES)] if rng.random() > 0.2 else rng.choice(NAME_STRATEGIES)


def build_prompt(row: dict, style: str, name_strategy: str, alias_map: dict[str, list[str]]) -> str:
    ids = [x for x in row["supplier_node_ids"].split("|") if x]
    names = [x for x in row["supplier_names"].split("|") if x]
    alias_blocks = []
    for sid, name in zip(ids, names):
        aliases = alias_map.get(sid, [])[:8]
        alias_blocks.append(
            f"- canonical: {name}\n  node_id: {sid}\n  known_aliases: {aliases or ['(none)']}"
        )
    companies_block = "\n".join(alias_blocks) if alias_blocks else f"- off-graph / fictional: {row['supplier_names'] or '(see original)'}"

    constraints = {
        "is_cyber": row["is_cyber"] == "true",
        "in_supply_chain": row["in_supply_chain"] == "true",
        "impact_level": row["impact_level"],
        "case_stratum": row["case_stratum"],
        "multi_supplier": row["is_multi_supplier"] == "true",
        "n_suppliers_required": max(1, len(names)) if names else 1,
    }

    impact_rules = {
        "severe": "Must clearly indicate production halt, plant shutdown, or major ops stoppage.",
        "minor": "Limited / contained disruption only (slowdowns, admin IT, brief shipping delays). Not a full plant halt.",
        "none": "No production stoppage. IT/data incident or claim only; plants continue running.",
    }

    cyber_rules = (
        "This MUST remain a cybersecurity incident (ransomware, malware, BEC, data theft claim, etc.)."
        if constraints["is_cyber"]
        else "This MUST NOT be a cyber incident. Keep it as a non-cyber disruption (strike, fire, recall, flood, bankruptcy/restructuring) matching the original topic."
    )

    return f"""You rewrite synthetic evaluation news for an automotive OEM supply-chain cyber-intel benchmark.

GOAL: Make the article feel like a real, independently written news item — NOT a template with swapped names.

STYLE TARGET: {style}
NAME SURFACE STRATEGY: {name_strategy}

CANONICAL COMPANIES (must still clearly refer to these entities; do not invent a different real company):
{companies_block}

ORIGINAL HEADLINE:
{row['headline']}

ORIGINAL BODY:
{row['body']}

HARD CONSTRAINTS:
- {cyber_rules}
- Impact level = {constraints['impact_level']}: {impact_rules[constraints['impact_level']]}
- If multiple canonical suppliers are listed, the rewrite MUST mention all of them (using varied surface forms is fine).
- Do NOT invent OEM disruption claims unless the original already implies customer notification.
- Do NOT add ransom amounts, threat-actor names, or CVE details unless already present.
- Length: headline ≤ 110 chars; body 70–160 words. Vary sentence rhythm.
- Use concrete but plausible details (plant geography hints, system types, sourcing language) without copying the original wording.
- For name strategy: use initials, shortened brand, ticker-like shorthand, "X's Y plant", or former names where natural — but a reader should still recognize the firm. Prefer known_aliases when helpful.
- NEVER use numeric database ids, MarkLines codes, ORG_* ids, or bare digit strings as company names.
- Off-graph / fictional companies: you may slightly vary spelling of short forms, but keep them recognizably the same entity.

Return ONLY valid JSON (no markdown) with keys:
{{
  "headline": "...",
  "body": "...",
  "surface_names": ["exact strings used in the article for each company, in canonical order"],
  "style_notes": "one short sentence on the voice used"
}}
"""


def call_rewrite(client: OpenAI, model: str, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.95,
        presence_penalty=0.4,
        frequency_penalty=0.3,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert news rewriter for evaluation datasets. "
                    "Output strict JSON only. Preserve factual constraints given by the user."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content or "{}"
    return json.loads(text)


def backup_templates() -> None:
    if BACKUP.exists():
        return
    shutil.copytree(ARTICLES, BACKUP)
    print(f"Backed up templates to {BACKUP}")


def rewrite_one(
    client: OpenAI,
    model: str,
    row: dict,
    alias_map: dict[str, list[str]],
    rng: random.Random,
) -> dict:
    style = pick_style(row["incident_id"], rng)
    name_strategy = pick_name_strategy(row["incident_id"], rng)
    prompt = build_prompt(row, style, name_strategy, alias_map)
    last_err = None
    for attempt in range(3):
        try:
            out = call_rewrite(client, model, prompt)
            headline = (out.get("headline") or "").strip()
            body = (out.get("body") or "").strip()
            if not headline or not body:
                raise ValueError("empty headline/body")
            surface = out.get("surface_names") or []
            if isinstance(surface, str):
                surface = [surface]
            return {
                "incident_id": row["incident_id"],
                "headline": headline,
                "body": body,
                "surface_names": [str(s) for s in surface],
                "rewrite_style": style,
                "name_strategy": name_strategy,
                "style_notes": out.get("style_notes", ""),
            }
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{row['incident_id']} failed: {last_err}")


def apply_result(row: dict, r: dict) -> None:
    row["headline"] = r["headline"]
    row["body"] = r["body"]
    row["surface_supplier_names"] = "|".join(r["surface_names"])
    row["rewrite_style"] = r["rewrite_style"]
    row["name_strategy"] = r["name_strategy"]
    row["style_notes"] = r["style_notes"]
    row["rewritten"] = "true"

    art_path = ARTICLES / f"{row['incident_id']}.json"
    payload = json.loads(art_path.read_text(encoding="utf-8"))
    payload["headline"] = r["headline"]
    payload["body"] = r["body"]
    payload["surface_supplier_names"] = r["surface_names"]
    payload["canonical_supplier_names"] = (
        row["supplier_names"].split("|") if row["supplier_names"] else []
    )
    payload["rewrite_style"] = r["rewrite_style"]
    payload["name_strategy"] = r["name_strategy"]
    payload["style_notes"] = r["style_notes"]
    payload["rewritten"] = True
    payload["synthetic"] = True
    payload["disclaimer"] = DISCLAIMER
    art_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_incidents_csv(rows: list[dict]) -> None:
    extras = ("surface_supplier_names", "rewrite_style", "name_strategy", "style_notes", "rewritten")
    fieldnames = list(rows[0].keys())
    for extra in extras:
        if extra not in fieldnames:
            fieldnames.append(extra)
        for row in rows:
            row.setdefault(extra, "")
    with INCIDENTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def update_manifest(n: int, model: str) -> None:
    data = {}
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["rewrite"] = {
        "status": "completed",
        "model": model,
        "n_rewritten": n,
        "template_backup": "articles_template_v0/",
        "fields_added": [
            "surface_supplier_names",
            "rewrite_style",
            "name_strategy",
            "style_notes",
            "rewritten",
        ],
    }
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="If >0, only rewrite first N (debug)")
    parser.add_argument("--ids", default="", help="Comma-separated INC_xxx to rewrite only")
    args = parser.parse_args()

    client = OpenAI()
    alias_map = load_aliases()
    rows = list(csv.DictReader(INCIDENTS.open(encoding="utf-8")))
    backup_templates()

    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        targets = [r for r in rows if r["incident_id"] in want]
    else:
        targets = rows
    if args.limit:
        targets = targets[: args.limit]

    rng = random.Random(7)
    results: dict[str, dict] = {}
    print(f"Rewriting {len(targets)} articles with {args.model} ({args.workers} workers)...")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(rewrite_one, client, args.model, row, alias_map, random.Random(rng.randint(0, 10**9))): row[
                "incident_id"
            ]
            for row in targets
        }
        done = 0
        for fut in as_completed(futs):
            iid = futs[fut]
            try:
                results[iid] = fut.result()
            except Exception as e:
                print(f"ERROR {iid}: {e}")
                raise
            done += 1
            if done % 10 == 0 or done == len(targets):
                print(f"  {done}/{len(targets)}")

    by_id = {r["incident_id"]: r for r in rows}
    for iid, result in results.items():
        apply_result(by_id[iid], result)

    write_incidents_csv(rows)
    update_manifest(len(results), args.model)
    print(f"Done. Rewrote {len(results)} articles. Templates in {BACKUP}")


if __name__ == "__main__":
    main()
