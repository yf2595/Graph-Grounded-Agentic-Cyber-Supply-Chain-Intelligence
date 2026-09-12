#!/usr/bin/env python3
"""Standalone integrity checker for silent_pair_dataset_v1.0.

Stdlib only. No project code, no network, no dependencies.

    cd silent_pair_dataset_v1.0 && python validate.py

Checks
  1. file counts       every article listed in dataset_manifest.json exists, and no
                       unlisted article files are present
  2. checksums         every entry in checksums.sha256 matches the file on disk
  3. byte identity     on 3 sampled pairs, substituting the crit supplier's name for
                       the foil supplier's name in the foil text reproduces the crit
                       text exactly (the identical-text construction property)
  4. gold recompute    every cases.csv gold_binary equals
                       escalate iff (halt_pin AND guaranteed_reach >= 30)
  5. disclaimer        every article JSON carries a non-empty disclaimer

Exit code 0 if all checks PASS, 1 otherwise.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THRESHOLD = 30
SAMPLE_PAIRS = ["pair_01", "pair_07", "pair_13"]

results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def load_manifest():
    return json.loads((ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))


# --- 1. file counts --------------------------------------------------------
def check_counts(dsm):
    listed = set(dsm["articles"])
    missing = sorted(r for r in listed if not (ROOT / r).is_file())
    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in list((ROOT / "articles").rglob("*.json"))
        + list((ROOT / "robustness_subset").rglob("*.json"))
    }
    unlisted = sorted(on_disk - listed)
    c = dsm["counts"]
    count_ok = (
        len(listed) == c["articles_total"]
        and sum(1 for r in listed if r.startswith("articles/")) == c["articles_main"]
        and sum(1 for r in listed if r.startswith("robustness_subset/")) == c["articles_robustness"]
    )
    n_cases = sum(1 for _ in csv.DictReader((ROOT / "cases.csv").open(encoding="utf-8")))
    n_pairs = sum(1 for _ in csv.DictReader((ROOT / "pairs.csv").open(encoding="utf-8")))
    n_seeds = sum(1 for _ in csv.DictReader((ROOT / "seeds/seeds_used.csv").open(encoding="utf-8")))
    tab_ok = (n_cases == c["cases"] and n_pairs == c["pairs"] and n_seeds == c["seeds"])
    ok = not missing and not unlisted and count_ok and tab_ok
    detail = (f"{len(listed)} articles listed, {len(missing)} missing, {len(unlisted)} unlisted; "
              f"cases={n_cases}/{c['cases']} pairs={n_pairs}/{c['pairs']} seeds={n_seeds}/{c['seeds']}")
    if missing:
        detail += f" | missing e.g. {missing[:3]}"
    if unlisted:
        detail += f" | unlisted e.g. {unlisted[:3]}"
    record("file counts match the manifest", ok, detail)


# --- 2. checksums ----------------------------------------------------------
def check_checksums():
    path = ROOT / "checksums.sha256"
    if not path.exists():
        record("checksums", False, "checksums.sha256 not found")
        return
    total = bad = absent = 0
    first_bad = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split("  ", 1)
        total += 1
        f = ROOT / rel
        if not f.is_file():
            absent += 1
            first_bad.append(f"{rel} (absent)")
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            bad += 1
            first_bad.append(f"{rel} (digest)")
    ok = bad == 0 and absent == 0
    detail = f"{total} files checked, {bad} digest mismatches, {absent} absent"
    if first_bad:
        detail += f" | e.g. {first_bad[:3]}"
    record("every checksum verifies", ok, detail)


# --- 3. byte identity ------------------------------------------------------
def check_byte_identity():
    checked = failures = 0
    notes = []
    for pd in SAMPLE_PAIRS:
        base = ROOT / "articles" / pd
        if not base.is_dir():
            notes.append(f"{pd} absent")
            continue
        crit = next((d for d in sorted(base.iterdir()) if d.name.startswith("crit_")), None)
        foil = next((d for d in sorted(base.iterdir()) if d.name.startswith("foil_")), None)
        if not crit or not foil:
            notes.append(f"{pd} missing an arm")
            continue
        for slot in range(1, 6):
            c = json.loads((crit / f"art_{slot:02d}.json").read_text(encoding="utf-8"))
            f = json.loads((foil / f"art_{slot:02d}.json").read_text(encoding="utf-8"))
            for field in ("headline", "body"):
                checked += 1
                if f[field].replace(f["supplier_name"], c["supplier_name"]) != c[field]:
                    failures += 1
                    notes.append(f"{pd}/art_{slot:02d}/{field}")
    ok = failures == 0 and checked > 0
    detail = (f"{checked} foil->crit substitutions on {len(SAMPLE_PAIRS)} sampled pairs "
              f"({', '.join(SAMPLE_PAIRS)}), {failures} failures")
    if notes:
        detail += f" | {notes[:3]}"
    record("byte-identity property holds", ok, detail)


# --- 4. gold recompute -----------------------------------------------------
def check_gold():
    n = bad = 0
    notes = []
    with (ROOT / "cases.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n += 1
            halt = r["halt_pin"].strip().lower() == "true"
            expected = "escalate" if (halt and int(r["guaranteed_reach"]) >= THRESHOLD) else "abstain"
            if expected != r["gold_binary"]:
                bad += 1
                notes.append(f"{r['case_id']} expected {expected} got {r['gold_binary']}")
    record("gold_binary recomputes from halt_pin and guaranteed_reach",
           bad == 0 and n > 0,
           f"{n} cases recomputed, {bad} mismatches" + (f" | {notes[:3]}" if notes else ""))


# --- 5. disclaimer ---------------------------------------------------------
def check_disclaimer(dsm):
    n = bad = 0
    notes = []
    required = set(dsm["article_schema"])
    for rel in dsm["articles"]:
        p = ROOT / rel
        if not p.is_file():
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        n += 1
        if not str(obj.get("disclaimer") or "").strip() or obj.get("synthetic") is not True \
                or not required.issubset(obj.keys()):
            bad += 1
            notes.append(rel)
    record("disclaimer present in every article JSON", bad == 0 and n > 0,
           f"{n} article JSONs inspected, {bad} missing disclaimer/synthetic flag/schema key"
           + (f" | e.g. {notes[:3]}" if notes else ""))


def main():
    print(f"validating {ROOT.name}\n")
    dsm = load_manifest()
    check_counts(dsm)
    check_checksums()
    check_byte_identity()
    check_gold()
    check_disclaimer(dsm)
    failed = [n for n, ok, _ in results if not ok]
    print()
    if failed:
        print(f"RESULT: FAIL ({len(failed)}/{len(results)} checks failed: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS ({len(results)}/{len(results)} checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
