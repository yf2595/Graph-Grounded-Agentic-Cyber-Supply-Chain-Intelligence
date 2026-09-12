# Supplementary silent-pair corpus

This directory is **not** the 129-report paper benchmark (`eval_corpus_v1/`).
It is a matched-pair construction used for sole-source discrimination checks.
Paper Tables 1–5 use `eval_corpus_v1/` only.

# silent_pair_dataset v1.0

## 1. What this dataset is

This is a set of **matched minimal pairs** for evaluating automated triage of cyber
incidents in an automotive supply chain. Each pair presents the same five-article news
story twice, differing in exactly one respect: the name of the supplier said to have been
attacked. One arm names a supplier that is a **sole source** for parts used in many
vehicle variants (so an outage is structurally guaranteed to propagate); the other names a
supplier of comparable overall size that is **second-sourced everywhere** (so an outage
propagates nowhere with certainty). Because the two arms are byte-identical apart from the
name, any system that treats them differently must be using the supply-network structure
rather than the wording of the article — and any system that treats them the same has
failed to. The dataset ships 15 pairs / 34 cases / 385 articles, plus a deterministic gold
label per case and the real incidents the stories were adapted from.

## 2. ⚠️ SYNTHETIC CONTENT WARNING — READ BEFORE USE

**Every article in this dataset is machine-generated. None of it is real reporting.**

The **companies named are real**. The **incidents attributed to them are fictional.** Each
story is a re-attribution: a real cyber incident that happened to *some other company* was
rewritten as though it had happened to the named supplier. The real victims are listed in
[`seeds/seeds_used.csv`](seeds/seeds_used.csv) — for example, the article naming
"HI-LEX Corporation" is an adaptation of the 2024 **Hoya Corporation** incident. HI-LEX
was not attacked. Every article JSON carries a `disclaimer` field and `"synthetic": true`,
and every `bundle.md` repeats the disclaimer at the top.

Two clarifications a reader must not miss:

- **Pair 01 is not even a re-attribution.** It uses a wholly invented, product-neutral
  ransomware narrative with no real underlying incident and no seed registry record
  (`seed_id = pilot_synthetic`).
- **Pair 13 is the one exception to the re-attribution rule.** Its seed incident really did
  happen to the named supplier's own corporate family (Nichirin). It is labelled `twin` and
  is excluded from pooled analysis for exactly that reason.

**Do not redistribute this content as factual reporting, do not index it as news, and do
not train retrieval or summarisation systems on it without preserving the disclaimer
field.** These texts are designed to be plausible; that is what makes them useful for
evaluation and hazardous if detached from this README.

## 3. Construction in brief

**Identical-text regime — `articles/`, the pooled corpus.** For each pair, a *single*
master set of five articles was generated once, referring to the supplier only by the
literal token `PLACEHOLDERCO`. That master is instantiated into each arm by one plain
string substitution of the token with the supplier's name — applied to the headline and
the body, all occurrences, nothing else. The arms are therefore byte-identical except for
the supplier name, by construction rather than by inspection. The master is shipped in
placeholder form at `articles/pair_NN/master/`, so this is independently checkable, and
`validate.py` checks it. **All pooled analyses use this regime only.**

**Independent-generation regime — `robustness_subset/`.** An earlier construction in which
each arm was generated separately from its own prompt, so the arms differ in wording as
well as in name. This confounds text with structure and is *not* suitable for the core
discrimination claim, which is why it was superseded. It is retained for cross-regime
robustness checks — 28 accepted arms across three waves. **Never pool these with
`articles/`.**

## 4. Gold definition

The label is a deterministic function of two facts. **No model is in the loop.**

```
gold_binary = escalate   iff   halt_pin AND guaranteed_reach >= 30
              abstain    otherwise
```

- **`halt_pin`** — whether the *real seed incident* involved a confirmed production halt,
  taken from the seed registry's evidence flags. It is a property of the real incident that
  the story was adapted from, and it is identical across both arms of a pair.
- **`guaranteed_reach`** — the number of distinct vehicle variants that would lose a source
  if this supplier went down **with no alternative supplier available**. It counts only
  products for which the supplier is the *sole* source, then counts the deduplicated set of
  variants those products feed. Computed from a real automotive supply network
  (~960 nodes). A second-source supplier scores 0 however large it is.
- **Threshold 30** — fixed in advance for the whole corpus.

The discriminating consequence: within a pair both arms share `halt_pin`, so the label
turns *entirely* on `guaranteed_reach`, which the article text never states. `total_reach`
is reported alongside as the supplier's total downstream fanout ignoring sole-sourcing; it
is deliberately matched across arms and is **not** part of the gold predicate. Pairs are
matched on `total_reach` precisely so that overall size cannot be used as a shortcut.

`validate.py` recomputes every label from `cases.csv` and fails if any disagrees.

## 5. File dictionary

```
silent_pair_dataset_v1.0/
  README.md                      this file
  dataset_manifest.json          machine-readable index of everything below
  cases.csv                      one row per case (arm)                     34 rows
  pairs.csv                      one row per pair                           15 rows
  seeds/seeds_used.csv           one row per real incident used             13 rows
  seeds/seeds_used.json          identical content, JSON mirror
  articles/pair_NN/
      master/art_01..05.json     placeholder form (supplier = PLACEHOLDERCO)
      crit_<Tag>/art_01..05.json + bundle.md      sole-source arm
      foil_<Tag>/art_01..05.json + bundle.md      second-sourced arm
      sharp_Stanley/...          present on pairs 02, 05, 09, 12 only
  robustness_subset/<wave>/pair_NN/<arm>/...      independent-generation, 28 arms
  checksums.sha256               sha256 of every file, "<digest>  <relative path>"
  validate.py                    standalone integrity checker (stdlib only)
```

`<Tag>` is a short filesystem-safe form of the supplier name (first significant word, legal
suffixes and parentheticals dropped) used only for directory naming. The authoritative
canonical name is always the `supplier_name` field.

⚠️ **`arm` is unique within a pair, not globally.** `crit_H-ONE` denotes *two different
companies*: H-ONE Co., Ltd. (`ORG_CF64B3D0E3C1`) in pair 01, and H-ONE Parts (Thailand)
Co., Ltd. (`ORG_A15B23A6BE55`) in pair 14. Likewise `foil_Enkei` appears in both pair 01
and pair 13. **Group by `(pair, arm)` or by `supplier_node_id` — never by `arm` alone**, or
you will silently merge distinct suppliers. `case_id` and `article_id` are globally unique
and safe to key on.

### Article JSON — every file under `articles/` and `robustness_subset/`

| field | meaning |
|---|---|
| `article_id` | unique id, `pair_NN__<arm>__art_SS` (robustness ids are wave-prefixed) |
| `pair` | pair number, 1–15 |
| `arm` | arm directory name, e.g. `crit_H-ONE`, `foil_Enkei`, `master` |
| `slot` | position of this article within its five-article set, 1–5 |
| `headline` | article headline, byte-verbatim from the frozen corpus |
| `body` | article body, byte-verbatim from the frozen corpus |
| `supplier_name` | canonical supplier name; the literal `PLACEHOLDERCO` in `master/` |
| `supplier_node_id` | supply-network node id; `null` in `master/` |
| `construction_regime` | `identical-text` or `independent-generation` |
| `wave` | generation run this text came from |
| `synthetic` | always `true` |
| `disclaimer` | one-sentence fictional-attribution notice |

`bundle.md` is the same five articles concatenated for human reading, disclaimer first. It
is a convenience rendering — the JSON is authoritative.

### cases.csv

| column | meaning |
|---|---|
| `case_id` | unique case id, `pair_NN__<arm>` |
| `pair` | pair number, 1–15 |
| `arm_role` | `crit` (sole-source), `foil` (second-sourced), `sharp` (see §7) |
| `class` | `discrimination`, `control`, or `twin` |
| `supplier_name` | canonical supplier name |
| `supplier_node_id` | supply-network node id |
| `guaranteed_reach` | variants losing a source, sole-source products only |
| `total_reach` | variants reachable via all supplied products (not in the gold rule) |
| `gold_binary` | `escalate` or `abstain` |
| `gold_basis` | the predicate string, constant: `halt_pin AND guaranteed_reach>=30` |
| `halt_pin` | did the real seed incident confirm a production halt |
| `exfil_pin` | did the real seed incident confirm data exfiltration (reported, unused by the gold rule) |
| `seed_id` | seed this case's story was adapted from; joins `seeds_used.csv` |
| `construction_regime` | `identical-text` for all rows in this file |
| `never_pooled` | `true` for the twin pair; those rows must be excluded from pooled analysis |
| `notes` | Δ-disclosures, seed and foil reuses, and other per-case caveats |

### pairs.csv

| column | meaning |
|---|---|
| `pair` | pair number, 1–15 |
| `class` | `discrimination`, `control`, or `twin` |
| `seed_id` | seed both arms share |
| `crit_supplier` / `foil_supplier` | canonical names of the two arms |
| `delta_exposure` | `abs(crit.total_reach − foil.total_reach)`; lower is a tighter size match |
| `sharp_arm_present` | whether a `sharp_Stanley` arm exists for this pair |
| `master_sha256` | sha256 of the pair's five master articles, over the canonical JSON `[{"body","headline"},…]` with sorted keys and `,`/`:` separators |

### seeds/seeds_used.csv

One row per **real** incident used as a seed, plus the synthetic pilot. Values are verbatim
from the project's seed registry; **no new research was performed for this export.**

| column | meaning |
|---|---|
| `seed_id` | registry case id, e.g. `legacy_inc_011`; `pilot_synthetic` for the invented seed |
| `victim_organization` | the company the incident **actually** happened to |
| `incident_year` | year of earliest public mention |
| `incident_type` | registry incident-type string |
| `production_halt_confirmed` | registry evidence flag; this is what becomes `halt_pin` |
| `data_exfiltration_confirmed` | registry evidence flag; becomes `exfil_pin` |
| `oem_disruption_confirmed` | registry evidence flag; `false` for all 12 real seeds here |
| `disclosure_class` | see derivation note below |
| `public_source_refs` | `Title — URL` per source, ` \| `-separated, verbatim (see §7) |
| `pairs_seeded` | which pairs use this seed |
| `reuse_note` | free text where a seed is reused or otherwise special |

**`disclosure_class` derivation.** The source registry has no field of this name. The value
is derived mechanically from two fields it does have — the `oem_disruption_confirmed`
evidence flag, and an internal "incident archetype" tag — using this rule, in order:

| value | condition | meaning |
|---|---|---|
| `oem_confirmed` | `oem_disruption_confirmed` is true | downstream OEM impact was publicly confirmed |
| `denied` | archetype tag `A2` | the victim publicly denied downstream impact |
| `silent` | archetype tag `A3` or `A7a` | the incident was disclosed, but downstream impact was never addressed either way — public silence |
| `other` | anything else | disclosed with a downstream picture that fits none of the above |
| `synthetic_no_seed` | the `pilot_synthetic` row | no real incident, so no disclosure behaviour |

The archetype tags `A2`/`A3`/`A7a` are internal codes from the source project and are given
here only to make the rule auditable; the `condition` and `meaning` columns above are their
complete definition for this dataset's purposes. In this release the observed distribution
is `silent` ×8, `other` ×4, `synthetic_no_seed` ×1 — no seed here is `oem_confirmed` or
`denied`. Note that `silent` is an inference from *absence* of reporting, which is weaker
evidence than a positive confirmation; see the last bullet of §7.

## 6. Quick start

```python
import json, pandas as pd

cases = pd.read_csv("cases.csv")
pairs = cases[cases["class"] == "discrimination"]        # drop controls and the twin
arts  = [json.load(open(f"articles/pair_01/crit_H-ONE/art_{i:02d}.json")) for i in range(1, 6)]
print(pairs[["case_id", "guaranteed_reach", "gold_binary"]].head(), arts[0]["headline"])
```

For pooled evaluation: use `cases.csv` only (all rows are `identical-text`), and drop
`never_pooled == True`.

## 7. Known properties & caveats

- **The twin pair (13) is never pooled.** Its seed incident belongs to the named supplier's
  own corporate family rather than being a re-attribution, so it is a case study, not a
  matched pair. `never_pooled = true` on both its rows. It also **reuses Enkei Corporation**
  as its foil, already used as the foil in pair 01 — acceptable *only* because pair 13 never
  enters a pooled analysis. Excluding it leaves 14 pooled pairs.
- **Sharp arms (`sharp_Stanley`, pairs 02, 05, 09, 12) probe size-vs-criticality.** Stanley
  Electric U.S. is a *large* supplier — `total_reach` 360, the largest in the dataset — with
  `guaranteed_reach` 0, because it is second-sourced on everything. Gold is therefore
  `abstain`. A system that escalates on sharp arms is using company size as a proxy for
  criticality. These arms share their pair's master text, so they are directly comparable to
  the crit and foil arms.
- **Δ-disclosures: pairs 02 and 15 have looser exposure matching.** Pair 02 is
  `total_reach` 204 vs 191 (Δ13) and pair 15 is 117 vs 127 (Δ10); every other pair is Δ≤7 and
  most are Δ≤2. In both cases the closer candidates carried a confound (household-name
  recognisability, or OEM-group affiliation with the network's focal manufacturer) and the
  wider-Δ but cleaner foil was chosen deliberately. Recorded in the `notes` column.
- **Two seeds are reused across two pairs each**: `legacy_inc_011` (Lacroix) seeds pairs 06
  and 14, and `legacy_inc_005` (Microchip) seeds pairs 05 and 15. The pairs use different
  suppliers and independently generated master text, but the underlying real incident — and
  therefore `halt_pin` — is shared, so the pairs are not fully independent.
- **Four seeds are sourced only from cross-reference articles.** For `legacy_inc_011`
  (Lacroix), `legacy_inc_013` (Wabtec), `legacy_inc_014` (Foxsemicon) and `legacy_inc_036`
  (MaxLinear), the URL in `public_source_refs` points to an article primarily about a
  *different* company that mentions the seed victim only in passing; the verbatim titles
  ("Related: …", "cross-ref: …", or a headline naming another firm) make this visible. These
  seeds have no primary source in the registry. The evidence flags were curated from that
  secondary mention. Treat `halt_pin` for these four as lower-confidence than the rest.
- **`oem_disruption_confirmed` is `false` for all 12 real seeds.** No seed in this dataset has
  publicly confirmed downstream OEM impact. Absence of confirmation is not evidence of
  absence — the gold rule deliberately does not depend on this flag, using the structural
  `guaranteed_reach` instead.
- **Controls carry a real halt-free incident.** `class = control` pairs are seeded from
  incidents with `production_halt_confirmed = false`, so gold is `abstain` on **both** arms
  regardless of reach — including crit arms with `guaranteed_reach` up to 49. They test
  whether a system escalates on exposure alone when the incident never halted production.
- **The supply network itself is not included** (see §8), so `guaranteed_reach` and
  `total_reach` cannot be recomputed from this package. They are shipped as data in
  `cases.csv`, and the gold rule is checkable from them.
- **Robustness-subset arms are not a complete grid.** 28 arms across three waves, not all
  15 pairs × 2; only arms that passed the original quality gate are included, some pairs
  appear in more than one wave, and rejected or verbatim-carried-over arms were dropped.
  See `dataset_manifest.json → robustness_subset.arms` for the exact inventory.

## 8. Provenance & version

| | |
|---|---|
| dataset | `silent_pair_dataset` v1.0 |
| corpus | `silent-critical-v1`, construction regime `identical-text` |
| corpus manifest id | `c8b5ffee3f9199ffa66db5e5a442931196fc2df807da23f81acac46583518c7b` (see note) |
| freeze commit | `5387707` (internal repository, not distributed) |
| source run | `wave4_symmetric_20260705_103038` |
| export date | 2026-07-19 |
| contact | `<CONTACT PLACEHOLDER — insert corresponding author e-mail before distribution>` |

**Note on the corpus manifest id.** This value is *not* the SHA-256 of any file shipped
here, and it is not the SHA-256 of the internal manifest file as it sits on disk. It is a
canonical-form digest recorded *inside* that manifest: SHA-256 over the manifest object with
its own `manifest_sha256` key removed, serialised as JSON with sorted keys and two-space
indentation. It is an opaque **identifier for the corpus revision**, useful for citing which
freeze this release came from, and it is **not verifiable from this package alone** — the
manifest it digests is not distributed. The integrity anchor you *can* check is
`checksums.sha256`, which covers every file here and is verified by `validate.py`. Do not
read the corpus manifest id as a checksum of this download.

Article bodies are byte-verbatim from the frozen corpus; nothing was regenerated for this
export and no model was invoked to produce it. All shipped text files use LF line endings;
`checksums.sha256` is computed over exactly those bytes, so any transport that rewrites line
endings (e.g. a git checkout with `core.autocrlf` enabled) will invalidate the checksums even
though the content is intact.

**Deliberately excluded from this release**, available from the authors on request:

- the supply-network **graph node, edge, and alias CSVs**
- serialized **graph slices** (the per-supplier network views shown to systems under test)
- **evaluation prompts and harness code**
- **results JSONLs** from evaluation runs

The exclusions keep the release to the corpus and its labels. In particular, without the
graph you can reproduce the gold rule from the shipped reach values but cannot recompute
those values, and without the prompts and results you cannot reproduce the published
evaluation numbers.

### Verifying this package

```
cd silent_pair_dataset_v1.0 && python validate.py
```

Checks file counts against the manifest, every checksum, the byte-identity property on
three sampled pairs, the gold recomputation on all 34 cases, and the presence of a
disclaimer in all 385 article JSONs. Prints `PASS`/`FAIL` per check; exit code 0 iff all
pass. Requires only the Python standard library.
