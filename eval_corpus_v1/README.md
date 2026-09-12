# Evaluation corpus (129 reports)

Controlled synthetic corpus used in the paper. Every article is machine-generated. Named organizations may be real; attributed incidents are fictional. Each JSON file includes `"synthetic": true` and a `disclaimer`.

The licensed OEM graph that these reports are grounded against is **not** shipped with the repository. See [`dataset/README.md`](../dataset/README.md).

## Composition

| Slice | n |
|---|---:|
| Actionable in-graph cyber (`is_cyber ∧ in_supply_chain`) | 99 |
| Off-graph cyber | 15 |
| In-graph non-cyber | 15 |
| **Total** | **129** |

The 129 reports are 100 base articles plus 29 cross-source variants of 25 actionable base reports. Of the 99 actionable reports, 45 are singleton incidents and 54 form 25 multi-report groups (34 positive merge pairs). The negative pair set includes 42 designed same-supplier / different-incident hard negatives.

Exposure strata among in-graph cyber reports include bottleneck/sole-source, critical-part sole-source, redundant high-reach, moderate-reach, low-reach, and multi-supplier cases.

## Layout

```
eval_corpus_v1/
  articles/INC_001.json … INC_129.json
  incidents.csv              gold table (relevance, risk, group_id, graph metrics)
  seeds.csv                  public incident seeds used as narrative templates
  dataset_manifest.json
  validate.py
  review/risk_bin_review.md  ordinal risk rubric
  tools/                     optional regenerators (need a graph + API key)
```

Exported leak-safe copies used by the harness:

- `evaluation/data/inputs.jsonl` — `article_id`, `headline`, `body`
- `evaluation/data/ground_truth.jsonl` — cyber / related / actionable / risk / group / supplier IDs

## Gold labels

```
gt_is_actionable = gt_is_cyber ∧ gt_is_related
```

- `gt_is_related` — at least one named organization is present in the OEM graph
- `gt_risk` — five-level OEM risk under a fixed rubric combining represented incident consequence with graph-derived exposure (`VERY_LOW` … `VERY_HIGH`)
- `gt_group_id` — shared identifier for reports that describe the same controlled event; `null` for singletons
- `gt_supplier_node_ids` — canonical graph IDs

Reference labels and group identifiers are never provided to the tested systems.

## Article schema

```json
{
  "article_id": "INC_001",
  "headline": "...",
  "body": "...",
  "supplier_names": ["..."],
  "supplier_node_ids": ["ORG_..."],
  "is_cyber": true,
  "in_supply_chain": true,
  "relevant": true,
  "case_stratum": "bottleneck_sole",
  "impact_level": "severe",
  "synthetic": true,
  "disclaimer": "SYNTHETIC ARTICLE: ..."
}
```

## Validation

```bash
python eval_corpus_v1/validate.py
```

## Regenerating on another graph

`tools/` can select suppliers and generate a new corpus if you have a compatible graph and an API key. That path is optional and is not required to run the paper evaluation on the shipped 129 reports.
