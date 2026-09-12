# OEM dependency graph

The evaluation graph used in the paper is a frozen Honda supply-chain graph derived from licensed MarkLines data (4,661 nodes, 26,035 edges, 8,886 aliases). **It is not redistributed with this repository.**

To reproduce the paper numbers, place the licensed extract in this directory as:

```
dataset/
  nodes.csv
  edges.csv
  aliases.csv
```

To run the agents on your own OEM, provide CSVs with the same interface. A fictional worked example is in [`examples/graph/`](../examples/graph/).

## Required columns

The graph adapter (`src/services/graph_service.py`) reads only these fields. Extra columns are ignored.

### `nodes.csv`

| Column | Values |
|---|---|
| `node_id` | Stable ID (`ORG_*`, `PRODUCT_*`, `VARIANT_*`, `CATEGORY_*`) |
| `node_name` | Display name |
| `node_kind` | `organization` \| `product` \| `vehicle_variant` \| `component_category` |

Organizations used as suppliers must have `node_kind=organization`.

### `edges.csv`

| Column | Values |
|---|---|
| `source_id` | Node ID |
| `target_id` | Node ID |
| `edge_type` | See below |
| `target_name` | Required for `product_in_category` (category display name) |

Consumed edge types:

| `edge_type` | Direction | Meaning |
|---|---|---|
| `supplies_product` | organization → product | Supplier supplies a part |
| `product_in_category` | product → category | Part category (used for criticality) |
| `product_used_in_variant` | product → vehicle variant | Part used on a program |

`variant_built_by_oem` may be present for documentation; the adapter does not require it.

### `aliases.csv`

| Column | Values |
|---|---|
| `node_id` | Organization ID (`ORG_*`) |
| `alias` | Surface form used in reports |

## Graph interface used by agents

Agents never see the CSVs. They call a fixed lookup API:

- organization / alias resolution
- supplied products and component categories
- vehicle-program dependencies
- sourcing alternatives and sole-source status
- criticality, bottleneck status, and downstream reach (`guaranteed_reach`, `total_reach`)

Point evaluation at a graph directory with:

```yaml
# evaluation/config/evaluation.yaml
dataset:
  graph_dir: dataset   # or path/to/your_graph
```

Point `dataset.graph_dir` in the evaluation YAML at your directory, or pass `--dataset` to `src.run_pipeline`. The shipped `INC_*` reports name Honda-graph suppliers; they will not ground on an unrelated OEM. The fictional example is checked by `tests/test_smoke.py::test_example_graph_loads`.
