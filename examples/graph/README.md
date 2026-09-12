# Example OEM graph (fictional)

Tiny Aether Motors graph for the CSV interface. **Not** the paper evaluation graph.

- **NorthForge Components Ltd.** is the sole source of a critical brake-control module used on two vehicle variants (`guaranteed_reach = 2`).
- **Ridgeway Plastics Inc.** and **DualCast Interiors GmbH** both supply the same door-trim product (redundant).

Load it with:

```python
from src.services.graph_service import SupplyChainGraphService
g = SupplyChainGraphService("examples/graph")
g.resolve_name("NorthForge")  # sole-source brake supplier
g.resolve_name("Acme Aerospace Castings")  # None — off-graph
```
