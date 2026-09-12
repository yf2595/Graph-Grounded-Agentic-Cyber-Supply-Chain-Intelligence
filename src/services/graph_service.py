"""OEM supply-chain graph adapter — deterministic lookups only."""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CRITICAL_CATEGORY_PATTERNS = [
    r"brake",
    r"abs\b",
    r"esc\b",
    r"ecu",
    r"airbag",
    r"steering",
    r"engine ass",
    r"engine ecu",
    r"cvt",
    r"transmission",
    r"radar",
    r"lidar",
    r"wiring harness",
    r"turbo",
    r"seat belt",
]


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(
        r"\b(co|ltd|limited|inc|corp|corporation|llc|gmbh|sa|plc|group|company)\b",
        " ",
        s,
    )
    return re.sub(r"\s+", " ", s).strip()


def is_critical_category(name: str) -> bool:
    low = (name or "").lower()
    return any(re.search(p, low) for p in CRITICAL_CATEGORY_PATTERNS)


@dataclass
class ResolveHit:
    supplier_node_id: str
    canonical_name: str
    surface_name: str
    match_method: str
    match_confidence: float


class SupplyChainGraphService:
    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.id2name: dict[str, str] = {}
        self.name2id: dict[str, str] = {}
        self.norm2id: dict[str, str] = {}
        self.alias2id: dict[str, str] = {}
        self.prod_sup: dict[str, set[str]] = defaultdict(set)
        self.sup_prods: dict[str, set[str]] = defaultdict(set)
        self.prod_vars: dict[str, set[str]] = defaultdict(set)
        self.prod_name: dict[str, str] = {}
        self.prod_cat: dict[str, str] = {}
        self.var_name: dict[str, str] = {}
        self.sole_prods: set[str] = set()
        self._metrics: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        nodes = list(csv.DictReader((self.dataset_dir / "nodes.csv").open(encoding="utf-8")))
        edges = list(csv.DictReader((self.dataset_dir / "edges.csv").open(encoding="utf-8")))
        aliases = list(csv.DictReader((self.dataset_dir / "aliases.csv").open(encoding="utf-8")))

        for n in nodes:
            nid, kind, name = n["node_id"], n["node_kind"], n["node_name"]
            if kind == "organization":
                self.id2name[nid] = name
                self.name2id[name] = nid
                self.norm2id[normalize_name(name)] = nid
            elif kind == "product":
                self.prod_name[nid] = name
            elif kind == "vehicle_variant":
                self.var_name[nid] = name

        for a in aliases:
            nid = a["node_id"]
            if not nid.startswith("ORG_"):
                continue
            alias = (a.get("alias") or "").strip()
            if not alias or alias.isdigit() or len(alias) >= 8 and alias.replace("-", "").isdigit():
                continue
            self.alias2id[alias.lower()] = nid
            nrm = normalize_name(alias)
            if nrm:
                self.alias2id[nrm] = nid

        for e in edges:
            et = e["edge_type"]
            if et == "supplies_product":
                self.prod_sup[e["target_id"]].add(e["source_id"])
                self.sup_prods[e["source_id"]].add(e["target_id"])
            elif et == "product_used_in_variant":
                self.prod_vars[e["source_id"]].add(e["target_id"])
            elif et == "product_in_category":
                self.prod_cat[e["source_id"]] = e["target_name"]

        self.sole_prods = {p for p, s in self.prod_sup.items() if len(s) == 1}
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        for sid, prods in self.sup_prods.items():
            gvars: set[str] = set()
            tvars: set[str] = set()
            sole_count = 0
            crit_sole = 0
            for p in prods:
                tvars |= self.prod_vars.get(p, set())
                if p in self.sole_prods:
                    sole_count += 1
                    gvars |= self.prod_vars.get(p, set())
                    cat = self.prod_cat.get(p, "")
                    if cat and is_critical_category(cat):
                        crit_sole += 1
            self._metrics[sid] = {
                "guaranteed_reach": len(gvars),
                "total_reach": len(tvars),
                "sole_source_product_count": sole_count,
                "critical_sole_product_count": crit_sole,
            }

    def get_supplier(self, node_id: str) -> dict | None:
        if node_id not in self.id2name:
            return None
        m = self._metrics.get(node_id, {})
        return {
            "supplier_node_id": node_id,
            "supplier_name": self.id2name[node_id],
            **m,
        }

    def resolve_name(self, surface: str) -> ResolveHit | None:
        if not surface or not surface.strip():
            return None
        raw = surface.strip()
        if raw in self.name2id:
            sid = self.name2id[raw]
            return ResolveHit(sid, self.id2name[sid], raw, "exact_name", 1.0)
        low = raw.lower()
        if low in self.alias2id:
            sid = self.alias2id[low]
            return ResolveHit(sid, self.id2name[sid], raw, "alias", 0.95)
        nrm = normalize_name(raw)
        if nrm and nrm in self.norm2id:
            sid = self.norm2id[nrm]
            return ResolveHit(sid, self.id2name[sid], raw, "normalized", 0.9)
        if nrm and nrm in self.alias2id:
            sid = self.alias2id[nrm]
            return ResolveHit(sid, self.id2name[sid], raw, "normalized_alias", 0.85)
        # Token containment for short brands (min length guard)
        if len(nrm) >= 4:
            best = None
            for cand_norm, sid in self.norm2id.items():
                if nrm == cand_norm:
                    best = (sid, 0.9)
                    break
                if len(nrm) >= 5 and (nrm in cand_norm or cand_norm.startswith(nrm)):
                    conf = 0.7 if nrm in cand_norm else 0.65
                    if not best or conf > best[1]:
                        best = (sid, conf)
            if best and best[1] >= 0.7:
                sid = best[0]
                return ResolveHit(sid, self.id2name[sid], raw, "fuzzy_norm", best[1])
        return None

    def resolve_entities(self, names: list[str]) -> list[ResolveHit]:
        hits: list[ResolveHit] = []
        seen: set[str] = set()
        for name in names:
            hit = self.resolve_name(name)
            if hit and hit.supplier_node_id not in seen:
                seen.add(hit.supplier_node_id)
                hits.append(hit)
        return hits

    def get_supplied_parts(self, node_id: str, limit: int = 25) -> list[dict]:
        out = []
        for pid in list(self.sup_prods.get(node_id, []))[:limit]:
            alts = sorted(self.prod_sup.get(pid, set()) - {node_id})
            cat = self.prod_cat.get(pid, "")
            sole = pid in self.sole_prods
            crit = "CRITICAL" if (cat and is_critical_category(cat)) else ("MEDIUM" if sole else "LOW")
            red = "SOLE_SOURCE" if sole else ("PARTIAL_REDUNDANCY" if alts else "UNKNOWN")
            out.append(
                {
                    "part_id": pid,
                    "part_name": self.prod_name.get(pid, pid),
                    "category": cat,
                    "criticality": crit,
                    "redundancy": red,
                    "alternative_supplier_ids": alts[:10],
                }
            )
        return out

    def get_affected_oem_products(self, node_id: str, limit: int = 40) -> list[dict]:
        vars_: set[str] = set()
        for pid in self.sup_prods.get(node_id, set()):
            vars_ |= self.prod_vars.get(pid, set())
        out = []
        for vid in list(vars_)[:limit]:
            out.append({"product_id": vid, "product_name": self.var_name.get(vid, vid)})
        return out

    def get_supplier_redundancy(self, node_id: str) -> dict:
        parts = self.get_supplied_parts(node_id, limit=100)
        sole = sum(1 for p in parts if p["redundancy"] == "SOLE_SOURCE")
        if sole == 0:
            level = "HIGH_REDUNDANCY"
        elif sole <= 2:
            level = "PARTIAL_REDUNDANCY"
        elif sole <= 8:
            level = "LOW_REDUNDANCY"
        else:
            level = "SOLE_SOURCE"
        return {"supplier_node_id": node_id, "redundancy": level, "sole_source_part_count": sole}

    def get_metrics(self, node_id: str) -> dict:
        return dict(self._metrics.get(node_id, {}))

    def build_propagation(self, node_id: str) -> dict:
        meta = self.get_supplier(node_id)
        if not meta:
            return {}
        parts = self.get_supplied_parts(node_id)
        products = self.get_affected_oem_products(node_id)
        red = self.get_supplier_redundancy(node_id)
        crit_parts = [p for p in parts if p["criticality"] == "CRITICAL"]
        sole_parts = [p for p in parts if p["redundancy"] == "SOLE_SOURCE"]
        g = meta.get("guaranteed_reach", 0)
        return {
            "supplier_node_id": node_id,
            "supplier_name": meta["supplier_name"],
            "minimum_graph_depth_to_oem": 1,
            "tier_label": "Tier-1",
            "supplied_parts": parts[:15],
            "affected_oem_products": products[:20],
            "affected_oem_sites": [],
            "path_count_to_oem_products": len(products),
            "guaranteed_reach": g,
            "total_reach": meta.get("total_reach", 0),
            "is_single_point_of_failure": g >= 30 and len(sole_parts) > 0,
            "redundancy_summary": red,
            "critical_part_count": len(crit_parts),
            "sole_source_part_count": len(sole_parts),
            "critical_sole_product_count": meta.get("critical_sole_product_count", 0),
        }
