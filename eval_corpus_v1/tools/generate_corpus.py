"""Generate the full 100-incident eval_corpus_v1 package."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from select_suppliers import load_graph, pick_strata

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = ROOT / "articles"
DISCLAIMER = (
    "SYNTHETIC ARTICLE: this text was machine-generated for research evaluation; "
    "the named company is real (or fictional as noted) but the incident described "
    "is fictional and is not a report of anything that company experienced."
)

IMPACT_W = {"none": 0.0, "minor": 1.0, "severe": 2.0}


def structure_weight(g: int, total: int, is_crit: bool) -> float:
    if g >= 50 or (is_crit and g >= 30):
        return 3.0
    if g >= 30:
        return 2.0
    if 10 <= g <= 29:
        return 1.0
    if 1 <= g <= 9 or (g == 0 and total >= 30):
        return 0.5
    return 0.25


def score_to_bin(score: float) -> str:
    if score <= 0.5:
        return "very_low"
    if score <= 1.5:
        return "low"
    if score <= 3.0:
        return "medium"
    if score <= 5.0:
        return "high"
    return "very_high"


def draft_risk(impact: str, g: int, total: int, is_crit: bool) -> tuple[str, float, str]:
    iw = IMPACT_W[impact]
    sw = structure_weight(g, total, is_crit)
    score = iw * sw
    # impact none always very_low regardless of structure (no OEM ops effect claimed)
    if impact == "none":
        score = min(score, 0.5)
        bin_ = "very_low"
    else:
        bin_ = score_to_bin(score)
    rationale = (
        f"impact={impact}(w={iw}) × structure(g={g},total={total},crit={is_crit},w={sw}) → score={score:.2f}"
    )
    return bin_, score, rationale


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

SEEDS = [
    {
        "seed_id": "seed_varta_2024",
        "victim_organization": "VARTA AG",
        "incident_year": 2024,
        "incident_type": "Ransomware",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": False,
        "default_impact": "severe",
        "public_source_refs": "https://www.securityweek.com/cyberattack-disrupts-production-in-varta-battery-factories/",
        "template_family": "ransomware_halt",
    },
    {
        "seed_id": "seed_hoya_2024",
        "victim_organization": "Hoya Corporation",
        "incident_year": 2024,
        "incident_type": "Ransomware",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": False,
        "default_impact": "severe",
        "public_source_refs": "https://www.securityweek.com/lens-maker-hoya-scrambling-to-restore-systems-following-cyberattack/",
        "template_family": "ransomware_halt",
    },
    {
        "seed_id": "seed_nexperia_2024",
        "victim_organization": "Nexperia",
        "incident_year": 2024,
        "incident_type": "Ransomware (data theft)",
        "production_halt_confirmed": False,
        "data_exfiltration_confirmed": True,
        "default_impact": "none",
        "public_source_refs": "https://www.securityweek.com/ransomware-group-claims-theft-of-data-from-chipmaker-nexperia/",
        "template_family": "exfil_only",
    },
    {
        "seed_id": "seed_microchip_2024",
        "victim_organization": "Microchip Technology Inc.",
        "incident_year": 2024,
        "incident_type": "Ransomware (Play)",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": True,
        "default_impact": "severe",
        "public_source_refs": "https://www.cnbc.com/2024/08/20/microchip-technology-says-certain-operations-disrupted-by-cyber-incident.html",
        "template_family": "ransomware_halt_exfil",
    },
    {
        "seed_id": "seed_abb_2023",
        "victim_organization": "ABB Ltd.",
        "incident_year": 2023,
        "incident_type": "Ransomware (Black Basta)",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": True,
        "default_impact": "severe",
        "public_source_refs": "https://www.cybersecuritydive.com/news/abb-confirms-ransomware-data-theft/651531/",
        "template_family": "ransomware_halt_exfil",
    },
    {
        "seed_id": "seed_continental_2022",
        "victim_organization": "Continental AG",
        "incident_year": 2022,
        "incident_type": "Ransomware (LockBit)",
        "production_halt_confirmed": False,
        "data_exfiltration_confirmed": True,
        "default_impact": "none",
        "public_source_refs": "https://www.bleepingcomputer.com/news/security/lockbit-ransomware-claims-attack-on-continental-automotive-giant/",
        "template_family": "exfil_only",
    },
    {
        "seed_id": "seed_tsmc_2018",
        "victim_organization": "TSMC",
        "incident_year": 2018,
        "incident_type": "Malware (WannaCry variant)",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": False,
        "default_impact": "severe",
        "public_source_refs": "https://www.securityweek.com/chip-giant-tsmc-says-wannacry-behind-production-halt/",
        "template_family": "malware_halt",
    },
    {
        "seed_id": "seed_nichirin_2022",
        "victim_organization": "Nichirin-Flex USA",
        "incident_year": 2022,
        "incident_type": "Ransomware",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": False,
        "default_impact": "severe",
        "public_source_refs": "https://www.securityweek.com/us-subsidiary-automotive-hose-maker-nichirin-hit-ransomware/",
        "template_family": "ransomware_halt",
    },
    {
        "seed_id": "seed_wabtec_2022",
        "victim_organization": "Wabtec Corporation",
        "incident_year": 2022,
        "incident_type": "Ransomware (LockBit)",
        "production_halt_confirmed": False,
        "data_exfiltration_confirmed": True,
        "default_impact": "none",
        "public_source_refs": "https://www.cybersecuritydive.com/news/estes-express-lines-cyberattack/695614/",
        "template_family": "exfil_only",
    },
    {
        "seed_id": "seed_maxlinear_2020",
        "victim_organization": "MaxLinear Inc.",
        "incident_year": 2020,
        "incident_type": "Ransomware (Maze)",
        "production_halt_confirmed": False,
        "data_exfiltration_confirmed": True,
        "default_impact": "none",
        "public_source_refs": "https://www.securityweek.com/",
        "template_family": "exfil_only",
    },
    {
        "seed_id": "seed_foxsemicon_2024",
        "victim_organization": "Foxsemicon Integrated Technology",
        "incident_year": 2024,
        "incident_type": "Ransomware (LockBit)",
        "production_halt_confirmed": False,
        "data_exfiltration_confirmed": True,
        "default_impact": "minor",
        "public_source_refs": "https://www.securityweek.com/",
        "template_family": "ransomware_minor",
    },
    {
        "seed_id": "seed_lacroix_2023",
        "victim_organization": "Lacroix Group",
        "incident_year": 2023,
        "incident_type": "Ransomware",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": False,
        "default_impact": "severe",
        "public_source_refs": "https://www.helpnetsecurity.com/",
        "template_family": "ransomware_halt",
    },
    {
        "seed_id": "seed_toyota_supplier_jtei_2022",
        "victim_organization": "Kojima Industries (Toyota supplier incident pattern)",
        "incident_year": 2022,
        "incident_type": "Cyberattack / system outage",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": False,
        "default_impact": "severe",
        "public_source_refs": "Public reporting on Toyota supplier cyber outages (2022 pattern)",
        "template_family": "ransomware_halt",
    },
    {
        "seed_id": "seed_synthetic_phish_minor",
        "victim_organization": "(synthetic)",
        "incident_year": 2024,
        "incident_type": "Business email compromise (contained)",
        "production_halt_confirmed": False,
        "data_exfiltration_confirmed": False,
        "default_impact": "minor",
        "public_source_refs": "",
        "template_family": "ransomware_minor",
    },
    {
        "seed_id": "seed_synthetic_exfil_none",
        "victim_organization": "(synthetic)",
        "incident_year": 2025,
        "incident_type": "Data breach claim (IT only)",
        "production_halt_confirmed": False,
        "data_exfiltration_confirmed": True,
        "default_impact": "none",
        "public_source_refs": "",
        "template_family": "exfil_only",
    },
    # Non-cyber templates
    {
        "seed_id": "seed_noncyber_strike",
        "victim_organization": "(template)",
        "incident_year": 2024,
        "incident_type": "Labor strike",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": False,
        "default_impact": "severe",
        "public_source_refs": "",
        "template_family": "noncyber_strike",
    },
    {
        "seed_id": "seed_noncyber_fire",
        "victim_organization": "(template)",
        "incident_year": 2023,
        "incident_type": "Plant fire",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": False,
        "default_impact": "severe",
        "public_source_refs": "",
        "template_family": "noncyber_fire",
    },
    {
        "seed_id": "seed_noncyber_recall",
        "victim_organization": "(template)",
        "incident_year": 2024,
        "incident_type": "Quality recall",
        "production_halt_confirmed": False,
        "data_exfiltration_confirmed": False,
        "default_impact": "minor",
        "public_source_refs": "",
        "template_family": "noncyber_recall",
    },
    {
        "seed_id": "seed_noncyber_bankruptcy",
        "victim_organization": "(template)",
        "incident_year": 2025,
        "incident_type": "Financial distress / restructuring",
        "production_halt_confirmed": False,
        "data_exfiltration_confirmed": False,
        "default_impact": "minor",
        "public_source_refs": "",
        "template_family": "noncyber_bankruptcy",
    },
    {
        "seed_id": "seed_noncyber_flood",
        "victim_organization": "(template)",
        "incident_year": 2024,
        "incident_type": "Flood / natural disaster",
        "production_halt_confirmed": True,
        "data_exfiltration_confirmed": False,
        "default_impact": "severe",
        "public_source_refs": "",
        "template_family": "noncyber_flood",
    },
]


OFF_GRAPH_COMPANIES = [
    ("Nordic Precision Castings AB", "Sweden"),
    ("Guangzhou Lianhai Stamping Co.", "China"),
    ("Midwest Forge & Fastener LLC", "United States"),
    ("Bavaria Trim Systems GmbH", "Germany"),
    ("Pune Auto Polymers Pvt. Ltd.", "India"),
    ("Osaka Secondary Wiring Co.", "Japan"),
    ("Queretaro Interior Modules SA de CV", "Mexico"),
    ("Busan Thermal Parts Co., Ltd.", "South Korea"),
    ("Lyon Gear Solutions SAS", "France"),
    ("Torino Sensor Labs S.p.A.", "Italy"),
    ("Changzhou Horizon Connectors", "China"),
    ("Detroit Legacy Stampings Inc.", "United States"),
    ("Silesia Rubber Compounds Sp. z o.o.", "Poland"),
    ("Nagoya Auxiliary Springs Co.", "Japan"),
    ("Valencia Cabin Electronics SL", "Spain"),
]


def render_article(
    company: str,
    family: str,
    impact: str,
    extra_companies: list[str] | None = None,
) -> tuple[str, str]:
    """Return (headline, body) for a synthetic article."""
    others = extra_companies or []
    multi_phrase = ""
    if others:
        joined = ", ".join(others[:-1]) + (" and " + others[-1] if len(others) > 1 else others[0])
        if len(others) == 1:
            multi_phrase = f" Related disruption was also reported at {others[0]}."
        else:
            multi_phrase = f" Similar disruption was also reported at {joined}."

    if family == "ransomware_halt":
        if impact == "severe":
            h = f"{company} Halts Production After Ransomware Attack on Plant Systems"
            b = (
                f"{company}, an automotive components supplier, confirmed a ransomware attack "
                f"that encrypted manufacturing execution and logistics systems. The company said "
                f"production lines at one or more plants were stopped while responders work to "
                f"restore operations. External cybersecurity specialists have been engaged. "
                f"No timeline for a full restart has been announced.{multi_phrase}"
            )
        elif impact == "minor":
            h = f"{company} Reports Contained Ransomware Incident, Limited Line Slowdowns"
            b = (
                f"{company} disclosed a ransomware incident affecting office and scheduling IT. "
                f"The firm said core machining cells continued with manual workarounds and that "
                f"only limited shipping delays are expected. Customer plants have been notified "
                f"as a precaution.{multi_phrase}"
            )
        else:
            h = f"{company} Investigating Suspected Ransomware Probe, Production Unaffected"
            b = (
                f"{company} said it is investigating suspicious activity consistent with an early "
                f"ransomware intrusion. Manufacturing systems were isolated as a precaution and "
                f"the company reported no production stoppage or confirmed data theft at this time."
                f"{multi_phrase}"
            )
    elif family == "ransomware_halt_exfil":
        h = f"{company} Confirms Ransomware Attack With Data Theft and Production Disruption"
        b = (
            f"{company} confirmed a ransomware attack that disrupted plant IT and industrial "
            f"control interfaces. The company said some production was halted and that attackers "
            f"may have exfiltrated business documents. Incident response teams are restoring "
            f"systems in stages.{multi_phrase}"
        )
        if impact == "minor":
            h = f"{company} Discloses Ransomware Breach; Partial Operations Resume"
            b = (
                f"{company} disclosed a ransomware incident involving possible data theft. "
                f"Some administrative systems remain offline, but the company said most "
                f"production cells have resumed with reduced throughput.{multi_phrase}"
            )
        elif impact == "none":
            h = f"Ransomware Group Claims Data Stolen From {company}; Plants Running"
            b = (
                f"A ransomware group claimed it stole files from {company}. The company "
                f"acknowledged a cyber incident affecting corporate IT but said manufacturing "
                f"operations continue without interruption.{multi_phrase}"
            )
    elif family == "malware_halt":
        h = f"Malware Outbreak Forces Temporary Shutdown at {company} Factories"
        b = (
            f"{company} said a malware outbreak on factory networks forced a temporary "
            f"shutdown of several production tools. The company is rebuilding affected hosts "
            f"and expects staggered restarts. No ransom demand was publicly confirmed."
            f"{multi_phrase}"
        )
    elif family == "exfil_only":
        h = f"Ransomware Gang Claims Theft of Files From {company}"
        b = (
            f"A ransomware group claimed it stole internal documents from {company}. "
            f"The company confirmed a cybersecurity incident involving corporate systems "
            f"but stated that production lines were not stopped and customer deliveries "
            f"remain on schedule.{multi_phrase}"
        )
    elif family == "ransomware_minor":
        h = f"{company} Contained Business Email Compromise, Brief Shipping Delays"
        b = (
            f"{company} said a business email compromise led to temporary disruption of "
            f"order-entry and shipping notifications. The company reported no encryption of "
            f"plant systems and expects only brief logistics delays.{multi_phrase}"
        )
    elif family == "noncyber_strike":
        h = f"Workers Strike at {company} Plant, Suspending Component Shipments"
        b = (
            f"Employees at a {company} manufacturing plant began a labor strike, suspending "
            f"shipments of automotive components. Management and union representatives are "
            f"in talks. The company warned customers of possible short-term supply gaps."
            f"{multi_phrase}"
        )
    elif family == "noncyber_fire":
        h = f"Fire at {company} Facility Disrupts Automotive Parts Production"
        b = (
            f"A fire at a {company} production facility damaged warehouse and assembly areas. "
            f"Authorities said there were no cyber-related causes. The company halted output "
            f"at the site while assessing rebuild timelines for affected part lines."
            f"{multi_phrase}"
        )
    elif family == "noncyber_recall":
        h = f"{company} Issues Quality Recall for Defective Automotive Components"
        b = (
            f"{company} initiated a quality recall after detecting a manufacturing defect in "
            f"a batch of automotive components. The action is a product-quality matter, not "
            f"a cybersecurity incident. Affected customers have been instructed to quarantine "
            f"suspect lots.{multi_phrase}"
        )
    elif family == "noncyber_bankruptcy":
        h = f"{company} Files for Restructuring Amid Cash-Flow Pressure"
        b = (
            f"{company} filed for court-supervised restructuring, citing cash-flow pressure "
            f"and rising input costs. The company said it intends to keep selected lines "
            f"running under creditor agreements. No cyberattack was reported."
            f"{multi_phrase}"
        )
    elif family == "noncyber_flood":
        h = f"Flooding Halts Operations at {company} Automotive Plant"
        b = (
            f"Severe flooding inundated a {company} plant, damaging equipment and inventory. "
            f"Production was stopped for safety and cleanup. The company described the outage "
            f"as weather-related, with no indication of a cyber incident.{multi_phrase}"
        )
    else:
        h = f"{company} Reports Operational Incident"
        b = f"{company} reported an operational incident under review by management.{multi_phrase}"

    return h, b


def seed_for_impact(impact: str, rng: random.Random) -> dict:
    if impact == "severe":
        pool = [s for s in SEEDS if s["template_family"] in ("ransomware_halt", "ransomware_halt_exfil", "malware_halt")]
    elif impact == "minor":
        pool = [s for s in SEEDS if s["template_family"] in ("ransomware_minor", "ransomware_halt")]
    else:
        pool = [s for s in SEEDS if s["template_family"] in ("exfil_only", "ransomware_minor")]
    return rng.choice(pool)


def metrics_for(suppliers: list[dict]) -> tuple[int, int, int, bool]:
    g = max(s["guaranteed_reach"] for s in suppliers)
    t = max(s["total_reach"] for s in suppliers)
    sole = max(s["sole_source_product_count"] for s in suppliers)
    crit = any(s["is_critical_part_sole"] or s.get("critical_sole_product_count", 0) > 0 for s in suppliers)
    return g, t, sole, crit


def make_row(
    incident_id: str,
    suppliers: list[dict],
    stratum: str,
    impact: str,
    seed: dict,
    is_cyber: bool,
    in_sc: bool,
    family: str | None = None,
    off_graph_name: str | None = None,
    off_graph_country: str | None = None,
) -> dict:
    family = family or seed["template_family"]
    if off_graph_name:
        names = [off_graph_name]
        ids = []
        g = t = sole = 0
        is_crit = False
        headline, body = render_article(off_graph_name, family, impact)
    else:
        names = [s["supplier_name"] for s in suppliers]
        ids = [s["supplier_node_id"] for s in suppliers]
        g, t, sole, is_crit = metrics_for(suppliers)
        primary = names[0]
        extras = names[1:]
        headline, body = render_article(primary, family, impact, extras if len(extras) else None)

    relevant = bool(is_cyber and in_sc)
    if relevant:
        bin_, score, rationale = draft_risk(impact, g, t, is_crit)
    else:
        bin_, score, rationale = "", 0.0, "not relevant — no OEM cyber risk gold"

    is_bottleneck = g >= 30
    has_no_alt = sole > 0 and g > 0
    is_multi = len(suppliers) > 1

    return {
        "incident_id": incident_id,
        "headline": headline,
        "body": body,
        "is_cyber": str(is_cyber).lower(),
        "in_supply_chain": str(in_sc).lower(),
        "relevant": str(relevant).lower(),
        "supplier_node_ids": "|".join(ids),
        "supplier_names": "|".join(names),
        "guaranteed_reach": g,
        "total_reach": t,
        "sole_source_product_count": sole,
        "case_stratum": stratum,
        "is_bottleneck": str(is_bottleneck).lower(),
        "is_sole_source_critical_part": str(is_crit).lower(),
        "is_multi_supplier": str(is_multi).lower(),
        "has_no_alternative": str(has_no_alt).lower(),
        "impact_level": impact,
        "risk_bin_draft": bin_,
        "risk_score_draft": f"{score:.2f}" if relevant else "",
        "risk_rationale": rationale,
        "risk_bin_gold": "",
        "seed_id": seed["seed_id"],
        "template_family": family,
        "off_graph_country": off_graph_country or "",
        "synthetic": "true",
        "disclaimer": DISCLAIMER,
    }


def write_article(row: dict) -> None:
    ARTICLES.mkdir(parents=True, exist_ok=True)
    payload = {
        "article_id": row["incident_id"],
        "incident_id": row["incident_id"],
        "headline": row["headline"],
        "body": row["body"],
        "supplier_names": row["supplier_names"].split("|") if row["supplier_names"] else [],
        "supplier_node_ids": row["supplier_node_ids"].split("|") if row["supplier_node_ids"] else [],
        "is_cyber": row["is_cyber"] == "true",
        "in_supply_chain": row["in_supply_chain"] == "true",
        "relevant": row["relevant"] == "true",
        "case_stratum": row["case_stratum"],
        "impact_level": row["impact_level"],
        "seed_id": row["seed_id"],
        "synthetic": True,
        "disclaimer": DISCLAIMER,
    }
    path = ARTICLES / f"{row['incident_id']}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def impact_schedule(n: int, severe: int, minor: int, none: int) -> list[str]:
    assert severe + minor + none == n
    return ["severe"] * severe + ["minor"] * minor + ["none"] * none


def build_incidents(strata: dict) -> list[dict]:
    rng = random.Random(42)
    rows: list[dict] = []
    n = 1

    def next_id() -> str:
        nonlocal n
        iid = f"INC_{n:03d}"
        n += 1
        return iid

    # --- 70 relevant ---
    # Order suppliers so severe×bottleneck without critical-part flag lands in `high`
    # (structure_weight=2 → score 4), while critical+bottleneck severe lands in `very_high`.
    def ordered(pool: list[dict], prefer_non_crit_first: bool = False) -> list[dict]:
        items = list(pool)
        if prefer_non_crit_first:
            items.sort(
                key=lambda r: (
                    1 if r.get("is_critical_part_sole") or r.get("critical_sole_product_count", 0) > 0 else 0,
                    -r["guaranteed_reach"],
                )
            )
        return items

    plans = [
        # severe-heavy bottlenecks (non-crit first → more `high` bins)
        ("bottleneck_sole", ordered(strata["bottleneck_sole"], True), impact_schedule(14, 10, 3, 1)),
        # critical sole: severe → very_high when g>=30; else medium/high
        ("critical_part_sole", ordered(strata["critical_part_sole"]), impact_schedule(12, 8, 3, 1)),
        # redundant: severe×0.5=1 → low; minor×0.5=0.5 → very_low; none → very_low
        ("redundant_high_reach", strata["redundant_high_reach"], impact_schedule(12, 8, 3, 1)),
        # moderate: severe×1=2 → medium; minor×1=1 → low
        ("moderate_reach", strata["moderate_reach"], impact_schedule(12, 7, 4, 1)),
        # low reach: severe×0.5=1 → low; minor → very_low/low
        ("low_reach", strata["low_reach"], impact_schedule(10, 4, 4, 2)),
    ]

    for stratum, pool, impacts in plans:
        if len(pool) < len(impacts):
            raise RuntimeError(f"Not enough suppliers for {stratum}: {len(pool)} < {len(impacts)}")
        for i, impact in enumerate(impacts):
            sup = pool[i]
            seed = seed_for_impact(impact, rng)
            # Align template to impact for clean narratives
            if impact == "severe":
                fam = rng.choice(["ransomware_halt", "ransomware_halt_exfil", "malware_halt"])
            elif impact == "minor":
                fam = rng.choice(["ransomware_minor", "ransomware_halt"])
            else:
                fam = "exfil_only"
            rows.append(
                make_row(next_id(), [sup], stratum, impact, seed, True, True, family=fam)
            )

    # multi_supplier: 10 incidents with 2–3 suppliers
    multi_impacts = impact_schedule(10, 7, 2, 1)
    bn = strata["bottleneck_sole"]
    rd = strata["redundant_high_reach"]
    md = strata["moderate_reach"]
    for i, impact in enumerate(multi_impacts):
        if i < 4:
            pair = [bn[i % len(bn)], rd[i % len(rd)]]
        elif i < 7:
            pair = [bn[(i + 3) % len(bn)], md[i % len(md)]]
        else:
            pair = [rd[(i + 2) % len(rd)], md[(i + 2) % len(md)], bn[(i + 5) % len(bn)]]
        # dedupe by id
        seen = set()
        uniq = []
        for s in pair:
            if s["supplier_node_id"] not in seen:
                seen.add(s["supplier_node_id"])
                uniq.append(s)
        seed = seed_for_impact(impact, rng)
        fam = (
            "ransomware_halt"
            if impact == "severe"
            else ("ransomware_minor" if impact == "minor" else "exfil_only")
        )
        rows.append(
            make_row(next_id(), uniq, "multi_supplier", impact, seed, True, True, family=fam)
        )

    assert sum(1 for r in rows if r["relevant"] == "true") == 70

    # --- 15 off-graph cyber ---
    for i, (name, country) in enumerate(OFF_GRAPH_COMPANIES):
        impact = ["severe", "minor", "none"][i % 3]
        seed = seed_for_impact(impact, rng)
        fam = (
            "ransomware_halt"
            if impact == "severe"
            else ("ransomware_minor" if impact == "minor" else "exfil_only")
        )
        rows.append(
            make_row(
                next_id(),
                [],
                "off_graph_cyber",
                impact,
                seed,
                True,
                False,
                family=fam,
                off_graph_name=name,
                off_graph_country=country,
            )
        )

    # --- 15 in-graph non-cyber ---
    noncyber_fams = [
        "noncyber_strike",
        "noncyber_fire",
        "noncyber_recall",
        "noncyber_bankruptcy",
        "noncyber_flood",
    ]
    noncyber_seeds = {f: next(s for s in SEEDS if s["template_family"] == f) for f in noncyber_fams}
    pool = strata["non_cyber_pool"]
    # Prefer suppliers not overused; fall back to redundant/moderate
    nc_pool = pool + strata["redundant_high_reach"] + strata["low_reach"]
    for i in range(15):
        fam = noncyber_fams[i % len(noncyber_fams)]
        seed = noncyber_seeds[fam]
        impact = "severe" if fam in ("noncyber_strike", "noncyber_fire", "noncyber_flood") else "minor"
        sup = nc_pool[i % len(nc_pool)]
        rows.append(
            make_row(next_id(), [sup], "in_graph_non_cyber", impact, seed, False, True, family=fam)
        )

    assert len(rows) == 100
    return rows


def write_seeds() -> None:
    path = ROOT / "seeds.csv"
    fields = [
        "seed_id",
        "victim_organization",
        "incident_year",
        "incident_type",
        "production_halt_confirmed",
        "data_exfiltration_confirmed",
        "default_impact",
        "public_source_refs",
        "template_family",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in SEEDS:
            w.writerow({k: s.get(k, "") for k in fields})


def write_incidents(rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    path = ROOT / "incidents.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        write_article(r)


def write_manifest(rows: list[dict]) -> None:
    from collections import Counter

    rel = [r for r in rows if r["relevant"] == "true"]
    manifest = {
        "corpus": "eval_corpus_v1",
        "version": "1.0",
        "n_incidents": len(rows),
        "n_relevant": len(rel),
        "n_irrelevant": len(rows) - len(rel),
        "strata": dict(Counter(r["case_stratum"] for r in rows)),
        "risk_bin_draft_counts": dict(Counter(r["risk_bin_draft"] for r in rel)),
        "impact_counts_relevant": dict(Counter(r["impact_level"] for r in rel)),
        "gold_relevance_rule": "relevant = is_cyber AND in_supply_chain",
        "risk_bin_gold_status": "pending_expert_review",
        "oem_graph": "dataset/ (Honda MarkLines oem_product_rooted)",
        "article_dir": "articles/",
    }
    (ROOT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def write_review(rows: list[dict]) -> None:
    rel = [r for r in rows if r["relevant"] == "true"]
    lines = [
        "# Risk bin review checklist",
        "",
        "Draft bins were computed from `impact_level × structure_weight` (see README).",
        "For each row, set **Approved bin** (or write `OK` to accept the draft), then we will",
        "copy values into `risk_bin_gold` in `incidents.csv`.",
        "",
        "| ID | Stratum | Suppliers | g_reach | Impact | Draft bin | Score | Rationale | Approved bin |",
        "|---|---|---|---:|---|---|---:|---|---|",
    ]
    for r in rel:
        names = r["supplier_names"].replace("|", "; ")
        if len(names) > 60:
            names = names[:57] + "..."
        lines.append(
            f"| {r['incident_id']} | {r['case_stratum']} | {names} | {r['guaranteed_reach']} | "
            f"{r['impact_level']} | **{r['risk_bin_draft']}** | {r['risk_score_draft']} | "
            f"{r['risk_rationale']} | |"
        )
    lines.extend(
        [
            "",
            "## How to approve",
            "",
            "1. Fill the Approved bin column with one of: `very_low`, `low`, `medium`, `high`, `very_high`, or `OK`.",
            "2. Tell the agent to freeze gold from this review (or paste overrides).",
            "3. Re-run `python validate.py` after `risk_bin_gold` is populated.",
            "",
        ]
    )
    (ROOT / "review" / "risk_bin_review.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    rows_metrics = load_graph()
    strata = pick_strata(rows_metrics)
    # persist candidates (same as select_suppliers)
    import select_suppliers as ss

    ss.OUT.mkdir(parents=True, exist_ok=True)
    with (ss.OUT / "supplier_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_metrics[0].keys()))
        w.writeheader()
        w.writerows(rows_metrics)
    for name, items in strata.items():
        if not items:
            continue
        with (ss.OUT / f"stratum_{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
            w.writeheader()
            w.writerows(items)

    incidents = build_incidents(strata)
    write_seeds()
    write_incidents(incidents)
    write_manifest(incidents)
    write_review(incidents)
    print(f"Generated {len(incidents)} incidents")
    from collections import Counter

    print("strata:", Counter(r["case_stratum"] for r in incidents))
    print(
        "draft bins:",
        Counter(r["risk_bin_draft"] for r in incidents if r["relevant"] == "true"),
    )


if __name__ == "__main__":
    main()
