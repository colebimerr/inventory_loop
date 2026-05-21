"""Synthetic data generator for the InventoryLoop demo.

Writes three CSVs to /data:
    - inventory_master.csv
    - variance_events.csv
    - branches.csv

Run: python generate_data.py
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration — tweak here to scale the demo dataset up or down.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    n_skus: int = 2500
    history_days: int = 180
    variance_events_total: int = 1800        # ~10/day across 5 branches
    open_event_ratio: float = 0.10            # share of events still mid-loop
    problem_branch_multiplier: float = 2.0    # one branch generates 2x events

    category_weights: dict = None
    detection_source_weights: dict = None
    velocity_weights: dict = None
    branches: list = None

    def __post_init__(self):
        self.category_weights = {
            "Refrigerant": 0.20,
            "Copper Tubing": 0.25,
            "Compressor": 0.10,
            "Motor": 0.15,
            "Controls": 0.20,
            "Miscellaneous": 0.10,
        }
        self.detection_source_weights = {
            "cycle_count": 0.40,
            "picking": 0.35,
            "stocking": 0.15,
            "branch_escape": 0.10,
        }
        self.velocity_weights = {"A": 0.20, "B": 0.30, "C": 0.50}
        # 1 DC + 4 branches. Branch 4 is the "problem branch."
        self.branches = [
            {"branch_id": "DC-01",  "branch_name": "Dallas Distribution Center",
             "branch_type": "DC",     "city": "Dallas",        "state": "TX"},
            {"branch_id": "BR-101", "branch_name": "Atlanta Branch",
             "branch_type": "branch", "city": "Atlanta",       "state": "GA"},
            {"branch_id": "BR-102", "branch_name": "Charlotte Branch",
             "branch_type": "branch", "city": "Charlotte",     "state": "NC"},
            {"branch_id": "BR-103", "branch_name": "Nashville Branch",
             "branch_type": "branch", "city": "Nashville",     "state": "TN"},
            {"branch_id": "BR-104", "branch_name": "Birmingham Branch",
             "branch_type": "branch", "city": "Birmingham",    "state": "AL"},
        ]


CFG = Config()
PROBLEM_BRANCH_ID = "BR-104"


# ---------------------------------------------------------------------------
# Authentic-feeling HVAC SKU descriptors.
# ---------------------------------------------------------------------------
CATEGORY_TEMPLATES = {
    "Refrigerant": {
        "names": ["R-410A", "R-32", "R-454B", "R-22", "R-407C", "R-134a", "R-404A", "R-448A"],
        "sizes": ["25lb cylinder", "50lb jug", "100lb cylinder", "12oz can", "30lb tank"],
        "cost_range": (95, 850),
    },
    "Copper Tubing": {
        "names": ["Soft Copper Tubing", "Hard Copper Tubing", "Line Set", "ACR Copper"],
        "sizes": ["1/4 in. x 50ft", "3/8 in. x 50ft", "1/2 in. x 50ft", "5/8 in. x 50ft",
                  "3/4 in. x 50ft", "7/8 in. x 50ft", "1-1/8 in. x 50ft"],
        "cost_range": (35, 425),
    },
    "Compressor": {
        "names": ["Scroll Compressor", "Rotary Compressor", "Reciprocating Compressor",
                  "Variable Speed Compressor"],
        "sizes": ["1.5 Ton 208V", "2 Ton 208V", "2.5 Ton 230V", "3 Ton 230V",
                  "3.5 Ton 230V", "4 Ton 460V", "5 Ton 460V"],
        "cost_range": (450, 4800),
    },
    "Motor": {
        "names": ["Condenser Fan Motor", "Blower Motor", "ECM Motor", "PSC Motor",
                  "Inducer Motor", "Direct Drive Motor"],
        "sizes": ["1/6 HP", "1/4 HP", "1/3 HP", "1/2 HP", "3/4 HP", "1 HP"],
        "cost_range": (85, 685),
    },
    "Controls": {
        "names": ["Programmable Thermostat", "Smart Thermostat", "Defrost Control Board",
                  "Furnace Control Board", "Contactor", "Capacitor", "Relay", "Pressure Switch"],
        "sizes": ["24V", "120V", "Single Stage", "Two Stage", "Heat Pump",
                  "30A", "40A", "45 MFD", "70 MFD"],
        "cost_range": (12, 385),
    },
    "Miscellaneous": {
        "names": ["Line Set Insulation", "Drain Pan", "Refrigerant Recovery Tank",
                  "Filter Drier", "Sight Glass", "Schrader Valve Core",
                  "Brazing Rod", "Nitrogen Regulator"],
        "sizes": ["1/2 in.", "3/4 in.", "1 in.", "Standard", "Heavy Duty"],
        "cost_range": (5, 245),
    },
}


def _category_choices(n: int) -> list[str]:
    cats = list(CFG.category_weights.keys())
    weights = list(CFG.category_weights.values())
    return random.choices(cats, weights=weights, k=n)


def _velocity_choices(n: int) -> list[str]:
    cls = list(CFG.velocity_weights.keys())
    weights = list(CFG.velocity_weights.values())
    return random.choices(cls, weights=weights, k=n)


def _make_description(category: str) -> tuple[str, str]:
    tpl = CATEGORY_TEMPLATES[category]
    name = random.choice(tpl["names"])
    size = random.choice(tpl["sizes"])
    desc = f"{name}, {size}"
    # Build a stable SKU id from category prefix + hash of description + counter
    return name, size, desc


def _sku_id(category: str, name: str, size: str, idx: int) -> str:
    prefix = {
        "Refrigerant":   "HVAC-REF",
        "Copper Tubing": "HVAC-CU",
        "Compressor":    "HVAC-COMP",
        "Motor":         "HVAC-MOT",
        "Controls":      "HVAC-CTL",
        "Miscellaneous": "HVAC-MSC",
    }[category]
    tag = name.split()[0].upper().replace("-", "")[:6]
    return f"{prefix}-{tag}-{idx:05d}"


def _sample_unit_cost(category: str) -> float:
    lo, hi = CATEGORY_TEMPLATES[category]["cost_range"]
    # Lognormal-ish skew toward the low end of the band.
    raw = np.random.lognormal(mean=np.log((lo + hi) / 6), sigma=0.7)
    cost = float(np.clip(raw + lo, lo, hi))
    return round(cost, 2)


def _reorder_point(velocity: str) -> int:
    base = {"A": 50, "B": 20, "C": 8}[velocity]
    return int(np.clip(np.random.normal(base, base * 0.4), 2, base * 4))


def generate_inventory(today: datetime) -> pd.DataFrame:
    rows = []
    categories = _category_choices(CFG.n_skus)
    velocities = _velocity_choices(CFG.n_skus)
    for i in range(CFG.n_skus):
        cat = categories[i]
        name, size, desc = _make_description(cat)
        sku_id = _sku_id(cat, name, size, i)
        velocity = velocities[i]
        cost = _sample_unit_cost(cat)
        rop = _reorder_point(velocity)
        # Last cycle count between 5 and 200 days ago — older for C class
        days_back = int(np.clip(
            np.random.normal({"A": 30, "B": 75, "C": 130}[velocity], 25),
            5, 200,
        ))
        last_count = (today - timedelta(days=days_back)).date()
        rows.append({
            "sku_id": sku_id,
            "description": desc,
            "category": cat,
            "unit_cost": cost,
            "velocity_class": velocity,
            "reorder_point": rop,
            "reorder_qty": int(rop * np.random.uniform(1.5, 3.0)),
            "last_cycle_count_date": last_count.isoformat(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Variance events
# ---------------------------------------------------------------------------
def _sample_loop_time_days() -> float:
    """Log-normal loop time, mean ~2.5d with ~5% long-tail above 7d."""
    val = np.random.lognormal(mean=0.5, sigma=0.85)
    return float(np.clip(val, 0.1, 30.0))


def _sample_variance_qty(velocity: str) -> int:
    base = {"A": 4, "B": 2, "C": 1}[velocity]
    sign = -1 if random.random() < 0.65 else 1  # shortages > overages
    mag = max(1, int(abs(np.random.normal(base, base))))
    return sign * mag


STAGES = ["detected", "investigating", "physical_count", "system_correction", "resolved"]


def _build_stage_history(detected_at: datetime, total_loop_hours: float,
                         is_open: bool) -> tuple[list[dict], str, datetime | None]:
    """Distribute the loop across stages. Returns (history, current_status, resolution_dt)."""
    # Stage share of total loop — investigating is the longest by design.
    shares = [0.0, 0.45, 0.30, 0.20, 0.05]  # detected is instantaneous
    if is_open:
        # Truncate the event at a random stage so it's still in-progress.
        stop_at = random.choices(
            [1, 2, 3, 4],  # investigating, physical_count, system_correction, "almost resolved"
            weights=[0.40, 0.35, 0.20, 0.05],
        )[0]
    else:
        stop_at = 5  # all stages complete

    history = []
    cursor = detected_at
    final_status = "open"
    for idx, stage in enumerate(STAGES):
        if idx >= stop_at:
            break
        duration = total_loop_hours * shares[idx]
        if stage == "detected":
            history.append({
                "stage": stage,
                "started_at": cursor.isoformat(),
                "completed_at": cursor.isoformat(),
            })
            continue
        completed_cursor = cursor + timedelta(hours=duration)
        # If we're at the last stage we'll execute AND this event is open,
        # leave completed_at null to signal in-progress.
        if is_open and idx == stop_at - 1:
            history.append({
                "stage": stage,
                "started_at": cursor.isoformat(),
                "completed_at": None,
            })
            final_status = {
                "investigating": "investigating",
                "physical_count": "physical_count_complete",
                "system_correction": "system_corrected",
            }.get(stage, "investigating")
            cursor = completed_cursor
        else:
            history.append({
                "stage": stage,
                "started_at": cursor.isoformat(),
                "completed_at": completed_cursor.isoformat(),
            })
            cursor = completed_cursor

    resolution_dt = None
    if not is_open:
        final_status = "resolved"
        resolution_dt = cursor  # cursor is end of "resolved" stage
    return history, final_status, resolution_dt


HANDLERS = ["M. Hernandez", "T. Wilson", "S. Patel", "J. Nguyen", "R. Johnson",
            "K. O'Brien", "D. Rivera", "L. Chen", "A. Brooks", "C. Martinez"]


def generate_variance_events(inventory: pd.DataFrame, today: datetime) -> pd.DataFrame:
    branches = CFG.branches
    branch_ids = [b["branch_id"] for b in branches]

    # Build a weighting that doubles the rate at the problem branch.
    weights = [CFG.problem_branch_multiplier if b == PROBLEM_BRANCH_ID else 1.0
               for b in branch_ids]
    weights = [w / sum(weights) for w in weights]

    sources = list(CFG.detection_source_weights.keys())
    src_weights = list(CFG.detection_source_weights.values())

    n_open = int(CFG.variance_events_total * CFG.open_event_ratio)
    n_closed = CFG.variance_events_total - n_open

    sku_records = inventory.to_dict("records")
    rows = []

    # --- Closed events: distributed throughout history -----------------------
    for _ in range(n_closed):
        sku = random.choice(sku_records)
        branch_id = random.choices(branch_ids, weights=weights)[0]
        source = random.choices(sources, weights=src_weights)[0]
        # Detection somewhere in [history_days, 1] days ago — but with enough
        # room for the loop to finish before "today".
        days_ago = random.uniform(2, CFG.history_days)
        detected_at = today - timedelta(days=days_ago)
        loop_hours = _sample_loop_time_days() * 24
        # Ensure resolution doesn't fall after today.
        if detected_at + timedelta(hours=loop_hours) > today:
            loop_hours = max(1.0, (today - detected_at).total_seconds() / 3600 - 1)
        history, status, resolved_at = _build_stage_history(detected_at, loop_hours, False)
        variance_qty = _sample_variance_qty(sku["velocity_class"])
        rows.append({
            "event_id": str(uuid.uuid4()),
            "sku_id": sku["sku_id"],
            "branch_id": branch_id,
            "detection_source": source,
            "detection_date": detected_at.isoformat(),
            "resolution_date": resolved_at.isoformat() if resolved_at else None,
            "expected_qty": int(np.random.randint(5, 200)),
            "actual_qty": None,  # filled below from expected + variance
            "variance_qty": variance_qty,
            "variance_cost": round(variance_qty * sku["unit_cost"], 2),
            "stage_history": json.dumps(history),
            "assigned_to": random.choice(HANDLERS),
            "current_status": status,
        })

    # --- Open events: clustered in the last ~10 days -------------------------
    for _ in range(n_open):
        sku = random.choice(sku_records)
        branch_id = random.choices(branch_ids, weights=weights)[0]
        source = random.choices(sources, weights=src_weights)[0]
        days_ago = random.uniform(0.2, 10)
        detected_at = today - timedelta(days=days_ago)
        # Loop is in progress — use a partial elapsed window.
        elapsed_hours = (today - detected_at).total_seconds() / 3600
        history, status, _ = _build_stage_history(detected_at, elapsed_hours, True)
        variance_qty = _sample_variance_qty(sku["velocity_class"])
        rows.append({
            "event_id": str(uuid.uuid4()),
            "sku_id": sku["sku_id"],
            "branch_id": branch_id,
            "detection_source": source,
            "detection_date": detected_at.isoformat(),
            "resolution_date": None,
            "expected_qty": int(np.random.randint(5, 200)),
            "actual_qty": None,
            "variance_qty": variance_qty,
            "variance_cost": round(variance_qty * sku["unit_cost"], 2),
            "stage_history": json.dumps(history),
            "assigned_to": random.choice(HANDLERS),
            "current_status": status,
        })

    df = pd.DataFrame(rows)
    df["actual_qty"] = (df["expected_qty"] + df["variance_qty"]).clip(lower=0)
    df = df.sort_values("detection_date").reset_index(drop=True)
    return df


def generate_branches() -> pd.DataFrame:
    return pd.DataFrame(CFG.branches)


def main() -> None:
    today = datetime.now().replace(microsecond=0)
    print(f"Generating synthetic InventoryLoop dataset for {today.date()}…")

    inventory = generate_inventory(today)
    print(f"  inventory_master: {len(inventory):,} SKUs")

    events = generate_variance_events(inventory, today)
    print(f"  variance_events:  {len(events):,} events "
          f"({(events['current_status'] != 'resolved').sum():,} open)")

    branches = generate_branches()
    print(f"  branches:         {len(branches):,} locations")

    inventory.to_csv(DATA_DIR / "inventory_master.csv", index=False)
    events.to_csv(DATA_DIR / "variance_events.csv", index=False)
    branches.to_csv(DATA_DIR / "branches.csv", index=False)
    print(f"Wrote CSVs to {DATA_DIR}")


if __name__ == "__main__":
    main()
