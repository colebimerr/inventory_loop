"""ROI math for the calculator view.

All assumptions are exposed as inputs so a customer can sanity-check the model
on the call. Industry defaults reflect mid-market HVAC distribution profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Active hands-on labor is charged PER VARIANCE (total), not per calendar day it
# sits open — most "open" time is queue/wait, not work. ~0.5 hr of real handling
# per variance is defensible. The tool's labor benefit is a direct prioritization
# saving (chase/count fewer, smarter things), set as an input the customer can dial.
ACTIVE_HOURS_PER_EVENT = 0.5

# NOTE: a lost-sales dollar line was deliberately removed from the ROI. A stockout
# is not a reliable lost sale (backorders, substitutions, partial fills, cross-branch
# pulls), and the data to confirm one basically doesn't exist in the ERP — validated
# directly by a sophisticated operator. ROI is built only on numbers a customer can
# actually defend: variance-loop labor saved and shrink reduction.


@dataclass
class ROIInputs:
    inventory_value: float           # $
    branches: int
    events_per_month: float
    labor_rate: float                # $/hr fully loaded
    shrink_rate_pct: float           # % e.g. 1.5
    shrink_reduction_pct: float      # % e.g. 10
    labor_reduction_pct: float       # % labor saved by prioritization, e.g. 20
    annual_price: float              # $ service fee


@dataclass
class ROIResult:
    current_shrink: float
    current_labor: float
    current_total: float

    future_shrink: float
    future_labor: float
    future_total: float

    annual_savings: float
    payback_months: float


def defaults_from_data(inv_df: pd.DataFrame, var_df: pd.DataFrame,
                       br_df: pd.DataFrame) -> dict:
    """Compute realistic defaults from the loaded dataset."""
    # Inventory value: sum of unit_cost × reorder_point × velocity factor as a
    # cheap proxy for avg on-hand value. (We don't have on-hand qty in v1.)
    velocity_factor = inv_df["velocity_class"].map({"A": 1.5, "B": 1.2, "C": 1.0}).fillna(1.0)
    inv_value = float(
        (inv_df["unit_cost"] * inv_df["reorder_point"].astype(float) * velocity_factor).sum()
    )

    # Events/month: scale observed count by the history window.
    var_dates = pd.to_datetime(var_df["detection_date"], format="ISO8601")
    if len(var_dates):
        span_days = max(1, (var_dates.max() - var_dates.min()).days)
        events_per_month = len(var_df) / span_days * 30.0
    else:
        events_per_month = 0.0

    return {
        "inventory_value": round(inv_value, 0),
        "branches": int(len(br_df)),
        "events_per_month": round(events_per_month, 1),
        "labor_rate": 45.0,
        "shrink_rate_pct": 1.5,
        "shrink_reduction_pct": 10.0,
        "labor_reduction_pct": 20.0,
        "annual_price": 50000.0,
    }


def compute(inputs: ROIInputs) -> ROIResult:
    events_per_year = inputs.events_per_month * 12.0

    current_shrink = inputs.inventory_value * (inputs.shrink_rate_pct / 100.0)
    # Labor = active hands-on hours PER VARIANCE (not per day it sits open).
    current_labor  = events_per_year * ACTIVE_HOURS_PER_EVENT * inputs.labor_rate
    current_total  = current_shrink + current_labor

    # Shrink lever: faster, more accurate counting reduces unaccounted loss.
    future_shrink_rate = inputs.shrink_rate_pct * (1 - inputs.shrink_reduction_pct / 100.0)
    future_shrink = inputs.inventory_value * (future_shrink_rate / 100.0)
    # Labor lever: prioritization = chase/count fewer, smarter things.
    future_labor  = current_labor * (1 - inputs.labor_reduction_pct / 100.0)
    future_total = future_shrink + future_labor

    savings = max(0.0, current_total - future_total)
    payback_months = (inputs.annual_price / savings * 12.0
                      if savings > 0 else float("inf"))
    return ROIResult(
        current_shrink, current_labor, current_total,
        future_shrink, future_labor, future_total,
        savings, payback_months,
    )
