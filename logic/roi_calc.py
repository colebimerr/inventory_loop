"""ROI math for the calculator view.

All assumptions are exposed as inputs so a customer can sanity-check the model
on the call. Industry defaults reflect mid-market HVAC distribution profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ACTIVE_HOURS_PER_DAY = 2.0      # time a person actually spends on a loop per calendar day
STOCKOUT_RATE = 0.15            # share of variance events that produce a lost sale


@dataclass
class ROIInputs:
    inventory_value: float           # $
    branches: int
    events_per_month: float
    current_loop_days: float
    labor_rate: float                # $/hr fully loaded
    lost_sale_value: float           # $/stockout
    shrink_rate_pct: float           # % e.g. 1.5
    target_loop_hours: float         # hours
    shrink_reduction_pct: float      # % e.g. 30
    annual_price: float              # $ tool list price


@dataclass
class ROIResult:
    current_shrink: float
    current_labor: float
    current_lost_sales: float
    current_total: float

    future_shrink: float
    future_labor: float
    future_lost_sales: float
    future_total: float

    annual_savings: float
    payback_months: float


def defaults_from_data(inv_df: pd.DataFrame, var_df: pd.DataFrame,
                       br_df: pd.DataFrame) -> dict:
    """Compute realistic defaults from the loaded dataset."""
    # Inventory value: sum of unit_cost × reorder_point × velocity factor as a
    # cheap proxy for avg on-hand value. (We don't have on-hand qty in v1.)
    velocity_factor = inv_df["velocity_class"].map({"A": 3.0, "B": 2.0, "C": 1.0}).fillna(1.0)
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

    # Avg loop time from closed events in the last 30 days.
    res_dates = pd.to_datetime(var_df["resolution_date"], format="ISO8601",
                                errors="coerce")
    closed = var_df.loc[res_dates.notna()].copy()
    if len(closed):
        closed["loop_days"] = (
            pd.to_datetime(closed["resolution_date"], format="ISO8601")
            - pd.to_datetime(closed["detection_date"], format="ISO8601")
        ).dt.total_seconds() / 86400.0
        loop_days = float(closed["loop_days"].mean())
    else:
        loop_days = 2.5

    return {
        "inventory_value": round(inv_value, 0),
        "branches": int(len(br_df)),
        "events_per_month": round(events_per_month, 1),
        "current_loop_days": round(loop_days, 2),
        "labor_rate": 45.0,
        "lost_sale_value": 250.0,
        "shrink_rate_pct": 1.5,
        "target_loop_hours": 4.0,
        "shrink_reduction_pct": 30.0,
        "annual_price": 50000.0,
    }


def compute(inputs: ROIInputs) -> ROIResult:
    events_per_year = inputs.events_per_month * 12.0

    current_shrink = inputs.inventory_value * (inputs.shrink_rate_pct / 100.0)
    current_labor  = (events_per_year * inputs.current_loop_days
                      * ACTIVE_HOURS_PER_DAY * inputs.labor_rate)
    current_lost   = events_per_year * STOCKOUT_RATE * inputs.lost_sale_value
    current_total  = current_shrink + current_labor + current_lost

    target_loop_days = inputs.target_loop_hours / 24.0
    future_shrink_rate = inputs.shrink_rate_pct * (1 - inputs.shrink_reduction_pct / 100.0)
    future_shrink = inputs.inventory_value * (future_shrink_rate / 100.0)
    future_labor  = (events_per_year * target_loop_days
                     * ACTIVE_HOURS_PER_DAY * inputs.labor_rate)
    future_lost   = (events_per_year * STOCKOUT_RATE * inputs.lost_sale_value
                     * (target_loop_days / max(inputs.current_loop_days, 0.001)))
    future_total = future_shrink + future_labor + future_lost

    savings = max(0.0, current_total - future_total)
    payback_months = (inputs.annual_price / savings * 12.0
                      if savings > 0 else float("inf"))
    return ROIResult(
        current_shrink, current_labor, current_lost, current_total,
        future_shrink, future_labor, future_lost, future_total,
        savings, payback_months,
    )
