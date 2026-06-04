"""Chase-vs-write-off decision support.

Answers the question an ops manager actually makes several times a week, today on
gut and mental math: a variance landed — is it worth my time to chase, or do I
write it off and move on? (If it turns up later, adjust it back in.)

The honest constraint isn't usually "is this one worth chasing in isolation" — at
typical distribution dollar values, most are. The constraint is *capacity*:
variances cluster ("feast and famine"), and chasing the whole open queue can be
hundreds of labor-hours nobody has. So the model is capacity-aware:

    For each variance:
        chase value  =  recoverable $  +  stockout-avoidance value (scaled by item class)
        chase cost   =  estimated hours  ×  loaded hourly rate
        net benefit  =  value  −  cost
        value/hour   =  net benefit  ÷  estimated hours      (ROI on your time)

    Then, against a weekly labor budget, spend hours on the highest value/hour
    variances first. What fits the budget → Chase. Everything else → Write off
    (either it loses money outright, or you're out of capacity this week).

Deliberately transparent arithmetic — no ML, no black box. Every number on screen
traces to an input the operator can see and change. Item class (A/B/C) weighting is
configurable, not hardcoded.

NOTE: this scores *whether* and *which* variances to chase. It does not tell you
*why* a variance happened — that's attribution, a separate, harder module built
against a real customer's transaction data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# An A item off the count hurts most — more dollars moving and higher stockout
# exposure. A C item rarely justifies a chase on dollars alone.
ABC_STOCKOUT_WEIGHT = {"A": 1.0, "B": 0.5, "C": 0.2}

DEFAULTS = {
    "hourly_rate":         38.0,   # loaded labor cost of whoever investigates
    "hours_quick":         0.5,    # quick case: an offset exists, easy backtrack
    "hours_deep":          2.5,    # no-offset case: full cradle-to-grave trace
    "deep_threshold":      250.0,  # $ abs variance at/above which we assume the deep trace
    "recovery_prob":       0.55,   # share of chased variances that recover value
    "stockout_value":      60.0,   # $ value of avoiding a stockout on a top-velocity miss
    "labor_budget_hours":  80.0,   # investigation hours available this week
}


def score_chase(events: pd.DataFrame, inv_df: pd.DataFrame,
                params: dict | None = None) -> pd.DataFrame:
    """Score and allocate each variance to Chase or Write off under a labor budget.

    Returns a copy of ``events`` with decision columns appended, sorted best-first
    by value-per-hour.
    """
    p = {**DEFAULTS, **(params or {})}
    df = events.copy()

    cols = ("abs_cost", "velocity_class", "est_hours", "chase_cost", "stockout_value",
            "expected_recovery", "chase_value", "net_benefit", "value_per_hour",
            "recommendation")
    if not len(df):
        for c in cols:
            df[c] = pd.Series(dtype="object" if c in
                              ("velocity_class", "recommendation") else "float64")
        return df

    inv_idx = inv_df.set_index("sku_id")
    df["abs_cost"] = df["variance_cost"].abs()
    df["velocity_class"] = df["sku_id"].map(inv_idx["velocity_class"]).fillna("C")
    if "description" not in df.columns:
        df["description"] = df["sku_id"].map(inv_idx["description"]).fillna("")
    if "category" not in df.columns:
        df["category"] = df["sku_id"].map(inv_idx["category"]).fillna("")

    # Time to investigate: dollar size stands in for complexity (big misses tend to
    # be the no-offset, deep-trace cases). Knowing up front whether an offset exists
    # — so you'd know it's a quick one — is the attribution module's job.
    df["est_hours"] = np.where(df["abs_cost"] >= p["deep_threshold"],
                               p["hours_deep"], p["hours_quick"])
    df["chase_cost"] = df["est_hours"] * p["hourly_rate"]

    df["stockout_value"] = (df["velocity_class"].map(ABC_STOCKOUT_WEIGHT).fillna(0.2)
                            * p["stockout_value"])
    df["expected_recovery"] = df["abs_cost"] * p["recovery_prob"]
    df["chase_value"] = df["expected_recovery"] + df["stockout_value"]
    df["net_benefit"] = df["chase_value"] - df["chase_cost"]
    df["value_per_hour"] = df["net_benefit"] / df["est_hours"].replace(0, np.nan)

    # Spend the weekly budget on the best value-per-hour variances first.
    df = df.sort_values("value_per_hour", ascending=False).reset_index(drop=True)
    eligible = df["net_benefit"] > 0
    cum_hours = df["est_hours"].where(eligible, 0).cumsum()
    within_budget = cum_hours <= p["labor_budget_hours"]
    df["recommendation"] = np.where(eligible & within_budget, "Chase", "Write off")
    return df


def summarize(scored: pd.DataFrame, params: dict | None = None) -> dict:
    """Headline numbers for the KPI row."""
    p = {**DEFAULTS, **(params or {})}
    if not len(scored):
        return {"chase_n": 0, "writeoff_n": 0, "chase_hours": 0.0, "chase_net": 0.0,
                "writeoff_exposure": 0.0, "total_hours_if_all": 0.0,
                "budget": p["labor_budget_hours"]}
    chase = scored[scored["recommendation"] == "Chase"]
    writeoff = scored[scored["recommendation"] == "Write off"]
    return {
        "chase_n": int(len(chase)),
        "writeoff_n": int(len(writeoff)),
        "chase_hours": float(chase["est_hours"].sum()),
        "chase_net": float(chase["net_benefit"].sum()),
        "writeoff_exposure": float(writeoff["abs_cost"].sum()),
        "total_hours_if_all": float(scored["est_hours"].sum()),
        "budget": float(p["labor_budget_hours"]),
    }
