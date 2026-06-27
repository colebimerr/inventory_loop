"""Per-SKU risk scoring heuristic — readable, no ML."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Weights are tuned so the leaderboard surfaces a believable mix of:
#   - class-A SKUs that are overdue against their (tighter) count cadence
#   - SKUs with recent variance churn
#   - high-dollar SKUs even if otherwise quiet
WEIGHTS = {
    "cadence_overdue":  0.35,
    "recent_variance":  0.30,
    "velocity":         0.15,
    "problem_branch":   0.10,
    "dollar_exposure":  0.10,
}

VELOCITY_SCORE = {"A": 1.0, "B": 0.6, "C": 0.3}

# Expected cycle-count cadence by class, in days between counts. A items are
# meant to be counted ~4×/yr, B ~3×, C ~2×, D ~1×. We score how overdue a SKU is
# *against its own cadence*, not by raw days — so an A item at 80 days (already
# past its ~90-day cadence) outranks a C item at 100 days (well inside its ~180).
# This is the item-class signal a seasoned ops lead uses; flat days-since-count
# misses it. Cadence per HARDI/distribution cycle-count norms.
COUNT_CADENCE_DAYS = {"A": 91, "B": 122, "C": 182, "D": 365}


def _normalize(s: pd.Series) -> pd.Series:
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def _identify_problem_branches(var_df: pd.DataFrame) -> set[str]:
    counts = var_df["branch_id"].value_counts()
    if not len(counts):
        return set()
    threshold = counts.median() * 1.5
    return set(counts[counts > threshold].index)


def score_skus(inv_df: pd.DataFrame, var_df: pd.DataFrame,
               br_df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return inventory rows enriched with risk_score plus contributing factors."""
    now = now or pd.Timestamp.now()
    inv = inv_df.copy()
    inv["last_cycle_count_date"] = pd.to_datetime(inv["last_cycle_count_date"],
                                                   format="ISO8601")
    inv["days_since_count"] = (now - inv["last_cycle_count_date"]).dt.days.clip(lower=0)

    # Overdue *relative to the item's class cadence* — the item-class signal.
    inv["expected_cadence_days"] = (
        inv["velocity_class"].map(COUNT_CADENCE_DAYS).fillna(COUNT_CADENCE_DAYS["C"])
    )
    inv["cadence_overdue_ratio"] = inv["days_since_count"] / inv["expected_cadence_days"]

    # Recent variance per SKU
    cutoff = now - pd.Timedelta(days=30)
    var_dates = pd.to_datetime(var_df["detection_date"], format="ISO8601")
    recent = var_df.loc[var_dates >= cutoff]
    recent_counts = recent.groupby("sku_id").size().rename("recent_variance_count")

    # Problem-branch involvement: did this SKU have a variance at a problem branch
    # in the last 90 days?
    problem_branches = _identify_problem_branches(var_df)
    pb_cutoff = now - pd.Timedelta(days=90)
    pb_recent = var_df.loc[var_dates >= pb_cutoff]
    pb_skus = set(pb_recent.loc[pb_recent["branch_id"].isin(problem_branches), "sku_id"])

    # Most-recent variance branch — used as the "branch" column in the leaderboard
    var_sorted = var_df.assign(detection_dt=var_dates).sort_values("detection_dt")
    last_branch = var_sorted.groupby("sku_id")["branch_id"].last()

    inv = inv.merge(recent_counts, left_on="sku_id", right_index=True, how="left")
    inv["recent_variance_count"] = inv["recent_variance_count"].fillna(0).astype(int)
    inv["last_variance_branch"] = inv["sku_id"].map(last_branch)
    inv["at_problem_branch"] = inv["sku_id"].isin(pb_skus).astype(int)
    inv["velocity_score"] = inv["velocity_class"].map(VELOCITY_SCORE).fillna(0.3)

    # Normalize each factor to [0, 1] then take a weighted sum.
    f_cadence = _normalize(inv["cadence_overdue_ratio"])
    f_var    = _normalize(inv["recent_variance_count"].clip(upper=5))
    f_vel    = inv["velocity_score"]
    f_pb     = inv["at_problem_branch"]
    f_dollar = _normalize(inv["unit_cost"])

    inv["risk_score"] = (
        WEIGHTS["cadence_overdue"] * f_cadence
        + WEIGHTS["recent_variance"] * f_var
        + WEIGHTS["velocity"]        * f_vel
        + WEIGHTS["problem_branch"]  * f_pb
        + WEIGHTS["dollar_exposure"] * f_dollar
    ) * 100.0

    # Crude exposure estimate: unit_cost × (reorder_point × velocity factor)
    inv["estimated_exposure"] = inv["unit_cost"] * (
        inv["reorder_point"] * inv["velocity_score"] * 0.5
    )
    inv = inv.merge(
        br_df[["branch_id", "branch_name"]],
        left_on="last_variance_branch", right_on="branch_id", how="left",
    )
    inv["branch_name"] = inv["branch_name"].fillna("All branches")
    return inv.drop(columns=["branch_id"], errors="ignore")
