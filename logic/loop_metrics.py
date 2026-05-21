"""Loop-time and stage-timing calculations for the variance queue view."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

STAGE_ORDER = ["detected", "investigating", "physical_count", "system_correction", "resolved"]
STAGE_LABELS = {
    "detected":          "Detected",
    "investigating":     "Investigating",
    "physical_count":    "Physical count",
    "system_correction": "System correction",
    "resolved":          "Resolved",
}
STATUS_TO_STAGE = {
    "open":                    "detected",
    "investigating":           "investigating",
    "physical_count_complete": "physical_count",
    "system_corrected":        "system_correction",
    "resolved":                "resolved",
}


# ---------------------------------------------------------------------------
@dataclass
class QueueKPIs:
    avg_loop_days_30d: float
    p90_loop_days_30d: float
    open_over_3_days: int
    open_dollar_exposure: float
    open_count: int


def enrich_events(var_df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Return a copy of var_df with derived columns used everywhere downstream."""
    now = now or pd.Timestamp.now()
    df = var_df.copy()

    df["detection_date"] = pd.to_datetime(df["detection_date"], format="ISO8601")
    df["resolution_date"] = pd.to_datetime(df["resolution_date"], format="ISO8601",
                                           errors="coerce")
    df["is_open"] = df["current_status"] != "resolved"
    df["loop_days"] = np.where(
        df["is_open"],
        (now - df["detection_date"]).dt.total_seconds() / 86400.0,
        (df["resolution_date"] - df["detection_date"]).dt.total_seconds() / 86400.0,
    )
    df["loop_hours"] = df["loop_days"] * 24
    df["abs_variance_cost"] = df["variance_cost"].abs()
    df["detection_week"] = df["detection_date"].dt.to_period("W").dt.start_time
    df["current_stage"] = df["current_status"].map(STATUS_TO_STAGE).fillna("detected")
    return df


def compute_kpis(enriched: pd.DataFrame, now: datetime | None = None) -> QueueKPIs:
    now = now or pd.Timestamp.now()
    cutoff = now - pd.Timedelta(days=30)
    recent_resolved = enriched[(~enriched["is_open"]) &
                               (enriched["resolution_date"] >= cutoff)]
    open_events = enriched[enriched["is_open"]]
    return QueueKPIs(
        avg_loop_days_30d=(float(recent_resolved["loop_days"].mean())
                           if len(recent_resolved) else 0.0),
        p90_loop_days_30d=(float(recent_resolved["loop_days"].quantile(0.9))
                           if len(recent_resolved) else 0.0),
        open_over_3_days=int((open_events["loop_days"] > 3).sum()),
        open_dollar_exposure=float(open_events["abs_variance_cost"].sum()),
        open_count=int(len(open_events)),
    )


def funnel_counts(enriched: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """Cumulative-style funnel: how many events have *reached at least* each stage,
    looking at events detected in the last N days."""
    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(days=lookback_days)
    recent = enriched[enriched["detection_date"] >= cutoff]
    reached_idx = {s: i for i, s in enumerate(STAGE_ORDER)}
    out = []
    for stage in STAGE_ORDER:
        count = int((recent["current_stage"].map(reached_idx) >= reached_idx[stage]).sum())
        out.append({"stage": STAGE_LABELS[stage], "count": count})
    return pd.DataFrame(out)


def loop_time_histogram(enriched: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(days=lookback_days)
    recent = enriched[(~enriched["is_open"]) &
                      (enriched["resolution_date"] >= cutoff)].copy()
    if not len(recent):
        return pd.DataFrame({"bucket_days": [], "count": []})
    recent["bucket_days"] = recent["loop_days"].clip(upper=14).round().astype(int)
    counts = recent.groupby("bucket_days").size().reset_index(name="count")
    return counts.sort_values("bucket_days")


def loop_time_trend(enriched: pd.DataFrame, weeks: int = 26) -> pd.DataFrame:
    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(weeks=weeks)
    recent = enriched[(~enriched["is_open"]) &
                      (enriched["resolution_date"] >= cutoff)].copy()
    if not len(recent):
        return pd.DataFrame({"resolution_week": [], "avg_loop_days": []})
    recent["resolution_week"] = recent["resolution_date"].dt.to_period("W").dt.start_time
    trend = (recent.groupby("resolution_week")["loop_days"].mean()
             .reset_index().rename(columns={"loop_days": "avg_loop_days"}))
    return trend.sort_values("resolution_week")


# ---------------------------------------------------------------------------
def stage_timing_for_event(stage_history_json: str,
                           now: datetime | None = None) -> list[dict]:
    """Return per-stage timing for a single event (used in the detail table)."""
    now = now or pd.Timestamp.now()
    try:
        stages = json.loads(stage_history_json) if stage_history_json else []
    except (TypeError, json.JSONDecodeError):
        return []
    out = []
    for s in stages:
        started = pd.to_datetime(s.get("started_at"))
        completed = s.get("completed_at")
        end = pd.to_datetime(completed) if completed else now
        out.append({
            "stage": s.get("stage"),
            "started_at": started,
            "completed_at": pd.to_datetime(completed) if completed else None,
            "hours": (end - started).total_seconds() / 3600.0 if pd.notna(started) else 0.0,
            "in_progress": completed is None,
        })
    return out


def detail_table(enriched: pd.DataFrame, inv_df: pd.DataFrame,
                 br_df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """Detail rows: all open events plus the last N days of resolved events."""
    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(days=lookback_days)
    recent_resolved = enriched[(~enriched["is_open"]) &
                               (enriched["resolution_date"] >= cutoff)]
    open_events = enriched[enriched["is_open"]]
    combined = pd.concat([open_events, recent_resolved], ignore_index=True)

    sku_lookup = inv_df.set_index("sku_id")[["description", "category", "velocity_class"]]
    br_lookup = br_df.set_index("branch_id")["branch_name"]

    combined = combined.merge(sku_lookup, left_on="sku_id", right_index=True, how="left")
    combined["branch_name"] = combined["branch_id"].map(br_lookup)
    combined = combined.sort_values(["is_open", "loop_days"],
                                    ascending=[False, False])
    return combined
