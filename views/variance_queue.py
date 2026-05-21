"""View 2 — Variance Queue."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic.loop_metrics import (
    STAGE_LABELS, STAGE_ORDER, compute_kpis, detail_table, enrich_events,
    funnel_counts, loop_time_histogram, loop_time_trend, stage_timing_for_event,
)

STAGE_COLORS = {
    "detected":          "#94A3B8",
    "investigating":     "#F59E0B",
    "physical_count":    "#3B82F6",
    "system_correction": "#8B5CF6",
    "resolved":          "#10B981",
}


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}"


def _fmt_days(v: float) -> str:
    if v < 1:
        return f"{v * 24:.1f}h"
    return f"{v:.1f}d"


def _kpi_row(kpis) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg loop time (last 30d)", _fmt_days(kpis.avg_loop_days_30d))
    with c2:
        st.metric("P90 loop time (last 30d)", _fmt_days(kpis.p90_loop_days_30d))
    with c3:
        st.metric("Open > 3 days", f"{kpis.open_over_3_days:,}",
                  delta=f"of {kpis.open_count} open", delta_color="off")
    with c4:
        st.metric("$ exposure in open queue", _fmt_money(kpis.open_dollar_exposure))


def _funnel_chart(funnel_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Funnel(
        y=funnel_df["stage"],
        x=funnel_df["count"],
        textinfo="value+percent initial",
        marker={"color": ["#94A3B8", "#F59E0B", "#3B82F6", "#8B5CF6", "#10B981"]},
    ))
    fig.update_layout(
        title="Where events are right now (events detected in last 30 days)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
    )
    return fig


def _histogram(hist_df: pd.DataFrame) -> go.Figure:
    if not len(hist_df):
        fig = go.Figure()
        fig.update_layout(title="Loop time distribution (last 90 days) — no data")
        return fig
    fig = px.bar(
        hist_df, x="bucket_days", y="count",
        labels={"bucket_days": "Loop time (days, capped at 14)", "count": "Events"},
    )
    fig.update_traces(marker_color="#0F5BD9")
    fig.update_layout(
        title="Loop time distribution (last 90 days)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
    )
    return fig


def _trend(trend_df: pd.DataFrame) -> go.Figure:
    if not len(trend_df):
        fig = go.Figure()
        fig.update_layout(title="Avg loop time by week — no data")
        return fig
    fig = px.line(
        trend_df, x="resolution_week", y="avg_loop_days",
        markers=True,
        labels={"resolution_week": "Week", "avg_loop_days": "Avg loop time (days)"},
    )
    fig.update_traces(line_color="#0F5BD9", marker_color="#0F5BD9")
    fig.add_hline(y=3.0, line_dash="dot", line_color="#94A3B8",
                  annotation_text="3 day SLA target", annotation_position="top left")
    fig.update_layout(
        title="Avg loop time by week (last 26 weeks)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
    )
    return fig


def _stage_bar(stage_records: list[dict]) -> str:
    """Compact unicode bar showing per-stage time. Used in the detail table."""
    if not stage_records:
        return "—"
    chars = []
    for s in stage_records:
        stage = s.get("stage", "")
        label = {
            "detected": "D", "investigating": "I", "physical_count": "P",
            "system_correction": "S", "resolved": "R",
        }.get(stage, "?")
        if s.get("in_progress"):
            chars.append(f"[{label}…]")
        else:
            hrs = s.get("hours", 0.0)
            chars.append(f"{label}:{hrs:.0f}h")
    return " → ".join(chars)


# ---------------------------------------------------------------------------
def render(inv_df: pd.DataFrame, var_df: pd.DataFrame, br_df: pd.DataFrame) -> None:
    st.subheader("Variance Queue")
    st.caption(
        "Live tracker of variance events from detection through resolution. "
        "The whole loop, on one page."
    )

    enriched = enrich_events(var_df)
    kpis = compute_kpis(enriched)
    _kpi_row(kpis)

    # Filters
    with st.expander("Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            branches = st.multiselect(
                "Branch",
                options=br_df["branch_id"].tolist(),
                default=br_df["branch_id"].tolist(),
                format_func=lambda b: br_df.set_index("branch_id").loc[b, "branch_name"],
            )
        with col2:
            sources = st.multiselect(
                "Detection source",
                options=["cycle_count", "picking", "stocking", "branch_escape"],
                default=["cycle_count", "picking", "stocking", "branch_escape"],
            )

    filtered = enriched[enriched["branch_id"].isin(branches) &
                        enriched["detection_source"].isin(sources)]

    # Charts row
    st.markdown("#### Where events are stuck")
    funnel_df = funnel_counts(filtered)
    hist_df = loop_time_histogram(filtered)
    col_f, col_h = st.columns(2, gap="medium")
    with col_f:
        st.plotly_chart(_funnel_chart(funnel_df), use_container_width=True)
    with col_h:
        st.plotly_chart(_histogram(hist_df), use_container_width=True)

    st.plotly_chart(_trend(loop_time_trend(filtered)), use_container_width=True)

    # Detail table
    st.markdown("#### Open events + recently resolved (last 30 days)")
    detail = detail_table(filtered, inv_df, br_df)
    detail["stage_bar"] = detail["stage_history"].apply(
        lambda h: _stage_bar(stage_timing_for_event(h))
    )
    detail["loop_label"] = detail["loop_days"].apply(_fmt_days)
    detail["status_label"] = detail.apply(
        lambda r: ("🟠 " + r["current_status"]) if r["is_open"] else "🟢 resolved",
        axis=1,
    )
    detail["$ exposure"] = detail["abs_variance_cost"].apply(_fmt_money)

    show_cols = [
        "status_label", "detection_date", "branch_name", "sku_id",
        "description", "category", "detection_source", "loop_label",
        "$ exposure", "stage_bar", "assigned_to",
    ]
    display = detail[show_cols].rename(columns={
        "status_label": "Status",
        "detection_date": "Detected",
        "branch_name": "Branch",
        "sku_id": "SKU",
        "description": "Description",
        "category": "Category",
        "detection_source": "Source",
        "loop_label": "Loop time",
        "stage_bar": "Stage timing",
        "assigned_to": "Owner",
    })
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=520,
    )

    st.caption(
        "Stage timing legend: D detected · I investigating · P physical count · "
        "S system correction · R resolved · […] in progress"
    )
