"""View 4 — Chase or Write-Off.

Makes the everyday chase-vs-write-off call explicit: for each open variance,
should the operator spend the time to chase it, or write it off and move on?
The real lever is capacity — you can't chase everything, so where do your hours
earn the most?
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from logic.chase_decision import DEFAULTS, score_chase, summarize


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}"


def _now_ref(detection: pd.Series) -> pd.Timestamp:
    """Anchor 'now' to the dataset so the demo reads sensibly regardless of the
    real calendar date."""
    return detection.max() if detection.notna().any() else pd.Timestamp.now()


def render(inv_df: pd.DataFrame, var_df: pd.DataFrame, br_df: pd.DataFrame) -> None:
    st.subheader("Chase or Write-Off")
    st.caption(
        "A variance landed — chase it or write it off and move on? Most variances are "
        "worth chasing in isolation. The real limit is hours: variances cluster, and "
        "chasing the whole queue is more labor than anyone has. This puts your "
        "available time where it earns the most."
    )

    df = var_df.copy()
    df["detection_date"] = pd.to_datetime(df["detection_date"], format="ISO8601",
                                          errors="coerce")
    open_df = df[df["current_status"] != "resolved"]

    scope = st.radio(
        "Show",
        ["Open variances", "All recent variances"],
        horizontal=True,
        help="Open variances are the calls sitting on your desk right now.",
    )
    if scope == "Open variances" and len(open_df) >= 5:
        events = open_df
    else:
        now_ref = _now_ref(df["detection_date"])
        events = df[df["detection_date"] >= now_ref - pd.Timedelta(days=30)]
        if scope == "Open variances":
            st.info("Not enough open variances here — showing all recent variances so "
                    "the model has something to rank.")

    # ---- Assumptions (configurable — different shops value time differently) --
    with st.expander("Assumptions — tune these to your shop", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            labor_budget = st.slider(
                "Investigation hours you have this week",
                4.0, 200.0, DEFAULTS["labor_budget_hours"], step=4.0,
                help="The real constraint. Drag it down and watch the queue triage "
                     "itself to what your hours can actually cover.",
            )
            hourly_rate = st.slider(
                "Loaded hourly rate of the investigator ($)",
                15.0, 90.0, DEFAULTS["hourly_rate"], step=1.0,
            )
        with c2:
            recovery_prob = st.slider(
                "Share of chases that actually recover value",
                0.10, 0.95, DEFAULTS["recovery_prob"], step=0.05,
            )
            stockout_value = st.slider(
                "Value of avoiding a stockout on a top-velocity item ($)",
                0.0, 250.0, DEFAULTS["stockout_value"], step=10.0,
                help="Item class (A/B/C) scales this — an A item being off risks the "
                     "next sale, not just its dollars.",
            )
        with c3:
            hours_quick = st.slider(
                "Hours for a quick case (offset, easy backtrack)",
                0.1, 2.0, DEFAULTS["hours_quick"], step=0.1,
            )
            hours_deep = st.slider(
                "Hours for a deep case (no offset, full trace)",
                0.5, 6.0, DEFAULTS["hours_deep"], step=0.5,
            )
            deep_threshold = st.slider(
                "$ variance that triggers a deep investigation",
                50.0, 1000.0, DEFAULTS["deep_threshold"], step=25.0,
            )

    params = {
        "labor_budget_hours": labor_budget,
        "hourly_rate": hourly_rate,
        "recovery_prob": recovery_prob,
        "stockout_value": stockout_value,
        "hours_quick": hours_quick,
        "hours_deep": hours_deep,
        "deep_threshold": deep_threshold,
    }

    scored = score_chase(events, inv_df, params)
    s = summarize(scored, params)
    total = s["chase_n"] + s["writeoff_n"]

    # ---- The capacity headline --------------------------------------------
    if s["total_hours_if_all"] > s["budget"]:
        st.warning(
            f"Chasing all **{total:,}** variances would take "
            f"**{s['total_hours_if_all']:.0f} hours**. You have **{s['budget']:.0f}**. "
            f"Below is where to spend them."
        )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Chase (fits your hours)", f"{s['chase_n']:,}",
                  delta=f"of {total:,}", delta_color="off")
    with k2:
        st.metric("Value you'd capture", _fmt_money(s["chase_net"]))
    with k3:
        st.metric("Hours used", f"{s['chase_hours']:.0f}h",
                  delta=f"of {s['budget']:.0f}h budget", delta_color="off")
    with k4:
        st.metric("Write off this week", f"{s['writeoff_n']:,}",
                  delta=_fmt_money(s["writeoff_exposure"]) + " exposure",
                  delta_color="off",
                  help="Either it loses money to chase, or you're out of capacity. "
                       "Flagged for adjust-back if they resurface.")

    # ---- The ranked table --------------------------------------------------
    st.markdown("#### Every variance, ranked by what your time earns")

    table = scored.copy()
    table["branch"] = table["branch_id"]
    if "branch_id" in br_df.columns and "branch_name" in br_df.columns:
        name_map = br_df.set_index("branch_id")["branch_name"].to_dict()
        table["branch"] = table["branch_id"].map(name_map).fillna(table["branch_id"])

    table["Rec"] = table["recommendation"].map(
        {"Chase": "🟢 Chase", "Write off": "⚪ Write off"})
    table["$ at stake"] = table["abs_cost"].map(_fmt_money)
    table["Est. time"] = table["est_hours"].map(lambda h: f"{h:.1f}h")
    table["Value/hr"] = table["value_per_hour"].map(
        lambda v: f"{'+' if v >= 0 else '-'}${abs(v):,.0f}")
    table["Net"] = table["net_benefit"].map(
        lambda v: f"{'+' if v >= 0 else '-'}${abs(v):,.0f}")

    show = table[[
        "Rec", "sku_id", "description", "velocity_class", "branch",
        "$ at stake", "Est. time", "Value/hr", "Net",
    ]].rename(columns={
        "sku_id": "SKU",
        "description": "Description",
        "velocity_class": "Class",
        "branch": "Branch",
    })

    st.dataframe(show, width="stretch", hide_index=True, height=520)

    st.caption(
        "Ranked by value-per-hour: net benefit (recoverable $ + stockout-avoidance "
        "value, scaled by item class, minus labor cost) divided by hours to chase. "
        "Your weekly hours fill from the top down — the line where they run out is "
        "the chase/write-off cut. This decides *whether* and *which* to chase; finding "
        "*why* a variance happened is attribution, the next module, built on real "
        "transaction data."
    )
