"""View 1 — Prioritized Count List."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from logic.risk_scoring import score_skus


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}"


def render(inv_df: pd.DataFrame, var_df: pd.DataFrame, br_df: pd.DataFrame) -> None:
    st.subheader("Prioritized Count List")
    st.caption(
        "Where the warehouse should start their day. Ranked by a composite "
        "risk score: days since last count, recent variance churn, velocity "
        "class, branch involvement, and unit cost exposure."
    )

    if "dismissed_skus" not in st.session_state:
        st.session_state["dismissed_skus"] = set()

    scored = score_skus(inv_df, var_df, br_df)

    # Filters
    branch_options = br_df["branch_name"].tolist() + ["All branches"]
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        branches = st.multiselect(
            "Branch (where most recent variance occurred)",
            options=branch_options,
            default=branch_options,
            key="cl_branches",
        )
    with col2:
        categories = st.multiselect(
            "Category",
            options=sorted(inv_df["category"].unique()),
            default=sorted(inv_df["category"].unique()),
            key="cl_categories",
        )
    with col3:
        n_show = st.number_input("Top N", min_value=10, max_value=100, value=25, step=5,
                                  key="cl_topn")

    filt = scored[
        scored["category"].isin(categories)
        & scored["branch_name"].isin(branches)
        & ~scored["sku_id"].isin(st.session_state["dismissed_skus"])
    ].sort_values("risk_score", ascending=False).head(int(n_show))

    # KPI cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("SKUs flagged today", f"{len(filt):,}")
    with c2:
        st.metric("Estimated $ exposure", _fmt_money(filt["estimated_exposure"].sum()))
    with c3:
        st.metric(
            "Avg days since last count (flagged)",
            f"{filt['days_since_count'].mean():.0f}" if len(filt) else "—",
        )

    if not len(filt):
        st.info("Nothing left to count under the current filters. ")
        if st.button("Reset dismissed SKUs"):
            st.session_state["dismissed_skus"] = set()
            st.rerun()
        return

    st.markdown("#### Today's prioritized list")
    # Render row-by-row so each row has its own "Mark counted" button.
    header = st.columns([0.8, 2.5, 1.5, 1.2, 1.0, 1.0, 1.2, 1.0, 1.0])
    headers = ["Rank", "SKU / description", "Branch", "Category", "Velocity",
               "Days since count", "Recent variance (30d)", "Risk score", ""]
    for col, h in zip(header, headers):
        col.markdown(f"**{h}**")

    for rank, (_, row) in enumerate(filt.iterrows(), start=1):
        cols = st.columns([0.8, 2.5, 1.5, 1.2, 1.0, 1.0, 1.2, 1.0, 1.0])
        cols[0].write(f"{rank}")
        cols[1].markdown(f"**{row['sku_id']}**  \n{row['description']}")
        cols[2].write(row["branch_name"])
        cols[3].write(row["category"])
        cols[4].write(row["velocity_class"])
        cols[5].write(f"{int(row['days_since_count'])}d")
        cols[6].write(int(row["recent_variance_count"]))
        cols[7].markdown(f"**{row['risk_score']:.0f}**")
        if cols[8].button("✓ Counted", key=f"mark_{row['sku_id']}"):
            st.session_state["dismissed_skus"].add(row["sku_id"])
            st.rerun()

    st.caption(
        "Marking a SKU as counted clears it from this list for the rest of "
        "this session. (No persistence — refresh resets.)"
    )

    with st.expander("How is the risk score calculated?"):
        st.markdown(
            "Weighted sum of normalized factors, each on [0,1], scaled to 100:\n"
            "- 35% cadence-overdue: days since count ÷ the item's expected count "
            "cadence by class (A ~4×/yr, B ~3×, C ~2×, D ~1×) — an A item past its "
            "tighter cadence outranks a C item with more raw days\n"
            "- 30% recent variance count (last 30 days, capped at 5)\n"
            "- 15% velocity class (A=1.0, B=0.6, C=0.3)\n"
            "- 10% recent variance at a problem branch (problem = >1.5× median branch count)\n"
            "- 10% unit cost (proxy for $ exposure per discrepancy)"
        )
