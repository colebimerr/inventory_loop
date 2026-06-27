"""View 3 — ROI Calculator."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from logic.roi_calc import ROIInputs, compute, defaults_from_data


def _money(v: float) -> str:
    return f"${v:,.0f}"


def render(inv_df: pd.DataFrame, var_df: pd.DataFrame, br_df: pd.DataFrame) -> None:
    st.subheader("ROI Calculator")
    st.caption(
        "Translate the time you save on the variance loop into dollars. "
        "Defaults are computed from the loaded dataset — override any of "
        "them to match your real operation."
    )

    defaults = defaults_from_data(inv_df, var_df, br_df)

    left, right = st.columns([1, 1.2], gap="large")

    # ---------------- Inputs ----------------
    with left:
        st.markdown("#### Your operation")
        inv_value = st.number_input(
            "Annual inventory value ($)",
            min_value=0.0, value=float(defaults["inventory_value"]),
            step=100_000.0, format="%.0f",
        )
        branches = st.number_input(
            "Number of branches",
            min_value=1, value=int(defaults["branches"]), step=1,
        )
        events_per_month = st.number_input(
            "Avg variance events per month",
            min_value=0.0, value=float(defaults["events_per_month"]), step=10.0,
        )
        current_loop_days = st.number_input(
            "Current avg loop time (days, detection → resolution)",
            min_value=0.1, value=float(defaults["current_loop_days"]), step=0.1,
        )

        st.markdown("#### Cost assumptions")
        labor_rate = st.number_input(
            "Fully-loaded labor cost ($/hr)",
            min_value=0.0, value=float(defaults["labor_rate"]), step=5.0,
        )
        shrink_rate = st.number_input(
            "Industry shrink rate (%)",
            min_value=0.0, max_value=10.0,
            value=float(defaults["shrink_rate_pct"]), step=0.1,
        )

        st.markdown("#### With InventoryLoop")
        target_loop_hours = st.number_input(
            "Target loop time (hours)",
            min_value=0.5, value=float(defaults["target_loop_hours"]), step=0.5,
        )
        shrink_reduction = st.slider(
            "Assumed shrink reduction with tool (%)",
            min_value=0, max_value=70,
            value=int(defaults["shrink_reduction_pct"]),
        )
        annual_price = st.number_input(
            "Tool price ($/yr)",
            min_value=0.0, value=float(defaults["annual_price"]), step=5_000.0,
        )

    inputs = ROIInputs(
        inventory_value=inv_value,
        branches=int(branches),
        events_per_month=events_per_month,
        current_loop_days=current_loop_days,
        labor_rate=labor_rate,
        shrink_rate_pct=shrink_rate,
        target_loop_hours=target_loop_hours,
        shrink_reduction_pct=float(shrink_reduction),
        annual_price=annual_price,
    )
    result = compute(inputs)

    # ---------------- Comparison ----------------
    with right:
        st.markdown("#### Current state vs. with InventoryLoop")
        col_a, col_b = st.columns(2, gap="medium")

        def _state_card(col, title, shrink, labor, total, color):
            with col:
                st.markdown(
                    f"<div style='padding:1rem;border:1px solid {color};"
                    f"border-radius:12px;background:#fff'>"
                    f"<div style='font-size:0.9rem;color:#64748B'>{title}</div>"
                    f"<div style='font-size:1.6rem;font-weight:700;color:{color};"
                    f"margin:0.25rem 0'>"
                    f"{_money(total)}<span style='font-size:0.9rem;color:#64748B;"
                    f"font-weight:400'> / yr</span></div>"
                    f"<div style='font-size:0.85rem;color:#475569;line-height:1.6'>"
                    f"Shrink: <b>{_money(shrink)}</b><br>"
                    f"Loop labor: <b>{_money(labor)}</b>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

        _state_card(col_a, "Current state",
                    result.current_shrink, result.current_labor,
                    result.current_total, "#94A3B8")
        _state_card(col_b, "With InventoryLoop",
                    result.future_shrink, result.future_labor,
                    result.future_total, "#10B981")

        st.markdown("")
        savings_col, payback_col = st.columns(2, gap="medium")
        with savings_col:
            st.markdown(
                "<div style='padding:1.25rem;border-radius:12px;"
                "background:#ECFDF5;border:1px solid #10B981'>"
                "<div style='font-size:0.9rem;color:#065F46'>Annual savings</div>"
                f"<div style='font-size:2.2rem;font-weight:800;color:#065F46'>"
                f"{_money(result.annual_savings)}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with payback_col:
            payback = ("n/a" if math.isinf(result.payback_months)
                       else f"{result.payback_months:.1f} months")
            st.markdown(
                "<div style='padding:1.25rem;border-radius:12px;"
                "background:#EFF6FF;border:1px solid #0F5BD9'>"
                f"<div style='font-size:0.9rem;color:#1E3A8A'>"
                f"Payback at {_money(annual_price)}/yr</div>"
                f"<div style='font-size:2.2rem;font-weight:800;color:#1E3A8A'>"
                f"{payback}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.caption(
            "Built only on numbers you can defend: variance-loop labor saved and "
            "shrink reduction. We deliberately leave out a lost-sales dollar figure — "
            "a stockout isn't a reliable lost sale (backorders, substitutions, partial "
            "fills, cross-branch pulls), and that data isn't capturable. Loop labor "
            "assumes 2 active hours of work per day a loop stays open; real savings "
            "depend on operational adoption."
        )
