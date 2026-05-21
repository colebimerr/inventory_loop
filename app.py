"""InventoryLoop — Streamlit entry point.

Run locally:    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.csv_validators import (
    cross_validate, validate_branches, validate_inventory, validate_variance,
)
from views import count_list, roi, variance_queue

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Inventory Variance Loop Accelerator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("data_loaded", False)
    st.session_state.setdefault("dataset_label", None)
    st.session_state.setdefault("inventory_df", None)
    st.session_state.setdefault("variance_df", None)
    st.session_state.setdefault("branches_df", None)


def _set_data(label: str, inv: pd.DataFrame, var: pd.DataFrame, br: pd.DataFrame) -> None:
    st.session_state["data_loaded"] = True
    st.session_state["dataset_label"] = label
    st.session_state["inventory_df"] = inv
    st.session_state["variance_df"] = var
    st.session_state["branches_df"] = br


def _reset_data() -> None:
    for k in ("data_loaded", "dataset_label", "inventory_df", "variance_df", "branches_df"):
        st.session_state[k] = None
    st.session_state["data_loaded"] = False


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
def _load_demo_data() -> bool:
    paths = {
        "inventory": DATA_DIR / "inventory_master.csv",
        "variance":  DATA_DIR / "variance_events.csv",
        "branches":  DATA_DIR / "branches.csv",
    }
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        st.error(
            "Demo data not found. Run `python generate_data.py` to create "
            f"the missing files: {', '.join(missing)}."
        )
        return False
    inv_res = validate_inventory(paths["inventory"])
    var_res = validate_variance(paths["variance"])
    br_res = validate_branches(paths["branches"])
    if not (inv_res.ok and var_res.ok and br_res.ok):
        st.error("Demo CSVs failed validation:")
        for r in (inv_res, var_res, br_res):
            for e in r.errors:
                st.write(f"• {e}")
        return False
    fk_errs = cross_validate(inv_res.df, var_res.df, br_res.df)
    if fk_errs:
        st.error("Cross-file integrity errors in demo data:")
        for e in fk_errs:
            st.write(f"• {e}")
        return False
    days = _days_span(var_res.df)
    label = (
        f"Demo data — {inv_res.rows:,} SKUs / {br_res.rows} locations / "
        f"{days} days"
    )
    _set_data(label, inv_res.df, var_res.df, br_res.df)
    return True


def _days_span(var_df: pd.DataFrame) -> int:
    if not len(var_df):
        return 0
    return int((var_df["detection_date"].max() - var_df["detection_date"].min()).days)


def render_landing() -> None:
    st.title("Inventory Variance Loop Accelerator")
    st.caption("Shorten your detection-to-resolution loop.")

    col_upload, col_demo = st.columns(2, gap="large")

    with col_upload:
        st.subheader("Upload your data")
        st.write(
            "Drop in your three CSVs. We validate the schema before loading "
            "anything into memory."
        )
        inv_file = st.file_uploader("inventory_master.csv", type="csv", key="up_inv")
        var_file = st.file_uploader("variance_events.csv",  type="csv", key="up_var")
        br_file  = st.file_uploader("branches.csv",         type="csv", key="up_br")

        if inv_file and var_file and br_file:
            inv_res = validate_inventory(inv_file)
            var_res = validate_variance(var_file)
            br_res  = validate_branches(br_file)
            ok_all = inv_res.ok and var_res.ok and br_res.ok

            def _show(name, res):
                if res.ok:
                    st.success(f"✓ {name} — {res.rows:,} rows")
                else:
                    st.error(f"✗ {name} — {res.rows:,} rows")
                    for e in res.errors:
                        st.write(f"  • {e}")

            _show("inventory_master.csv", inv_res)
            _show("variance_events.csv",  var_res)
            _show("branches.csv",         br_res)

            if ok_all:
                fk_errs = cross_validate(inv_res.df, var_res.df, br_res.df)
                if fk_errs:
                    st.error("Cross-file integrity:")
                    for e in fk_errs:
                        st.write(f"  • {e}")
                else:
                    if st.button("Load this dataset", type="primary",
                                 width="stretch"):
                        days = _days_span(var_res.df)
                        label = (
                            f"Your data — {inv_res.rows:,} SKUs / "
                            f"{br_res.rows} locations / {days} days"
                        )
                        _set_data(label, inv_res.df, var_res.df, br_res.df)
                        st.rerun()

    with col_demo:
        st.subheader("Use demo data")
        st.write(
            "Walk through the experience on a realistic mid-market HVAC "
            "distributor profile: 2,500 SKUs, 5 locations, 180 days of "
            "variance history."
        )
        st.write("")
        if st.button("Load demo data", type="primary", width="stretch"):
            if _load_demo_data():
                st.rerun()

        with st.expander("What's inside the demo dataset?"):
            st.markdown(
                "- **2,500 SKUs** across Refrigerant, Copper Tubing, "
                "Compressors, Motors, Controls, Miscellaneous\n"
                "- **5 locations** — 1 DC + 4 branches across the Southeast US\n"
                "- **180 days** of detected variance events\n"
                "- **~1,800 events** with a realistic spread of detection sources, "
                "loop times, and currently-open queue\n"
                "- One branch generates ~2× the variance rate so the branch-level "
                "story shows through"
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Dataset")
        if st.session_state.get("data_loaded"):
            st.success(st.session_state["dataset_label"])
            if st.button("← Switch dataset"):
                _reset_data()
                st.rerun()
        else:
            st.info("No dataset loaded yet.")

        st.markdown("---")
        st.markdown("### About")
        st.caption(
            "Demo prototype. No data is persisted — refresh the browser tab "
            "to reset to the upload screen."
        )


# ---------------------------------------------------------------------------
# Loaded-data shell with the three views
# ---------------------------------------------------------------------------
def render_loaded_shell() -> None:
    inv = st.session_state["inventory_df"]
    var = st.session_state["variance_df"]
    br  = st.session_state["branches_df"]

    tab_queue, tab_count, tab_roi = st.tabs([
        "📋 Variance Queue",
        "🎯 Prioritized Count List",
        "💵 ROI Calculator",
    ])
    with tab_queue:
        variance_queue.render(inv, var, br)
    with tab_count:
        count_list.render(inv, var, br)
    with tab_roi:
        roi.render(inv, var, br)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()
    render_sidebar()
    if st.session_state.get("data_loaded"):
        render_loaded_shell()
    else:
        render_landing()


if __name__ == "__main__":
    main()
