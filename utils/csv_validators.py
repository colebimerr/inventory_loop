"""Schema validation for the three CSVs InventoryLoop accepts on upload."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pandas as pd

INVENTORY_REQUIRED = {
    "sku_id": "string",
    "description": "string",
    "category": "string",
    "unit_cost": "float",
    "velocity_class": "string",
    "reorder_point": "int",
    "reorder_qty": "int",
    "last_cycle_count_date": "date",
}

VARIANCE_REQUIRED = {
    "event_id": "string",
    "sku_id": "string",
    "branch_id": "string",
    "detection_source": "string",
    "detection_date": "datetime",
    "resolution_date": "datetime_nullable",
    "expected_qty": "int",
    "actual_qty": "int",
    "variance_qty": "int",
    "variance_cost": "float",
    "stage_history": "json",
    "assigned_to": "string",
    "current_status": "string",
}

BRANCHES_REQUIRED = {
    "branch_id": "string",
    "branch_name": "string",
    "branch_type": "string",
    "city": "string",
    "state": "string",
}

VALID_VELOCITY_CLASSES = {"A", "B", "C"}
VALID_DETECTION_SOURCES = {"cycle_count", "picking", "stocking", "branch_escape"}
VALID_STATUSES = {
    "open", "investigating", "physical_count_complete",
    "system_corrected", "resolved",
}
VALID_BRANCH_TYPES = {"DC", "branch"}


@dataclass
class ValidationResult:
    ok: bool
    rows: int
    errors: list[str]
    df: pd.DataFrame | None = None


def _check_columns(df: pd.DataFrame, required: dict) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    return [f"Missing required column: '{c}'" for c in missing]


def _coerce_types(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    df = df.copy()
    for col, kind in schema.items():
        if col not in df.columns:
            continue
        try:
            if kind == "string":
                df[col] = df[col].astype(str)
            elif kind == "int":
                df[col] = pd.to_numeric(df[col], errors="raise").astype("Int64")
            elif kind == "float":
                df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
            elif kind == "date":
                df[col] = pd.to_datetime(df[col], format="ISO8601", errors="raise").dt.date
            elif kind == "datetime":
                df[col] = pd.to_datetime(df[col], format="ISO8601", errors="raise")
            elif kind == "datetime_nullable":
                df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce")
            elif kind == "json":
                df[col].apply(lambda v: json.loads(v) if pd.notna(v) else [])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Could not coerce column '{col}' to {kind}: {exc}")
    return df, errors


def validate_inventory(file_or_df) -> ValidationResult:
    df = _read(file_or_df)
    if df is None:
        return ValidationResult(False, 0, ["File could not be read as CSV."])
    errs = _check_columns(df, INVENTORY_REQUIRED)
    if errs:
        return ValidationResult(False, len(df), errs)
    df, type_errs = _coerce_types(df, INVENTORY_REQUIRED)
    errs.extend(type_errs)
    bad_vel = df.loc[~df["velocity_class"].isin(VALID_VELOCITY_CLASSES)]
    if len(bad_vel):
        errs.append(f"{len(bad_vel)} rows have velocity_class not in A/B/C")
    if (df["unit_cost"] < 0).any():
        errs.append("unit_cost has negative values")
    if df["sku_id"].duplicated().any():
        errs.append("sku_id is not unique")
    return ValidationResult(len(errs) == 0, len(df), errs, df)


def validate_variance(file_or_df) -> ValidationResult:
    df = _read(file_or_df)
    if df is None:
        return ValidationResult(False, 0, ["File could not be read as CSV."])
    errs = _check_columns(df, VARIANCE_REQUIRED)
    if errs:
        return ValidationResult(False, len(df), errs)
    df, type_errs = _coerce_types(df, VARIANCE_REQUIRED)
    errs.extend(type_errs)
    bad_src = df.loc[~df["detection_source"].isin(VALID_DETECTION_SOURCES)]
    if len(bad_src):
        errs.append(
            f"{len(bad_src)} rows have detection_source not in "
            f"{sorted(VALID_DETECTION_SOURCES)}"
        )
    bad_status = df.loc[~df["current_status"].isin(VALID_STATUSES)]
    if len(bad_status):
        errs.append(
            f"{len(bad_status)} rows have current_status not in {sorted(VALID_STATUSES)}"
        )
    if df["event_id"].duplicated().any():
        errs.append("event_id is not unique")
    return ValidationResult(len(errs) == 0, len(df), errs, df)


def validate_branches(file_or_df) -> ValidationResult:
    df = _read(file_or_df)
    if df is None:
        return ValidationResult(False, 0, ["File could not be read as CSV."])
    errs = _check_columns(df, BRANCHES_REQUIRED)
    if errs:
        return ValidationResult(False, len(df), errs)
    df, type_errs = _coerce_types(df, BRANCHES_REQUIRED)
    errs.extend(type_errs)
    bad_type = df.loc[~df["branch_type"].isin(VALID_BRANCH_TYPES)]
    if len(bad_type):
        errs.append(f"{len(bad_type)} rows have branch_type not in DC/branch")
    if df["branch_id"].duplicated().any():
        errs.append("branch_id is not unique")
    return ValidationResult(len(errs) == 0, len(df), errs, df)


def cross_validate(inv_df: pd.DataFrame, var_df: pd.DataFrame,
                   br_df: pd.DataFrame) -> list[str]:
    """Foreign-key integrity checks across the three files."""
    errs: list[str] = []
    unknown_skus = set(var_df["sku_id"]) - set(inv_df["sku_id"])
    if unknown_skus:
        errs.append(
            f"{len(unknown_skus)} sku_id in variance_events not found in inventory_master"
        )
    unknown_branches = set(var_df["branch_id"]) - set(br_df["branch_id"])
    if unknown_branches:
        errs.append(
            f"{len(unknown_branches)} branch_id in variance_events not found in branches"
        )
    return errs


def _read(file_or_df) -> pd.DataFrame | None:
    if isinstance(file_or_df, pd.DataFrame):
        return file_or_df
    try:
        if hasattr(file_or_df, "read"):
            content = file_or_df.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            return pd.read_csv(io.StringIO(content))
        return pd.read_csv(file_or_df)
    except Exception:  # noqa: BLE001
        return None
