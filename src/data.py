"""Data utilities: generate realistic sample sales and parse user-uploaded CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when an uploaded sales dataset cannot be forecast safely."""


def generate_sample_sales(
    n_months: int = 60,
    start: str = "2020-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Create a realistic monthly sales series for a small retail business.

    The series intentionally contains the three signals a forecaster must learn:
    a rising trend (growing business), a 12-month seasonal cycle (holiday peaks),
    and random noise. This is the demo data shipped with the app so a recruiter
    can click "Run" without uploading anything.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    t = np.arange(n_months)

    base = 10_000                      # starting monthly revenue
    trend = 180 * t                    # steady growth
    seasonal = 2_200 * np.sin(2 * np.pi * (t % 12) / 12 - np.pi / 2)  # yearly cycle
    holiday = np.where((t % 12) == 11, 3_500, 0)  # December spike
    noise = rng.normal(0, 700, n_months)

    sales = base + trend + seasonal + holiday + noise
    sales = np.clip(sales, 500, None).round(0)

    return pd.DataFrame({"date": dates, "sales": sales})


def load_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """Turn a two-column dataframe into a clean, monthly, sorted sales series."""
    if df is None or df.empty:
        raise DataValidationError("The uploaded CSV is empty.")
    if date_col not in df.columns or value_col not in df.columns:
        raise DataValidationError("Choose one date column and one sales column.")
    if date_col == value_col:
        raise DataValidationError("The date and sales columns must be different.")

    raw = df[[date_col, value_col]].copy()
    dates = pd.to_datetime(raw[date_col], errors="coerce")
    values = pd.to_numeric(raw[value_col], errors="coerce")
    valid = dates.notna() & values.notna() & np.isfinite(values)
    dropped_rows = int((~valid).sum())
    if not valid.any():
        raise DataValidationError("No valid date and sales rows were found.")
    if (values[valid] < 0).any():
        raise DataValidationError("Sales values must be non-negative.")

    clean = pd.DataFrame({
        "date": dates[valid].dt.to_period("M").dt.to_timestamp(),
        "sales": values[valid].astype(float),
    })
    monthly = clean.groupby("date")["sales"].sum(min_count=1).sort_index().asfreq("MS")
    interpolated_months = int(monthly.isna().sum())
    if interpolated_months:
        monthly = monthly.interpolate(method="linear", limit_area="inside")
    if monthly.isna().any():
        raise DataValidationError("The sales history contains gaps that cannot be filled safely.")

    monthly.name = "sales"
    monthly.index.name = "date"
    monthly.attrs["dropped_rows"] = dropped_rows
    monthly.attrs["interpolated_months"] = interpolated_months
    return monthly


def guess_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Best-effort auto-detection of the date and value columns in an upload."""
    if df is None or len(df.columns) < 2:
        raise DataValidationError("The CSV must contain at least two columns.")

    date_col, value_col = None, None
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.7:
                date_col = c
                break
        except Exception:
            continue
    for c in df.columns:
        if c == date_col:
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.7:
            value_col = c
            break
    return date_col or df.columns[0], value_col or df.columns[-1]
