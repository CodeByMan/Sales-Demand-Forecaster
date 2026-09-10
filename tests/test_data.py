import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError, generate_sample_sales, guess_columns, load_series


def test_sample_data_is_monthly_and_complete():
    series = load_series(generate_sample_sales(), "date", "sales")

    assert len(series) == 60
    assert not series.isna().any()
    assert series.index.freqstr == "MS"


def test_missing_month_is_interpolated_after_aggregation():
    frame = pd.DataFrame({"date": ["2024-01-01", "2024-03-01"], "sales": [100, 300]})

    series = load_series(frame, "date", "sales")

    assert series.loc["2024-02-01"] == 200
    assert series.attrs["interpolated_months"] == 1


def test_invalid_rows_are_reported():
    frame = pd.DataFrame({"date": ["bad", "2024-02-01"], "sales": ["bad", 200]})

    series = load_series(frame, "date", "sales")

    assert series.iloc[0] == 200
    assert series.attrs["dropped_rows"] == 1


def test_invalid_dataset_is_rejected():
    with pytest.raises(DataValidationError):
        guess_columns(pd.DataFrame())

    with pytest.raises(DataValidationError):
        load_series(pd.DataFrame({"date": ["2024-01-01"], "sales": [-1]}), "date", "sales")


def test_numeric_sales_column_is_not_guessed_as_date():
    frame = pd.DataFrame({"sales": [100, 200], "date": ["2024-01-01", "2024-02-01"]})

    assert guess_columns(frame) == ("date", "sales")
