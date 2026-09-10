import numpy as np

from src.data import generate_sample_sales, load_series
from src.models import holt_winters_forecast, run_model, seasonal_naive_forecast


def test_holt_winters_returns_requested_horizon():
    series = load_series(generate_sample_sales(), "date", "sales")

    forecast = holt_winters_forecast(series, horizon=3)

    assert len(forecast) == 3
    assert np.isfinite(forecast).all()
    assert (forecast >= 0).all()


def test_run_model_produces_backtest_and_future_series():
    series = load_series(generate_sample_sales(), "date", "sales")

    result = run_model("Holt-Winters", series, horizon=3)

    assert len(result["backtest"]) == 12
    assert len(result["future"]) == 3
    assert result["test_size"] == 12
    assert np.isfinite(result["future"]).all()


def test_seasonal_naive_repeats_last_year_without_training():
    series = load_series(generate_sample_sales(), "date", "sales")

    forecast = seasonal_naive_forecast(series, horizon=15)

    assert len(forecast) == 15
    assert np.array_equal(forecast[:12], series.iloc[-12:].to_numpy())
    assert np.array_equal(forecast[12:], series.iloc[-12:-9].to_numpy())
