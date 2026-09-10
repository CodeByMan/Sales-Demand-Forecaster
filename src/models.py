"""Forecasting models.

Four approaches are implemented on purpose so the app can *compare* them:

  - LSTM            : a recurrent neural net (deep learning) — the headline model.
  - Holt-Winters    : classical triple exponential smoothing (trend + seasonality).
  - SARIMA          : seasonal ARIMA, the statistical workhorse for monthly data.
  - Seasonal Naive  : a transparent seasonal baseline that repeats the last year.

The point of the comparison is to demonstrate model selection judgement: on short
monthly business series the classical baselines are often competitive with — or
better than — an LSTM, and a good forecaster reports that honestly instead of
assuming "deep learning = best".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

from .utils import evaluate  # noqa: E402

SEASONAL_PERIODS = 12


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
def _make_windows(values: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i : i + look_back])
        y.append(values[i + look_back])
    return np.array(X), np.array(y)


def _build_lstm(look_back: int):
    # Imported lazily so the classical models work even if TF is unavailable.
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=(look_back, 1)),
            LSTM(64, activation="tanh", return_sequences=True),
            Dropout(0.15),
            LSTM(32, activation="tanh"),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def lstm_forecast(series: pd.Series, horizon: int, look_back: int = 12, epochs: int = 120, seed: int = 42):
    """Fit an LSTM and forecast `horizon` future months (recursive multi-step)."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if len(series) < 6 or not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("series must contain at least 6 finite observations")

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    look_back = min(look_back, max(3, len(series) // 2))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    X, y = _make_windows(scaled, look_back)
    X = X.reshape(-1, look_back, 1)

    model = _build_lstm(look_back)
    fit_kwargs = {
        "epochs": epochs,
        "batch_size": 8,
        "verbose": 0,
        "shuffle": False,
    }
    if len(X) >= 10:
        fit_kwargs["validation_split"] = 0.2
        monitor = "val_loss"
    else:
        monitor = "loss"
    fit_kwargs["callbacks"] = [
        EarlyStopping(monitor=monitor, patience=15, restore_best_weights=True)
    ]
    model.fit(X, y, **fit_kwargs)

    window = scaled[-look_back:].tolist()
    preds = []
    for _ in range(horizon):
        x = np.array(window[-look_back:]).reshape(1, look_back, 1)
        nxt = float(model.predict(x, verbose=0)[0, 0])
        preds.append(nxt)
        window.append(nxt)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return np.clip(preds, 0, None), model, scaler


# --------------------------------------------------------------------------- #
# Classical baselines
# --------------------------------------------------------------------------- #
def holt_winters_forecast(series: pd.Series, horizon: int):
    seasonal = "add" if len(series) >= 2 * SEASONAL_PERIODS else None
    fit = ExponentialSmoothing(
        series, trend="add", seasonal=seasonal,
        seasonal_periods=SEASONAL_PERIODS if seasonal else None,
        initialization_method="estimated",
    ).fit()
    return np.clip(fit.forecast(horizon).values, 0, None)


def seasonal_naive_forecast(series: pd.Series, horizon: int):
    """Repeat the most recent seasonal cycle as a transparent baseline."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    values = np.asarray(series, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("series must contain finite observations")
    cycle = values[-SEASONAL_PERIODS:] if len(values) >= SEASONAL_PERIODS else values[-1:]
    return np.resize(cycle, horizon).astype(float)


def sarima_forecast(series: pd.Series, horizon: int):
    seasonal_order = (1, 1, 1, SEASONAL_PERIODS) if len(series) >= 2 * SEASONAL_PERIODS else (0, 0, 0, 0)
    fit = SARIMAX(
        series, order=(1, 1, 1), seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    return np.clip(fit.forecast(horizon).values, 0, None)


# --------------------------------------------------------------------------- #
# Unified runner: backtest (for accuracy) + future forecast (for the chart)
# --------------------------------------------------------------------------- #
MODELS = {
    "LSTM": lstm_forecast,
    "Holt-Winters": holt_winters_forecast,
    "SARIMA": sarima_forecast,
    "Seasonal Naive": seasonal_naive_forecast,
}


def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:]


def run_model(name: str, series: pd.Series, horizon: int, test_size: int = 12) -> dict:
    """Backtest a model on a hold-out tail, then forecast the real future."""
    test_size = min(test_size, max(3, len(series) // 4))
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]

    def _call(fn, s, h):
        out = fn(s, h)
        return out[0] if isinstance(out, tuple) else out

    fn = MODELS[name]
    backtest_pred = _call(fn, train, len(test))
    metrics = evaluate(test.values, backtest_pred)

    future_pred = _call(fn, series, horizon)
    future = pd.Series(future_pred, index=_future_index(series, horizon), name=name)

    return {"name": name, "metrics": metrics, "backtest": pd.Series(backtest_pred, index=test.index),
            "future": future, "test_size": test_size}


def compare_models(series: pd.Series, horizon: int, names=None, test_size: int = 12) -> list[dict]:
    names = names or list(MODELS.keys())
    results = []
    for n in names:
        try:
            results.append(run_model(n, series, horizon, test_size))
        except Exception as e:  # a model failing should never crash the app
            results.append({"name": n, "metrics": {"MAPE": float("nan"), "RMSE": float("nan")},
                            "backtest": None, "future": None, "error": str(e),
                            "error_type": type(e).__name__})
    return results
