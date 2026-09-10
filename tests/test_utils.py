import numpy as np

from src.utils import evaluate, mape


def test_evaluate_returns_expected_metrics():
    assert evaluate([100, 200], [110, 180]) == {"MAPE": 10.0, "RMSE": 15.8}


def test_mape_excludes_zero_actuals_without_dividing_by_zero():
    value = mape([0, 100], [25, 110])

    assert np.isclose(value, 10.0)
