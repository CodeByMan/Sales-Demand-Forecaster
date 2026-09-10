"""Offline training + evaluation pipeline.

Run once to:
  1. generate the bundled sample dataset (data/sample_sales.csv),
  2. train every model and back-test it on a hold-out year,
  3. save the trained LSTM + scaler and a metrics report into models/.

    python train.py
"""
from __future__ import annotations

import json
import os

import joblib
import pandas as pd

from src.data import generate_sample_sales, load_series
from src.models import compare_models, lstm_forecast

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "sample_sales.csv")
MODELS_DIR = os.path.join(HERE, "models")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("→ Generating realistic sample sales data ...")
    df = generate_sample_sales()
    df.to_csv(DATA, index=False)
    series = load_series(df, "date", "sales")
    print(f"  {len(series)} months, {series.index.min():%Y-%m} → {series.index.max():%Y-%m}")

    print("\n→ Training & back-testing models (hold-out: last 12 months) ...")
    results = compare_models(series, horizon=6)

    rows = []
    for r in results:
        m = r["metrics"]
        flag = "  (FAILED: %s)" % r["error"] if r.get("error") else ""
        print(f"  {r['name']:<14}  MAPE {m['MAPE']:>6}%   RMSE {m['RMSE']:>9}{flag}")
        rows.append({"model": r["name"], **m})

    best = min((r for r in rows if r["MAPE"] == r["MAPE"]), key=lambda x: x["MAPE"])
    print(f"\n  Best on back-test: {best['model']} (MAPE {best['MAPE']}%)")

    print("\n→ Saving trained LSTM + scaler ...")
    _, model, scaler = lstm_forecast(series, horizon=6)
    model.save(os.path.join(MODELS_DIR, "lstm.keras"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"results": rows, "best": best}, f, indent=2)

    print("\n✓ Done. Artifacts written to models/.")


if __name__ == "__main__":
    main()
