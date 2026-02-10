import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from chronos import Chronos2Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--horizon", default=96, type=int)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.results_dir, "figures"), exist_ok=True)

    # 1) Load Chronos-2 pretrained pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=args.device,
    )

    # 2) Load a benchmark dataset 
    ds = load_dataset("autogluon/fev_datasets", "ETT_1ST")
    df = ds["train"].to_pandas()

    # The dataset contains multiple series; use the first ID and its target
   
    first_id = df["id"].iloc[0]
    series = df[df["id"] == first_id].sort_values("timestamp")["target"].astype(float).values

    # 3) Forecast next horizon from the last context window
    # Chronos pipeline expects a list/array of past values
    context = series[:-args.horizon]
    true_future = series[-args.horizon:]

    forecast = pipeline.predict(
        context,
        prediction_length=args.horizon,
    )

    # forecast can be samples or quantiles depending on pipeline; take mean if samples
    # Safe approach: convert to numpy and average over sample dimension if needed
    pred = np.array(forecast)
    if pred.ndim > 1:
        pred_mean = pred.mean(axis=0)
    else:
        pred_mean = pred

    # 4) Simple metrics
    mae = np.mean(np.abs(true_future - pred_mean))
    mse = np.mean((true_future - pred_mean) ** 2)
    print(f"MAE={mae:.4f} MSE={mse:.4f} (series_id={first_id})")

    # 5) Plot
    plt.figure()
    plt.plot(range(len(series)), series, label="series")
    start = len(series) - args.horizon
    plt.plot(range(start, len(series)), pred_mean, label="forecast")
    plt.title(f"Chronos-2 baseline (id={first_id})")
    plt.legend()

    outpath = os.path.join(args.results_dir, "figures", "chronos2_baseline.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {outpath}")


if name == "__main__":
    main()