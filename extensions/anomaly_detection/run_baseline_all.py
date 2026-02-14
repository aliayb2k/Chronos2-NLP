import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chronos import Chronos2Pipeline
from transformers import pipeline
from chronos2_nlp.data.datasets import load_domain_series


def forecast_last_window(pipeline, values: np.ndarray, context: int, horizon: int):
    context_arr = values[-(context + horizon):-horizon].astype(np.float32)
    true_future = values[-horizon:].astype(np.float32)

    # Chronos-2 expects (n_series, n_variates, history_length)
    context_in = context_arr[None, None, :]  # (1, 1, L)

    forecast = pipeline.predict(context_in, prediction_length=horizon)

    pred = np.array(forecast)
    pred = np.squeeze(pred)  # remove singleton dims if present
    print("DEBUG pred shape:", pred.shape)

    if pred.ndim == 2:
        pred_mean = pred.mean(axis=0)
        pred_median = np.median(pred, axis=0)
    elif pred.ndim == 1:
        pred_mean = pred
        pred_median = pred
    else:
        raise ValueError(f"Unexpected forecast shape: {pred.shape}")

    return context_arr, true_future, pred_mean, pred_median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--context", default=512, type=int)
    parser.add_argument("--horizon", default=16, type=int)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "tables"), exist_ok=True)

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)

    rows = []
    for domain in ["finance_spy", "energy_solar_1D"]:
        sd = load_domain_series(domain)
        df = sd.df
        values = df["value"].astype(float).values
        ts = df["timestamp"].values

        # adapt context if the series is shorter than requested
        min_needed = args.horizon + 32  # keep at least some history
        if len(values) < min_needed:
            raise ValueError(f"{domain} too short even for minimal run: T={len(values)} < {min_needed}")

        ctx_len = min(args.context, len(values) - args.horizon)  # max possible without leakage
        # optional: cap to something reasonable so solar isn't using 349 which is weird
        if domain.startswith("energy_"):
            ctx_len = min(ctx_len, 256)

        ctx, y_true, y_mean, y_med = forecast_last_window(
            pipeline, values, context=ctx_len, horizon=args.horizon
        )

        ctx, y_true, y_mean, y_med = forecast_last_window(
            pipeline, values, context=args.context, horizon=args.horizon
        )

        mae = float(np.mean(np.abs(y_true - y_med)))  # median is robust
        mse = float(np.mean((y_true - y_med) ** 2))

        rows.append({"domain": domain, "context": ctx_len, "horizon": args.horizon, "mae": mae, "mse": mse})
        print(f"[{domain}] MAE={mae:.6f} MSE={mse:.6f}")

        # Plot only the tail (context + horizon)
        tail_values = values[-(ctx_len + args.horizon):]
        tail_ts = ts[-(ctx_len + args.horizon):]
        fut_ts = tail_ts[-args.horizon:]

        plt.figure(figsize=(10, 4))
        plt.plot(tail_ts, tail_values, label="true")
        plt.plot(fut_ts, y_med, label="forecast(median)")
        plt.axvline(fut_ts[0], linestyle="--")
        plt.title(f"Chronos-2 baseline - {domain}")
        plt.legend()
        plt.xticks(rotation=20)

        out_fig = os.path.join(args.results_dir, "figures", f"baseline_{domain}.png")
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out_fig)

    out_csv = os.path.join(args.results_dir, "tables", "baseline_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()