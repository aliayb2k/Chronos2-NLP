import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from chronos import Chronos2Pipeline


def to_list_if_array(x):
    """Convert numpy arrays to Python lists (needed before explode)."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", type=str)
    parser.add_argument("--dataset_config", default="ETT_15T", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--horizon", default=96, type=int)
    parser.add_argument("--value_col", default="OT", type=str, help="Which column to forecast (ETT: OT)")
    args = parser.parse_args()

    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Load Chronos-2 pretrained pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=args.device,
    )

    # 2) Load dataset
    ds = load_dataset("autogluon/fev_datasets", args.dataset_config)
    df_raw = ds["train"].to_pandas()

    # Sanity: show columns once
    print("Raw columns:", df_raw.columns.tolist())

    # 3) Build long-format dataframe: id, timestamp, target
    if args.value_col not in df_raw.columns:
        raise ValueError(
            f"value_col='{args.value_col}' not found. Available columns: {df_raw.columns.tolist()}"
        )

    df = df_raw[["id", "timestamp", args.value_col]].copy()
    df = df.rename(columns={args.value_col: "target"})

    # Convert numpy arrays -> lists so explode works
    df["timestamp"] = df["timestamp"].apply(to_list_if_array)
    df["target"] = df["target"].apply(to_list_if_array)

    # Explode arrays so each row is one timestamp
    df = df.explode(["timestamp", "target"], ignore_index=True)

    # Types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).sort_values(["id", "timestamp"])

    # 4) Choose one series id
    series_id = df["id"].iloc[0]
    series_df = df[df["id"] == series_id].copy()

    # Split into context and ground-truth for the last horizon points
    if len(series_df) <= args.horizon + 10:
        raise ValueError(f"Series too short: len={len(series_df)} horizon={args.horizon}")

    context_df = series_df.iloc[:-args.horizon].copy()
    ground_truth = series_df.iloc[-args.horizon:].copy()

    # 5) Forecast using predict_df (same as notebook)
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=args.horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # Join with GT and compute metrics on median forecast
    comparison = ground_truth.copy()
    comparison["pred_median"] = pred_df["0.5"].values

    mae = (comparison["target"] - comparison["pred_median"]).abs().mean()
    mse = ((comparison["target"] - comparison["pred_median"]) ** 2).mean()

    print(f"MAE={mae:.6f} MSE={mse:.6f} (series_id={series_id}, value_col={args.value_col})")

    # 6) Plot GT vs prediction
    plt.figure(figsize=(10, 4))
    plt.plot(comparison["timestamp"], comparison["target"], label="Ground Truth", marker="o")
    plt.plot(comparison["timestamp"], comparison["pred_median"], label="Prediction (q=0.5)", marker="x")
    plt.xticks(rotation=30)
    plt.title(f"Chronos-2 baseline ({args.dataset_config}, id={series_id}, col={args.value_col})")
    plt.legend()

    outpath = os.path.join(fig_dir, "chronos2_baseline.png")
    print("Saving to:", os.path.abspath(outpath))
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {outpath}")

    # Optional: save the comparison table
    csv_path = os.path.join(args.results_dir, "tables")
    os.makedirs(csv_path, exist_ok=True)
    comparison_out = os.path.join(csv_path, "chronos2_baseline_comparison.csv")

    comparison.to_csv(comparison_out, index=False)
    print(f"Saved table: {comparison_out}")


if __name__ == "__main__":
    main()