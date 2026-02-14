import argparse
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from datasets import load_dataset


def ensure_dirs(root: str):
    raw = Path(root) / "raw"
    processed = Path(root) / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return raw, processed


def summarize_series(df: pd.DataFrame, name: str):
    # expects columns: timestamp, value
    df = df.sort_values("timestamp")
    n = len(df)
    miss = df["value"].isna().mean()

    # rough frequency guess
    if n >= 3:
        deltas = pd.Series(df["timestamp"].values[1:]) - pd.Series(df["timestamp"].values[:-1])
        freq = deltas.mode().iloc[0] if len(deltas.mode()) > 0 else deltas.median()
    else:
        freq = None

    print(
        f"[{name}] n={n}, missing={miss:.3f}, "
        f"min={df['value'].min():.3f}, max={df['value'].max():.3f}, freq≈{freq}"
    )


def download_spy(processed_dir: Path, start="2010-01-01", end=None):
    import yfinance as yf

    ticker = "SPY"
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",  # makes columns more consistent
    )

    if df is None or df.empty:
        raise RuntimeError("SPY download returned empty dataframe. Check internet or yfinance.")

    # Robustly extract Close as 1D
    close = None

    # Case 1: normal columns
    if "Close" in df.columns:
        close = df["Close"]

    # Case 2: MultiIndex columns
    if close is None and isinstance(df.columns, pd.MultiIndex):
        candidates = [col for col in df.columns if "Close" in col]
        if len(candidates) > 0:
            close = df[candidates[0]]

    if close is None:
        raise ValueError(f"Could not find Close column. Columns: {df.columns}")

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df.index),
            "value": close.values,
        }
    ).dropna()

    out_path = processed_dir / "finance_spy.csv"
    out.to_csv(out_path, index=False)
    summarize_series(out, "finance_spy")
    print("Saved:", out_path)

    
def download_solar(processed_dir: Path):
    ds = load_dataset("autogluon/fev_datasets", "solar_1D")
    df = ds["train"].to_pandas()
    print("[solar_1D] raw columns:", df.columns.tolist())

    # Similar to ETT: timestamp/value columns may be arrays -> explode
    series_id = df["id"].iloc[0]
    sub = df[df["id"] == series_id].sort_values("timestamp")

    def to_list_if_array(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    sub = sub.copy()
    sub["timestamp"] = sub["timestamp"].apply(to_list_if_array)

    if "target" in sub.columns:
        sub["target"] = sub["target"].apply(to_list_if_array)
        long_df = sub[["timestamp", "target"]].explode(["timestamp", "target"], ignore_index=True)
        long_df = long_df.rename(columns={"target": "value"})
    else:
        candidate_cols = [c for c in sub.columns if c not in ["id", "timestamp"]]
        val_col = candidate_cols[-1]
        sub[val_col] = sub[val_col].apply(to_list_if_array)
        long_df = sub[["timestamp", val_col]].explode(["timestamp", val_col], ignore_index=True)
        long_df = long_df.rename(columns={val_col: "value"})

    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna().sort_values("timestamp")

    out_path = processed_dir / "energy_solar_1D.csv"
    long_df.to_csv(out_path, index=False)
    summarize_series(long_df, "energy_solar_1D")
    print("Saved:", out_path)


def download_yahoo_s5(processed_dir: Path):
    """
    Yahoo S5 download will be implemented next step.
    Placeholder is kept for now.
    """
    out_path = processed_dir / "anomaly_yahoo_s5_PLACEHOLDER.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("TODO: add Yahoo S5 download source and parser.\n")
    print("Created placeholder:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--spy_start", default="2010-01-01", type=str)

    args = parser.parse_args()

    raw_dir, processed_dir = ensure_dirs(args.data_dir)

    # Forecasting domains
    download_spy(processed_dir, start=args.spy_start)
    download_solar(processed_dir)

    # Anomaly benchmark placeholder (we implement Yahoo S5 next step)
    download_yahoo_s5(processed_dir)

    print("Done.")


if __name__ == "__main__":
    main()