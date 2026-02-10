from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class SeriesData:
    name: str
    df: pd.DataFrame  # columns: timestamp, value (timestamp is datetime64)

def load_csv_series(path: str | Path, name: str) -> SeriesData:
    path = Path(path)
    df = pd.read_csv(path)

    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{path} must have columns ['timestamp','value'], got {df.columns.tolist()}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["value"] = pd.to_numeric(df["value"], errors="raise")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    return SeriesData(name=name, df=df)

def load_domain_series(domain: str, data_dir: str | Path = "data/processed") -> SeriesData:
    data_dir = Path(data_dir)

    mapping = {
        "finance_spy": data_dir / "finance_spy.csv",
        "energy_solar_1D": data_dir / "energy_solar_1D.csv",
    }

    if domain not in mapping:
        raise ValueError(f"Unknown domain={domain}. Available: {list(mapping.keys())}")

    return load_csv_series(mapping[domain], name=domain)