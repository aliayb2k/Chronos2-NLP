import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chronos import Chronos2Pipeline
from chronos2_nlp.data.datasets import load_domain_series
from pathlib import Path



# Forecast helper
def forecast_1step_samples(pipeline, context_1d: np.ndarray, n_horizon: int = 1) -> np.ndarray:
    x = context_1d.astype(np.float32)[None, None, :]  # (1, 1, L)
    forecast = pipeline.predict(x, prediction_length=n_horizon)
    pred = np.squeeze(np.array(forecast))  # expected (n_samples, n_horizon)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.ndim != 2:
        raise ValueError(f"Unexpected pred shape: {pred.shape}")
    return pred  # (S, H)



# Injection (realistic + auditable)
def inject_anomalies(
    values: np.ndarray,
    timestamps: np.ndarray,
    n_anoms: int = 10,
    seed: int = 0,
    spike_k: float = 6.0,
    shift_k: float = 4.0,
    shift_len: int = 8,
    *,
    inject_start: int | None = None,
    inject_end: int | None = None,
    # More realistic for positive-valued series (prices/energy):
    pct_mode: bool = True,
    spike_pct: float = 0.10,   # 10% one-point shock
    shift_pct: float = 0.06,   # 6% level shift
    clip_min: float | None = 0.0,  # keep non-negative by default
):
    """
    Inject synthetic anomalies into a COPY of values.
    Returns: (values_inj, labels, inj_idxs)

      - labels: {0,1} aligned with original indices
      - inj_idxs: list of starting indices where we injected (spike/drop index,
                 shift start index)

    Injection types:
      - spike/drop: single-point shock
      - level shift: window of length shift_len
    """
    rng = np.random.default_rng(seed)
    x = values.astype(np.float32).copy()
    labels = np.zeros(len(x), dtype=np.int32)

    # Additive scale 
    std = float(np.nanstd(x))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0

    T = len(x)

    # Decide whether to use multiplicative % shocks (safer + more realistic)
    x_finite = x[np.isfinite(x)]
    is_positive = (len(x_finite) > 0) and (float(np.min(x_finite)) > 0.0)
    use_pct = bool(pct_mode and is_positive)

    # Choose injection range (prefer evaluation window), but keep buffers
    buffer_left = 5
    buffer_right = max(5, shift_len + 1)

    start = buffer_left if inject_start is None else max(buffer_left, int(inject_start))
    end = (T - buffer_right) if inject_end is None else min(T - buffer_right, int(inject_end))

    if end <= start:
        return x, labels, []

    valid = np.arange(start, end)
    idxs = rng.choice(valid, size=min(n_anoms, len(valid)), replace=False)
    inj_idxs: list[int] = []

    for t in sorted(idxs):
        inj_idxs.append(int(t))
        kind = rng.choice(
            ["spike", "drop", "shift_up", "shift_down"],
            p=[0.35, 0.35, 0.15, 0.15],
        )

        if use_pct:
            if kind == "spike":
                x[t] = x[t] * (1.0 + spike_pct)
                labels[t] = 1
            elif kind == "drop":
                x[t] = x[t] * (1.0 - spike_pct)
                labels[t] = 1
            elif kind == "shift_up":
                t2 = min(T, t + shift_len)
                x[t:t2] = x[t:t2] * (1.0 + shift_pct)
                labels[t:t2] = 1
            elif kind == "shift_down":
                t2 = min(T, t + shift_len)
                x[t:t2] = x[t:t2] * (1.0 - shift_pct)
                labels[t:t2] = 1
        else:
            if kind == "spike":
                x[t] = x[t] + spike_k * std
                labels[t] = 1
            elif kind == "drop":
                x[t] = x[t] - spike_k * std
                labels[t] = 1
            elif kind == "shift_up":
                t2 = min(T, t + shift_len)
                x[t:t2] = x[t:t2] + shift_k * std
                labels[t:t2] = 1
            elif kind == "shift_down":
                t2 = min(T, t + shift_len)
                x[t:t2] = x[t:t2] - shift_k * std
                labels[t:t2] = 1

    # Safety: keep series valid (e.g., no negative energy/price)
    if clip_min is not None:
        x = np.maximum(x, float(clip_min))

    return x, labels, inj_idxs



# Rolling 1-step
def rolling_1step(
    pipeline,
    values: np.ndarray,
    timestamps: np.ndarray,
    context: int,
    max_steps: int,
    stride: int,
):
    """
    Produces rolling 1-step probabilistic forecasts for the last max_steps points.

    Returns DataFrame with one row per predicted point t:
      predict y[t] from values[t-context:t]
    Includes t_idx for safe label alignment.
    """
    T = len(values)
    start_t = max(context, T - max_steps)

    rows = []
    eps = 1e-6

    for t in range(start_t, T, stride):
        if t - context < 0:
            continue

        ctx = values[t - context : t]
        y_true = values[t]
        ts = timestamps[t]

        samples = forecast_1step_samples(pipeline, ctx, n_horizon=1)  # (S, 1)
        s = samples[:, 0]

        q10 = float(np.quantile(s, 0.10))
        q50 = float(np.quantile(s, 0.50))
        q90 = float(np.quantile(s, 0.90))

        pi_width = float(q90 - q10)
        abs_err = float(abs(y_true - q50))

        coverage_out = 0 if (q10 <= y_true <= q90) else 1
        score_z = float(abs(y_true - q50) / (pi_width + eps))

        rows.append(
            {
                "timestamp": ts,
                "t_idx": int(t),
                "y_true": float(y_true),
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "pi_width": pi_width,
                "abs_err": abs_err,
                "coverage_out": int(coverage_out),
                "score_z": score_z,
                "residual": float(y_true - q50),
            }
        )

    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out



# Detectors  
#1
def detector_pi_scorez(df: pd.DataFrame, rolling_window: int, q: float) -> pd.Series:
    thr = (
        df["score_z"]
        .rolling(window=rolling_window, min_periods=max(20, rolling_window // 5))
        .quantile(q)
    )
    thr = thr.fillna(float(df["score_z"].quantile(q)))
    return ((df["score_z"] > thr) & (df["coverage_out"] == 1)).astype(int)


#2
def detector_resid_z(df: pd.DataFrame, rolling_window: int, q: float) -> pd.Series:
    r = df["residual"]
    mu = r.rolling(window=rolling_window, min_periods=max(20, rolling_window // 5)).mean()
    sd = r.rolling(window=rolling_window, min_periods=max(20, rolling_window // 5)).std(ddof=0)
    sd = sd.replace(0.0, np.nan)

    z = (r - mu).abs() / (sd + 1e-6)

    thr = z.rolling(window=rolling_window, min_periods=max(20, rolling_window // 5)).quantile(q)
    thr = thr.fillna(float(z.quantile(q)))
    return (z > thr).astype(int)


#3
def detector_resid_mad(df: pd.DataFrame, rolling_window: int, q: float) -> pd.Series:
    r = df["residual"]
    med = r.rolling(window=rolling_window, min_periods=max(20, rolling_window // 5)).median()
    mad = (r - med).abs().rolling(window=rolling_window, min_periods=max(20, rolling_window // 5)).median()
    scale = (1.4826 * mad).replace(0.0, np.nan)

    z = (r - med).abs() / (scale + 1e-6)

    thr = z.rolling(window=rolling_window, min_periods=max(20, rolling_window // 5)).quantile(q)
    thr = thr.fillna(float(z.quantile(q)))
    return (z > thr).astype(int)


# Ensemble 
def majority_vote(*flags: pd.Series, k: int = 2) -> pd.Series:
    s = None
    for f in flags:
        s = f.astype(int) if s is None else (s + f.astype(int))
    return (s >= k).astype(int)


def eval_pointwise(yhat: np.ndarray, y: np.ndarray):
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn



# Plotting
def save_plot(df: pd.DataFrame, domain: str, results_dir: str, anomaly_col: str = "anom_ens"):
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    plt.figure(figsize=(11, 4))
    plt.plot(df["timestamp"], df["y_true"], label="true")
    plt.plot(df["timestamp"], df["q50"], label="forecast(q50)")
    plt.fill_between(df["timestamp"], df["q10"], df["q90"], alpha=0.2, label="PI(10-90)")

    if anomaly_col in df.columns:
        anom = df[df[anomaly_col] == 1]
        if len(anom) > 0:
            plt.scatter(anom["timestamp"], anom["y_true"], s=20, label=f"anomaly({anomaly_col})")

    if "label" in df.columns:
        lab = df[df["label"] == 1]
        if len(lab) > 0:
            plt.scatter(lab["timestamp"], lab["y_true"], s=35, marker="x", label="label")

    plt.title(f"Rolling 1-step forecast + anomaly flags - {domain}")
    plt.legend()

    outpath = os.path.join(results_dir, "figures", f"rolling_1step_{domain}.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print("Saved plot:", outpath)



# Yahoo loader
def load_yahoo_a1_csv(path: str | Path):
    """
    Yahoo S5 A1Benchmark format:
    timestamp,value,is_anomaly
    """
    df = pd.read_csv(path)
    needed = {"timestamp", "value", "is_anomaly"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Found: {list(df.columns)}")

    t = df["timestamp"].astype(int).values
    base = pd.Timestamp("2000-01-01")
    ts = np.array([base + pd.Timedelta(days=int(i) - 1) for i in t], dtype="datetime64[ns]")

    values = df["value"].astype(np.float32).values
    labels = df["is_anomaly"].astype(np.int32).values
    return ts, values, labels




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rolling_window", type=int, default=200)
    parser.add_argument("--quantile", type=float, default=0.99)

    # injection
    parser.add_argument("--inject", action="store_true")
    parser.add_argument("--n_anoms", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_inject", action="store_true")

    # realistic injection controls
    parser.add_argument("--pct_mode", action="store_true", help="Use percent shocks instead of k*std")
    parser.add_argument("--spike_pct", type=float, default=0.12)
    parser.add_argument("--shift_pct", type=float, default=0.08)

    # stress injection controls (if pct_mode is off)
    parser.add_argument("--spike_k", type=float, default=6.0)
    parser.add_argument("--shift_k", type=float, default=4.0)
    parser.add_argument("--shift_len", type=int, default=8)

    # yahoo
    parser.add_argument("--yahoo", action="store_true", help="Run Yahoo S5 A1Benchmark evaluation")
    parser.add_argument("--yahoo_dir", default="data/raw/yahoo_s5/A1Benchmark", help="Path to A1Benchmark folder")
    parser.add_argument("--yahoo_n", type=int, default=20, help="How many real_*.csv files to use")
    parser.add_argument("--yahoo_glob", default="real_*.csv", help="File glob inside yahoo_dir")
    parser.add_argument("--yahoo_plots", type=int, default=3, help="How many series to save plots for (0 disables)")

    args = parser.parse_args()

    os.makedirs(os.path.join(args.results_dir, "tables"), exist_ok=True)

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)

    metrics_rows = []

    
    # Yahoo S5 A1Benchmark 
    if args.yahoo:
        yahoo_dir = Path(args.yahoo_dir)
        files = sorted(yahoo_dir.glob(args.yahoo_glob))
        if len(files) == 0:
            raise FileNotFoundError(f"No files found in {yahoo_dir} with pattern {args.yahoo_glob}")

        files = files[: min(args.yahoo_n, len(files))]
        print(f"[YAHOO A1] Using {len(files)} files from {yahoo_dir}")

        agg = {}  # detector -> dict(tp,fp,fn)

        for k, fpath in enumerate(files):
            series_id = fpath.stem  
            ts, values, labels = load_yahoo_a1_csv(fpath)

            ctx = min(args.context, len(values) - 2)
            df = rolling_1step(pipeline, values, ts, context=ctx, max_steps=args.max_steps, stride=args.stride)

            rw = min(args.rolling_window, max(30, len(df) // 3))

            df["anom_pi"] = detector_pi_scorez(df, rolling_window=rw, q=args.quantile)
            df["anom_resid_z"] = detector_resid_z(df, rolling_window=rw, q=args.quantile)
            df["anom_resid_mad"] = detector_resid_mad(df, rolling_window=rw, q=args.quantile)
            df["anom_ens"] = majority_vote(df["anom_pi"], df["anom_resid_z"], df["anom_resid_mad"], k=2)

            df["label"] = df["t_idx"].apply(lambda i: int(labels[int(i)]))
            df["series_id"] = series_id

            y = df["label"].astype(int).values

            for det_name, col in [
                ("pi_scorez", "anom_pi"),
                ("resid_z", "anom_resid_z"),
                ("resid_mad", "anom_resid_mad"),
                ("ensemble", "anom_ens"),
            ]:
                yhat = df[col].astype(int).values
                prec, rec, f1, tp, fp, fn = eval_pointwise(yhat, y)

                metrics_rows.append(
                    {
                        "domain": f"yahoo_a1/{series_id}",
                        "mode": "yahoo",
                        "detector": det_name,
                        "seed": args.seed,
                        "context": ctx,
                        "max_steps": args.max_steps,
                        "quantile": args.quantile,
                        "rolling_window": rw,
                        "anoms_pred": int(yhat.sum()),
                        "anoms_true": int(y.sum()),
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )

                if det_name not in agg:
                    agg[det_name] = {"tp": 0, "fp": 0, "fn": 0}
                agg[det_name]["tp"] += tp
                agg[det_name]["fp"] += fp
                agg[det_name]["fn"] += fn

            out_csv = os.path.join(args.results_dir, "tables", f"rolling_1step_yahoo_{series_id}.csv")
            df.to_csv(out_csv, index=False)

            if args.yahoo_plots > 0 and k < args.yahoo_plots:
                save_plot(df, f"yahoo_{series_id}", args.results_dir, anomaly_col="anom_ens")

        rows = []
        for det, c in agg.items():
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            rows.append(
                {
                    "domain": "yahoo_a1_micro",
                    "mode": "yahoo",
                    "detector": det,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
            )

        met_y = pd.DataFrame(rows)
        met_y_path = os.path.join(args.results_dir, "tables", "yahoo_a1_micro_metrics.csv")
        met_y.to_csv(met_y_path, index=False)
        print("Saved Yahoo micro metrics:", met_y_path)

    
    #injection benchmark
    
    for domain in ["finance_spy", "energy_solar_1D"]:
        sd = load_domain_series(domain, data_dir="data/processed")
        ts = sd.df["timestamp"].values
        values = sd.df["value"].values.astype(np.float32)

        ctx = min(args.context, len(values) - 2)
        if domain.startswith("energy_"):
            ctx = min(ctx, 256)

        T = len(values)
        start_t = max(ctx, T - args.max_steps)
        end_t = T

        labels = None
        inj_idxs = []

        pct_mode = args.pct_mode  # user-controlled; if not passed, uses additive mode

        if args.inject:
            values, labels, inj_idxs = inject_anomalies(
                values=values,
                timestamps=ts,
                n_anoms=args.n_anoms,
                seed=args.seed,
                spike_k=args.spike_k,
                shift_k=args.shift_k,
                shift_len=args.shift_len,
                inject_start=start_t,
                inject_end=end_t,
                pct_mode=pct_mode,
                spike_pct=args.spike_pct,
                shift_pct=args.shift_pct,
                clip_min=0.0,
            )

        df = rolling_1step(pipeline, values, ts, context=ctx, max_steps=args.max_steps, stride=args.stride)
        rw = min(args.rolling_window, max(30, len(df) // 3))

        df["anom_pi"] = detector_pi_scorez(df, rolling_window=rw, q=args.quantile)
        df["anom_resid_z"] = detector_resid_z(df, rolling_window=rw, q=args.quantile)
        df["anom_resid_mad"] = detector_resid_mad(df, rolling_window=rw, q=args.quantile)
        df["anom_ens"] = majority_vote(df["anom_pi"], df["anom_resid_z"], df["anom_resid_mad"], k=2)

        if labels is not None:
            df["label"] = df["t_idx"].apply(lambda i: int(labels[int(i)]))

            if args.debug_inject:
                inj_ts = [pd.to_datetime(ts[i]) for i in inj_idxs]
                inside = [i for i in inj_idxs if (start_t <= i < end_t)]
                print(f"\n[{domain}] DEBUG eval window: start_t={start_t} end_t={end_t} (T={T}, ctx={ctx}, max_steps={args.max_steps})")
                print(f"[{domain}] DEBUG window timestamps: {pd.to_datetime(ts[start_t])} -> {pd.to_datetime(ts[end_t-1])}")
                print(f"[{domain}] DEBUG injected idxs: {inj_idxs}")
                print(f"[{domain}] DEBUG injected timestamps: {inj_ts}")
                print(f"[{domain}] DEBUG injected inside eval window: {len(inside)}/{len(inj_idxs)}")
                print(f"[{domain}] DEBUG labels in df (sum): {int(df['label'].sum())} rows={len(df)}")

            y = df["label"].astype(int).values
            for det_name, col in [
                ("pi_scorez", "anom_pi"),
                ("resid_z", "anom_resid_z"),
                ("resid_mad", "anom_resid_mad"),
                ("ensemble", "anom_ens"),
            ]:
                yhat = df[col].astype(int).values
                prec, rec, f1, tp, fp, fn = eval_pointwise(yhat, y)
                metrics_rows.append(
                    {
                        "domain": domain,
                        "mode": "inject",
                        "detector": det_name,
                        "seed": args.seed,
                        "context": ctx,
                        "max_steps": args.max_steps,
                        "quantile": args.quantile,
                        "rolling_window": rw,
                        "anoms_pred": int(yhat.sum()),
                        "anoms_true": int(y.sum()),
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )

            out_csv = os.path.join(args.results_dir, "tables", f"rolling_1step_{domain}_INJECT.csv")
            save_plot(df, f"{domain}_INJECT", args.results_dir, anomaly_col="anom_ens")
        else:
            out_csv = os.path.join(args.results_dir, "tables", f"rolling_1step_{domain}.csv")
            save_plot(df, domain, args.results_dir, anomaly_col="anom_pi")

        df.to_csv(out_csv, index=False)
        print(
            f"[{domain}] saved: {out_csv} rows={len(df)} "
            f"anoms(pi)={int(df['anom_pi'].sum())} anoms(ens)={int(df['anom_ens'].sum())}"
        )

    # Save metrics 
    if len(metrics_rows) > 0:
        met = pd.DataFrame(metrics_rows)
        met_path = os.path.join(args.results_dir, "tables", "inject_eval_metrics.csv")
        met.to_csv(met_path, index=False)
        print("Saved metrics:", met_path)


if __name__ == "__main__":
    main()