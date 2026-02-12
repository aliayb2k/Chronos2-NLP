import os
import argparse
import json
import glob
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from groq import Groq


def build_event_payload(df: pd.DataFrame, idx: int, win_left: int, win_right: int) -> dict:
    """
    Build a compact payload for LLM: local window + key stats.
    idx is row index within df (NOT t_idx).
    """
    lo = max(0, idx - win_left)
    hi = min(len(df), idx + win_right + 1)
    w = df.iloc[lo:hi].copy()

    center = df.iloc[idx]

    payload = {
        "series_id": str(center.get("series_id", "")),
        "timestamp": str(center["timestamp"]),
        "t_idx": int(center["t_idx"]),
        "y_true": float(center["y_true"]),
        "q50": float(center.get("q50", np.nan)),
        "q10": float(center.get("q10", np.nan)),
        "q90": float(center.get("q90", np.nan)),
        "pi_width": float(center.get("pi_width", np.nan)),
        "residual": float(center.get("residual", np.nan)),
        "score_z": float(center.get("score_z", np.nan)),
        "coverage_out": int(center.get("coverage_out", 0)),
        "detectors": {
            "anom_pi": int(center.get("anom_pi", 0)),
            "anom_resid_z": int(center.get("anom_resid_z", 0)),
            "anom_resid_mad": int(center.get("anom_resid_mad", 0)),
            "anom_ens": int(center.get("anom_ens", 0)),
        },
        "window": {
            "start_ts": str(w["timestamp"].iloc[0]),
            "end_ts": str(w["timestamp"].iloc[-1]),
            "y_true": [float(x) for x in w["y_true"].values],
            "q50": [float(x) for x in w["q50"].values] if "q50" in w.columns else [],
            "residual": [float(x) for x in w["residual"].values] if "residual" in w.columns else [],
            "score_z": [float(x) for x in w["score_z"].values] if "score_z" in w.columns else [],
        },
    }
    return payload


def groq_explain(client: Groq, model: str, payload: dict, max_tokens: int = 220) -> str:
    """
    Ask Groq for a short anomaly explanation.
    """
    system = (
        "You are a time-series anomaly analysis assistant. "
        "Explain briefly and concretely for an engineer. "
        "DO NOT invent external context (no domain story). "
        "Use only the provided numbers and window."
    )

    user = (
    "Given this JSON payload, produce EXACTLY 4 bullet points:\n"
    "- Type: spike/drop/level-shift/noise (choose one)\n"
    "- Evidence: compare y_true vs q50 and whether it is outside [q10,q90]\n"
    "- Magnitude: give |y_true-q50| and PI width (q90-q10)\n"
    "- Detectors: list which are 1 and what they imply\n"
    "Rules: Do NOT mention IQR. Do NOT invent context.\n\n"
    f"{json.dumps(payload)}"
)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def load_rolling_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")

    # Default behavior: scan Yahoo rolling tables
    ap.add_argument("--input_glob", default="results/tables/rolling_1step_yahoo_real_*.csv")

    # Optional: curated list of events (recommended)
    ap.add_argument(
        "--events_csv",
        default="results/tables/yahoo_llm_shortlist.csv",
        help="CSV with at least columns: series_id, t_idx (and optionally label). "
            "Default points to the Yahoo shortlist; set empty to scan all files."
    )

    ap.add_argument("--detector_col", default="anom_resid_mad", help="detector column to explain (flagged=1)")
    ap.add_argument("--max_events", type=int, default=30, help="cap number of events explained (cost control)")
    ap.add_argument("--win_left", type=int, default=50)
    ap.add_argument("--win_right", type=int, default=10)
    ap.add_argument("--groq_model",default="llama-3.1-8b-instant")
    ap.add_argument("--max_tokens", type=int, default=220)
    args = ap.parse_args()

    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY in environment (set it or use .env).")

    client = Groq(api_key=key)

    out_rows = []
    explained = 0

    # -------------------------
    # Mode 1: Curated events file (recommended)
    # -------------------------
    if args.events_csv:
        events_path = Path(args.events_csv)
        if not events_path.exists():
            raise FileNotFoundError(f"--events_csv not found: {events_path}")

        events = pd.read_csv(events_path)
        if "series_id" not in events.columns or "t_idx" not in events.columns:
            raise ValueError("--events_csv must contain columns: series_id, t_idx (and optionally label)")

        # cache rolling tables in memory by series_id to avoid repeated reads
        cache: dict[str, pd.DataFrame] = {}

        for _, row in events.iterrows():
            if explained >= args.max_events:
                break

            series_id = str(row["series_id"])
            t_idx = int(row["t_idx"])
            label = int(row["label"]) if "label" in events.columns and pd.notna(row["label"]) else None

            rolling_path = Path(args.results_dir) / "tables" / f"rolling_1step_yahoo_{series_id}.csv"
            if not rolling_path.exists():
                print(f"[SKIP] Missing rolling file: {rolling_path}")
                continue

            if series_id not in cache:
                cache[series_id] = load_rolling_table(rolling_path)

            df = cache[series_id]

            # find the row index corresponding to t_idx
            hit = df.index[df["t_idx"].astype(int) == t_idx].tolist()
            if not hit:
                print(f"[SKIP] {series_id} t_idx={t_idx} not found in {rolling_path.name}")
                continue

            i = int(hit[0])

            # build payload + run LLM
            payload = build_event_payload(df, i, args.win_left, args.win_right)
            text = groq_explain(client, args.groq_model, payload, max_tokens=args.max_tokens)

            det_flag = int(df.loc[i, args.detector_col]) if args.detector_col in df.columns else None

            out_rows.append({
                "file": rolling_path.name,
                "series_id": series_id,
                "timestamp": payload["timestamp"],
                "t_idx": payload["t_idx"],
                "label": label,
                "detector_col": args.detector_col,
                "detector_flag": det_flag,
                "explanation": text,
            })
            explained += 1

    # -------------------------
    # Mode 2: old behavior (scan all rolling files, explain flagged points)
    # -------------------------
    else:
        paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
        if not paths:
            raise FileNotFoundError(f"No files matched: {args.input_glob}")

        for p in paths:
            df = load_rolling_table(p)

            if args.detector_col not in df.columns:
                print(f"[SKIP] {p.name}: missing {args.detector_col}")
                continue

            anom_idx = df.index[df[args.detector_col].astype(int) == 1].tolist()

            for i in anom_idx:
                if explained >= args.max_events:
                    break

                payload = build_event_payload(df, int(i), args.win_left, args.win_right)
                text = groq_explain(client, args.groq_model, payload, max_tokens=args.max_tokens)

                out_rows.append({
                    "file": p.name,
                    "series_id": payload["series_id"],
                    "timestamp": payload["timestamp"],
                    "t_idx": payload["t_idx"],
                    "label": int(df.loc[i, "label"]) if "label" in df.columns else None,
                    "detector_col": args.detector_col,
                    "detector_flag": int(df.loc[i, args.detector_col]),
                    "explanation": text,
                })
                explained += 1

            if explained >= args.max_events:
                break

    out = pd.DataFrame(out_rows)
    out_path = Path(args.results_dir) / "tables" / "yahoo_llm_explanations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("Saved:", out_path, "rows=", len(out))


if __name__ == "__main__":
    main()