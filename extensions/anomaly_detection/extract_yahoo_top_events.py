import os
import argparse
import glob
import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", help="Root results dir (default: results)")
    parser.add_argument("--detector_col", default="anom_resid_mad", help="Which detector column to rank by")
    parser.add_argument("--top_k", type=int, default=80, help="How many top events to export")
    parser.add_argument("--only_flagged", action="store_true", help="Keep only rows where detector_col == 1")
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional explicit glob pattern. If not set, uses results_dir/tables/rolling_1step_yahoo_*.csv",
    )
    args = parser.parse_args()

    if args.pattern is None:
        pattern = os.path.join(args.results_dir, "tables", "rolling_1step_yahoo_*.csv")
    else:
        pattern = args.pattern

    files = sorted(glob.glob(pattern))
    print(f"[extract] pattern: {pattern}")
    print(f"[extract] found files: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError(
            "No Yahoo rolling files found.\n"
            f"Tried pattern: {pattern}\n"
            "Make sure you have files like results/tables/rolling_1step_yahoo_real_1.csv"
        )

    rows = []

    # Read + collect
    for fpath in files:
        df = pd.read_csv(fpath)

        # sanity check once
        if len(rows) == 0:
            print(f"[extract] first file: {os.path.basename(fpath)}")
            print(f"[extract] columns: {list(df.columns)}")

        if args.detector_col not in df.columns:
            print(f"[extract] WARNING: {os.path.basename(fpath)} missing detector_col={args.detector_col} -> skipped")
            continue

        # Ranking score (simple + consistent): absolute residual
        if "residual" in df.columns:
            df["rank_score"] = df["residual"].abs()
        else:
            if "score_z" not in df.columns:
                print(f"[extract] WARNING: {os.path.basename(fpath)} missing residual and score_z -> skipped")
                continue
            df["rank_score"] = df["score_z"].astype(float)

        if args.only_flagged:
            df = df[df[args.detector_col].astype(int) == 1]

        if len(df) == 0:
            continue

        keep = [
            "series_id",
            "timestamp",
            "t_idx",
            "y_true",
            "label",
            args.detector_col,
            "residual",
            "score_z",
            "rank_score",
        ]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].copy()

        rows.append(out)

    if len(rows) == 0:
        raise ValueError(
            "No rows collected from Yahoo rolling files.\n"
            "This usually means:\n"
            "1) All files were skipped because detector_col was missing (check warnings above), OR\n"
            "2) --only_flagged filtered everything (try running without --only_flagged), OR\n"
            "3) Files are empty / malformed.\n"
            f"detector_col used: {args.detector_col}"
        )

    all_df = pd.concat(rows, ignore_index=True)

    # sort by rank_score desc and take top_k
    all_df = all_df.sort_values("rank_score", ascending=False).head(args.top_k)

    out_path = os.path.join(args.results_dir, "tables", f"yahoo_top_events_{args.detector_col}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_df.to_csv(out_path, index=False)
    print(f"[extract] saved: {out_path} rows={len(all_df)}")


if __name__ == "__main__":
    main()