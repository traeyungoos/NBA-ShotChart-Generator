"""
Download shotdetail data from github.com/shufinskiy/nba_data and prep into parquet.

Usage:
    python prep_data.py                  # WNBA 2020-2025
    python prep_data.py --league nba     # NBA 2020-2025
    python prep_data.py --start 2015     # WNBA 2015-2025
"""

import argparse
import tarfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from itertools import product

import pandas as pd


def load_nba_data(seasons, league="wnba"):
    """Download shotdetail CSVs from shufinskiy/nba_data, return single DataFrame."""

    data_type = "shotdetail"
    need_data = [f"{data_type}_{s}" for s in seasons]
    if league.lower() == "wnba":
        need_data = [f"wnba_{n}" for n in need_data]

    # Fetch the manifest mapping names → download URLs
    with urlopen(
        "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"
    ) as f:
        lines = f.read().decode("utf-8").strip().split("\n")

    manifest = {}
    for line in lines:
        name, url = line.split("=", 1)
        manifest[name] = url

    frames = []
    for name in need_data:
        if name not in manifest:
            print(f"  ⚠ {name} not found in manifest, skipping")
            continue
        url = manifest[name]
        print(f"  ↓ downloading {name} ...")
        try:
            with urlopen(url) as resp:
                content = resp.read()
            with tarfile.open(fileobj=BytesIO(content), mode="r:xz") as tar:
                csv_name = f"{name}.csv"
                csv_file = tar.extractfile(csv_name)
                if csv_file is None:
                    print(f"  ⚠ {csv_name} not in archive, skipping")
                    continue
                df = pd.read_csv(csv_file)
                frames.append(df)
        except Exception as e:
            print(f"  ⚠ error on {name}: {e}")
            continue

    if not frames:
        raise RuntimeError("No data downloaded. Check seasons/league.")
    return pd.concat(frames, ignore_index=True)


def prep(df, league):
    """Add derived columns and clean up."""

    # Derive season from GAME_DATE (format varies: 20230615 or 2023-06-15)
    df["GAME_DATE"] = df["GAME_DATE"].astype(str)
    if df["GAME_DATE"].str.contains("-").any():
        df["_year"] = df["GAME_DATE"].str[:4].astype(int)
        df["_month"] = df["GAME_DATE"].str[5:7].astype(int)
    else:
        df["_year"] = df["GAME_DATE"].str[:4].astype(int)
        df["_month"] = df["GAME_DATE"].str[4:6].astype(int)

    # WNBA season = calendar year. NBA season: Oct-Dec = season start, Jan-Jun = season end.
    if league.lower() == "wnba":
        df["SEASON"] = df["_year"]
    else:
        df["SEASON"] = df.apply(
            lambda r: r["_year"] if r["_month"] >= 9 else r["_year"] - 1, axis=1
        )

    df.drop(columns=["_year", "_month"], inplace=True)

    # Drop backcourt heaves (noisy, never enough volume to color)
    df = df[df["SHOT_ZONE_BASIC"] != "Backcourt"].copy()

    # Ensure numeric types
    df["LOC_X"] = pd.to_numeric(df["LOC_X"], errors="coerce")
    df["LOC_Y"] = pd.to_numeric(df["LOC_Y"], errors="coerce")
    df["SHOT_MADE_FLAG"] = pd.to_numeric(df["SHOT_MADE_FLAG"], errors="coerce")
    df.dropna(subset=["LOC_X", "LOC_Y", "SHOT_MADE_FLAG"], inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Prep shot chart data")
    parser.add_argument("--league", default="nba", choices=["nba", "wnba"])
    parser.add_argument("--start", type=int, default=2020)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    seasons = list(range(args.start, args.end + 1))
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"shots_{args.league}_{args.start}_{args.end}.parquet"

    print(f"Downloading {args.league.upper()} shotdetail for seasons {seasons}")
    df = load_nba_data(seasons, league=args.league)
    print(f"  → {len(df):,} rows downloaded")

    print("Prepping data ...")
    df = prep(df, args.league)
    print(f"  → {len(df):,} rows after cleanup")

    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
