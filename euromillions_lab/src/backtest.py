from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compute_matches


def run_backtest(
    draws_df: pd.DataFrame,
    strategy,
    start_date=None,
    end_date=None,
    window_k=None,
    seed: int = 42,
) -> pd.DataFrame:
    df = draws_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if start_date is not None:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    df = df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    rows = []

    full = draws_df.sort_values("date").reset_index(drop=True).copy()
    full["date"] = pd.to_datetime(full["date"])

    for _, row in df.iterrows():
        current_date = row["date"]
        history_df = full[full["date"] < current_date]
        if window_k is not None and window_k > 0:
            history_df = history_df.tail(window_k)

        pick_nums, pick_stars = strategy.pick(history_df, rng)
        real_nums = [int(row[f"n{i}"]) for i in range(1, 6)]
        real_stars = [int(row["s1"]), int(row["s2"])]

        mn, ms = compute_matches(pick_nums, pick_stars, real_nums, real_stars)

        out = {
            "date": current_date,
            **{f"pick_n{i}": pick_nums[i - 1] for i in range(1, 6)},
            "pick_s1": pick_stars[0],
            "pick_s2": pick_stars[1],
            **{f"real_n{i}": real_nums[i - 1] for i in range(1, 6)},
            "real_s1": real_stars[0],
            "real_s2": real_stars[1],
            "matches_numbers": mn,
            "matches_stars": ms,
        }
        rows.append(out)

    return pd.DataFrame(rows)


def save_results(results_df: pd.DataFrame, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    return out
