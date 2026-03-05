from __future__ import annotations

import numpy as np
import pandas as pd


def compute_matches(
    pick_nums: list[int], pick_stars: list[int], real_nums: list[int], real_stars: list[int]
) -> tuple[int, int]:
    matches_numbers = len(set(pick_nums).intersection(real_nums))
    matches_stars = len(set(pick_stars).intersection(real_stars))
    return matches_numbers, matches_stars


def aggregate_metrics(results_df: pd.DataFrame) -> dict:
    num_dist = results_df["matches_numbers"].value_counts().sort_index().to_dict()
    star_dist = results_df["matches_stars"].value_counts().sort_index().to_dict()
    top_outcomes = (
        results_df.groupby(["matches_numbers", "matches_stars"]).size().sort_values(ascending=False).head(10).to_dict()
    )

    return {
        "distribution_matches_numbers": num_dist,
        "distribution_matches_stars": star_dist,
        "mean_matches_numbers": float(results_df["matches_numbers"].mean()),
        "median_matches_numbers": float(results_df["matches_numbers"].median()),
        "mean_matches_stars": float(results_df["matches_stars"].mean()),
        "median_matches_stars": float(results_df["matches_stars"].median()),
        "top_outcomes": top_outcomes,
    }


def random_baseline_stats(results_df: pd.DataFrame, n_simulations: int = 5000, seed: int = 42) -> dict:
    """Baseline distribution under random tickets using vectorized hypergeometric sampling."""
    rng = np.random.default_rng(seed)
    draw_count = len(results_df)
    if draw_count == 0:
        return {
            "expected_mean_numbers": 0.0,
            "expected_mean_stars": 0.0,
            "percentile_of_strategy_by_draw": [],
            "overall_percentile": 0.0,
        }

    # Random match counts are draw-invariant for fixed Euromillions combinatorics.
    # numbers: from population 50 with 5 winning numbers, draw 5
    # stars: from population 12 with 2 winning stars, draw 2
    sim_num_hits = rng.hypergeometric(ngood=5, nbad=45, nsample=5, size=n_simulations)
    sim_star_hits = rng.hypergeometric(ngood=2, nbad=10, nsample=2, size=n_simulations)

    strategy_hits = results_df["matches_numbers"].to_numpy(dtype=float)
    per_draw_percentiles = [float((sim_num_hits <= hit).mean() * 100.0) for hit in strategy_hits]

    # Compare strategy mean against simulated means over same number of draws.
    sim_mean_hits = rng.hypergeometric(
        ngood=5,
        nbad=45,
        nsample=5,
        size=(n_simulations, draw_count),
    ).mean(axis=1)
    strategy_mean = float(strategy_hits.mean())
    overall_percentile = float((sim_mean_hits <= strategy_mean).mean() * 100.0)

    return {
        "expected_mean_numbers": float(sim_num_hits.mean()),
        "expected_mean_stars": float(sim_star_hits.mean()),
        "percentile_of_strategy_by_draw": per_draw_percentiles,
        "overall_percentile": overall_percentile,
    }
