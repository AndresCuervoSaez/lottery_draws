from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


NUMBERS = np.arange(1, 51)
BASE_RATE = 5.0 / 50.0
WINDOWS = (5, 10, 20, 50, 100, 200)
EWMA_HALFLIVES = (5, 10, 20, 50)


@dataclass(frozen=True)
class CandidateFeatureConfig:
    min_history: int = 30
    graph_alpha: float = 20.0
    gap_cap: int = 100


def _binary_draw_matrix(draws: pd.DataFrame) -> np.ndarray:
    y = np.zeros((len(draws), 50), dtype=np.int8)
    for i, row in draws.iterrows():
        for c in ("n1", "n2", "n3", "n4", "n5"):
            y[i, int(row[c]) - 1] = 1
    return y


def _window_count(cumsum: np.ndarray, t: int, w: int) -> np.ndarray:
    start = max(0, t - w)
    if start == 0:
        return cumsum[t - 1].astype(float)
    return (cumsum[t - 1] - cumsum[start - 1]).astype(float)


def build_candidate_frame(
    draws_df: pd.DataFrame,
    config: CandidateFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Build one row per (draw, candidate number).

    All features for draw t use only draws < t.
    No feature uses the current draw or future draws.
    """
    cfg = config or CandidateFeatureConfig()

    draws = draws_df.copy()
    draws["date"] = pd.to_datetime(draws["date"])
    draws = draws.sort_values("date").reset_index(drop=True)

    y = _binary_draw_matrix(draws)
    cumsum = np.cumsum(y, axis=0)

    # EWMA state before the current draw.
    ewma = {h: np.zeros(50, dtype=float) for h in EWMA_HALFLIVES}
    ewma_alpha = {h: 1.0 - np.exp(np.log(0.5) / h) for h in EWMA_HALFLIVES}

    # Hit history for gap features.
    hit_history: list[list[int]] = [[] for _ in range(50)]

    # Transition graph:
    # source number appeared in draw s -> candidate appeared in draw s+1.
    transition_counts = np.zeros((50, 50), dtype=float)
    source_counts = np.zeros(50, dtype=float)

    rows: list[dict] = []

    for t in range(len(draws)):
        if t >= cfg.min_history:
            prev_draw = np.flatnonzero(y[t - 1]) if t > 0 else np.array([], dtype=int)
            prev2 = np.flatnonzero(y[t - 2]) if t > 1 else np.array([], dtype=int)
            prev5_start = max(0, t - 5)
            prev5_counts = y[prev5_start:t].sum(axis=0)

            # Precompute frequency families.
            frequency_features: dict[str, np.ndarray] = {}
            for w in WINDOWS:
                available = min(t, w)
                counts = _window_count(cumsum, t, w)
                freq = counts / max(available, 1)
                expected_count = available * BASE_RATE
                sd = np.sqrt(max(available * BASE_RATE * (1.0 - BASE_RATE), 1e-12))
                z = (counts - expected_count) / sd
                frequency_features[f"freq_{w}"] = freq
                frequency_features[f"zfreq_{w}"] = z

            all_counts = cumsum[t - 1].astype(float)
            frequency_features["freq_all"] = all_counts / t
            expected_all = t * BASE_RATE
            sd_all = np.sqrt(max(t * BASE_RATE * (1.0 - BASE_RATE), 1e-12))
            frequency_features["zfreq_all"] = (all_counts - expected_all) / sd_all

            # Graph score from the immediately preceding draw.
            if len(prev_draw):
                smoothed = (
                    transition_counts[prev_draw]
                    + cfg.graph_alpha * BASE_RATE
                ) / (
                    source_counts[prev_draw, None]
                    + cfg.graph_alpha
                )
                graph_probability = smoothed.mean(axis=0)
                graph_lift = graph_probability - BASE_RATE
                graph_max_lift = smoothed.max(axis=0) - BASE_RATE
            else:
                graph_probability = np.full(50, BASE_RATE)
                graph_lift = np.zeros(50)
                graph_max_lift = np.zeros(50)

            prev_deciles = np.zeros(5, dtype=float)
            prev_parity = np.zeros(2, dtype=float)
            for idx in prev_draw:
                prev_deciles[idx // 10] += 1
                prev_parity[(idx + 1) % 2] += 1

            for idx in range(50):
                number = idx + 1
                hits = hit_history[idx]

                gap1 = t - hits[-1] if len(hits) >= 1 else cfg.gap_cap
                gap2 = t - hits[-2] if len(hits) >= 2 else cfg.gap_cap
                gap3 = t - hits[-3] if len(hits) >= 3 else cfg.gap_cap
                gap1 = min(gap1, cfg.gap_cap)
                gap2 = min(gap2, cfg.gap_cap)
                gap3 = min(gap3, cfg.gap_cap)

                if len(hits) >= 2:
                    recent_hits = hits[-6:]
                    gaps = np.diff(recent_hits)
                    mean_gap_5 = float(gaps.mean()) if len(gaps) else 10.0
                else:
                    mean_gap_5 = 10.0

                same_decile_prev = prev_deciles[idx // 10] / 5.0
                same_parity_prev = prev_parity[number % 2] / 5.0

                if len(prev_draw):
                    prev_values = prev_draw + 1
                    min_distance_prev = float(np.min(np.abs(prev_values - number))) / 49.0
                    mean_distance_prev = float(np.mean(np.abs(prev_values - number))) / 49.0
                else:
                    min_distance_prev = 1.0
                    mean_distance_prev = 0.5

                row = {
                    "date": draws.at[t, "date"],
                    "draw_index": t,
                    "number": number,
                    "target": int(y[t, idx]),
                    "number_scaled": (number - 25.5) / 14.43,
                    "number_parity": float(number % 2),
                    "number_decile": float(idx // 10),
                    "gap1": gap1 / 10.0,
                    "gap2": gap2 / 20.0,
                    "gap3": gap3 / 30.0,
                    "gap_ratio": gap1 / max(mean_gap_5, 1.0),
                    "mean_gap_5": mean_gap_5 / 10.0,
                    "appeared_prev1": float(y[t - 1, idx]) if t >= 1 else 0.0,
                    "appeared_prev2": float(y[t - 2:t, idx].sum() > 0) if t >= 2 else 0.0,
                    "appeared_prev3": float(y[max(0, t - 3):t, idx].sum() > 0),
                    "appearances_prev5": float(prev5_counts[idx]) / 5.0,
                    "momentum_5_50": frequency_features["freq_5"][idx] - frequency_features["freq_50"][idx],
                    "momentum_10_100": frequency_features["freq_10"][idx] - frequency_features["freq_100"][idx],
                    "momentum_20_200": frequency_features["freq_20"][idx] - frequency_features["freq_200"][idx],
                    "same_decile_prev": same_decile_prev,
                    "same_parity_prev": same_parity_prev,
                    "min_distance_prev": min_distance_prev,
                    "mean_distance_prev": mean_distance_prev,
                    "graph_probability": float(graph_probability[idx]),
                    "graph_lift": float(graph_lift[idx]),
                    "graph_max_lift": float(graph_max_lift[idx]),
                }

                for name, arr in frequency_features.items():
                    row[name] = float(arr[idx])

                for h in EWMA_HALFLIVES:
                    row[f"ewma_{h}"] = float(ewma[h][idx])

                rows.append(row)

        # Update state after feature construction for draw t.
        current = y[t].astype(float)

        for idx in np.flatnonzero(current):
            hit_history[idx].append(t)

        for h in EWMA_HALFLIVES:
            a = ewma_alpha[h]
            ewma[h] = (1.0 - a) * ewma[h] + a * current

        if t >= 1:
            sources = np.flatnonzero(y[t - 1])
            targets = np.flatnonzero(y[t])
            for s in sources:
                source_counts[s] += 1.0
                transition_counts[s, targets] += 1.0

    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "number_scaled",
    "number_parity",
    "number_decile",
    "gap1",
    "gap2",
    "gap3",
    "gap_ratio",
    "mean_gap_5",
    "appeared_prev1",
    "appeared_prev2",
    "appeared_prev3",
    "appearances_prev5",
    "momentum_5_50",
    "momentum_10_100",
    "momentum_20_200",
    "same_decile_prev",
    "same_parity_prev",
    "min_distance_prev",
    "mean_distance_prev",
    "graph_probability",
    "graph_lift",
    "graph_max_lift",
    "freq_5",
    "zfreq_5",
    "freq_10",
    "zfreq_10",
    "freq_20",
    "zfreq_20",
    "freq_50",
    "zfreq_50",
    "freq_100",
    "zfreq_100",
    "freq_200",
    "zfreq_200",
    "freq_all",
    "zfreq_all",
    "ewma_5",
    "ewma_10",
    "ewma_20",
    "ewma_50",
]
