from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import normalize_ticket


class Strategy:
    name: str = "base"

    def pick(self, history_df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[int], list[int]]:
        raise NotImplementedError


@dataclass
class RandomStrategy(Strategy):
    name: str = "Random"

    def pick(self, history_df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[int], list[int]]:
        nums = rng.choice(np.arange(1, 51), size=5, replace=False).tolist()
        stars = rng.choice(np.arange(1, 13), size=2, replace=False).tolist()
        return normalize_ticket(nums, stars)


@dataclass
class FrequencyStrategy(Strategy):
    window_k: int = 200
    name: str = "Frequency"

    def pick(self, history_df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[int], list[int]]:
        if history_df.empty:
            return RandomStrategy().pick(history_df, rng)
        window = history_df.tail(self.window_k)

        num_counts = np.zeros(50, dtype=int)
        star_counts = np.zeros(12, dtype=int)

        for col in ["n1", "n2", "n3", "n4", "n5"]:
            values = window[col].to_numpy(dtype=int)
            np.add.at(num_counts, values - 1, 1)
        for col in ["s1", "s2"]:
            values = window[col].to_numpy(dtype=int)
            np.add.at(star_counts, values - 1, 1)

        nums = _top_k_with_random_tie_break(num_counts, 5, rng, offset=1)
        stars = _top_k_with_random_tie_break(star_counts, 2, rng, offset=1)
        return normalize_ticket(nums, stars)


@dataclass
class DecileBucketStrategy(Strategy):
    window_k: int = 200
    name: str = "DecileBucket"

    def pick(self, history_df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[int], list[int]]:
        if history_df.empty:
            return RandomStrategy().pick(history_df, rng)
        window = history_df.tail(self.window_k)

        num_counts = np.zeros(50, dtype=float)
        for col in ["n1", "n2", "n3", "n4", "n5"]:
            values = window[col].to_numpy(dtype=int)
            np.add.at(num_counts, values - 1, 1)

        bucket_weights = []
        buckets: list[np.ndarray] = []
        for b in range(5):
            start = b * 10
            end = start + 10
            bucket_vals = np.arange(start + 1, end + 1)
            buckets.append(bucket_vals)
            bucket_weights.append(num_counts[start:end].sum() + 1e-6)

        bucket_weights = np.array(bucket_weights, dtype=float)
        bucket_probs = bucket_weights / bucket_weights.sum()

        picks = set()
        attempts = 0
        while len(picks) < 5 and attempts < 200:
            b_idx = int(rng.choice(np.arange(5), p=bucket_probs))
            candidates = buckets[b_idx]
            c_weights = num_counts[candidates - 1] + 1e-6
            c_probs = c_weights / c_weights.sum()
            chosen = int(rng.choice(candidates, p=c_probs))
            picks.add(chosen)
            attempts += 1

        if len(picks) < 5:
            remain = [n for n in range(1, 51) if n not in picks]
            extras = rng.choice(remain, size=5 - len(picks), replace=False).tolist()
            picks.update(extras)

        # stars by frequency
        star_counts = np.zeros(12, dtype=int)
        for col in ["s1", "s2"]:
            values = window[col].to_numpy(dtype=int)
            np.add.at(star_counts, values - 1, 1)
        stars = _top_k_with_random_tie_break(star_counts, 2, rng, offset=1)
        return normalize_ticket(sorted(list(picks)), stars)


@dataclass
class BirthdayStrategy(Strategy):
    dates_input: list[pd.Timestamp] = field(default_factory=list)
    manual_numbers: list[int] = field(default_factory=list)
    use_year_mod: bool = False
    name: str = "Birthday"

    def pick(self, history_df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[int], list[int]]:
        number_candidates: set[int] = set()
        star_candidates: set[int] = set()

        for dt in self.dates_input:
            if pd.isna(dt):
                continue
            day = int(dt.day)
            month = int(dt.month)
            year_mod = int(dt.year % 50)
            if year_mod == 0:
                year_mod = 50

            if 1 <= day <= 50:
                number_candidates.add(day)
            if 1 <= month <= 50:
                number_candidates.add(month)
            if self.use_year_mod:
                number_candidates.add(year_mod)

            if 1 <= day <= 12:
                star_candidates.add(day)
            if 1 <= month <= 12:
                star_candidates.add(month)

        for n in self.manual_numbers:
            if 1 <= n <= 50:
                number_candidates.add(int(n))
            if 1 <= n <= 12:
                star_candidates.add(int(n))

        nums = sorted(number_candidates)[:5]
        stars = sorted(star_candidates)[:2]

        if len(nums) < 5:
            remaining = [n for n in range(1, 51) if n not in nums]
            nums += rng.choice(remaining, size=5 - len(nums), replace=False).tolist()
        if len(stars) < 2:
            remaining = [s for s in range(1, 13) if s not in stars]
            stars += rng.choice(remaining, size=2 - len(stars), replace=False).tolist()

        return normalize_ticket(nums[:5], stars[:2])


def _top_k_with_random_tie_break(counts: np.ndarray, k: int, rng: np.random.Generator, offset: int = 0) -> list[int]:
    noise = rng.random(len(counts)) * 1e-6
    score = counts.astype(float) + noise
    idx = np.argsort(-score)[:k]
    return (idx + offset).tolist()
