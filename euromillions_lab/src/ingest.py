from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .schema import NUM_MAX, NUM_MIN, STAR_MAX, STAR_MIN
from .utils import normalize_column_name


@dataclass
class ColumnMapping:
    date_col: str
    number_cols: list[str]
    star_cols: list[str]


def load_excel_any(source: str | Path | BinaryIO, sheet_name: str | int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load excel and return selected sheet + available sheets."""
    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names
    selected = 0 if sheet_name is None else sheet_name
    df = pd.read_excel(xls, sheet_name=selected)
    return df, sheets


def auto_detect_mapping(df: pd.DataFrame) -> ColumnMapping | None:
    cols = list(df.columns)
    norm_map = {c: normalize_column_name(c) for c in cols}

    date_candidates = [
        c
        for c, nc in norm_map.items()
        if any(k in nc for k in ["fecha", "date", "draw_date", "sorteo"])
    ]
    date_col = date_candidates[0] if date_candidates else None

    number_cols = [
        c
        for c, nc in norm_map.items()
        if any(k in nc for k in ["n1", "n2", "n3", "n4", "n5", "num", "bola", "number"])
        and not any(k in nc for k in ["star", "estrella", "s1", "s2"])
    ]
    star_cols = [
        c
        for c, nc in norm_map.items()
        if any(k in nc for k in ["star", "estrella", "s1", "s2"])
    ]

    if len(number_cols) > 5:
        number_cols = number_cols[:5]
    if len(star_cols) > 2:
        star_cols = star_cols[:2]

    if date_col and len(number_cols) == 5 and len(star_cols) == 2:
        return ColumnMapping(date_col=date_col, number_cols=number_cols, star_cols=star_cols)
    return None


def normalize_draws(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    work = df[[mapping.date_col, *mapping.number_cols, *mapping.star_cols]].copy()
    work.columns = ["date", "n1", "n2", "n3", "n4", "n5", "s1", "s2"]

    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    for col in ["n1", "n2", "n3", "n4", "n5", "s1", "s2"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["date", "n1", "n2", "n3", "n4", "n5", "s1", "s2"])

    for col in ["n1", "n2", "n3", "n4", "n5", "s1", "s2"]:
        work[col] = work[col].astype(int)

    def is_valid_row(row: pd.Series) -> bool:
        nums = [row[f"n{i}"] for i in range(1, 6)]
        stars = [row["s1"], row["s2"]]
        if len(set(nums)) != 5 or len(set(stars)) != 2:
            return False
        if not all(NUM_MIN <= n <= NUM_MAX for n in nums):
            return False
        if not all(STAR_MIN <= s <= STAR_MAX for s in stars):
            return False
        return True

    work = work[work.apply(is_valid_row, axis=1)].copy()

    nums_sorted = work[["n1", "n2", "n3", "n4", "n5"]].to_numpy()
    nums_sorted.sort(axis=1)
    work[["n1", "n2", "n3", "n4", "n5"]] = nums_sorted

    stars_sorted = work[["s1", "s2"]].to_numpy()
    stars_sorted.sort(axis=1)
    work[["s1", "s2"]] = stars_sorted

    work = work.sort_values("date", ascending=True).reset_index(drop=True)
    work["date"] = pd.to_datetime(work["date"])
    return work


def save_normalized_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
