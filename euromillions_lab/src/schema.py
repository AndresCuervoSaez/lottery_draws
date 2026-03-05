from __future__ import annotations

from typing import Iterable

NUM_MIN = 1
NUM_MAX = 50
STAR_MIN = 1
STAR_MAX = 12


def normalize_ticket(nums: Iterable[int], stars: Iterable[int]) -> tuple[list[int], list[int]]:
    """Sort and cast ticket values."""
    nums_list = sorted(int(x) for x in nums)
    stars_list = sorted(int(x) for x in stars)
    return nums_list, stars_list


def validate_ticket(nums: Iterable[int], stars: Iterable[int]) -> bool:
    """Return True when ticket has correct cardinality, uniqueness and ranges."""
    nums_list, stars_list = normalize_ticket(nums, stars)

    if len(nums_list) != 5 or len(stars_list) != 2:
        return False
    if len(set(nums_list)) != 5 or len(set(stars_list)) != 2:
        return False
    if not all(NUM_MIN <= n <= NUM_MAX for n in nums_list):
        return False
    if not all(STAR_MIN <= s <= STAR_MAX for s in stars_list):
        return False
    return True
