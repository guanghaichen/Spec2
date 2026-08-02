"""Pure schedule primitives for paired temporal-control experiments."""

from __future__ import annotations

import math
from typing import Sequence


def balanced_gap_multiset(period: int, target_count: int) -> tuple[int, ...]:
    """Return the unique nearest-uniform integer gap multiset for a budget."""
    if period <= 0 or target_count <= 0 or target_count > period:
        raise ValueError("Require 0 < target_count <= period.")
    short_gap, long_count = divmod(int(period), int(target_count))
    return tuple(
        sorted(
            (short_gap + 1,) * long_count
            + (short_gap,) * (target_count - long_count),
            reverse=True,
        )
    )


def extremal_gap_orders(
    period: int, target_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive the low-discrepancy and maximally concentrated gap orders.

    Both orders use the same nearest-uniform integer gap multiset. The first
    minimizes the largest target-count prefix discrepancy. The second
    minimizes cyclic total variation, concentrating equal gaps into runs.
    Deterministic tie-breaking fixes rotation/reflection symmetries.
    """
    gaps = balanced_gap_multiset(period, target_count)
    density = target_count / period
    phase = 1.0 - density
    offsets = [
        step
        for step in range(period)
        if math.floor((step + 1) * density + phase)
        > math.floor(step * density + phase)
    ]
    if len(offsets) != target_count or offsets[0] != 0:
        raise RuntimeError("Failed to construct the canonical mechanical schedule.")
    low_discrepancy = tuple(
        offsets[index + 1] - offsets[index]
        for index in range(len(offsets) - 1)
    ) + (period - offsets[-1],)
    max_concentration = tuple(sorted(gaps, reverse=True))
    return low_discrepancy, max_concentration


def target_steps(intervals: Sequence[int], horizon: int) -> list[int]:
    """Return target-frame indices, always grounding the first control step."""
    if horizon <= 0:
        return []
    normalized = tuple(int(value) for value in intervals)
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("Target intervals must contain positive integers.")
    steps = [0]
    cursor = 0
    interval_index = 0
    while True:
        cursor += normalized[interval_index]
        interval_index = (interval_index + 1) % len(normalized)
        if cursor >= int(horizon):
            break
        steps.append(cursor)
    return steps


def target_indicator(intervals: Sequence[int], horizon: int) -> list[bool]:
    selected = set(target_steps(intervals, horizon))
    return [index in selected for index in range(max(0, int(horizon)))]


def power_law_authority_scale(*, exponent: float, hold_depth: int) -> float:
    """Return the continuous-action authority at one open-loop depth."""
    exponent = float(exponent)
    if exponent < 0.0:
        raise ValueError("Authority exponent must be non-negative.")
    return max(1, int(hold_depth)) ** (-exponent)


def exact_mcnemar_p(gains: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(int(gains), int(losses)) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)
