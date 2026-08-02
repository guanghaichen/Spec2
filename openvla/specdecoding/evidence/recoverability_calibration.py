"""Pure configuration and risk-selection logic for embodied speculation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Optional

from scipy.stats import beta

from specdecoding.evidence.temporal_schedule_design import (
    extremal_gap_orders,
    target_steps,
)


@dataclass(frozen=True)
class RecoveryConfiguration:
    name: str
    schedule_kind: str
    schedule_period: int
    schedule_target_count: int
    schedule_offsets: tuple[int, ...]
    schedule_phase: float
    authority_exponent: float
    max_consecutive_holds: int
    depth_visual_bounds: tuple[Optional[float], ...] = tuple()

    @property
    def target_density(self) -> float:
        return self.schedule_target_count / self.schedule_period

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["target_density"] = self.target_density
        return payload


def enumerate_temporal_control_family(
    *,
    schedule_resolution: int,
    max_hold_depth: int,
    min_target_density: float,
    max_target_density: float,
    authority_exponents: Iterable[float] = (0.0, 1.0),
) -> tuple[RecoveryConfiguration, ...]:
    """Enumerate budget-derived schedules crossed with an authority grid.

    ``schedule_resolution`` is shared across suites. Each admissible target
    count induces a nearest-uniform gap multiset; we retain its two canonical
    extremal orders when they differ.  No suite-specific schedule is encoded.
    """
    if schedule_resolution < 1 or max_hold_depth < 0:
        raise ValueError(
            "schedule_resolution must be positive and max_hold_depth non-negative."
        )
    if not 0.0 < min_target_density <= max_target_density <= 1.0:
        raise ValueError("Target-density range must lie in (0, 1].")

    configurations = []
    period = int(schedule_resolution)
    minimum_count = max(1, math.ceil(min_target_density * period - 1e-12))
    maximum_count = min(period, math.floor(max_target_density * period + 1e-12))
    for target_count in range(minimum_count, maximum_count + 1):
        low_discrepancy, max_concentration = extremal_gap_orders(
            period, target_count
        )
        orders = (
            ("minimum_prefix_discrepancy", low_discrepancy),
            ("maximum_gap_concentration", max_concentration),
        )
        distinct_orders = []
        seen_offsets = set()
        for schedule_kind, intervals in orders:
            offsets = tuple(target_steps(intervals, period))
            if offsets in seen_offsets:
                continue
            seen_offsets.add(offsets)
            if max(intervals) - 1 > int(max_hold_depth):
                continue
            distinct_orders.append((schedule_kind, intervals, offsets))

        for schedule_kind, intervals, offsets in distinct_orders:
            max_depth = max(intervals) - 1
            kind_label = (
                "ld" if schedule_kind == "minimum_prefix_discrepancy" else "mc"
            )
            for exponent in sorted(set(float(value) for value in authority_exponents)):
                if exponent < 0.0:
                    raise ValueError("Authority exponents must be non-negative.")
                exponent_label = str(exponent).replace(".", "p")
                name = (
                    f"m{target_count}of{period}-{kind_label}"
                    f"-p{exponent_label}"
                )
                configurations.append(
                    RecoveryConfiguration(
                        name=name,
                        schedule_kind=schedule_kind,
                        schedule_period=period,
                        schedule_target_count=target_count,
                        schedule_offsets=offsets,
                        schedule_phase=-1.0,
                        authority_exponent=exponent,
                        max_consecutive_holds=max_depth,
                        depth_visual_bounds=(None,) * max_depth,
                    )
                )
    return tuple(configurations)


def bonferroni_clopper_pearson_upper(
    *, harm_count: int, sample_count: int, family_size: int, alpha: float
) -> float:
    """One-sided exact binomial upper bound with family-wise correction."""
    if sample_count <= 0 or family_size <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("Invalid sample count, family size, or alpha.")
    if not 0 <= harm_count <= sample_count:
        raise ValueError("harm_count must lie in [0, sample_count].")
    corrected_alpha = alpha / family_size
    if harm_count == sample_count:
        return 1.0
    return float(
        beta.ppf(
            1.0 - corrected_alpha,
            harm_count + 1,
            sample_count - harm_count,
        )
    )


def select_lowest_cost_feasible(
    rows: list[dict], *, risk_budget: float, alpha: float
) -> tuple[dict, list[dict]]:
    """Attach uniform risk bounds and select the lowest-cost feasible row."""
    if not rows:
        raise ValueError("No calibration rows supplied.")
    # The deterministic target reference is always feasible by construction and
    # is not a data-selected hypothesis.  Only candidate policies consume the
    # family-wise error budget.
    family_size = sum(
        not bool(row.get("deterministic_reference", False)) for row in rows
    )
    if family_size <= 0:
        family_size = 1
    evaluated = []
    for row in rows:
        enriched = dict(row)
        if bool(row.get("deterministic_reference", False)):
            enriched["harm_upper_bound"] = 0.0
        else:
            enriched["harm_upper_bound"] = bonferroni_clopper_pearson_upper(
                harm_count=int(row["harm_count"]),
                sample_count=int(row["paired_episodes"]),
                family_size=family_size,
                alpha=alpha,
            )
        enriched["feasible"] = enriched["harm_upper_bound"] <= risk_budget
        evaluated.append(enriched)
    feasible = [row for row in evaluated if row["feasible"]]
    if not feasible:
        raise RuntimeError("No configuration satisfies the uniform risk bound.")
    selected = min(
        feasible,
        key=lambda row: (
            float(row["mean_target_rate"]),
            -float(row["success_rate"]),
            str(row["configuration"]),
        ),
    )
    return selected, evaluated
