"""Frozen matched-budget design for the Spatial mechanism experiment.

This module belongs to the evidence scaffold, not to the online candidate
generator. Keeping the study design separate prevents its matched budget from
becoming an implicit method hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass

from specdecoding.evidence.temporal_schedule_design import extremal_gap_orders


FACTORIAL_PERIOD = 10
FACTORIAL_TARGET_COUNT = 4


@dataclass(frozen=True)
class TemporalCondition:
    name: str
    intervals: tuple[int, ...]
    authority_exponent: float

    @property
    def harmonic(self) -> bool:
        """Retain the label used by the archived evidence schema."""
        return self.authority_exponent == 1.0


LOW_DISCREPANCY_INTERVALS, MAX_CONCENTRATION_INTERVALS = extremal_gap_orders(
    FACTORIAL_PERIOD, FACTORIAL_TARGET_COUNT
)


PAIRED_CONDITIONS = (
    TemporalCondition("low_discrepancy_linear", LOW_DISCREPANCY_INTERVALS, 0.0),
    TemporalCondition("low_discrepancy_critical", LOW_DISCREPANCY_INTERVALS, 1.0),
    TemporalCondition("max_concentration_linear", MAX_CONCENTRATION_INTERVALS, 0.0),
    TemporalCondition("max_concentration_critical", MAX_CONCENTRATION_INTERVALS, 1.0),
)
