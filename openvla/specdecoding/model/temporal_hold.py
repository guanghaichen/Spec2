"""Pure decision logic for bounded temporal action holds.

The module intentionally has no dependency on the VLA model or CUDA.  It only
decides whether a previously target-verified action may be reused before the
expensive multimodal prefill starts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


def settle_extension_debt(
    *, policy: str, debt_active: bool, holds_before_target: int
) -> bool:
    """Update paced-budget debt when a real target keyframe is completed."""
    if normalize_temporal_hold_policy(policy) != "paced_budget":
        return bool(debt_active)
    if debt_active and int(holds_before_target) < 2:
        return False
    return bool(debt_active)


def normalize_temporal_hold_policy(value) -> str:
    """Normalize launcher aliases while keeping the legacy policy as default."""
    normalized = str(value or "fixed").strip().lower()
    aliases = {
        "fixed": "fixed",
        "legacy": "fixed",
        "adaptive": "adaptive",
        "risk_bounded": "adaptive",
        "risk-bounded": "adaptive",
        "visual_budget": "visual_budget",
        "visual-budget": "visual_budget",
        "paced_budget": "paced_budget",
        "paced-budget": "paced_budget",
        "calibrated": "calibrated",
        "risk_calibrated": "calibrated",
        "risk-calibrated": "calibrated",
    }
    if normalized not in aliases:
        raise ValueError(
            "dflash_temporal_hold_policy must be 'fixed', 'adaptive', "
            "'visual_budget', 'paced_budget', or 'calibrated'."
        )
    return aliases[normalized]


@dataclass(frozen=True)
class TemporalHoldDecision:
    """Decision made before any target-model forward for the current action."""

    allow: bool
    reason: str
    hold_depth: int
    adaptive_extension: bool
    anchor_pixel_relative_l2: Optional[float]

    def as_record(self) -> dict:
        return {
            "allow": self.allow,
            "reason": self.reason,
            "hold_depth": self.hold_depth,
            "adaptive_extension": self.adaptive_extension,
            "anchor_pixel_relative_l2": self.anchor_pixel_relative_l2,
        }


def mechanical_target_due(
    *, control_step: int, period: int, target_count: int, phase: Optional[float] = None
) -> bool:
    """Return the minimum-prefix-discrepancy target decision at one step."""
    period = int(period)
    target_count = int(target_count)
    control_step = max(0, int(control_step))
    if period <= 0 or target_count <= 0 or target_count > period:
        raise ValueError("Require 0 < target_count <= period.")
    density = target_count / period
    phase = 1.0 - density if phase is None or float(phase) < 0.0 else float(phase)
    if not 0.0 <= phase < 1.0:
        raise ValueError("Mechanical-sequence phase must lie in [0, 1).")
    return math.floor((control_step + 1) * density + phase) > math.floor(
        control_step * density + phase
    )


def parse_target_offsets(value) -> tuple[int, ...]:
    """Parse one period's target-frame offsets from a comma-separated value."""
    normalized_value = "" if value is None else str(value).strip()
    if normalized_value.lower() in {"", "none", "null", "off", "mechanical"}:
        return tuple()
    offsets = tuple(
        int(item.strip()) for item in normalized_value.split(",") if item.strip()
    )
    if any(offset < 0 for offset in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("Target offsets must be distinct non-negative integers.")
    return tuple(sorted(offsets))


def periodic_target_due(
    *, control_step: int, period: int, target_offsets: tuple[int, ...]
) -> bool:
    """Return whether a calibrated periodic schedule calls the target now."""
    period = int(period)
    if period <= 0:
        raise ValueError("Schedule period must be positive.")
    offsets = tuple(int(value) for value in target_offsets)
    if not offsets:
        raise ValueError("A periodic schedule needs at least one target offset.")
    if offsets[0] < 0 or offsets[-1] >= period or len(set(offsets)) != len(offsets):
        raise ValueError("Target offsets must be unique and lie inside the period.")
    return max(0, int(control_step)) % period in set(offsets)


def parse_depth_visual_bounds(value) -> tuple[Optional[float], ...]:
    """Parse comma-separated per-depth visual bounds; `inf` disables a bound."""
    normalized_value = "" if value is None else str(value).strip()
    if normalized_value.lower() in {"", "none", "null", "off"}:
        return tuple()
    bounds = []
    for item in normalized_value.split(","):
        normalized = item.strip().lower()
        if normalized in {"inf", "infinity", "none", "off"}:
            bounds.append(None)
            continue
        bound = float(normalized)
        if bound < 0.0 or not math.isfinite(bound):
            raise ValueError("Finite visual bounds must be non-negative.")
        bounds.append(bound)
    return tuple(bounds)


def temporal_hold_action_scale(
    mode: str, hold_depth: int, exponent: float = 1.0
) -> float:
    """Return a bounded continuous-action scale for an aged held command."""
    normalized = str(mode or "none").strip().lower()
    if normalized == "none":
        return 1.0
    if normalized == "inverse_age":
        exponent = 1.0
    elif normalized != "power_law":
        raise ValueError(
            "dflash_temporal_hold_action_decay must be 'none', 'inverse_age', "
            "or 'power_law'."
        )
    exponent = float(exponent)
    if exponent < 0.0:
        raise ValueError("Power-law authority exponent must be non-negative.")
    return max(1, int(hold_depth)) ** (-exponent)


def decide_temporal_hold(
    *,
    policy: str,
    base_eligible: bool,
    consecutive_holds: int,
    max_consecutive_holds: int,
    verified_action_run_length: int,
    adaptive_min_verified_run: int,
    anchor_pixel_relative_l2: Optional[float],
    adaptive_max_anchor_pixel_relative_l2: float,
    extension_budget_available: bool = True,
    schedule_target_due: bool = False,
    calibrated_visual_bound: Optional[float] = None,
) -> TemporalHoldDecision:
    """Apply the fixed or risk-bounded hold policy.

    ``fixed`` exactly preserves the old behavior: every eligible hold is
    allowed until ``max_consecutive_holds`` is reached.

    ``adaptive`` preserves the first hold, but a second consecutive hold is
    allowed only when two independent online signals agree: multiple target
    keyframes produced the same complete action, and the current image remains
    close to the last target keyframe.  A third hold is never allowed.

    ``visual_budget`` is an isolated speed-oriented policy.  It preserves the
    first hold and spends a second hold only while cumulative image drift from
    the last target keyframe remains within a registered budget.  It does not
    reinterpret that hold as target-verified evidence, and still forces target
    after at most two holds.

    ``paced_budget`` keeps the same visual criterion but adds temporal debt:
    after spending a second hold, the next target interval may contain only one
    hold. This caps the long-run target cadence at T-H-H, T-H without relying
    on a learned confidence score.

    """
    policy = normalize_temporal_hold_policy(policy)
    consecutive_holds = max(0, int(consecutive_holds))
    hold_depth = consecutive_holds + 1

    if not base_eligible:
        return TemporalHoldDecision(
            False, "base_gate_rejected", hold_depth, False, anchor_pixel_relative_l2
        )
    if consecutive_holds >= int(max_consecutive_holds):
        return TemporalHoldDecision(
            False, "max_consecutive_reached", hold_depth, False, anchor_pixel_relative_l2
        )
    if policy == "calibrated" and schedule_target_due:
        return TemporalHoldDecision(
            False,
            "scheduled_regrounding",
            hold_depth,
            False,
            anchor_pixel_relative_l2,
        )

    # A configured per-depth bound always refers to the most recent Target
    # keyframe. This lets H1 and H2 share one causal anchor while retaining
    # different admissible drift budgets.
    if calibrated_visual_bound is not None:
        if anchor_pixel_relative_l2 is None:
            return TemporalHoldDecision(
                False,
                "missing_anchor_visual_signal",
                hold_depth,
                False,
                None,
            )
        if float(anchor_pixel_relative_l2) > float(calibrated_visual_bound):
            return TemporalHoldDecision(
                False,
                "anchor_visual_drift",
                hold_depth,
                False,
                anchor_pixel_relative_l2,
            )

    if policy == "calibrated":
        return TemporalHoldDecision(
            True,
            "calibrated_open_loop",
            hold_depth,
            hold_depth > 1,
            anchor_pixel_relative_l2,
        )
    if policy == "fixed":
        return TemporalHoldDecision(
            True, "fixed_budget", hold_depth, False, anchor_pixel_relative_l2
        )
    if hold_depth == 1:
        return TemporalHoldDecision(
            True, "base_hold", hold_depth, False, anchor_pixel_relative_l2
        )
    if hold_depth > 2:
        return TemporalHoldDecision(
            False, "adaptive_hard_limit", hold_depth, False, anchor_pixel_relative_l2
        )
    if policy == "paced_budget" and not bool(extension_budget_available):
        return TemporalHoldDecision(
            False, "extension_debt", hold_depth, False, anchor_pixel_relative_l2
        )
    if (
        policy == "adaptive"
        and int(verified_action_run_length) < int(adaptive_min_verified_run)
    ):
        return TemporalHoldDecision(
            False, "insufficient_verified_run", hold_depth, False, anchor_pixel_relative_l2
        )
    if anchor_pixel_relative_l2 is None:
        return TemporalHoldDecision(
            False, "missing_anchor_visual_signal", hold_depth, False, None
        )
    effective_visual_bound = (
        float(calibrated_visual_bound)
        if calibrated_visual_bound is not None
        else float(adaptive_max_anchor_pixel_relative_l2)
    )
    if float(anchor_pixel_relative_l2) > effective_visual_bound:
        return TemporalHoldDecision(
            False, "anchor_visual_drift", hold_depth, False, anchor_pixel_relative_l2
        )
    reason = (
        "adaptive_extension"
        if policy == "adaptive"
        else "visual_budget_extension"
    )
    return TemporalHoldDecision(
        True, reason, hold_depth, True, anchor_pixel_relative_l2
    )
