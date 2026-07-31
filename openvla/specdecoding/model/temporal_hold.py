"""Pure decision logic for bounded temporal action holds.

The module intentionally has no dependency on the VLA model or CUDA.  It only
decides whether a previously target-verified action may be reused before the
expensive multimodal prefill starts.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    }
    if normalized not in aliases:
        raise ValueError(
            "dflash_temporal_hold_policy must be 'fixed', 'adaptive', "
            "'visual_budget', or 'paced_budget'."
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


def temporal_hold_action_scale(mode: str, hold_depth: int) -> float:
    """Return a bounded continuous-action scale for an aged held command."""
    normalized = str(mode or "none").strip().lower()
    if normalized == "none":
        return 1.0
    if normalized != "inverse_age":
        raise ValueError(
            "dflash_temporal_hold_action_decay must be 'none' or 'inverse_age'."
        )
    return 1.0 / max(1, int(hold_depth))


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
    if float(anchor_pixel_relative_l2) > float(
        adaptive_max_anchor_pixel_relative_l2
    ):
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
