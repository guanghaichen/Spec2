import unittest

from openvla.specdecoding.model.temporal_hold import (
    decide_temporal_hold,
    normalize_temporal_hold_policy,
)


class TemporalHoldPolicyTest(unittest.TestCase):
    def _decide(self, **overrides):
        values = {
            "policy": "adaptive",
            "base_eligible": True,
            "consecutive_holds": 0,
            "max_consecutive_holds": 2,
            "verified_action_run_length": 2,
            "adaptive_min_verified_run": 2,
            "anchor_pixel_relative_l2": 0.01,
            "adaptive_max_anchor_pixel_relative_l2": 0.03,
        }
        values.update(overrides)
        return decide_temporal_hold(**values)

    def test_policy_aliases_do_not_change_legacy_default(self):
        self.assertEqual(normalize_temporal_hold_policy(None), "fixed")
        self.assertEqual(normalize_temporal_hold_policy("legacy"), "fixed")
        self.assertEqual(normalize_temporal_hold_policy("risk-bounded"), "adaptive")
        with self.assertRaises(ValueError):
            normalize_temporal_hold_policy("unbounded")

    def test_first_adaptive_hold_preserves_fast_path(self):
        decision = self._decide(
            verified_action_run_length=1,
            anchor_pixel_relative_l2=0.9,
        )
        self.assertTrue(decision.allow)
        self.assertEqual(decision.hold_depth, 1)
        self.assertFalse(decision.adaptive_extension)

    def test_second_hold_requires_both_target_stability_and_visual_stability(self):
        accepted = self._decide(consecutive_holds=1)
        self.assertTrue(accepted.allow)
        self.assertTrue(accepted.adaptive_extension)
        self.assertEqual(accepted.hold_depth, 2)

        weak_history = self._decide(
            consecutive_holds=1, verified_action_run_length=1
        )
        self.assertFalse(weak_history.allow)
        self.assertEqual(weak_history.reason, "insufficient_verified_run")

        visual_drift = self._decide(
            consecutive_holds=1, anchor_pixel_relative_l2=0.031
        )
        self.assertFalse(visual_drift.allow)
        self.assertEqual(visual_drift.reason, "anchor_visual_drift")

    def test_target_is_forced_after_two_holds(self):
        decision = self._decide(consecutive_holds=2)
        self.assertFalse(decision.allow)
        self.assertEqual(decision.reason, "max_consecutive_reached")

    def test_fixed_policy_keeps_existing_budget_semantics(self):
        decision = self._decide(
            policy="fixed",
            consecutive_holds=1,
            verified_action_run_length=1,
            anchor_pixel_relative_l2=None,
        )
        self.assertTrue(decision.allow)
        self.assertEqual(decision.reason, "fixed_budget")


if __name__ == "__main__":
    unittest.main()
