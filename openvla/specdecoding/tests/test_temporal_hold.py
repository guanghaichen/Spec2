import unittest

from openvla.specdecoding.model.temporal_hold import (
    decide_temporal_hold,
    normalize_temporal_hold_policy,
    settle_extension_debt,
    temporal_hold_action_scale,
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
        self.assertEqual(
            normalize_temporal_hold_policy("visual-budget"), "visual_budget"
        )
        self.assertEqual(normalize_temporal_hold_policy("paced-budget"), "paced_budget")
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

    def test_visual_budget_uses_drift_without_exact_action_run(self):
        accepted = self._decide(
            policy="visual_budget",
            consecutive_holds=1,
            verified_action_run_length=1,
            anchor_pixel_relative_l2=0.03,
            adaptive_max_anchor_pixel_relative_l2=0.05,
        )
        self.assertTrue(accepted.allow)
        self.assertEqual(accepted.reason, "visual_budget_extension")

        rejected = self._decide(
            policy="visual_budget",
            consecutive_holds=1,
            verified_action_run_length=1,
            anchor_pixel_relative_l2=0.051,
            adaptive_max_anchor_pixel_relative_l2=0.05,
        )
        self.assertFalse(rejected.allow)
        self.assertEqual(rejected.reason, "anchor_visual_drift")

    def test_fixed_policy_keeps_existing_budget_semantics(self):
        decision = self._decide(
            policy="fixed",
            consecutive_holds=1,
            verified_action_run_length=1,
            anchor_pixel_relative_l2=None,
        )
        self.assertTrue(decision.allow)
        self.assertEqual(decision.reason, "fixed_budget")

    def test_paced_budget_requires_repaid_extension_credit(self):
        accepted = self._decide(
            policy="paced_budget",
            consecutive_holds=1,
            extension_budget_available=True,
        )
        self.assertTrue(accepted.allow)
        self.assertEqual(accepted.reason, "visual_budget_extension")

        rejected = self._decide(
            policy="paced_budget",
            consecutive_holds=1,
            extension_budget_available=False,
        )
        self.assertFalse(rejected.allow)
        self.assertEqual(rejected.reason, "extension_debt")

        # The target immediately following T-H-H keeps the debt. After the
        # subsequent T-H interval, its target keyframe repays it.
        self.assertTrue(
            settle_extension_debt(
                policy="paced_budget", debt_active=True, holds_before_target=2
            )
        )
        self.assertFalse(
            settle_extension_debt(
                policy="paced_budget", debt_active=True, holds_before_target=1
            )
        )

    def test_paced_budget_repeats_two_one_hold_cadence(self):
        debt_active = False
        interval_holds = []
        for _ in range(4):
            holds = 0
            while True:
                decision = self._decide(
                    policy="paced_budget",
                    consecutive_holds=holds,
                    extension_budget_available=not debt_active,
                )
                if not decision.allow:
                    break
                holds += 1
                if decision.adaptive_extension:
                    debt_active = True
            interval_holds.append(holds)
            debt_active = settle_extension_debt(
                policy="paced_budget",
                debt_active=debt_active,
                holds_before_target=holds,
            )

        self.assertEqual(interval_holds, [2, 1, 2, 1])

    def test_inverse_age_decay_preserves_first_hold_and_damps_second(self):
        self.assertEqual(temporal_hold_action_scale("none", 2), 1.0)
        self.assertEqual(temporal_hold_action_scale("inverse_age", 1), 1.0)
        self.assertEqual(temporal_hold_action_scale("inverse_age", 2), 0.5)
        with self.assertRaises(ValueError):
            temporal_hold_action_scale("learned", 2)

if __name__ == "__main__":
    unittest.main()
