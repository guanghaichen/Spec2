import unittest

from specdecoding.evidence.temporal_schedule_design import (
    balanced_gap_multiset,
    exact_mcnemar_p,
    extremal_gap_orders,
    power_law_authority_scale,
    target_steps,
)


class TemporalScheduleDesignTest(unittest.TestCase):
    def test_extremal_orders_are_derived_from_one_budget(self):
        self.assertEqual(balanced_gap_multiset(10, 4), (3, 3, 2, 2))
        low_discrepancy, max_concentration = extremal_gap_orders(10, 4)
        self.assertEqual(low_discrepancy, (3, 2, 3, 2))
        self.assertEqual(max_concentration, (3, 3, 2, 2))
        self.assertEqual(
            sorted(low_discrepancy),
            sorted(max_concentration),
        )
        self.assertEqual(sum(low_discrepancy), 10)

    def test_extremal_orders_preserve_equal_target_budget(self):
        low, concentrated = extremal_gap_orders(10, 4)
        self.assertEqual(target_steps(low, 10), [0, 3, 5, 8])
        self.assertEqual(target_steps(concentrated, 10), [0, 3, 6, 8])

    def test_extremal_orders_scale_without_permutation_enumeration(self):
        low, concentrated = extremal_gap_orders(97, 40)
        self.assertEqual(len(low), 40)
        self.assertEqual(sum(low), 97)
        self.assertEqual(sorted(low), sorted(concentrated))
        self.assertLessEqual(max(low) - min(low), 1)

    def test_schedule_rejects_invalid_intervals(self):
        with self.assertRaises(ValueError):
            target_steps((3, 0), 10)

    def test_harmonic_authority_is_inverse_age(self):
        self.assertEqual(
            power_law_authority_scale(exponent=1.0, hold_depth=1), 1.0
        )
        self.assertEqual(
            power_law_authority_scale(exponent=1.0, hold_depth=2), 0.5
        )
        self.assertEqual(
            power_law_authority_scale(exponent=0.0, hold_depth=2), 1.0
        )

    def test_exact_mcnemar_uses_only_discordant_pairs(self):
        self.assertAlmostEqual(exact_mcnemar_p(3, 0), 0.25)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)

if __name__ == "__main__":
    unittest.main()
