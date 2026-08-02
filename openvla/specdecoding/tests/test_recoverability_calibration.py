import unittest

from specdecoding.evidence.recoverability_calibration import (
    bonferroni_clopper_pearson_upper,
    enumerate_temporal_control_family,
    select_lowest_cost_feasible,
)


class RecoverabilityCalibrationTest(unittest.TestCase):
    def test_family_is_derived_from_one_shared_budget_resolution(self):
        family = enumerate_temporal_control_family(
            schedule_resolution=10,
            max_hold_depth=2,
            min_target_density=0.3,
            max_target_density=0.5,
            authority_exponents=(0.0, 1.0),
        )
        densities = sorted({item.target_density for item in family})
        self.assertEqual(densities, [0.4, 0.5])
        self.assertEqual(len(family), 6)
        four_of_ten = [item for item in family if item.schedule_target_count == 4]
        self.assertEqual(
            {item.schedule_kind for item in four_of_ten},
            {"minimum_prefix_discrepancy", "maximum_gap_concentration"},
        )
        self.assertEqual(
            {item.schedule_offsets for item in four_of_ten},
            {(0, 3, 5, 8), (0, 3, 6, 8)},
        )

    def test_uniform_bound_tightens_with_more_pairs(self):
        small = bonferroni_clopper_pearson_upper(
            harm_count=0, sample_count=10, family_size=4, alpha=0.05
        )
        large = bonferroni_clopper_pearson_upper(
            harm_count=0, sample_count=100, family_size=4, alpha=0.05
        )
        self.assertGreater(small, large)

    def test_zero_harm_bound_matches_exact_binomial_formula(self):
        bound = bonferroni_clopper_pearson_upper(
            harm_count=0, sample_count=50, family_size=9, alpha=0.05
        )
        expected = 1.0 - (0.05 / 9.0) ** (1.0 / 50.0)
        self.assertAlmostEqual(bound, expected, places=12)

    def test_selector_uses_cost_only_inside_feasible_set(self):
        rows = [
            {
                "configuration": "reference",
                "harm_count": 0,
                "paired_episodes": 100,
                "mean_target_rate": 1.0,
                "success_rate": 0.9,
                "deterministic_reference": True,
            },
            {
                "configuration": "efficient",
                "harm_count": 0,
                "paired_episodes": 100,
                "mean_target_rate": 0.4,
                "success_rate": 0.9,
            },
        ]
        selected, _ = select_lowest_cost_feasible(
            rows, risk_budget=0.1, alpha=0.05
        )
        self.assertEqual(selected["configuration"], "efficient")


if __name__ == "__main__":
    unittest.main()
