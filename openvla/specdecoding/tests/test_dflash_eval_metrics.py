import unittest

from openvla.experiments.robot.libero.eval_metrics import (
    parse_tree_calibration_positions,
    select_tree_branch_position,
)


class DFlashEvalMetricsTest(unittest.TestCase):
    def test_calibration_positions_are_unique_and_include_off(self):
        self.assertEqual(parse_tree_calibration_positions("3,2,3,5"), [0, 3, 2, 5])

    def test_significant_full_action_speedup_selects_fastest_fork(self):
        samples = 64
        times = {
            0: [1.0] * samples,
            2: [0.91] * samples,
            3: [0.96] * samples,
            4: [1.02] * samples,
            5: [1.05] * samples,
        }
        triggered = {0: 0, 2: samples, 3: samples, 4: samples, 5: samples}
        selected, diagnostics = select_tree_branch_position(times, triggered)
        self.assertEqual(selected, 2)
        self.assertLess(diagnostics["candidates"]["2"]["sign_test_pvalue"], 0.0125)

    def test_untriggered_or_slower_forks_fall_back_to_off(self):
        samples = 64
        times = {0: [1.0] * samples, 2: [0.9] * samples, 3: [1.1] * samples}
        triggered = {0: 0, 2: 0, 3: samples}
        selected, _ = select_tree_branch_position(times, triggered)
        self.assertEqual(selected, 0)


if __name__ == "__main__":
    unittest.main()
