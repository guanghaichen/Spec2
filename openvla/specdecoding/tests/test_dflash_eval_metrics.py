import unittest

from openvla.experiments.robot.libero.eval_metrics import summarize_generation_stats


class DFlashEvalMetricsTest(unittest.TestCase):
    def test_dynamic_tree_metrics_are_aggregated_by_block_count(self):
        stats = [
            {
                "backend": "dflash",
                "block_size": 7,
                "num_blocks": 2,
                "generated_tokens": 7,
                "progressed_tokens": 5,
                "accept_lengths": [2, 1],
                "main_path_accept_lengths": [1, 1],
                "progress_lengths": [3, 2],
                "tree_mode": "ddtree",
                "tree_budget": 0,
                "tree_triggered_blocks": 2,
                "tree_selected_alternate_blocks": 1,
                "tree_extra_verified_nodes": 0,
                "tree_extra_accepted": 1,
                "tree_average_verified_nodes": 5.0,
                "tree_average_max_depth": 3.0,
            },
            {
                "backend": "dflash",
                "block_size": 7,
                "num_blocks": 1,
                "generated_tokens": 7,
                "progressed_tokens": 4,
                "accept_lengths": [3],
                "main_path_accept_lengths": [2],
                "progress_lengths": [4],
                "tree_mode": "ddtree",
                "tree_budget": 0,
                "tree_triggered_blocks": 1,
                "tree_selected_alternate_blocks": 1,
                "tree_extra_verified_nodes": 0,
                "tree_extra_accepted": 1,
                "tree_average_verified_nodes": 6.0,
                "tree_average_max_depth": 4.0,
            },
        ]

        summary = summarize_generation_stats(stats)

        self.assertEqual(summary["tree_mode"], "ddtree")
        self.assertEqual(summary["tree_triggered_blocks"], 3)
        self.assertEqual(summary["tree_selected_alternate_blocks"], 2)
        self.assertAlmostEqual(summary["tree_average_verified_nodes"], 16.0 / 3.0)
        self.assertAlmostEqual(summary["tree_average_max_depth"], 10.0 / 3.0)

    def test_temporal_prefill_fusion_metrics_keep_each_action_record(self):
        stats = [
            {
                "backend": "dflash",
                "block_size": 7,
                "num_blocks": 1,
                "generated_tokens": 7,
                "progressed_tokens": 7,
                "accept_lengths": [7],
                "progress_lengths": [7],
                "temporal_prefill_fusion_record": {
                    "accept_length": 7,
                    "progress_length": 7,
                    "full_match": True,
                },
            },
            {
                "backend": "dflash",
                "block_size": 7,
                "num_blocks": 2,
                "generated_tokens": 7,
                "progressed_tokens": 7,
                "accept_lengths": [2, 3],
                "progress_lengths": [3, 4],
                "temporal_prefill_fusion_record": {
                    "accept_length": 2,
                    "progress_length": 3,
                    "full_match": False,
                },
            },
        ]

        summary = summarize_generation_stats(stats)

        self.assertEqual(summary["temporal_prefill_fused_actions"], 2)
        self.assertEqual(summary["temporal_prefill_full_match_actions"], 1)
        self.assertAlmostEqual(summary["temporal_prefill_avg_accept_length"], 4.5)
        self.assertEqual(summary["temporal_prefill_accept_histogram"], {2: 1, 7: 1})

    def test_temporal_prefill_tree_reports_branch_coverage_and_gain(self):
        stats = [
            {
                "backend": "dflash",
                "block_size": 7,
                "num_blocks": 1,
                "generated_tokens": 7,
                "progressed_tokens": 7,
                "accept_lengths": [7],
                "progress_lengths": [7],
                "temporal_prefill_fusion_record": {
                    "mode": "temporal_tree",
                    "accept_length": 7,
                    "full_match": True,
                    "full_exact_match": False,
                    "candidate_count": 3,
                    "verified_node_count": 12,
                    "selected_alternate": True,
                    "extra_accepted_over_hold": 4,
                },
            }
        ]

        summary = summarize_generation_stats(stats)

        self.assertEqual(summary["temporal_prefill_tree_actions"], 1)
        self.assertEqual(summary["temporal_prefill_tree_full_exact_actions"], 0)
        self.assertEqual(summary["temporal_prefill_tree_selected_alternate_actions"], 1)
        self.assertEqual(summary["temporal_prefill_tree_extra_accepted"], 4)
        self.assertEqual(summary["temporal_prefill_tree_avg_candidates"], 3.0)
        self.assertEqual(summary["temporal_prefill_tree_avg_verified_nodes"], 12.0)


if __name__ == "__main__":
    unittest.main()
