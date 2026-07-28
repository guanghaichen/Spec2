import unittest

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.configuration_llama import LlamaConfig

from openvla.prismatic.extern.hf.modeling_speculation import (
    LlamaSpecForCausalLM,
    SpecVLAforActionPrediction,
    normalize_dflash_tree_mode,
    normalize_dflash_verify_skip_mode,
)


class DFlashVerificationTest(unittest.TestCase):
    def setUp(self):
        self.verifier = SpecVLAforActionPrediction.__new__(SpecVLAforActionPrediction)
        torch.nn.Module.__init__(self.verifier)
        self.verifier.dflash_acceptance_mode = "token"

    def test_tree_mode_normalizes_yaml_boolean_aliases(self):
        for value in (False, None, "off", "false", "0", "none", "disabled"):
            self.assertEqual(normalize_dflash_tree_mode(value), "off")
        self.assertEqual(normalize_dflash_tree_mode("ddtree"), "ddtree")
        self.assertEqual(normalize_dflash_tree_mode("single_fork"), "ddtree")
        with self.assertRaises(ValueError):
            normalize_dflash_tree_mode(True)

    def test_verify_skip_mode_distinguishes_route_from_active(self):
        self.assertEqual(normalize_dflash_verify_skip_mode("route"), "route")
        self.assertEqual(normalize_dflash_verify_skip_mode("active"), "active")
        with self.assertRaises(ValueError):
            normalize_dflash_verify_skip_mode("unsafe")

    @staticmethod
    def _tree_logits():
        return torch.tensor(
            [
                [4.0, 3.0, -4.0, -5.0],
                [0.0, 0.0, 0.0, -5.0],
                [5.0, 0.0, -2.0, -5.0],
                [5.0, 0.0, -2.0, -5.0],
            ],
            dtype=torch.float32,
        )

    def test_ddtree_best_first_builder_obeys_budget_and_ancestor_visibility(self):
        flat_tokens, child_maps, tree_mask, relative_positions, greedy_tokens = (
            self.verifier._build_ddtree_from_logits(
                self._tree_logits(), node_budget=5, token_id_offset=100
            )
        )

        self.assertEqual(flat_tokens.shape, (1, 5))
        self.assertEqual(len(child_maps), 6)
        self.assertEqual(greedy_tokens.tolist(), [[100, 100, 100, 100]])
        self.assertIn(100, child_maps[0])
        self.assertIn(101, child_maps[0])
        self.assertTrue(torch.all(relative_positions >= 0))
        self.assertTrue(torch.all(relative_positions < 4))
        mask = tree_mask[0, 0]
        self.assertTrue(torch.all(mask.diagonal()))
        root_children = list(child_maps[0].values())
        self.assertFalse(mask[root_children[0] - 1, root_children[1] - 1])
        self.assertFalse(mask[root_children[1] - 1, root_children[0] - 1])

    def test_ddtree_follows_target_tokens_instead_of_longest_path(self):
        _, child_maps, _, _, _ = self.verifier._build_ddtree_from_logits(
            self._tree_logits(), node_budget=5, token_id_offset=100
        )
        alternate_index = child_maps[0][101]
        node_posteriors = torch.full((1, 5), 177, dtype=torch.long)

        accepted_nodes, next_token = self.verifier._follow_ddtree_target_path(
            child_maps,
            root_posterior_token=torch.tensor([[101]]),
            node_posterior_tokens=node_posteriors,
            max_accept_length=4,
        )

        self.assertEqual(accepted_nodes.tolist(), [alternate_index - 1])
        self.assertEqual(next_token, 177)

    def test_ddtree_uses_target_correction_when_root_has_no_child(self):
        _, child_maps, _, _, _ = self.verifier._build_ddtree_from_logits(
            self._tree_logits(), node_budget=5, token_id_offset=100
        )

        accepted_nodes, next_token = self.verifier._follow_ddtree_target_path(
            child_maps,
            root_posterior_token=torch.tensor([[131]]),
            node_posterior_tokens=torch.zeros((1, 5), dtype=torch.long),
            max_accept_length=3,
        )

        self.assertEqual(accepted_nodes.numel(), 0)
        self.assertEqual(next_token, 131)

    def test_action_group_budget_extends_motion_but_not_gripper(self):
        proposed = torch.tensor([[100, 100, 100, 100, 100, 100]])
        posterior = torch.tensor([[111, 100, 100, 100, 100, 101]])
        positions = torch.tensor([[1, 2, 3, 4, 5, 6]])

        token_mask = self.verifier._compute_dflash_accept_mask(
            proposed,
            posterior,
            accept_threshold=9,
            action_position_ids=positions,
            acceptance_mode="token",
        )
        group_mask = self.verifier._compute_dflash_accept_mask(
            proposed,
            posterior,
            accept_threshold=9,
            action_position_ids=positions,
            acceptance_mode="action_group",
        )

        self.assertEqual(token_mask.tolist(), [[0, 1, 1, 1, 1, 1]])
        self.assertEqual(group_mask.tolist(), [[1, 1, 1, 1, 1, 0]])

    def test_temporal_prefix_certificate_requires_complete_exact_prefix(self):
        certified, accept_length = self.verifier._evaluate_temporal_prefix_certificate(
            torch.tensor([[100, 101, 102, 103]]),
            torch.tensor([[100, 101, 102, 103]]),
        )
        self.assertTrue(certified)
        self.assertEqual(accept_length, 4)

        certified, accept_length = self.verifier._evaluate_temporal_prefix_certificate(
            torch.tensor([[100, 101, 102, 103]]),
            torch.tensor([[100, 101, 199, 103]]),
        )
        self.assertFalse(certified)
        self.assertEqual(accept_length, 2)

    def test_relaxed_ddtree_selects_longest_group_valid_leaf_and_correction(self):
        # Virtual root -> token 100 -> token 101 is the longest group-valid
        # branch. Token 110 is a shorter sibling branch.
        child_maps = [
            {100: 1, 110: 2},
            {101: 3},
            {},
            {},
        ]
        verify_input_ids = torch.tensor([[100, 110, 101]])
        node_posteriors = torch.tensor([[110, 130, 120]])

        selected_nodes, correction, accept_length = (
            self.verifier._select_relaxed_ddtree_path(
                child_maps=child_maps,
                verify_input_ids=verify_input_ids,
                root_posterior_token=torch.tensor([[109]]),
                node_posterior_tokens=node_posteriors,
                action_start_position=0,
                max_action_tokens=3,
                accept_threshold=9,
            )
        )

        self.assertEqual(selected_nodes.tolist(), [0, 2])
        self.assertEqual(correction, 120)
        self.assertEqual(accept_length, 2)

    def test_tree_cache_gather_commits_only_winning_nodes(self):
        key = torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1)
        value = (100 + torch.arange(7, dtype=torch.float32)).view(1, 1, 7, 1)
        selected = self.verifier._select_tree_past_key_values(
            ((key, value),),
            base_length=2,
            tree_node_indices=torch.tensor([0, 3]),
        )

        self.assertEqual(selected[0][0].flatten().tolist(), [0.0, 1.0, 2.0, 5.0])
        self.assertEqual(selected[0][1].flatten().tolist(), [100.0, 101.0, 102.0, 105.0])
        self.assertEqual(self.verifier._past_key_values_length(selected), 4)

    def test_temporal_prefill_trie_shares_prefix_and_selects_better_branch(self):
        candidates = torch.tensor(
            [
                [100, 101, 102],
                [100, 101, 103],
                [100, 104, 105],
            ],
            dtype=torch.long,
        )
        flat_tokens, _, tree_mask, _, candidate_paths = (
            self.verifier._build_temporal_prefill_trie(candidates)
        )
        self.assertEqual(flat_tokens.shape[1], 3)
        self.assertEqual(candidate_paths[0].tolist(), candidate_paths[1].tolist())
        self.assertNotEqual(candidate_paths[0].tolist(), candidate_paths[2].tolist())
        self.assertTrue(torch.all(tree_mask[0, 0].diagonal()))

        node_posteriors = torch.zeros((1, flat_tokens.shape[1]), dtype=torch.long)
        shared_path = candidate_paths[0]
        node_posteriors[0, shared_path[0]] = 101
        node_posteriors[0, shared_path[1]] = 103
        selected = self.verifier._select_temporal_prefill_path(
            candidate_tokens=candidates,
            candidate_paths=candidate_paths,
            root_posterior_token=torch.tensor([[100]]),
            node_posterior_tokens=node_posteriors,
            accept_threshold=0,
        )

        self.assertEqual(selected["candidate_index"], 1)
        self.assertEqual(selected["accept_length"], 3)
        self.assertEqual(selected["candidate_accept_lengths"], [2, 3, 1])

    def test_temporal_prefill_candidates_use_continuous_velocity_and_deduplicate(self):
        self.verifier.dflash_temporal_prefill_tree_min_history = 2
        self.verifier.dflash_temporal_prefill_tree_max_candidates = 3
        self.verifier.vocab_size = 1000
        self.verifier.bin_centers = torch.linspace(-1.0, 1.0, 9).numpy()
        previous_indices = torch.tensor([2, 2, 2, 2, 2, 2, 0])
        latest_indices = torch.tensor([3, 3, 3, 3, 3, 3, 8])
        self.verifier._dflash_action_history_cpu = [
            1000 - previous_indices - 1,
            1000 - latest_indices - 1,
        ]

        candidates, sources = self.verifier._build_temporal_prefill_candidates(7)

        self.assertEqual(sources, ["hold", "constant_velocity", "recent"])
        expected_velocity_indices = torch.tensor([4, 4, 4, 4, 4, 4, 8])
        torch.testing.assert_close(candidates[1], 1000 - expected_velocity_indices - 1)

    def test_dynamic_tree_cache_gather_commits_only_winning_nodes(self):
        cache = DynamicCache()
        key = torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1)
        value = (100 + torch.arange(7, dtype=torch.float32)).view(1, 1, 7, 1)
        cache.update(key, value, layer_idx=0)

        selected = self.verifier._select_tree_past_key_values(
            cache,
            base_length=2,
            tree_node_indices=torch.tensor([0, 3]),
        )

        self.assertEqual(selected.key_cache[0].flatten().tolist(), [0.0, 1.0, 2.0, 5.0])
        self.assertEqual(
            selected.value_cache[0].flatten().tolist(),
            [100.0, 101.0, 102.0, 105.0],
        )
        self.assertEqual(self.verifier._past_key_values_length(selected), 4)

    def test_tree_target_logits_match_every_linear_leaf_path(self):
        torch.manual_seed(17)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        target = LlamaSpecForCausalLM(config, "eager").eval()
        prefix = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            prefix_outputs = target(input_ids=prefix, use_cache=True, return_dict=True)
        flat_tokens, child_maps, tree_mask, relative_positions, _ = (
            self.verifier._build_ddtree_from_logits(
                self._tree_logits(), node_budget=6, token_id_offset=10
            )
        )

        target.tree_mask = tree_mask
        try:
            with torch.no_grad():
                tree_outputs = target(
                    input_ids=flat_tokens,
                    past_key_values=prefix_outputs.past_key_values,
                    position_ids=prefix.shape[1] + relative_positions,
                    use_cache=True,
                    return_dict=True,
                )
        finally:
            target.tree_mask = None

        for path in self.verifier._enumerate_ddtree_leaf_paths(child_maps):
            tensor_indices = torch.tensor([node - 1 for node in path])
            path_tokens = flat_tokens.index_select(1, tensor_indices)
            with torch.no_grad():
                linear_outputs = target(
                    input_ids=path_tokens,
                    past_key_values=prefix_outputs.past_key_values,
                    position_ids=torch.arange(
                        prefix.shape[1], prefix.shape[1] + len(path)
                    ).unsqueeze(0),
                    use_cache=True,
                    return_dict=True,
                )
            tree_path_logits = tree_outputs.logits[0].index_select(
                0,
                tensor_indices,
            )
            torch.testing.assert_close(
                tree_path_logits,
                linear_outputs.logits[0],
                rtol=2e-4,
                atol=2e-4,
            )


if __name__ == "__main__":
    unittest.main()
