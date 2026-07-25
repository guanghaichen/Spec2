import unittest

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.configuration_llama import LlamaConfig

from openvla.prismatic.extern.hf.modeling_speculation import (
    LlamaSpecForCausalLM,
    SpecVLAforActionPrediction,
    normalize_dflash_tree_mode,
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
