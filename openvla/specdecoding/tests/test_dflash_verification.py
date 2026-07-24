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
        self.assertEqual(normalize_dflash_tree_mode("single_fork"), "single_fork")
        with self.assertRaises(ValueError):
            normalize_dflash_tree_mode(True)

    def test_single_fork_tree_has_two_isolated_causal_paths(self):
        token_paths = torch.tensor(
            [
                [11, 12, 13, 14, 15, 16],
                [11, 22, 23, 24, 25, 26],
            ]
        )
        flat_tokens, path_nodes, tree_mask, relative_positions = (
            self.verifier._build_single_fork_tree(token_paths, branch_index=1)
        )

        self.assertEqual(flat_tokens.tolist(), [[11, 12, 13, 14, 15, 22, 23, 24, 25]])
        self.assertEqual(path_nodes.tolist(), [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8]])
        self.assertEqual(relative_positions.tolist(), [[0, 1, 2, 3, 4, 1, 2, 3, 4]])
        mask = tree_mask[0, 0]
        self.assertTrue(mask[8, 0])
        self.assertTrue(mask[8, 5])
        self.assertFalse(mask[8, 1])
        self.assertFalse(mask[4, 5])

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

    def test_tree_target_logits_match_two_linear_verifications(self):
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
        paths = torch.tensor(
            [
                [11, 12, 13, 14, 15, 16],
                [11, 22, 23, 24, 25, 26],
            ]
        )
        with torch.no_grad():
            prefix_outputs = target(input_ids=prefix, use_cache=True, return_dict=True)
        flat_tokens, path_nodes, tree_mask, relative_positions = (
            self.verifier._build_single_fork_tree(paths, branch_index=1)
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

        for path_index in range(2):
            with torch.no_grad():
                linear_outputs = target(
                    input_ids=paths[path_index : path_index + 1, :-1],
                    past_key_values=prefix_outputs.past_key_values,
                    position_ids=torch.arange(3, 8).unsqueeze(0),
                    use_cache=True,
                    return_dict=True,
                )
            tree_path_logits = tree_outputs.logits[0].index_select(
                0,
                path_nodes[path_index],
            )
            torch.testing.assert_close(
                tree_path_logits,
                linear_outputs.logits[0],
                rtol=2e-4,
                atol=2e-4,
            )


if __name__ == "__main__":
    unittest.main()
