from types import SimpleNamespace
import unittest
import importlib.util
from pathlib import Path

import torch

from specdecoding.model.dflash import DFlashDraftModel, build_evenly_spaced_target_layer_ids


def load_training_module():
    script_path = Path(__file__).resolve().parents[1] / "train-scripts" / "train_dflash_libero_goal.py"
    spec = importlib.util.spec_from_file_location("train_dflash_libero_goal", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_config():
    return SimpleNamespace(
        hidden_size=32,
        intermediate_size=64,
        hidden_act="silu",
        pretraining_tp=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=3,
        max_position_embeddings=64,
        rope_theta=10000,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_target_layers=32,
        dflash_target_layer_ids=[1, 8, 15, 22, 29],
        dflash_selected_hidden_variant="target_layers",
        dflash_block_size=7,
        dflash_action_dim=7,
        dflash_causal_residual_type="none",
        dflash_logit_markov_type="none",
        dflash_action_head_type="slot_rnn",
        dflash_action_head_rank=8,
        dflash_action_token_start=96,
        dflash_action_vocab_size=32,
        dflash_action_confidence_enabled=True,
    )


class DFlashActionHeadTest(unittest.TestCase):
    def test_even_layer_selection_keeps_endpoints(self):
        self.assertEqual(
            build_evenly_spaced_target_layer_ids(32, num_feature_layers=5, first_layer_id=1),
            [1, 9, 16, 24, 31],
        )

    def test_teacher_forcing_and_inference_shapes(self):
        torch.manual_seed(7)
        model = DFlashDraftModel(tiny_config())
        batch_size, block_len = 2, 6
        base_logits = torch.randn(batch_size, block_len, 128, requires_grad=True)
        hidden = torch.randn(batch_size, block_len, 32, requires_grad=True)
        previous_tokens = torch.randint(96, 128, (batch_size, block_len))
        action_positions = torch.arange(block_len).unsqueeze(0).expand(batch_size, -1)

        action_logits, confidence_logits = model.apply_action_sequential_head(
            base_logits,
            hidden,
            previous_tokens,
            action_positions,
        )
        self.assertEqual(action_logits.shape, (batch_size, block_len, 32))
        self.assertEqual(confidence_logits.shape, (batch_size, block_len))
        # Residual head starts at zero, so initialization exactly preserves frozen lm_head logits.
        self.assertTrue(torch.equal(action_logits, model.action_logits_from_full(base_logits)))
        (action_logits.mean() + confidence_logits.mean()).backward()
        self.assertIsNotNone(model.action_head_out.weight.grad)

        sampled_tokens, sampled_logits, confidence = model.sample_action_block(
            base_logits.detach(),
            hidden.detach(),
            previous_tokens[:, :1],
            action_positions,
            confidence_threshold=0.0,
        )
        self.assertEqual(sampled_tokens.shape, (batch_size, block_len))
        self.assertEqual(sampled_logits.shape, (batch_size, block_len, 32))
        self.assertEqual(confidence.shape, (batch_size, block_len))
        self.assertTrue(torch.all(sampled_tokens >= 96))
        self.assertTrue(torch.all(sampled_tokens < 128))

    def test_new_even_layer_sample_does_not_require_duplicate_prompt_last(self):
        train_module = load_training_module()
        parser = train_module.OfflineDFlashSampleMixin()
        layer_ids = [1, 9, 16, 24, 31]
        parser._configure_format(5, layer_ids, "target_layers")
        data = {
            "dflash_data_format": "full_prefix_plus_action_hidden_v4",
            "predicted_tokens": torch.arange(7) + 31744,
            "hidden_state": {
                "prompt_selected": torch.randn(3, 5 * 8),
                "prompt_position_ids": torch.arange(3),
                "prompt_length": 3,
                "action_selected": torch.randn(6, 5 * 8),
                "action_last": torch.randn(6, 8),
                "layer_ids": layer_ids,
            },
        }
        sample = parser._format_sample(data, "memory")
        self.assertEqual(sample["prompt_selected"].shape, (3, 40))
        self.assertEqual(sample["target_hidden"].shape, (6, 8))

    def test_new_losses_are_finite_and_reach_action_head(self):
        train_module = load_training_module()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        model = DFlashDraftModel(tiny_config()).to(device=device, dtype=dtype)
        embed_tokens = torch.nn.Embedding(128, 32).to(device=device, dtype=dtype)
        lm_head = torch.nn.Linear(32, 128, bias=False).to(device=device, dtype=dtype)
        embed_tokens.requires_grad_(False)
        lm_head.requires_grad_(False)

        batch_size, prompt_len, action_hidden_len = 2, 3, 6
        tokens = torch.randint(96, 128, (batch_size, action_hidden_len + 1))
        batch = {
            "prompt_selected": torch.randn(batch_size, prompt_len, 5 * 32, dtype=dtype),
            "prompt_position_ids": torch.arange(prompt_len).unsqueeze(0).expand(batch_size, -1),
            "prompt_attention_mask": torch.ones(batch_size, prompt_len, dtype=torch.bool),
            "prompt_lengths": torch.full((batch_size,), prompt_len, dtype=torch.long),
            "action_selected": torch.randn(batch_size, action_hidden_len, 5 * 32, dtype=dtype),
            "target_hidden": torch.randn(batch_size, action_hidden_len, 32, dtype=dtype),
            "tokens": tokens,
            "lengths": torch.full((batch_size,), action_hidden_len, dtype=torch.long),
        }
        args = SimpleNamespace(
            include_anchor_hidden=True,
            block_size=7,
            mask_token_id=0,
            hidden_noise=0.0,
            causal_residual_type="none",
            causal_residual_start_index=0,
            logit_markov_type="none",
            slot_decay=1.0,
            position_balance=True,
            soft_w=0.0,
            soft_temperature=2.0,
            action_token_ce_w=0.1,
            action_distill_l1_w=0.9,
            action_distill_temperature=1.0,
            prefix_survival_w=0.5,
            action_confidence_w=0.1,
            hidden_loss_type="smooth_l1",
            hidden_w=0.3,
            cos_w=0.02,
            refined_hidden_w=0.0,
            refined_hidden_min_position=1,
            refined_hidden_max_position=6,
            refined_hidden_loss_type="smooth_l1",
            residual_token_ce_w=0.0,
            residual_token_ce_min_position=1,
            residual_token_ce_max_position=6,
            residual_token_ce_label_smoothing=0.0,
            anchor_consistency_w=0.0,
            anchor_consistency_type="cosine",
            causal_residual_cad_w=0.0,
            causal_residual_cad_correct_teacher_only=False,
            causal_residual_min_position=2,
            causal_residual_max_position=6,
            causal_residual_cad_type="cosine",
            anchor_logit_distill_w=0.0,
            anchor_logit_distill_temperature=2.0,
            anchor_logit_distill_min_position=2,
            anchor_logit_distill_max_position=6,
            anchor_logit_distill_correct_teacher_only=True,
        )
        metrics = train_module.compute_loss_and_accuracy(
            model,
            embed_tokens,
            lm_head,
            batch,
            args,
            device,
        )
        self.assertTrue(torch.isfinite(metrics["loss"]))
        self.assertGreater(metrics["action_token_ce_loss"].item(), 0.0)
        self.assertGreaterEqual(metrics["expected_prefix_length"].item(), 0.0)
        metrics["loss"].backward()
        self.assertIsNotNone(model.action_head_out.weight.grad)


if __name__ == "__main__":
    unittest.main()
