from types import SimpleNamespace
import unittest
from unittest import mock
import importlib.util
from pathlib import Path
import tempfile

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
        num_hidden_layers=1,
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
    def test_stage2_model_initialization_needs_no_training_state(self):
        train_module = load_training_module()
        source_model = DFlashDraftModel(tiny_config())
        target_model = DFlashDraftModel(tiny_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            stage1_root = Path(tmpdir) / "stage1_representation"
            checkpoint_dir = stage1_root / "epoch_100_step_001000"
            checkpoint_dir.mkdir(parents=True)
            torch.save(source_model.state_dict(), checkpoint_dir / "pytorch_model.bin")
            (stage1_root / "latest_checkpoint.txt").write_text(
                str(checkpoint_dir),
                encoding="utf-8",
            )
            resolved = train_module.resolve_model_init_checkpoint(str(stage1_root))
            self.assertEqual(resolved, checkpoint_dir)
            self.assertFalse((checkpoint_dir / "training_state.pt").exists())
            train_module.load_model_initialization(resolved, target_model, torch.device("cpu"))
            for source_parameter, target_parameter in zip(
                source_model.parameters(),
                target_model.parameters(),
            ):
                self.assertTrue(torch.equal(source_parameter, target_parameter))

    def test_stage1_freezes_only_action_head_parameters(self):
        train_module = load_training_module()
        model = DFlashDraftModel(tiny_config())
        train_module.configure_training_phase_trainability(model, "representation")
        for name, parameter in model.named_parameters():
            if train_module.is_action_head_parameter(name):
                self.assertFalse(parameter.requires_grad, name)
            else:
                self.assertTrue(parameter.requires_grad, name)
        train_module.configure_training_phase_trainability(model, "refinement")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_independent_two_stage_controls_are_fixed_and_final_only(self):
        train_module = load_training_module()
        common = dict(
            unified_cosine_curriculum=False,
            staged_training=False,
            hidden_w=1.0,
            cos_w=0.05,
            soft_w=0.05,
            backbone_anchor_logit_distill_w=0.05,
            action_token_ce_w=0.05,
            action_distill_l1_w=0.10,
            prefix_survival_w=0.05,
            action_confidence_w=0.0,
            anchor_logit_distill_w=0.0,
        )
        stage1 = train_module.build_training_control(
            SimpleNamespace(training_phase="representation", **common),
            epoch=50,
            global_step=500,
            total_optimizer_steps=1000,
        )
        self.assertEqual(stage1["base_scale"], 1.0)
        self.assertEqual(stage1["final_scale"], 0.0)
        self.assertEqual(stage1["weights"]["hidden_w"], 1.0)
        self.assertEqual(stage1["weights"]["cos_w"], 0.05)
        for name in train_module.STAGED_LOSS_WEIGHT_NAMES:
            if name not in ("hidden_w", "cos_w"):
                self.assertEqual(stage1["weights"][name], 0.0)

        stage2 = train_module.build_training_control(
            SimpleNamespace(training_phase="refinement", **common),
            epoch=1,
            global_step=0,
            total_optimizer_steps=1000,
        )
        self.assertEqual(stage2["base_scale"], 0.0)
        self.assertEqual(stage2["final_scale"], 1.0)
        self.assertEqual(stage2["weights"]["soft_w"], 0.05)
        self.assertEqual(stage2["weights"]["action_token_ce_w"], 0.05)
        self.assertEqual(stage2["weights"]["backbone_anchor_logit_distill_w"], 0.05)

    def test_unified_cosine_curriculum_is_smooth_and_uses_one_clock(self):
        train_module = load_training_module()
        args = SimpleNamespace(
            unified_cosine_curriculum=True,
            staged_training=False,
            hidden_w=1.0,
            cos_w=0.05,
            soft_w=0.05,
            backbone_anchor_logit_distill_w=0.05,
            action_token_ce_w=0.10,
            action_distill_l1_w=0.40,
            prefix_survival_w=0.20,
            action_confidence_w=0.0,
            anchor_logit_distill_w=0.0,
            token_curriculum_power=4.0,
        )

        start = train_module.build_training_control(args, 1, 0, 100)
        middle = train_module.build_training_control(args, 100, 50, 100)
        end = train_module.build_training_control(args, 200, 100, 100)
        self.assertEqual(start["base_scale"], 1.0)
        self.assertEqual(start["final_scale"], 0.0)
        self.assertAlmostEqual(middle["base_scale"], 0.5)
        self.assertAlmostEqual(middle["final_scale"], 0.5)
        self.assertEqual(end["base_scale"], 0.0)
        self.assertEqual(end["final_scale"], 1.0)
        self.assertEqual(start["token_envelope"], 0.0)
        self.assertAlmostEqual(middle["token_envelope"], 0.0625)
        self.assertEqual(end["token_envelope"], 1.0)
        self.assertEqual(start["weights"]["soft_w"], 0.0)
        self.assertEqual(start["weights"]["backbone_anchor_logit_distill_w"], 0.0)
        self.assertEqual(start["weights"]["action_token_ce_w"], 0.0)
        self.assertAlmostEqual(middle["weights"]["soft_w"], 0.003125)
        self.assertAlmostEqual(
            middle["weights"]["backbone_anchor_logit_distill_w"],
            0.003125,
        )
        self.assertAlmostEqual(middle["weights"]["action_token_ce_w"], 0.00625)
        self.assertAlmostEqual(middle["weights"]["action_distill_l1_w"], 0.0125)
        self.assertEqual(end["weights"]["prefix_survival_w"], 0.20)

    def test_action_head_lr_uses_delayed_curriculum_clock(self):
        train_module = load_training_module()
        draft_parameter = torch.nn.Parameter(torch.ones(()))
        action_parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW(
            [
                {"params": [draft_parameter], "lr": 2e-5, "group_name": "draft_backbone"},
                {"params": [action_parameter], "lr": 5e-5, "group_name": "action_head"},
            ]
        )
        scheduler = train_module.build_scheduler(
            optimizer,
            total_steps=200,
            warmup_steps=10,
            warmup_ratio=0.03,
            action_warmup_steps=10,
            action_curriculum_power=4.0,
        )
        self.assertEqual(scheduler.get_last_lr()[1], 0.0)
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        self.assertGreater(scheduler.get_last_lr()[0], 0.0)
        self.assertLess(scheduler.get_last_lr()[1], 5e-5)
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        self.assertGreater(scheduler.get_last_lr()[1], 0.0)

    def test_three_stage_loss_schedule(self):
        train_module = load_training_module()
        args = SimpleNamespace(
            staged_training=True,
            stage2_start_epoch=21,
            stage3_start_epoch=101,
            stage_weight_ramp_epochs=5,
            hidden_w=1.0,
            cos_w=0.05,
            soft_w=0.05,
            backbone_anchor_logit_distill_w=0.05,
            action_token_ce_w=0.10,
            action_distill_l1_w=0.40,
            prefix_survival_w=0.20,
            action_confidence_w=0.0,
            anchor_logit_distill_w=0.0,
        )

        stage1 = train_module.build_training_stage(args, 20)
        self.assertEqual(stage1["id"], 1)
        self.assertEqual(stage1["weights"]["soft_w"], 0.0)
        self.assertEqual(stage1["weights"]["action_token_ce_w"], 0.0)

        stage2_start = train_module.build_training_stage(args, 21)
        self.assertEqual(stage2_start["id"], 2)
        self.assertAlmostEqual(stage2_start["weights"]["soft_w"], 0.01)
        self.assertAlmostEqual(
            stage2_start["weights"]["backbone_anchor_logit_distill_w"],
            0.01,
        )
        self.assertEqual(stage2_start["weights"]["action_token_ce_w"], 0.0)

        stage2_full = train_module.build_training_stage(args, 25)
        self.assertAlmostEqual(stage2_full["weights"]["soft_w"], 0.05)
        self.assertAlmostEqual(
            stage2_full["weights"]["backbone_anchor_logit_distill_w"],
            0.05,
        )

        stage3_start = train_module.build_training_stage(args, 101)
        self.assertEqual(stage3_start["id"], 3)
        self.assertAlmostEqual(stage3_start["weights"]["action_token_ce_w"], 0.02)
        self.assertAlmostEqual(stage3_start["weights"]["action_distill_l1_w"], 0.08)
        self.assertAlmostEqual(stage3_start["weights"]["prefix_survival_w"], 0.04)
        self.assertEqual(stage3_start["weights"]["anchor_logit_distill_w"], 0.0)

        stage3_full = train_module.build_training_stage(args, 105)
        self.assertAlmostEqual(stage3_full["weights"]["action_token_ce_w"], 0.10)
        self.assertAlmostEqual(stage3_full["weights"]["action_distill_l1_w"], 0.40)
        self.assertAlmostEqual(stage3_full["weights"]["prefix_survival_w"], 0.20)

    def test_action_head_lr_starts_only_in_stage_three(self):
        train_module = load_training_module()
        draft_parameter = torch.nn.Parameter(torch.ones(()))
        action_parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW(
            [
                {"params": [draft_parameter], "lr": 2e-5, "group_name": "draft_backbone"},
                {"params": [action_parameter], "lr": 5e-5, "group_name": "action_head"},
            ]
        )
        scheduler = train_module.build_scheduler(
            optimizer,
            total_steps=200,
            warmup_steps=10,
            warmup_ratio=0.03,
            action_start_step=100,
            action_warmup_steps=10,
        )
        self.assertEqual(scheduler.get_last_lr()[1], 0.0)

        for _ in range(50):
            optimizer.step()
            scheduler.step()
        self.assertGreater(scheduler.get_last_lr()[0], 0.0)
        self.assertEqual(scheduler.get_last_lr()[1], 0.0)

        for _ in range(51):
            optimizer.step()
            scheduler.step()
        self.assertGreater(scheduler.get_last_lr()[1], 0.0)

    def test_gradient_clipping_uses_one_global_norm(self):
        train_module = load_training_module()
        draft_parameter = torch.nn.Parameter(torch.ones(()))
        action_parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW(
            [
                {"params": [draft_parameter], "lr": 2e-5, "group_name": "draft_backbone"},
                {"params": [action_parameter], "lr": 5e-5, "group_name": "action_head"},
            ]
        )
        draft_parameter.grad = torch.tensor(0.25)
        action_parameter.grad = torch.tensor(100.0)

        norms = train_module.clip_optimizer_gradients_global(optimizer, max_norm=0.5)

        self.assertAlmostEqual(norms["draft_backbone"].item(), 0.25, places=5)
        self.assertAlmostEqual(norms["action_head"].item(), 100.0, places=3)
        self.assertAlmostEqual(norms["global"].item(), 100.0003, places=3)
        self.assertLess(draft_parameter.grad.item(), 0.002)
        self.assertAlmostEqual(action_parameter.grad.item(), 0.5, places=5)

        draft_parameter.grad = torch.tensor(float("nan"))
        action_parameter.grad = None
        with self.assertRaises(RuntimeError):
            train_module.clip_optimizer_gradients_global(optimizer, max_norm=0.5)

    def test_inactive_action_group_has_no_update_or_adam_state(self):
        train_module = load_training_module()
        draft_parameter = torch.nn.Parameter(torch.ones(()))
        action_parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW(
            [
                {"params": [draft_parameter], "lr": 2e-5, "group_name": "draft_backbone"},
                {"params": [action_parameter], "lr": 5e-5, "group_name": "action_head"},
            ],
            weight_decay=0.1,
        )
        draft_before = draft_parameter.detach().clone()
        action_before = action_parameter.detach().clone()
        draft_parameter.grad = torch.ones_like(draft_parameter)
        action_parameter.grad = None

        train_module.clip_optimizer_gradients_global(optimizer, max_norm=0.5)
        optimizer.step()

        self.assertNotEqual(draft_parameter.item(), draft_before.item())
        self.assertEqual(action_parameter.item(), action_before.item())
        self.assertIn(draft_parameter, optimizer.state)
        self.assertNotIn(action_parameter, optimizer.state)

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

    def test_single_fork_reuses_main_path_and_rolls_alternate_suffix(self):
        torch.manual_seed(11)
        model = DFlashDraftModel(tiny_config())
        block_len = 6
        base_logits = torch.randn(1, block_len, 128)
        hidden = torch.randn(1, block_len, 32)
        first_token = torch.tensor([[101]])
        action_positions = torch.arange(block_len).unsqueeze(0)

        main_tokens, main_logits, _ = model.sample_action_block(
            base_logits,
            hidden,
            first_token,
            action_positions,
        )
        proposal = model.sample_action_tree(
            base_logits,
            hidden,
            first_token,
            action_positions,
            branch_index=2,
        )

        self.assertEqual(proposal.token_paths.shape, (2, block_len))
        self.assertTrue(torch.equal(proposal.token_paths[0], main_tokens[0]))
        self.assertTrue(torch.equal(proposal.main_logits, main_logits))
        self.assertTrue(torch.equal(proposal.token_paths[1, :2], main_tokens[0, :2]))
        self.assertNotEqual(
            proposal.token_paths[1, 2].item(),
            main_tokens[0, 2].item(),
        )
        self.assertEqual(proposal.branch_index, 2)
        self.assertIsNotNone(proposal.branch_score)

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
            detach_action_head_inputs=True,
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
            backbone_anchor_logit_distill_w=0.0,
            anchor_logit_distill_temperature=2.0,
            anchor_logit_distill_min_position=2,
            anchor_logit_distill_max_position=6,
            anchor_logit_distill_correct_teacher_only=True,
        )

        # 独立阶段一必须完全绕过 Action-RNN 的 teacher-forcing 与自回滚前向。
        args.training_phase = "representation"
        with mock.patch.object(
            model,
            "apply_action_sequential_head",
            side_effect=AssertionError("representation phase called Action-RNN"),
        ), mock.patch.object(
            model,
            "sample_action_block",
            side_effect=AssertionError("representation phase called Action-RNN rollout"),
        ):
            representation_metrics = train_module.compute_loss_and_accuracy(
                model,
                embed_tokens,
                lm_head,
                batch,
                args,
                device,
                loss_weights={
                    "soft_w": 0.0,
                    "backbone_anchor_logit_distill_w": 0.0,
                    "action_token_ce_w": 0.0,
                    "action_distill_l1_w": 0.0,
                    "prefix_survival_w": 0.0,
                    "action_confidence_w": 0.0,
                    "anchor_logit_distill_w": 0.0,
                },
            )
        representation_metrics["loss"].backward()
        self.assertIsNone(model.action_head_out.weight.grad)
        model.zero_grad(set_to_none=True)
        args.training_phase = "legacy"

        # 阶段一只计算 Draft hidden/cos；尚未入场的 raw loss 必须严格为 0。
        args.soft_w = 0.05
        args.backbone_anchor_logit_distill_w = 0.05
        stage1_metrics = train_module.compute_loss_and_accuracy(
            model,
            embed_tokens,
            lm_head,
            batch,
            args,
            device,
            loss_weights={
                "soft_w": 0.0,
                "backbone_anchor_logit_distill_w": 0.0,
                "action_token_ce_w": 0.0,
                "action_distill_l1_w": 0.0,
                "prefix_survival_w": 0.0,
                "action_confidence_w": 0.0,
                "anchor_logit_distill_w": 0.0,
            },
        )
        for metric_name in (
            "soft_loss",
            "backbone_anchor_logit_distill_loss",
            "action_token_ce_loss",
            "action_distill_l1_loss",
            "prefix_survival_loss",
        ):
            self.assertEqual(stage1_metrics[metric_name].item(), 0.0)
        stage1_metrics["loss"].backward()
        draft_grad = model.layers[0].self_attn.q_proj.weight.grad
        self.assertIsNotNone(draft_grad)
        self.assertGreater(draft_grad.abs().sum().item(), 0.0)
        action_grad = model.action_head_out.weight.grad
        self.assertIsNotNone(action_grad)
        self.assertEqual(torch.count_nonzero(action_grad).item(), 0)
        model.zero_grad(set_to_none=True)
        args.soft_w = 0.0
        args.backbone_anchor_logit_distill_w = 0.0

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
        self.assertGreaterEqual(metrics["rollout_accuracy"].item(), 0.0)
        self.assertGreaterEqual(metrics["rollout_top2_accuracy"].item(), metrics["rollout_accuracy"].item())
        self.assertGreaterEqual(metrics["base_rollout_expected_prefix_length"].item(), 0.0)
        self.assertGreaterEqual(metrics["rollout_expected_prefix_length"].item(), 0.0)
        self.assertGreaterEqual(metrics["base_distribution_overlap"].item(), 0.0)
        self.assertLessEqual(metrics["base_distribution_overlap"].item(), 1.0)
        self.assertGreaterEqual(metrics["rollout_distribution_overlap"].item(), 0.0)
        self.assertLessEqual(metrics["rollout_distribution_overlap"].item(), 1.0)
        self.assertGreaterEqual(metrics["base_expected_accept_length_proxy"].item(), 0.0)
        self.assertGreaterEqual(metrics["rollout_expected_accept_length_proxy"].item(), 0.0)
        prefix_rates = metrics["rollout_prefix_correct"] / metrics[
            "rollout_prefix_total"
        ].clamp_min(1.0)
        self.assertTrue(torch.all(prefix_rates[:-1] >= prefix_rates[1:]))
        self.assertAlmostEqual(
            prefix_rates.sum().item(),
            metrics["rollout_expected_prefix_length"].item(),
            places=5,
        )
        metrics["loss"].backward()
        self.assertIsNotNone(model.action_head_out.weight.grad)

        # With hidden/cos disabled, detached action objectives must update only
        # the Action-RNN instead of using the shared Draft hidden as a shortcut.
        model.zero_grad(set_to_none=True)
        args.hidden_w = 0.0
        args.cos_w = 0.0
        isolated_metrics = train_module.compute_loss_and_accuracy(
            model,
            embed_tokens,
            lm_head,
            batch,
            args,
            device,
        )
        isolated_metrics["loss"].backward()
        self.assertGreater(model.action_head_out.weight.grad.abs().sum().item(), 0.0)
        backbone_grad = model.layers[0].self_attn.q_proj.weight.grad
        self.assertTrue(
            backbone_grad is None or torch.count_nonzero(backbone_grad).item() == 0
        )

        # 独立阶段二的 Final CE 是联合目标：修正头和 Draft 主干必须同时收到梯度。
        model.zero_grad(set_to_none=True)
        args.training_phase = "refinement"
        args.unified_cosine_curriculum = False
        args.detach_action_head_inputs = True
        args.action_distill_l1_w = 0.0
        args.prefix_survival_w = 0.0
        args.action_confidence_w = 0.0
        joint_metrics = train_module.compute_loss_and_accuracy(
            model,
            embed_tokens,
            lm_head,
            batch,
            args,
            device,
            curriculum_state={"base_scale": 0.5, "final_scale": 0.5},
        )
        joint_metrics["loss"].backward()
        self.assertGreater(model.action_head_out.weight.grad.abs().sum().item(), 0.0)
        backbone_grad = model.layers[0].self_attn.q_proj.weight.grad
        self.assertIsNotNone(backbone_grad)
        self.assertGreater(backbone_grad.abs().sum().item(), 0.0)
        self.assertGreater(joint_metrics["base_action_token_ce_loss"].item(), 0.0)
        self.assertGreater(joint_metrics["action_token_ce_loss"].item(), 0.0)
        self.assertAlmostEqual(
            joint_metrics["token_curriculum_component"].item(),
            args.action_token_ce_w * joint_metrics["action_token_ce_loss"].item(),
            places=5,
        )

        # Base Soft 只训练 Draft；阶段二 Final Soft 沿联合图同时训练 Draft 与 RNN。
        model.zero_grad(set_to_none=True)
        args.action_token_ce_w = 0.0
        args.action_distill_l1_w = 0.0
        args.prefix_survival_w = 0.0
        args.action_confidence_w = 0.0
        args.soft_w = 0.05
        args.soft_loss_type = "kl"
        backbone_soft_metrics = train_module.compute_loss_and_accuracy(
            model,
            embed_tokens,
            lm_head,
            batch,
            args,
            device,
            curriculum_state={"base_scale": 1.0, "final_scale": 0.0},
        )
        backbone_soft_metrics["loss"].backward()
        backbone_grad = model.layers[0].self_attn.q_proj.weight.grad
        self.assertIsNotNone(backbone_grad)
        self.assertGreater(backbone_grad.abs().sum().item(), 0.0)
        action_head_grad = model.action_head_out.weight.grad
        self.assertTrue(
            action_head_grad is None or torch.count_nonzero(action_head_grad).item() == 0
        )
        self.assertGreater(backbone_soft_metrics["base_soft_loss"].item(), 0.0)

        model.zero_grad(set_to_none=True)
        final_soft_metrics = train_module.compute_loss_and_accuracy(
            model,
            embed_tokens,
            lm_head,
            batch,
            args,
            device,
            curriculum_state={"base_scale": 0.0, "final_scale": 1.0},
        )
        final_soft_metrics["loss"].backward()
        self.assertGreater(model.action_head_out.weight.grad.abs().sum().item(), 0.0)
        backbone_grad = model.layers[0].self_attn.q_proj.weight.grad
        self.assertIsNotNone(backbone_grad)
        self.assertGreater(backbone_grad.abs().sum().item(), 0.0)
        self.assertGreater(final_soft_metrics["final_soft_loss"].item(), 0.0)

        # The separate backbone cross-anchor KL must also train the Draft while
        # keeping the Action-RNN outside its gradient graph.
        model.zero_grad(set_to_none=True)
        args.soft_w = 0.0
        args.backbone_anchor_logit_distill_w = 0.05
        args.anchor_logit_distill_correct_teacher_only = False
        backbone_kl_metrics = train_module.compute_loss_and_accuracy(
            model,
            embed_tokens,
            lm_head,
            batch,
            args,
            device,
        )
        self.assertGreater(backbone_kl_metrics["backbone_anchor_logit_distill_pairs"].item(), 0.0)
        backbone_kl_metrics["loss"].backward()
        backbone_grad = model.layers[0].self_attn.q_proj.weight.grad
        self.assertIsNotNone(backbone_grad)
        self.assertGreater(backbone_grad.abs().sum().item(), 0.0)
        action_head_grad = model.action_head_out.weight.grad
        self.assertTrue(
            action_head_grad is None or torch.count_nonzero(action_head_grad).item() == 0
        )

    def test_swanlab_payload_is_chinese_and_filtered(self):
        train_module = load_training_module()
        args = SimpleNamespace(
            swanlab_log_all_metrics=False,
            swanlab_log_detail_metrics=True,
            swanlab_detail_every_steps=200,
        )
        payload = {
            "train/loss": 1.2,
            "train/hidden_loss": 0.8,
            "train/hidden_component": 0.8,
            "train/anchor_0_acc": 0.9,
            "train/rollout_position_2_acc": 0.7,
            "train/rollout_prefix_2_success_rate": 0.6,
            "train/rollout_expected_accept_length_proxy": 2.4,
            "train/curriculum_token_envelope": 0.0625,
            "train/lr": 1e-5,
        }
        swan_payload = train_module.build_swanlab_metrics_payload(
            payload,
            args,
            default_prefix="train",
            step=200,
        )
        self.assertEqual(swan_payload["训练损失/总损失"], 1.2)
        self.assertEqual(swan_payload["训练损失/主干Hidden损失"], 0.8)
        self.assertEqual(swan_payload["训练逐位置自回滚/P2命中率"], 0.7)
        self.assertEqual(swan_payload["训练连续前缀成功率/连续命中至少2步"], 0.6)
        self.assertEqual(swan_payload["训练推理代理/Action-RNN理论期望接受长度"], 2.4)
        self.assertEqual(swan_payload["训练课程/Token监督渐入比例"], 0.0625)
        self.assertEqual(swan_payload["训练优化器/Draft学习率"], 1e-5)
        self.assertNotIn("train/hidden_component", swan_payload)
        self.assertFalse(any("anchor_0" in key for key in swan_payload))


if __name__ == "__main__":
    unittest.main()
