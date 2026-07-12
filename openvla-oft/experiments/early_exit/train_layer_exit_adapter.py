"""Train the OpenVLA-OFT layer-exit residual adapter from frozen teacher features."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from experiments.early_exit.adapter import LayerExitAdapterConfig, LayerExitResidualAdapter
from experiments.early_exit.feature_store import TeacherFeatureDataset
from prismatic.models.action_heads import L1RegressionActionHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--action-head-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bottleneck-size", type=int, default=512)
    parser.add_argument("--mixer-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.05)
    parser.add_argument("--action-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


def load_action_head(path: str, hidden_size: int, device: torch.device) -> L1RegressionActionHead:
    action_head = L1RegressionActionHead(input_dim=hidden_size, hidden_dim=hidden_size, action_dim=7)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    action_head.load_state_dict(state_dict, strict=True)
    action_head.to(device=device, dtype=torch.bfloat16).eval()
    for parameter in action_head.parameters():
        parameter.requires_grad_(False)
    return action_head


def mean_across_ranks(value: torch.Tensor, world_size: int) -> float:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return float(value.item())


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_distributed()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    dataset = TeacherFeatureDataset(args.feature_file)
    metadata = dataset.metadata
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    config = LayerExitAdapterConfig(
        hidden_size=int(metadata["hidden_size"]),
        num_action_tokens=int(metadata["num_action_tokens"]),
        bottleneck_size=args.bottleneck_size,
        num_mixer_layers=args.mixer_layers,
        num_attention_heads=args.attention_heads,
        early_exit_layer=int(metadata["early_exit_layer"]),
    )
    adapter = LayerExitResidualAdapter(config).to(device)
    trainable = adapter
    if world_size > 1:
        trainable = DistributedDataParallel(adapter, device_ids=[device.index], broadcast_buffers=False)

    action_head = load_action_head(args.action_head_checkpoint, config.hidden_size, device)
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        trainable.train()
        sums = torch.zeros(4, device=device)
        steps = 0
        for batch in loader:
            early = batch["early_hidden"].to(device, non_blocking=True)
            final = batch["final_hidden"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predicted = trainable(early)
                hidden_loss = F.smooth_l1_loss(predicted.float(), final.float())
                cosine_loss = 1.0 - F.cosine_similarity(predicted.float(), final.float(), dim=-1).mean()
                with torch.no_grad():
                    teacher_actions = action_head(final.to(dtype=torch.bfloat16)).float()
                predicted_actions = action_head(predicted.to(dtype=torch.bfloat16)).float()
                action_loss = F.smooth_l1_loss(predicted_actions, teacher_actions)
                loss = (
                    args.hidden_weight * hidden_loss
                    + args.cosine_weight * cosine_loss
                    + args.action_weight * action_loss
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), args.grad_clip)
            optimizer.step()
            sums += torch.stack((loss.detach(), hidden_loss.detach(), cosine_loss.detach(), action_loss.detach()))
            steps += 1

        epoch_metrics = sums / max(steps, 1)
        epoch_metrics = torch.tensor([mean_across_ranks(item, world_size) for item in epoch_metrics], device=device)
        if rank == 0:
            record = {
                "epoch": epoch,
                "loss": float(epoch_metrics[0]),
                "hidden_loss": float(epoch_metrics[1]),
                "cosine_loss": float(epoch_metrics[2]),
                "action_loss": float(epoch_metrics[3]),
            }
            print(json.dumps(record), flush=True)
            with metrics_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            adapter.save_pretrained(output_dir / f"epoch_{epoch:03d}")
            adapter.save_pretrained(output_dir / "latest")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
