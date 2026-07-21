#!/usr/bin/env python3
"""Four-process cold-read benchmark for legacy and packed DFlash HDF5."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dflash_hdf5_utils import DistributedBlockSampler, is_packed_hdf5
from train_dflash_libero_goal import (
    DataCollatorForOfflineDFlash,
    OfflineDFlashHDF5Dataset,
    OfflineDFlashPackedHDF5Dataset,
)


class LegacyIndependentBlockSampler(Sampler[int]):
    """The pre-v2 sampler, retained only for an honest A/B benchmark."""

    def __init__(self, size: int, replicas: int, rank: int, block_size: int, seed: int):
        self.size = size
        self.replicas = replicas
        self.rank = rank
        self.block_size = block_size
        self.seed = seed
        self.num_blocks = math.ceil(size / block_size)
        self.blocks_per_replica = math.ceil(self.num_blocks / replicas)
        self.num_samples = self.blocks_per_replica * block_size

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        blocks = [
            list(range(start, min(start + self.block_size, self.size)))
            for start in range(0, self.size, self.block_size)
        ]
        missing = self.block_size - len(blocks[-1])
        if missing:
            blocks[-1].extend(index % self.size for index in range(missing))
        random.Random(self.seed).shuffle(blocks)
        required = self.blocks_per_replica * self.replicas
        if len(blocks) < required:
            blocks.extend(blocks[: required - len(blocks)])
        rank_blocks = blocks[self.rank : required : self.replicas]
        return iter([index for block in rank_blocks for index in block])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--sampler", choices=["legacy", "super"], required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--benchmark_batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def tensor_bytes(batch: dict) -> int:
    return sum(value.numel() * value.element_size() for value in batch.values() if torch.is_tensor(value))


def main() -> None:
    args = parse_args()
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dataset_class = OfflineDFlashPackedHDF5Dataset if is_packed_hdf5(args.datapath) else OfflineDFlashHDF5Dataset
    dataset = dataset_class(
        args.datapath,
        expected_selected_layers=5,
        target_layer_ids=[1, 9, 16, 24, 31],
        selected_hidden_variant="target_layers",
    )
    if args.sampler == "super":
        sampler = DistributedBlockSampler(
            len(dataset), world_size, rank, args.block_size, shuffle=True, seed=args.seed
        )
    else:
        sampler = LegacyIndependentBlockSampler(
            len(dataset), world_size, rank, args.block_size, seed=args.seed
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        prefetch_factor=1 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=False,
        collate_fn=DataCollatorForOfflineDFlash(),
    )
    iterator = iter(loader)
    for _ in range(args.warmup_batches):
        next(iterator)
    dist.barrier()
    started = time.perf_counter()
    local_bytes = 0
    completed = 0
    for _ in range(args.benchmark_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        local_bytes += tensor_bytes(batch)
        completed += 1
    elapsed = torch.tensor(time.perf_counter() - started, dtype=torch.float64)
    byte_count = torch.tensor(local_bytes, dtype=torch.float64)
    batch_count = torch.tensor(completed, dtype=torch.int64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
    if rank == 0:
        global_samples = int(batch_count.item()) * args.batch_size
        seconds = float(elapsed.item())
        print(
            f"RESULT format={'packed_v2' if is_packed_hdf5(args.datapath) else 'legacy_v1'} "
            f"sampler={args.sampler} rank_block={args.block_size} "
            f"samples={global_samples} seconds={seconds:.3f} "
            f"samples_per_second={global_samples / seconds:.2f} "
            f"tensor_GiB_per_second={float(byte_count.item()) / 2**30 / seconds:.3f}",
            flush=True,
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
