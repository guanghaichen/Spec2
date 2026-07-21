import importlib.util
from pathlib import Path


UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "train-scripts"
    / "dflash_hdf5_utils.py"
)
SPEC = importlib.util.spec_from_file_location("dflash_hdf5_utils", UTILS_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
DistributedBlockSampler = MODULE.DistributedBlockSampler


def test_block_sampler_partitions_evenly_and_covers_dataset():
    samplers = [
        DistributedBlockSampler(
            dataset_size=103,
            num_replicas=4,
            rank=rank,
            block_size=16,
            seed=7,
        )
        for rank in range(4)
    ]
    rank_indices = [list(sampler) for sampler in samplers]

    assert {len(indices) for indices in rank_indices} == {32}
    flattened = [index for indices in rank_indices for index in indices]
    assert set(range(103)).issubset(flattened)
    assert len(flattened) == 128
    for indices in rank_indices:
        for block_start in range(0, len(indices), 16):
            block = indices[block_start : block_start + 16]
            assert all(
                right == left + 1 or (left == 102 and right == 0)
                for left, right in zip(block, block[1:])
            )


def test_block_sampler_is_reproducible_and_changes_each_epoch():
    sampler = DistributedBlockSampler(
        dataset_size=128,
        num_replicas=1,
        rank=0,
        block_size=16,
        seed=11,
    )
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    sampler.set_epoch(0)

    assert list(sampler) == epoch_zero
    assert epoch_one != epoch_zero
    assert all(
        right == left + 1
        for left, right in zip(epoch_zero[:15], epoch_zero[1:16])
    )
