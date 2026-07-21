"""Single-file HDF5 storage helpers for DFlash offline data.

The HDF5 format keeps each sample as a group inside one physical file. BF16 tensors
are stored as int16 raw bits because h5py/numpy do not have native bfloat16.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Sampler

HDF5_FORMAT = "dflash_hdf5_v1"
HDF5_DATA_FORMAT = "full_prefix_plus_action_hidden_v4"
DEFAULT_HDF5_NAMES = ("dflash_goal_dataset.h5", "dflash_dataset.h5")


class DistributedBlockSampler(Sampler[int]):
    """Shuffle contiguous sample blocks while keeping reads sequential inside each block.

    A regular ``DistributedSampler(shuffle=True)`` turns a large HDF5 file into many
    concurrent random reads. This sampler changes only the sample order: every epoch
    shuffles physical blocks, splits the resulting stream evenly across ranks, and
    keeps indices inside each block contiguous. It therefore preserves epoch-level
    stochasticity without making every sample a separate disk seek.
    """

    def __init__(
        self,
        dataset_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        block_size: int = 16,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_blocks = int(math.ceil(self.dataset_size / self.block_size))
        self.blocks_per_replica = int(math.ceil(self.num_blocks / self.num_replicas))
        self.num_samples = self.blocks_per_replica * self.block_size

    def __iter__(self):
        blocks = [
            list(range(start, min(start + self.block_size, self.dataset_size)))
            for start in range(0, self.dataset_size, self.block_size)
        ]
        if len(blocks[-1]) < self.block_size:
            blocks[-1].extend(range(self.block_size - len(blocks[-1])))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(blocks)
        required_blocks = self.blocks_per_replica * self.num_replicas
        if len(blocks) < required_blocks:
            blocks.extend(blocks[: required_blocks - len(blocks)])
        rank_blocks = blocks[self.rank : required_blocks : self.num_replicas]
        return iter([index for block in rank_blocks for index in block])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def resolve_hdf5_path(path: str | Path) -> Optional[Path]:
    dataset_path = Path(path)
    if dataset_path.is_file() and dataset_path.suffix.lower() in {".h5", ".hdf5"}:
        return dataset_path
    if dataset_path.is_dir():
        for name in DEFAULT_HDF5_NAMES:
            candidate = dataset_path / name
            if candidate.exists():
                return candidate
    return None


def get_hdf5_sample_count(path: str | Path) -> int:
    hdf5_path = resolve_hdf5_path(path)
    if hdf5_path is None:
        raise FileNotFoundError(f"No DFlash HDF5 dataset found at {path}")
    with h5py.File(hdf5_path, "r") as f:
        if f.attrs.get("complete", False) is not True and str(f.attrs.get("complete", "False")) != "True":
            raise RuntimeError(f"{hdf5_path} is incomplete; wait until generation/packing finishes before training")
        return int(f.attrs["num_samples"])


def init_hdf5_file(path: str | Path, source: str, overwrite: bool = False) -> h5py.File:
    hdf5_path = Path(path)
    if hdf5_path.suffix.lower() not in {".h5", ".hdf5"}:
        hdf5_path.mkdir(parents=True, exist_ok=True)
        hdf5_path = hdf5_path / DEFAULT_HDF5_NAMES[0]
    else:
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    if hdf5_path.exists() and not overwrite:
        raise FileExistsError(f"{hdf5_path} already exists; pass overwrite=True to rebuild it")
    f = h5py.File(hdf5_path, "w")
    f.attrs["format"] = HDF5_FORMAT
    f.attrs["dflash_data_format"] = HDF5_DATA_FORMAT
    f.attrs["source"] = source
    f.attrs["complete"] = False
    f.attrs["num_samples"] = 0
    f.create_group("samples")
    return f


def finalize_hdf5_file(f: h5py.File, num_samples: int) -> None:
    f.attrs["num_samples"] = int(num_samples)
    f.attrs["complete"] = True
    f.flush()


def _normalise_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _write_tensor(group: h5py.Group, name: str, value: Any) -> None:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        array = tensor.view(torch.int16).numpy()
        torch_dtype = "bfloat16"
    else:
        array = tensor.numpy()
        torch_dtype = str(tensor.dtype).replace("torch.", "")
    ds = group.create_dataset(name, data=array)
    ds.attrs["torch_dtype"] = torch_dtype
    ds.attrs["shape"] = list(tensor.shape)


def _read_tensor(group: h5py.Group, name: str) -> torch.Tensor:
    ds = group[name]
    torch_dtype = _normalise_attr(ds.attrs.get("torch_dtype", ""))
    array = np.array(ds[()], copy=True)
    tensor = torch.from_numpy(array)
    if torch_dtype == "bfloat16":
        return tensor.view(torch.bfloat16).reshape(tuple(ds.attrs["shape"]))
    if torch_dtype == "float32":
        return tensor.float()
    if torch_dtype == "float16":
        return tensor.half()
    if torch_dtype == "float64":
        return tensor.double()
    if torch_dtype == "int64":
        return tensor.long()
    if torch_dtype == "int32":
        return tensor.int()
    if torch_dtype == "bool":
        return tensor.bool()
    return tensor


def write_sample(group_root: h5py.Group, index: int, sample: Dict[str, Any], source_file: Optional[str] = None) -> None:
    sample_group = group_root.create_group(f"{index:08d}")
    sample_group.attrs["dflash_data_format"] = sample.get("dflash_data_format", HDF5_DATA_FORMAT)
    if source_file is not None:
        sample_group.attrs["source_file"] = source_file

    for key in ("input_ids", "pixel_values", "loss_mask", "predicted_tokens"):
        if key in sample:
            _write_tensor(sample_group, key, sample[key])

    hidden = sample["hidden_state"]
    hidden_group = sample_group.create_group("hidden_state")
    for key in ("prompt_selected", "prompt_last", "prompt_position_ids"):
        if key in hidden:
            _write_tensor(hidden_group, key, hidden[key])
    hidden_group.attrs["prompt_length"] = int(hidden["prompt_length"])
    if "layer_ids" in hidden:
        _write_tensor(hidden_group, "layer_ids", torch.tensor(hidden["layer_ids"], dtype=torch.long))
    for key in ("action_selected", "action_last"):
        if key in hidden:
            value = hidden[key]
            if isinstance(value, list):
                value = torch.stack([torch.as_tensor(item) for item in value], dim=0)
            _write_tensor(hidden_group, key, value)


def read_sample(f: h5py.File, index: int) -> Dict[str, Any]:
    sample_group = f["samples"][f"{index:08d}"]
    sample: Dict[str, Any] = {
        "dflash_data_format": _normalise_attr(sample_group.attrs.get("dflash_data_format", HDF5_DATA_FORMAT))
    }
    for key in ("input_ids", "pixel_values", "loss_mask", "predicted_tokens"):
        if key in sample_group:
            sample[key] = _read_tensor(sample_group, key)

    hidden_group = sample_group["hidden_state"]
    hidden: Dict[str, Any] = {
        "prompt_length": int(hidden_group.attrs["prompt_length"]),
    }
    for key in ("prompt_selected", "prompt_last", "prompt_position_ids", "action_selected", "action_last"):
        if key in hidden_group:
            hidden[key] = _read_tensor(hidden_group, key)
    if "layer_ids" in hidden_group:
        hidden["layer_ids"] = _read_tensor(hidden_group, "layer_ids").long().tolist()
    sample["hidden_state"] = hidden
    return sample
