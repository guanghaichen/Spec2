"""Single-file HDF5 storage for OpenVLA-OFT layer-exit teacher features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TeacherFeatureWriter:
    """Append teacher pairs without creating a large number of small files."""

    def __init__(
        self,
        path: str | Path,
        *,
        num_action_tokens: int,
        hidden_size: int,
        metadata: Mapping[str, Any],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w")
        shape = (0, num_action_tokens, hidden_size)
        maxshape = (None, num_action_tokens, hidden_size)
        # One sample per chunk keeps appends robust and avoids an explosion of
        # filesystem objects; training opens this single HDF5 file per rank.
        self.early = self.file.create_dataset(
            "early_hidden", shape=shape, maxshape=maxshape, dtype=np.float16, chunks=(1, num_action_tokens, hidden_size)
        )
        self.final = self.file.create_dataset(
            "final_hidden", shape=shape, maxshape=maxshape, dtype=np.float16, chunks=(1, num_action_tokens, hidden_size)
        )
        self.actions = self.file.create_dataset(
            "teacher_actions", shape=(0, 8, 7), maxshape=(None, 8, 7), dtype=np.float16, chunks=(1, 8, 7)
        )
        self.file.attrs["metadata_json"] = json.dumps(dict(metadata), ensure_ascii=False)
        self.file.attrs["num_samples"] = 0

    def append(self, features: Mapping[str, torch.Tensor], actions: np.ndarray) -> int:
        early = features["early_hidden"].detach().float().cpu().numpy().astype(np.float16, copy=False)
        final = features["final_hidden"].detach().float().cpu().numpy().astype(np.float16, copy=False)
        if early.shape[0] != 1 or final.shape != early.shape:
            raise ValueError("Teacher feature writer currently expects batch size 1 and matching hidden tensors")
        actions = np.asarray(actions, dtype=np.float16)
        if actions.shape != (8, 7):
            raise ValueError(f"Expected OFT action chunk [8, 7], got {actions.shape}")
        index = self.early.shape[0]
        for dataset, value in ((self.early, early), (self.final, final), (self.actions, actions[None])):
            dataset.resize(index + 1, axis=0)
            dataset[index] = value[0]
        self.file.attrs["num_samples"] = index + 1
        self.file.flush()
        return index + 1

    @property
    def count(self) -> int:
        return int(self.early.shape[0])

    def close(self) -> None:
        self.file.close()


class TeacherFeatureDataset(Dataset):
    """Lazily open a single HDF5 file in each DataLoader worker/rank."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as file:
            self.length = int(file["early_hidden"].shape[0])
            self.metadata = json.loads(file.attrs["metadata_json"])

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file = self._get_file()
        return {
            "early_hidden": torch.from_numpy(np.asarray(file["early_hidden"][index], dtype=np.float32)),
            "final_hidden": torch.from_numpy(np.asarray(file["final_hidden"][index], dtype=np.float32)),
            "teacher_actions": torch.from_numpy(np.asarray(file["teacher_actions"][index], dtype=np.float32)),
        }
