#!/usr/bin/env python3
"""Losslessly repack sample-group DFlash HDF5 into contiguous training arrays."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

from dflash_hdf5_utils import HDF5_FORMAT, PACKED_HDF5_FORMAT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="现有 dflash_hdf5_v1 文件")
    parser.add_argument("--output", required=True, help="新建 packed v2 文件")
    parser.add_argument("--copy_batch_size", type=int, default=16, help="每次顺序搬运的样本数")
    parser.add_argument("--progress_every", type=int, default=256, help="每多少个样本打印一次进度")
    parser.add_argument("--overwrite", action="store_true", help="删除既有输出和 .partial 后重建")
    return parser.parse_args()


def copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def scan_layout(source: h5py.File) -> dict:
    if str(source.attrs.get("format", "")) != HDF5_FORMAT:
        raise ValueError(f"input format must be {HDF5_FORMAT!r}, got {source.attrs.get('format')!r}")
    if not bool(source.attrs.get("complete", False)):
        raise RuntimeError("input HDF5 is not marked complete")

    samples = source["samples"]
    num_samples = int(source.attrs["num_samples"])
    first = samples["00000000"]
    first_hidden = first["hidden_state"]
    action_length, selected_dim = first_hidden["action_selected"].shape
    target_length, hidden_dim = first_hidden["action_last"].shape
    token_length = first["predicted_tokens"].shape[0]
    if action_length != target_length or token_length != action_length + 1:
        raise ValueError("first sample has inconsistent action/token lengths")

    prompt_offsets = np.zeros(num_samples + 1, dtype=np.int64)
    started = time.time()
    for index in range(num_samples):
        sample = samples[f"{index:08d}"]
        hidden = sample["hidden_state"]
        prompt_length = int(hidden.attrs["prompt_length"])
        prompt_shape = hidden["prompt_selected"].shape
        if prompt_shape != (prompt_length, selected_dim):
            raise ValueError(f"sample {index} prompt shape {prompt_shape} is inconsistent")
        if hidden["prompt_position_ids"].shape != (prompt_length,):
            raise ValueError(f"sample {index} prompt position shape is inconsistent")
        if hidden["action_selected"].shape != (action_length, selected_dim):
            raise ValueError(f"sample {index} action_selected shape is inconsistent")
        if hidden["action_last"].shape != (target_length, hidden_dim):
            raise ValueError(f"sample {index} action_last shape is inconsistent")
        if sample["predicted_tokens"].shape != (token_length,):
            raise ValueError(f"sample {index} token shape is inconsistent")
        prompt_offsets[index + 1] = prompt_offsets[index] + prompt_length

    return {
        "num_samples": num_samples,
        "prompt_offsets": prompt_offsets,
        "total_prompt_tokens": int(prompt_offsets[-1]),
        "selected_dim": int(selected_dim),
        "hidden_dim": int(hidden_dim),
        "action_length": int(action_length),
        "token_length": int(token_length),
        "scan_seconds": time.time() - started,
    }


def create_packed_file(path: Path, source: h5py.File, layout: dict) -> h5py.File:
    target = h5py.File(path, "w", libver="latest")
    copy_attrs(source.attrs, target.attrs)
    target.attrs["format"] = PACKED_HDF5_FORMAT
    target.attrs["complete"] = False
    target.attrs["packed_from"] = str(Path(source.filename).resolve())
    target.attrs["omitted_training_unused_fields"] = "input_ids,loss_mask"

    num_samples = layout["num_samples"]
    total_prompt_tokens = layout["total_prompt_tokens"]
    selected_dim = layout["selected_dim"]
    hidden_dim = layout["hidden_dim"]
    action_length = layout["action_length"]
    token_length = layout["token_length"]

    target.create_dataset("prompt_offsets", data=layout["prompt_offsets"])
    target.create_dataset(
        "prompt_selected",
        shape=(total_prompt_tokens, selected_dim),
        dtype=np.int16,
    )
    target.create_dataset(
        "prompt_position_ids",
        shape=(total_prompt_tokens,),
        dtype=np.int64,
    )
    target.create_dataset(
        "action_selected",
        shape=(num_samples, action_length, selected_dim),
        dtype=np.int16,
    )
    target.create_dataset(
        "action_last",
        shape=(num_samples, action_length, hidden_dim),
        dtype=np.int16,
    )
    target.create_dataset(
        "predicted_tokens",
        shape=(num_samples, token_length),
        dtype=np.int64,
    )
    layer_ids = source["samples/00000000/hidden_state/layer_ids"][()]
    target.create_dataset("layer_ids", data=layer_ids)
    for name in ("prompt_selected", "action_selected", "action_last"):
        target[name].attrs["torch_dtype"] = "bfloat16"
    target["prompt_position_ids"].attrs["torch_dtype"] = "int64"
    target["predicted_tokens"].attrs["torch_dtype"] = "int64"
    target["layer_ids"].attrs["torch_dtype"] = "int64"
    return target


def copy_data(source: h5py.File, target: h5py.File, layout: dict, args: argparse.Namespace) -> None:
    samples = source["samples"]
    offsets = layout["prompt_offsets"]
    num_samples = layout["num_samples"]
    started = time.time()
    for batch_start in range(0, num_samples, args.copy_batch_size):
        batch_end = min(batch_start + args.copy_batch_size, num_samples)
        groups = [samples[f"{index:08d}"] for index in range(batch_start, batch_end)]
        hidden_groups = [group["hidden_state"] for group in groups]

        prompt_start = int(offsets[batch_start])
        prompt_end = int(offsets[batch_end])
        target["prompt_selected"][prompt_start:prompt_end] = np.concatenate(
            [hidden["prompt_selected"][()] for hidden in hidden_groups], axis=0
        )
        target["prompt_position_ids"][prompt_start:prompt_end] = np.concatenate(
            [hidden["prompt_position_ids"][()] for hidden in hidden_groups], axis=0
        )
        target["action_selected"][batch_start:batch_end] = np.stack(
            [hidden["action_selected"][()] for hidden in hidden_groups], axis=0
        )
        target["action_last"][batch_start:batch_end] = np.stack(
            [hidden["action_last"][()] for hidden in hidden_groups], axis=0
        )
        target["predicted_tokens"][batch_start:batch_end] = np.stack(
            [group["predicted_tokens"][()] for group in groups], axis=0
        )

        if batch_end % args.progress_every == 0 or batch_end == num_samples:
            elapsed = max(time.time() - started, 1e-6)
            rate = batch_end / elapsed
            eta_minutes = (num_samples - batch_end) / max(rate, 1e-6) / 60.0
            target.flush()
            print(
                f"packed {batch_end}/{num_samples} samples "
                f"({100.0 * batch_end / num_samples:.2f}%), "
                f"{rate:.2f} samples/s, ETA {eta_minutes:.1f} min",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if args.copy_batch_size <= 0:
        raise ValueError("--copy_batch_size must be > 0")
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if input_path == output_path:
        raise ValueError("input and output must be different files")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
    if output_path.exists() or partial_path.exists():
        raise FileExistsError(f"output or partial file already exists: {output_path}")

    total_started = time.time()
    with h5py.File(input_path, "r") as source:
        layout = scan_layout(source)
        print(
            "layout: "
            f"samples={layout['num_samples']} "
            f"prompt_tokens={layout['total_prompt_tokens']} "
            f"selected_dim={layout['selected_dim']} "
            f"scan={layout['scan_seconds']:.1f}s",
            flush=True,
        )
        with create_packed_file(partial_path, source, layout) as target:
            copy_data(source, target, layout, args)
            target.attrs["complete"] = True
            target.attrs["packing_seconds"] = time.time() - total_started
            target.flush()

    os.replace(partial_path, output_path)
    print(
        f"completed: {output_path} in {(time.time() - total_started) / 60.0:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
