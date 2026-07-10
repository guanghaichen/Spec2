"""Pack per-sample DFlash .ckpt files into one HDF5 file.

This is an IO maintenance tool. It preserves each sample dict semantically and only
changes the physical storage format from many small files to one .h5 container.
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from dflash_hdf5_utils import finalize_hdf5_file, init_hdf5_file, write_sample


def list_ckpt_files(input_dir: Path) -> List[Path]:
    files = []
    for root, _, names in os.walk(input_dir, followlinks=True):
        for name in names:
            if name.endswith(".ckpt"):
                files.append(Path(root) / name)
    files.sort()
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack DFlash per-sample ckpt files into one HDF5 file.")
    parser.add_argument("--input_dir", type=Path, required=True, help="旧版每样本 .ckpt 数据目录")
    parser.add_argument("--output_file", type=Path, required=True, help="输出 .h5 文件路径")
    parser.add_argument("--max_samples", type=int, default=None, help="只打包前 N 条，用于 smoke test")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 .h5 文件")
    parser.add_argument("--flush_every", type=int, default=32, help="每写多少条 flush 一次 HDF5；默认 32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.flush_every <= 0:
        raise ValueError("--flush_every must be > 0")
    input_dir = args.input_dir.resolve()
    output_file = args.output_file.resolve()
    files = list_ckpt_files(input_dir)
    if args.max_samples is not None:
        files = files[: args.max_samples]
    if not files:
        raise ValueError(f"No .ckpt files found in {input_dir}")

    h5 = init_hdf5_file(output_file, source=str(input_dir), overwrite=args.overwrite)
    try:
        samples_group = h5["samples"]
        for idx, file_path in enumerate(tqdm(files, desc="packing DFlash HDF5", unit="sample")):
            sample = torch.load(file_path, map_location="cpu")
            write_sample(samples_group, idx, sample, source_file=str(file_path))
            h5.attrs["num_samples"] = idx + 1
            if (idx + 1) % args.flush_every == 0:
                h5.flush()
        finalize_hdf5_file(h5, len(files))
    finally:
        h5.close()

    print(f"packed samples: {len(files)}")
    print(f"output file   : {output_file}")


if __name__ == "__main__":
    main()
