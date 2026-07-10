"""Pack per-sample DFlash .ckpt files into sequential shards.

This is an IO maintenance tool, not a data transformation: each sample dict is kept
unchanged. The resulting directory contains shard_*.pt files plus
`dflash_shards_manifest.json`, which train_dflash_libero_goal.py can read directly.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm


MANIFEST_NAME = "dflash_shards_manifest.json"


def list_ckpt_files(input_dir: Path) -> List[Path]:
    files = []
    for root, _, names in os.walk(input_dir, followlinks=True):
        for name in names:
            if name.endswith(".ckpt"):
                files.append(Path(root) / name)
    files.sort()
    return files


def write_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    tmp_path = output_dir / f".{MANIFEST_NAME}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_dir / MANIFEST_NAME)


def flush_shard(output_dir: Path, shard_idx: int, samples: List[Dict[str, Any]], source_files: List[str]) -> Dict[str, Any]:
    shard_name = f"shard_{shard_idx:06d}.pt"
    shard_path = output_dir / shard_name
    tmp_path = output_dir / f".{shard_name}.tmp"
    torch.save(
        {
            "format": "dflash_shard_v1",
            "dflash_data_format": "full_prefix_plus_action_hidden_v4",
            "source_files": source_files,
            "samples": samples,
        },
        tmp_path,
    )
    os.replace(tmp_path, shard_path)
    return {"file": shard_name, "count": len(samples)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack DFlash per-sample ckpt files into sequential shards.")
    parser.add_argument("--input_dir", type=Path, required=True, help="旧版每样本 .ckpt 数据目录")
    parser.add_argument("--output_dir", type=Path, default=None, help="新版 shard 输出目录；默认 input_dir + '_sharded'")
    parser.add_argument("--samples_per_shard", type=int, default=32, help="每个 shard 的样本数；默认 32")
    parser.add_argument("--max_samples", type=int, default=None, help="只打包前 N 条，用于 smoke test")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_shard <= 0:
        raise ValueError("--samples_per_shard must be > 0")
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or Path(str(input_dir) + "_sharded")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite if you really want to rebuild it")

    files = list_ckpt_files(input_dir)
    if args.max_samples is not None:
        files = files[: args.max_samples]
    if not files:
        raise ValueError(f"No .ckpt files found in {input_dir}")

    manifest: Dict[str, Any] = {
        "format": "dflash_shards_v1",
        "dflash_data_format": "full_prefix_plus_action_hidden_v4",
        "source_dir": str(input_dir),
        "num_samples": 0,
        "samples_per_shard": args.samples_per_shard,
        "complete": False,
        "shards": [],
    }
    write_manifest(output_dir, manifest)

    buffer: List[Dict[str, Any]] = []
    source_buffer: List[str] = []
    shard_idx = 0
    valid = 0
    for file_path in tqdm(files, desc="packing DFlash shards", unit="sample"):
        sample = torch.load(file_path, map_location="cpu")
        buffer.append(sample)
        source_buffer.append(str(file_path))
        valid += 1
        if len(buffer) >= args.samples_per_shard:
            shard_info = flush_shard(output_dir, shard_idx, buffer, source_buffer)
            manifest["shards"].append(shard_info)
            manifest["num_samples"] = valid
            write_manifest(output_dir, manifest)
            shard_idx += 1
            buffer = []
            source_buffer = []

    if buffer:
        shard_info = flush_shard(output_dir, shard_idx, buffer, source_buffer)
        manifest["shards"].append(shard_info)
        manifest["num_samples"] = valid
        write_manifest(output_dir, manifest)

    manifest["complete"] = True
    manifest["num_samples"] = valid
    write_manifest(output_dir, manifest)

    print(f"packed samples: {valid}")
    print(f"packed shards : {len(manifest['shards'])}")
    print(f"output dir    : {output_dir}")
    print(f"manifest      : {output_dir / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
