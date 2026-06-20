import argparse
import copy
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main")

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig# 导入VLA配置类
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction# 导入VLA模型
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from openvla.specdecoding.model.dflash import DFlashDraftModel, build_target_layer_ids# 导入DFlash Draft模型与自适应选层函数

try:
    import swanlab
except ImportError:
    swanlab = None

# 超参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash draft on offline OpenVLA hidden states.")
    parser.add_argument(
        "--vla_path",
        type=str,
        default="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal",
        help="OpenVLA 目标模型权重路径（用于加载 embedding 和 lm_head 权重）",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset",
        help="离线数据路径（预计算的 .ckpt 文件）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash",
        help="输出目录（用于保存模型权重）",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="每张卡的 micro batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数；effective batch = batch_size * gradient_accumulation_steps")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--num_epochs", type=int, default=200, help="最大训练 epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="AdamW 学习率")
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="Adam beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="学习率 warmup 步数；<=0 时退回 warmup_ratio")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="当 warmup_steps<=0 时使用")
    parser.add_argument("--save_every", type=int, default=5, help="保存间隔epochs")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--block_size", type=int, default=7, help="块大小")
    parser.add_argument("--num_draft_layers", type=int, default=3, help="DFlash Draft 层数（原默认 5）")
    parser.add_argument("--target_layer_ids", type=int, nargs="*", default=[1, 8, 15, 22, 29], help="捕捉的 OpenVLA 目标层；留空时按 Draft 层数均匀选取，恢复 5 层时用 [1,8,15,22,29]")
    parser.add_argument("--mask_token_id", type=int, default=None, help="加噪声的 token ID，不指定也会自适应取pad_token_id")
    parser.add_argument("--hidden_w", type=float, default=0.03, help="hidden states 蒸馏权重")
    parser.add_argument("--ce_w", type=float, default=1.0, help="token CE 权重")
    parser.add_argument("--kl_w", type=float, default=0.05, help="teacher soft distribution KL 蒸馏权重")
    parser.add_argument("--kl_temperature", type=float, default=1.0, help="KL 蒸馏温度")
    parser.add_argument(
        "--hidden_loss_type",
        type=str,
        default="cosine",
        choices=["cosine", "norm_mse", "raw_mse"],
        help="hidden 蒸馏损失类型；默认 cosine，避免 raw MSE 过度约束 hidden 幅值",
    )
    parser.add_argument("--action_dim", type=int, default=7, help="OpenVLA action token 维度数，用于 action-dimension embedding")
    parser.add_argument("--hidden_noise", type=float, default=0.01, help="训练时 context hidden 加噪标准差（0=不加，推荐 0.02）")
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--log_every_steps", type=int, default=20, help="每多少个 optimizer step 记录一次训练日志")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="恢复训练；可传具体 checkpoint 目录，或 latest")
    parser.add_argument("--run_name", type=str, default=None, help="实验名；默认自动生成")
    parser.add_argument("--use_swanlab", action=argparse.BooleanOptionalAction, default=True, help="是否启用 SwanLab 记录")
    parser.add_argument("--swanlab_project", type=str, default="dflash-libero-goal", help="dflash_openvla")
    parser.add_argument("--swanlab_workspace", type=str, default=None, help="SwanLab workspace/org")
    parser.add_argument("--swanlab_mode", type=str, default="cloud", choices=["cloud", "local", "offline", "disabled"], help="SwanLab 模式")
    parser.add_argument("--refresh_file_cache", action="store_true", help="强制重新扫描数据目录并刷新 .ckpt 文件清单缓存")
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集比例，0 表示不划分验证集")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值（epoch 数）；验证 loss 不下降多少个 epoch 后停止")
    parser.add_argument("--eval_every", type=int, default=1, help="每隔多少个 epoch 进行一次验证")
    return parser.parse_args()

# 固定随机种子
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataset_file_cache_path(path: str) -> Path:
    return Path(path) / ".dflash_ckpt_index.json"


# 递归遍历 datapath 目录，找到所有 .ckpt 结尾的文件（离线数据），按文件名排序后返回
def list_files(path: str, refresh_cache: bool = False) -> List[str]:
    cache_path = get_dataset_file_cache_path(path)
    if not refresh_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cached_files = payload.get("files", [])
        if cached_files:
            return cached_files

    datapath = []
    for root, _, files in os.walk(path, followlinks=True):
        for file in files:
            if file.endswith(".ckpt"):
                datapath.append(os.path.join(root, file))
    datapath.sort()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"root": path, "count": len(datapath), "files": datapath}, f, ensure_ascii=False, indent=2)
    return datapath

# 数据集类
class OfflineDFlashDataset(Dataset):
    def __init__(self, datapath: List[str], expected_selected_layers: Optional[int] = None):
        self.data = datapath# 所有 .ckpt 文件的路径列表
        self.expected_selected_layers = expected_selected_layers# 期望的 selected layers 数量（用于校验数据兼容性）

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _collapse_step_hidden(step_hidden_list: List[torch.Tensor], key: str, file_path: str) -> torch.Tensor:
        """将每个解码步的 hidden 压成 [num_steps, hidden_dim]。

        当前 DFlash 训练只接受新版导出格式：每步已经是 [hidden_dim]，
        即“当前新生成 token”的 hidden，不再兼容旧的整段前缀快照格式。
        """
        if len(step_hidden_list) == 0:
            raise ValueError(f"{file_path} hidden_state[{key}] is empty.")

        collapsed = []
        for step_hidden in step_hidden_list:
            if step_hidden.ndim == 1:
                collapsed.append(step_hidden)
            else:
                raise ValueError(
                    f"{file_path} hidden_state[{key}] contains unsupported tensor shape {tuple(step_hidden.shape)}. "
                    "Please regenerate the DFlash dataset with step_last_token_hidden_v2 format."
                )
        return torch.stack(collapsed, dim=0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = torch.load(self.data[index], map_location="cpu")
        hidden_state = data["hidden_state"]# 目标模型的 hidden states
        tokens = torch.tensor(data["predicted_tokens"], dtype=torch.long)# 目标模型生成的 token 序列
        data_format = data.get("dflash_data_format")

        if data_format != "full_prefix_plus_action_hidden_v4":
            raise ValueError(
                f"{self.data[index]} uses unsupported dflash_data_format={data_format!r}. "
                "Please regenerate the offline DFlash dataset with full_prefix_plus_action_hidden_v4 format."
            )

        if not isinstance(hidden_state, dict):
            raise ValueError(
                f"{self.data[index]} uses an unsupported legacy hidden_state format. "
                "Please regenerate the offline DFlash dataset with the current exporter."
            )
        required_keys = {"prompt_selected", "prompt_position_ids", "prompt_length", "action_last", "action_selected"}
        if not required_keys.issubset(hidden_state):
            raise ValueError(
                f"{self.data[index]} is missing one of {sorted(required_keys)} in hidden_state. "
                "Please regenerate the offline DFlash dataset."
            )

        action_last_hidden = self._collapse_step_hidden(hidden_state["action_last"], "action_last", self.data[index])
        action_selected_hidden = self._collapse_step_hidden(
            hidden_state["action_selected"], "action_selected", self.data[index]
        )
        if action_last_hidden.shape[0] + 1 != tokens.shape[0] or action_selected_hidden.shape[0] + 1 != tokens.shape[0]:
            raise ValueError(
                f"{self.data[index]} hidden/tokens length mismatch: "
                f"action_selected={action_selected_hidden.shape[0]}, "
                f"action_last={action_last_hidden.shape[0]}, tokens={tokens.shape[0]}."
            )
        if self.expected_selected_layers is not None:
            expected_hidden = self.expected_selected_layers * action_last_hidden.shape[-1]
            if action_selected_hidden.shape[-1] != expected_hidden:# 确保加载的数据的 selected_hidden 维度与预期一致
                raise ValueError(
                    f"{self.data[index]} action_selected dim={action_selected_hidden.shape[-1]} "
                    f"!= expected {expected_hidden}. Please regenerate data with matching hidden_layer_ids."
                )
            if hidden_state["prompt_selected"].shape[-1] != expected_hidden:
                raise ValueError(
                    f"{self.data[index]} prompt_selected dim={hidden_state['prompt_selected'].shape[-1]} "
                    f"!= expected {expected_hidden}. Please regenerate data with matching hidden_layer_ids."
                )
        prompt_length = int(hidden_state["prompt_length"])
        if hidden_state["prompt_selected"].shape[0] != prompt_length:
            raise ValueError(
                f"{self.data[index]} prompt_length={prompt_length} but "
                f"prompt_selected has length={hidden_state['prompt_selected'].shape[0]}."
            )
        if hidden_state["prompt_position_ids"].shape[0] != prompt_length:
            raise ValueError(
                f"{self.data[index]} prompt_position_ids length={hidden_state['prompt_position_ids'].shape[0]} "
                f"!= prompt_length={prompt_length}."
            )
        # 返回单个样本的dict
        return {
            "prompt_selected": hidden_state["prompt_selected"],# 完整 prefill/prefix 目标层 [prefix_len, L*hidden]
            "prompt_position_ids": hidden_state["prompt_position_ids"].long(),
            "prompt_length": prompt_length,
            "action_selected": action_selected_hidden,# token0..token5 的目标层
            "target_hidden": action_last_hidden,# token0..token5 的最后层，用于预测 token1..token6
            "tokens": tokens,# 目标模型生成的 token 序列
            "length": tokens.shape[0] - 1,# 可监督 hidden/logit 的长度
        }

# 将不定长的样本 padding 到 batch 内最长的长度，形成 [batch_size, max_len, ...] 的统一张量
class DataCollatorForOfflineDFlash:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(item["length"] for item in features)
        max_prompt_len = max(item["prompt_length"] for item in features)
        selected_dim = features[0]["action_selected"].shape[-1]
        hidden_dim = features[0]["target_hidden"].shape[-1]

        # 创建 padding 后的 batch 张量
        batch_prompt_selected = torch.zeros(len(features), max_prompt_len, selected_dim)
        batch_prompt_position_ids = torch.zeros(len(features), max_prompt_len, dtype=torch.long)
        batch_prompt_attention_mask = torch.zeros(len(features), max_prompt_len, dtype=torch.bool)
        batch_prompt_lengths = torch.zeros(len(features), dtype=torch.long)
        batch_selected = torch.zeros(len(features), max_len, selected_dim)
        batch_target = torch.zeros(len(features), max_len, hidden_dim)
        batch_tokens = torch.zeros(len(features), max_len + 1, dtype=torch.long)
        batch_lengths = torch.zeros(len(features), dtype=torch.long)

        for i, item in enumerate(features):
            length = item["length"]
            prompt_length = item["prompt_length"]
            batch_prompt_selected[i, :prompt_length] = item["prompt_selected"]
            batch_prompt_position_ids[i, :prompt_length] = item["prompt_position_ids"]
            batch_prompt_attention_mask[i, :prompt_length] = True
            batch_prompt_lengths[i] = prompt_length
            batch_selected[i, :length] = item["action_selected"]
            batch_target[i, :length] = item["target_hidden"]
            batch_tokens[i, : length + 1] = item["tokens"]
            batch_lengths[i] = length

        return {
            "prompt_selected": batch_prompt_selected,# 完整 prefill/prefix 目标层
            "prompt_position_ids": batch_prompt_position_ids,
            "prompt_attention_mask": batch_prompt_attention_mask,
            "prompt_lengths": batch_prompt_lengths,
            "action_selected": batch_selected,# token0..token5 目标层
            "target_hidden": batch_target,# token0..token5 最后一层
            "tokens": batch_tokens,# 目标模型生成的 token 序列
            "lengths": batch_lengths,# 可监督 hidden/logit 的长度，通常为 6
        }

def build_dflash_config_dict(args) -> Dict[str, Any]:
    return {
        "run_name": args.run_name,
        "block_size": args.block_size,
        "num_draft_layers": args.num_draft_layers,
        "target_layer_ids": args.target_layer_ids,
        "mask_token_id": args.mask_token_id,
        "hidden_w": args.hidden_w,
        "ce_w": args.ce_w,
        "kl_w": args.kl_w,
        "kl_temperature": args.kl_temperature,
        "hidden_loss_type": args.hidden_loss_type,
        "action_dim": args.action_dim,
        "hidden_noise": args.hidden_noise,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "grad_clip": args.grad_clip,
        "log_every_steps": args.log_every_steps,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "val_split": args.val_split,
        "patience": args.patience,
        "eval_every": args.eval_every,
        "use_swanlab": args.use_swanlab,
        "swanlab_project": args.swanlab_project,
        "swanlab_workspace": args.swanlab_workspace,
        "swanlab_mode": args.swanlab_mode,
    }


def save_checkpoint(
    output_dir: str,
    epoch: int,
    global_step: int,
    model: DFlashDraftModel,
    optimizer: AdamW,
    scheduler,
    args,
    swanlab_run_id: Optional[str] = None,
    best_val_loss: Optional[float] = None,
    best_val_acc: Optional[float] = None,
    patience_counter: int = 0,
):
    save_dir = Path(output_dir) / f"epoch_{epoch:03d}_step_{global_step:06d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    config_payload = build_dflash_config_dict(args)
    # 保存模型权重
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
    # 保存训练状态（用于恢复训练，包含早停与最优权重信息）
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "swanlab_run_id": swanlab_run_id,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "patience_counter": patience_counter,
        },
        save_dir / "training_state.pt",
    )
    # 保存超参数配置
    with open(save_dir / "dflash_config.json", "w") as f:
        json.dump(config_payload, f, indent=2)
    torch.save(model.state_dict(), Path(output_dir) / "pytorch_model.bin")
    with open(Path(output_dir) / "dflash_config.json", "w") as f:
        json.dump(config_payload, f, indent=2)
    with open(Path(output_dir) / "latest_checkpoint.txt", "w") as f:
        f.write(str(save_dir))


def save_best_checkpoint(output_dir: str, epoch: int, global_step: int, model: DFlashDraftModel, args) -> None:
    """保存当前最优权重，自动覆盖上一个最优权重。"""
    best_dir = Path(output_dir) / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best_dir / "pytorch_model.bin")
    with open(best_dir / "dflash_config.json", "w") as f:
        json.dump(build_dflash_config_dict(args), f, indent=2)
    info = {"best_epoch": epoch, "best_global_step": global_step, "timestamp": datetime.now().isoformat()}
    with open(best_dir / "best_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 最优权重已保存到 {best_dir} (epoch={epoch}, step={global_step})")


def append_jsonl(log_path: Path, payload: Dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_resume_checkpoint(output_dir: str, resume_from_checkpoint: Optional[str]) -> Optional[Path]:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        latest_path = Path(output_dir) / "latest_checkpoint.txt"
        if not latest_path.exists():
            raise ValueError(f"Cannot resume: {latest_path} does not exist.")
        checkpoint_dir = Path(latest_path.read_text(encoding="utf-8").strip())
    else:
        checkpoint_dir = Path(resume_from_checkpoint)
    if not checkpoint_dir.exists():
        raise ValueError(f"Resume checkpoint {checkpoint_dir} does not exist.")
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: Path,
    model: DFlashDraftModel,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> Dict[str, Any]:
    model.load_state_dict(torch.load(checkpoint_dir / "pytorch_model.bin", map_location=device))
    training_state = torch.load(checkpoint_dir / "training_state.pt", map_location=device)
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])
    return training_state  # 包含 epoch, global_step, swanlab_run_id, best_val_loss, best_val_acc, patience_counter


def init_swanlab_run(args, config_payload: Dict[str, Any], output_dir: str, resume_run_id: Optional[str] = None):
    if not args.use_swanlab:
        return None
    if swanlab is None:
        raise ImportError("SwanLab 未安装。请先执行 `pip install swanlab`，或关闭 --use_swanlab。")

    run = swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=args.run_name,
        config=config_payload,
        logdir=str(Path(output_dir) / "swanlog"),
        mode=args.swanlab_mode,
        id=resume_run_id,
        resume="allow" if resume_run_id else None,
        reinit=True,
    )
    return run


def get_swanlab_run_id(run) -> Optional[str]:
    if run is None:
        return None
    for attr in ("id", "run_id", "experiment_id", "_id"):
        value = getattr(run, attr, None)
        if value:
            return str(value)
    return None


def safe_swanlab_log(run, payload: Dict[str, Any], step: Optional[int] = None):
    """兼容旧版 SwanLab 的脆弱事务实现；日志失败时不让训练中断。"""
    if run is None:
        return None
    try:
        swanlab.log(payload, step=step)
        return run
    except Exception as exc:  # pragma: no cover - 防御性降级
        print(f"SwanLab 记录失败（已忽略，训练继续）：{exc}")
        return run


def numeric_payload_for_swanlab(payload: Dict[str, Any], default_prefix: str = "run") -> Dict[str, float]:
    """把 jsonl payload 中所有数值字段转换成 SwanLab 可记录的 metric。

    已经带 "/" 的 key 会原样保留，例如 train/loss、val/anchor_0_acc。
    不带 "/" 的运行状态字段会挂到 default_prefix 下，例如 epoch -> train/epoch。
    """
    swan_payload: Dict[str, float] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.item()
        if isinstance(value, bool):
            value = float(value)
        elif isinstance(value, int):
            value = float(value)
        elif isinstance(value, float):
            value = float(value)
        else:
            continue
        if math.isfinite(value):
            metric_key = key if "/" in key else f"{default_prefix}/{key}"
            swan_payload[metric_key] = value
    return swan_payload


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 学习率调度策略，先warmup，再linear decay
def build_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int, warmup_ratio: float):
    if warmup_steps <= 0:
        warmup_steps = max(1, int(total_steps * warmup_ratio))
    else:
        warmup_steps = min(total_steps, warmup_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def new_detail_accumulator(num_positions: int) -> Dict[str, torch.Tensor]:
    return {
        "anchor_correct": torch.zeros(num_positions, dtype=torch.float32),
        "anchor_total": torch.zeros(num_positions, dtype=torch.float32),
        "position_correct": torch.zeros(num_positions, dtype=torch.float32),
        "position_total": torch.zeros(num_positions, dtype=torch.float32),
        "anchor_position_correct": torch.zeros(num_positions, num_positions, dtype=torch.float32),
        "anchor_position_total": torch.zeros(num_positions, num_positions, dtype=torch.float32),
    }


def accumulate_detail_metrics(accumulator: Optional[Dict[str, torch.Tensor]], metrics: Dict[str, torch.Tensor]):
    detail_keys = [
        "anchor_correct",
        "anchor_total",
        "position_correct",
        "position_total",
        "anchor_position_correct",
        "anchor_position_total",
    ]
    if accumulator is None:
        accumulator = new_detail_accumulator(metrics["anchor_correct"].numel())
    for key in detail_keys:
        accumulator[key] += metrics[key].detach().float().cpu()
    return accumulator


def detail_metrics_to_log(prefix: str, accumulator: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    if accumulator is None:
        return {}
    payload: Dict[str, float] = {}
    anchor_correct = accumulator["anchor_correct"]
    anchor_total = accumulator["anchor_total"]
    position_correct = accumulator["position_correct"]
    position_total = accumulator["position_total"]
    anchor_position_correct = accumulator["anchor_position_correct"]
    anchor_position_total = accumulator["anchor_position_total"]

    for idx in range(anchor_total.numel()):
        if anchor_total[idx] > 0:
            payload[f"{prefix}/anchor_{idx}_acc"] = (anchor_correct[idx] / anchor_total[idx]).item()
        if position_total[idx] > 0:
            payload[f"{prefix}/position_{idx + 1}_acc"] = (position_correct[idx] / position_total[idx]).item()

    for anchor in range(anchor_position_total.shape[0]):
        for position in range(anchor_position_total.shape[1]):
            if anchor_position_total[anchor, position] > 0:
                payload[f"{prefix}/anchor_{anchor}_to_position_{position + 1}_acc"] = (
                    anchor_position_correct[anchor, position] / anchor_position_total[anchor, position]
                ).item()
    return payload


# 损失函数与准确计算函数
def compute_loss_and_accuracy(
    model: DFlashDraftModel,# Draft Model
    embed_tokens: nn.Module,
    lm_head: nn.Module,# 目标模型的 lm head 层（冻结）
    batch: Dict[str, torch.Tensor],# 一个 batch 数据
    args,# 训练参数
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    prompt_selected = batch["prompt_selected"].to(device=device, dtype=torch.bfloat16)# 完整 prefill/prefix 上下文 [B, P, L*hidden]
    prompt_position_ids = batch["prompt_position_ids"].to(device=device)# 完整 prefill/prefix 的绝对 position [B, P]
    prompt_attention_mask = batch["prompt_attention_mask"].to(device=device)# prefix padding mask [B, P]
    prompt_lengths = batch["prompt_lengths"].to(device=device)# 每条样本真实 prefix 长度 [B]
    action_selected = batch["action_selected"].to(device=device, dtype=torch.bfloat16)# action历史上下文[B, seq, L*hidden]
    target_hidden = batch["target_hidden"].to(device=device, dtype=torch.bfloat16)# token0..token5最后层[B, seq, hidden]
    tokens = batch["tokens"].to(device=device)# 目标模型生成的 token 序列[B, seq]
    lengths = batch["lengths"].to(device=device)# 目标模型生成的 序列 的长度[B, ]

    batch_size, seq_len, _ = target_hidden.shape
    ce_sum = torch.zeros((), device=device, dtype=torch.float32)
    kl_sum = torch.zeros((), device=device, dtype=torch.float32)
    hidden_sum = torch.zeros((), device=device, dtype=torch.float32)
    total_positions = torch.zeros((), device=device, dtype=torch.float32)
    total_correct = torch.zeros((), device=device, dtype=torch.float32)
    anchor_correct = torch.zeros(seq_len, device=device, dtype=torch.float32)
    anchor_total = torch.zeros(seq_len, device=device, dtype=torch.float32)
    position_correct = torch.zeros(seq_len, device=device, dtype=torch.float32)
    position_total = torch.zeros(seq_len, device=device, dtype=torch.float32)
    anchor_position_correct = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    anchor_position_total = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    # 主循环：anchor token 作为当前已知 token，DFLASH 并行预测 anchor+1 开始的若干 token。
    # 输出槽位 0 对齐 teacher H(token_anchor)，经 lm_head 预测 token_{anchor+1}。
    for anchor in range(seq_len):# anchor：一个滑动窗口的起始位置
        if int((lengths > anchor).sum().item()) == 0:
            continue
        max_block_len = min(args.block_size - 1, seq_len - anchor)
        if max_block_len <= 0:
            continue
        # 构造输入序列
        block_ids = torch.full(
            (batch_size, max_block_len),
            args.mask_token_id,# 指定填充噪声的位置（token id）
            dtype=tokens.dtype,
            device=device,
        )
        anchor_active = lengths > anchor
        block_ids[anchor_active, 0] = tokens[anchor_active, anchor]# 给输入序列的第 0 位放已知 anchor token
        noise_embedding = embed_tokens(block_ids)# 加噪

        ctx_hidden = torch.cat(
            [prompt_selected, action_selected[:, :anchor, :]],
            dim=1,
        )# [B, prefix_len+anchor, L*hidden] 上下文：完整 prefill/prefix + anchor前action hidden
        if anchor > 0:
            action_ctx_mask = (
                torch.arange(anchor, device=device, dtype=torch.long).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            ctx_attention_mask = torch.cat([prompt_attention_mask, action_ctx_mask], dim=1)
        else:
            ctx_attention_mask = prompt_attention_mask
        # 数据增强：训练时给 context hidden 加噪，防止模型死记 exact hidden（效仿 SpecVLA）
        if model.training and args.hidden_noise > 0:
            noise = torch.randn_like(ctx_hidden) * args.hidden_noise
            ctx_hidden = torch.where(ctx_attention_mask.unsqueeze(-1), ctx_hidden + noise, ctx_hidden)
        action_base_positions = prompt_lengths
        if anchor > 0:
            action_ctx_positions = (
                action_base_positions.unsqueeze(1)
                + torch.arange(anchor, device=device, dtype=torch.long).unsqueeze(0)
            )
            ctx_position_ids = torch.cat([prompt_position_ids, action_ctx_positions], dim=1)
        else:
            ctx_position_ids = prompt_position_ids
        noise_position_ids = (
            action_base_positions.unsqueeze(1)
            + anchor
            + torch.arange(max_block_len, device=device, dtype=torch.long).unsqueeze(0)
        )
        action_position_ids = (
            anchor + torch.arange(max_block_len, device=device, dtype=torch.long).unsqueeze(0)
        ).expand(batch_size, -1)

        # Draft 模型推理
        pred_hidden = model(# 输出去噪序列[B, block_size, hidden]
            noise_embedding=noise_embedding, # [B, max_block_len, hidden]
            target_hidden=ctx_hidden,# [B, prefix_len+anchor, L*hidden]
            ctx_position_ids=ctx_position_ids,# [B, prefix_len+anchor]
            noise_position_ids=noise_position_ids,# [B, max_block_len]
            ctx_attention_mask=ctx_attention_mask,# [B, prefix_len+anchor]
            action_position_ids=action_position_ids,# [B, max_block_len] action维度/槽位位置
        )

        student_hidden = pred_hidden[:, :max_block_len, :].float()# 草稿预测的最终层 hidden [B, block, hidden]
        teacher_hidden = target_hidden[:, anchor : anchor + max_block_len, :].float()# H(token_anchor..)
        # 草稿预测最终层hidden过lm头取logits
        student_logits = lm_head(student_hidden.to(torch.bfloat16)).float()
        target_tokens = tokens[:, anchor + 1 : anchor + 1 + max_block_len]

        valid_mask = (
            torch.arange(max_block_len, device=device, dtype=torch.long).unsqueeze(0)
            < (lengths - anchor).unsqueeze(1)
        ).float()
        valid_count = valid_mask.sum()
        if valid_count.item() == 0:
            continue
        # token块交叉熵损失
        ce = F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            target_tokens.reshape(-1),
            reduction="none",
        ).view(batch_size, -1)
        ce_sum += (ce * valid_mask).sum()
        # soft distribution 蒸馏：不只逼近argmax token，也让草稿学习目标模型的相对偏好。
        if args.kl_w > 0:
            with torch.no_grad():
                teacher_logits = lm_head(teacher_hidden.to(torch.bfloat16)).float()
                teacher_probs = F.softmax(teacher_logits / args.kl_temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / args.kl_temperature, dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            kl_sum += (kl * (args.kl_temperature ** 2) * valid_mask).sum()

        # hidden蒸馏默认用cosine，避免raw MSE强行匹配幅值；必要时可切回norm_mse/raw_mse做消融。
        if args.hidden_loss_type == "cosine":
            hidden_reg = 1.0 - F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)
        elif args.hidden_loss_type == "norm_mse":
            hidden_reg = F.mse_loss(
                F.normalize(student_hidden, dim=-1),
                F.normalize(teacher_hidden, dim=-1),
                reduction="none",
            ).sum(dim=-1)
        else:
            hidden_reg = F.mse_loss(student_hidden, teacher_hidden, reduction="none").mean(dim=-1)
        hidden_sum += (hidden_reg * valid_mask).sum()

        pred_tokens = student_logits.argmax(dim=-1)# 草稿模型走贪婪解码
        correct_mask = (pred_tokens == target_tokens) & valid_mask.bool()
        total_correct += correct_mask.sum().float()
        total_positions += valid_count
        anchor_total[anchor] += valid_count
        anchor_correct[anchor] += correct_mask.sum().float()
        for local_pos in range(max_block_len):
            target_pos = anchor + local_pos
            pos_mask = valid_mask[:, local_pos].bool()
            pos_count = pos_mask.sum().float()
            if pos_count.item() == 0:
                continue
            pos_correct = correct_mask[:, local_pos].sum().float()
            position_total[target_pos] += pos_count
            position_correct[target_pos] += pos_correct
            anchor_position_total[anchor, target_pos] += pos_count
            anchor_position_correct[anchor, target_pos] += pos_correct

    denom = total_positions.clamp_min(1.0)
    ce_loss = ce_sum / denom
    kl_loss = kl_sum / denom
    hidden_loss = hidden_sum / denom
    total_loss = args.ce_w * ce_loss + args.kl_w * kl_loss + args.hidden_w * hidden_loss
    accuracy = total_correct / denom

    return {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "kl_loss": kl_loss,
        "hidden_loss": hidden_loss,
        "accuracy": accuracy,
        "anchor_correct": anchor_correct.detach(),
        "anchor_total": anchor_total.detach(),
        "position_correct": position_correct.detach(),
        "position_total": position_total.detach(),
        "anchor_position_correct": anchor_position_correct.detach(),
        "anchor_position_total": anchor_position_total.detach(),
    }


@torch.no_grad()
def evaluate(
    model: DFlashDraftModel,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    val_loader: DataLoader,
    args,
    device: torch.device,
) -> Dict[str, float]:
    """在验证集上评估 Draft 模型，返回平均指标。"""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_hidden = 0.0
    total_acc = 0.0
    total_samples = 0
    detail_accumulator = None

    for batch in val_loader:
        metrics = compute_loss_and_accuracy(model, embed_tokens, lm_head, batch, args, device)
        bs = batch["lengths"].shape[0]
        total_loss += metrics["loss"].item() * bs
        total_ce += metrics["ce_loss"].item() * bs
        total_kl += metrics["kl_loss"].item() * bs
        total_hidden += metrics["hidden_loss"].item() * bs
        total_acc += metrics["accuracy"].item() * bs
        total_samples += bs
        detail_accumulator = accumulate_detail_metrics(detail_accumulator, metrics)

    model.train()
    denom = max(total_samples, 1)
    result = {
        "val/loss": total_loss / denom,
        "val/ce_loss": total_ce / denom,
        "val/kl_loss": total_kl / denom,
        "val/hidden_loss": total_hidden / denom,
        "val/accuracy": total_acc / denom,
    }
    result.update(detail_metrics_to_log("val", detail_accumulator))
    return result


def main():
    args = parse_args()
    if args.kl_temperature <= 0:
        raise ValueError("--kl_temperature must be > 0.")
    if args.action_dim <= 0:
        raise ValueError("--action_dim must be > 0.")
    if args.lr > 1e-3:
        print(
            f"WARNING: lr={args.lr:g} is very high for DFLASH AdamW training. "
            "This can quickly improve early metrics and then destabilize the draft model."
        )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_name is None:
        args.run_name = f"dflash-libero-goal-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_log_path = Path(args.output_dir) / "metrics.jsonl"

    # transformer库内置的模型、配置、分词器加载器
    AutoConfig.register("openvla", OpenVLAConfig)# 遇到"openvla"时，使用本地写好的OpenVLAConfig加载配置
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)# 使用PrismaticImageProcessor图像处理器
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)# 使用PrismaticProcessor分词器
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)# 使用 OpenVLAForActionPrediction 来实例化VLA模型

    print("正在加载 OpenVLA ...")
    vla = AutoModelForVision2Seq.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,# 不量化
        low_cpu_mem_usage=True,# 在加载模型时优化 CPU 内存使用
        trust_remote_code=False,# 使用本仓库已注册的本地 OpenVLA 类，避免联网拉 HF dynamic module
    )
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=False)# 加载本地 PrismaticProcessor，避免联网拉 HF dynamic module
    # 如果用户没有通过命令行参数指定噪声掩码
    if args.mask_token_id is None:
        args.mask_token_id = processor.tokenizer.pad_token_id# 则使用加载的 OpenVLA 模型对应的 tokenizer 的 pad_token_id 作为默认值

    target_config = copy.deepcopy(vla.language_model.config)# 只保留草稿模型真正需要的结构配置
    num_target_layers = target_config.num_hidden_layers
    embed_tokens = vla.language_model.model.embed_tokens.to(device=device, dtype=torch.bfloat16).eval()# 取编码层（设置为评估模式）
    lm_head = vla.language_model.lm_head.to(device=device, dtype=torch.bfloat16).eval()# 取lm头（设置为评估模式）
    embed_tokens.requires_grad_(False)# 冻结
    lm_head.requires_grad_(False)# 冻结
    del vla, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 如果用户没有通过命令行指定要从目标模型的哪些层提取特征
    if args.target_layer_ids is None or len(args.target_layer_ids) == 0:
        args.target_layer_ids = build_target_layer_ids(# 那就自动取
            num_target_layers=num_target_layers,
            num_draft_layers=args.num_draft_layers,
        )
    # Draft初始化
    draft_config = copy.deepcopy(target_config)# 从目标模型的配置复制而来，继承大部分结构参数，确保其与目标模型兼容
    draft_config.num_hidden_layers = args.num_draft_layers# 覆盖 草稿模型的层数
    draft_config.num_target_layers = num_target_layers# 将目标模型的总层数也存入草稿配置，供 build_target_layer_ids 使用
    draft_config.dflash_target_layer_ids = args.target_layer_ids# 5
    draft_config.dflash_block_size = args.block_size# 7
    draft_config.dflash_action_dim = args.action_dim# action token维度数，用于action-dimension embedding
    model = DFlashDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)# 实例化草稿模型
    trainable_params = count_trainable_parameters(model)

    cache_path = get_dataset_file_cache_path(args.datapath)
    if args.refresh_file_cache:
        print(f"正在刷新数据文件缓存: {cache_path}")
    elif cache_path.exists():
        print(f"正在加载数据文件缓存: {cache_path}")
    else:
        print(f"首次扫描数据目录并建立缓存: {args.datapath}")
    datapath = list_files(args.datapath, refresh_cache=args.refresh_file_cache)
    if not datapath:
        raise ValueError(f"No .ckpt files found in {args.datapath}")
    random.Random(args.seed).shuffle(datapath)

    # 划分训练集 / 验证集
    val_loader = None
    if args.val_split > 0:
        val_size = max(1, int(len(datapath) * args.val_split))
        train_files = datapath[:-val_size] if val_size < len(datapath) else datapath
        val_files = datapath[-val_size:]
    else:
        train_files = datapath
        val_files = []

    train_dataset = OfflineDFlashDataset(train_files, expected_selected_layers=len(args.target_layer_ids))
    collator = DataCollatorForOfflineDFlash()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    if val_files:
        val_dataset = OfflineDFlashDataset(val_files, expected_selected_layers=len(args.target_layer_ids))
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    steps_per_epoch = max(1, (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    total_optimizer_steps = max(1, args.num_epochs * steps_per_epoch)# 计算总的优化器步数，用于学习率调度器
    scheduler = build_scheduler(optimizer, total_optimizer_steps, args.warmup_steps, args.warmup_ratio)

    global_step = 0
    resume_checkpoint_dir = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    resume_run_id = None
    start_epoch = 1
    best_val_loss = None
    best_val_acc = None
    patience_counter = 0
    if resume_checkpoint_dir is not None:
        state = load_checkpoint(resume_checkpoint_dir, model, optimizer, scheduler, device)
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        resume_run_id = state.get("swanlab_run_id")
        best_val_loss = state.get("best_val_loss")
        best_val_acc = state.get("best_val_acc")
        patience_counter = state.get("patience_counter", 0)
        print(f"已从 {resume_checkpoint_dir} 恢复：epoch={start_epoch} global_step={global_step} "
              f"best_val_loss={best_val_loss} best_val_acc={best_val_acc} patience_counter={patience_counter}")

    config_payload = build_dflash_config_dict(args)
    config_payload.update(
        {
            "train_files": len(train_files),
            "val_files": len(val_files),
            "trainable_params": trainable_params,
            "trainable_params_m": round(trainable_params / 1e6, 2),
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
        }
    )
    with open(Path(args.output_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)
    append_jsonl(
        metrics_log_path,
        {
            "event": "run_start",
            "timestamp": datetime.now().isoformat(),
            "run_name": args.run_name,
            "train_files": len(train_files),
            "val_files": len(val_files),
            "trainable_params": trainable_params,
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
        },
    )
    print(
        f"训练集={len(train_files)} "
        f"验证集={len(val_files)} "
        f"Draft参数={trainable_params/1e6:.2f}M "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps} "
        f"steps_per_epoch={steps_per_epoch}"
    )

    swanlab_run = init_swanlab_run(args, config_payload, args.output_dir, resume_run_id=resume_run_id)
    if swanlab_run is not None:
        swanlab_run_id = get_swanlab_run_id(swanlab_run)
        if swanlab_run_id is not None:
            print(f"SwanLab 已启动: mode={args.swanlab_mode}, run_id={swanlab_run_id}")
        else:
            print(f"SwanLab 已启动: mode={args.swanlab_mode}")
        swanlab_run = safe_swanlab_log(
            swanlab_run,
            numeric_payload_for_swanlab(
                {
                    "train_files": len(train_files),
                    "val_files": len(val_files),
                    "trainable_params": trainable_params,
                    "trainable_params_m": round(trainable_params / 1e6, 2),
                    "steps_per_epoch": steps_per_epoch,
                    "total_optimizer_steps": total_optimizer_steps,
                    "effective_batch": args.batch_size * args.gradient_accumulation_steps,
                },
                default_prefix="run",
            ),
            step=global_step,
        )

    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            train_ce_sum = 0.0
            train_kl_sum = 0.0
            train_hidden_sum = 0.0
            train_acc_sum = 0.0
            train_detail_accumulator = None
            train_log_steps = 0
            pbar = tqdm(train_loader, desc=f"train {epoch}/{args.num_epochs}", dynamic_ncols=True)
            for batch_idx, batch in enumerate(pbar, start=1):
                metrics = compute_loss_and_accuracy(model, embed_tokens, lm_head, batch, args, device)
                (metrics["loss"] / args.gradient_accumulation_steps).backward()

                train_loss_sum += metrics["loss"].item()
                train_ce_sum += metrics["ce_loss"].item()
                train_kl_sum += metrics["kl_loss"].item()
                train_hidden_sum += metrics["hidden_loss"].item()
                train_acc_sum += metrics["accuracy"].item()
                train_detail_accumulator = accumulate_detail_metrics(train_detail_accumulator, metrics)
                train_log_steps += 1

                should_step = (
                    batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == len(train_loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % args.log_every_steps == 0:
                        train_payload = {
                            "event": "train_step",
                            "timestamp": datetime.now().isoformat(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "train/loss": train_loss_sum / max(1, train_log_steps),
                            "train/ce_loss": train_ce_sum / max(1, train_log_steps),
                            "train/kl_loss": train_kl_sum / max(1, train_log_steps),
                            "train/hidden_loss": train_hidden_sum / max(1, train_log_steps),
                            "train/accuracy": train_acc_sum / max(1, train_log_steps),
                            "train/lr": scheduler.get_last_lr()[0],
                        }
                        train_payload.update(detail_metrics_to_log("train", train_detail_accumulator))
                        append_jsonl(metrics_log_path, train_payload)
                        if swanlab_run is not None:
                            swan_payload = numeric_payload_for_swanlab(train_payload, default_prefix="train")
                            swan_payload["train/log_steps"] = float(train_log_steps)
                            swanlab_run = safe_swanlab_log(
                                swanlab_run,
                                swan_payload,
                                step=global_step,
                            )
                        train_loss_sum = 0.0
                        train_ce_sum = 0.0
                        train_kl_sum = 0.0
                        train_hidden_sum = 0.0
                        train_acc_sum = 0.0
                        train_detail_accumulator = None
                        train_log_steps = 0

                pbar.set_postfix(
                    loss=f"{metrics['loss'].item():.4f}",
                    ce=f"{metrics['ce_loss'].item():.4f}",
                    kl=f"{metrics['kl_loss'].item():.4f}",
                    h=f"{metrics['hidden_loss'].item():.4f}",
                    acc=f"{metrics['accuracy'].item():.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=global_step,
                )

            # ── 验证 + 早停 + 最优权重 ──
            do_eval = (
                val_loader is not None
                and epoch % args.eval_every == 0
            )
            if do_eval:
                val_metrics = evaluate(model, embed_tokens, lm_head, val_loader, args, device)

                current_val_loss = val_metrics["val/loss"]
                current_val_acc = val_metrics["val/accuracy"]

                # 最优权重保存：按 accuracy（越高越好）
                is_best_acc = best_val_acc is None or current_val_acc > best_val_acc
                if is_best_acc:
                    best_val_acc = current_val_acc
                    save_best_checkpoint(args.output_dir, epoch, global_step, model, args)

                # 早停：按 loss（越低越好）
                is_best_loss = best_val_loss is None or current_val_loss < best_val_loss
                if is_best_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                val_payload = {
                    "event": "val_epoch",
                    "timestamp": datetime.now().isoformat(),
                    "epoch": epoch,
                    "global_step": global_step,
                    **val_metrics,
                    "val/best_loss": best_val_loss,
                    "val/best_accuracy": best_val_acc,
                    "val/patience_counter": patience_counter,
                    "val/patience": args.patience,
                    "val/is_best_accuracy": is_best_acc,
                    "val/is_best_loss": is_best_loss,
                }
                append_jsonl(metrics_log_path, val_payload)
                if swanlab_run is not None:
                    swanlab_run = safe_swanlab_log(
                        swanlab_run,
                        numeric_payload_for_swanlab(val_payload, default_prefix="val"),
                        step=global_step,
                    )

                print(
                    f"验证 epoch={epoch} | loss={current_val_loss:.4f} ce={val_metrics['val/ce_loss']:.4f} "
                    f"kl={val_metrics['val/kl_loss']:.4f} "
                    f"hidden={val_metrics['val/hidden_loss']:.4f} acc={current_val_acc:.3f} "
                    f"| best_loss={best_val_loss:.4f} best_acc={best_val_acc:.3f} "
                    f"patience={patience_counter}/{args.patience}"
                )

                if patience_counter >= args.patience:
                    print(f"早停触发：验证 loss 连续 {args.patience} 个 epoch 未下降，停止训练。")
                    break

            # 按 epoch 保存 checkpoint
            if epoch % args.save_every == 0:
                save_checkpoint(
                    args.output_dir,
                    epoch,
                    global_step,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    swanlab_run_id=(get_swanlab_run_id(swanlab_run) if swanlab_run is not None else resume_run_id),
                    best_val_loss=best_val_loss,
                    best_val_acc=best_val_acc,
                    patience_counter=patience_counter,
                )
    except KeyboardInterrupt:
        print("检测到手动中断，正在保存 latest checkpoint 以便续训 ...")
        save_checkpoint(
            args.output_dir,
            max(start_epoch, min(args.num_epochs, epoch if 'epoch' in locals() else start_epoch)),
            global_step,
            model,
            optimizer,
            scheduler,
            args,
            swanlab_run_id=(get_swanlab_run_id(swanlab_run) if swanlab_run is not None else resume_run_id),
            best_val_loss=best_val_loss if 'best_val_loss' in locals() else None,
            best_val_acc=best_val_acc if 'best_val_acc' in locals() else None,
            patience_counter=patience_counter if 'patience_counter' in locals() else 0,
        )
        raise
    finally:
        if swanlab_run is not None:
            swanlab_run.finish()


if __name__ == "__main__":
    main()
