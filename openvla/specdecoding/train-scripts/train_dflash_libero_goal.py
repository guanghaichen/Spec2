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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig# 导入VLA配置类
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction# 导入VLA模型
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from openvla.specdecoding.model.dflash import (
    SELECTED_HIDDEN_VARIANTS,
    DFlashDraftModel,
    apply_selected_hidden_variant,
    build_target_layer_ids,
    normalize_selected_hidden_variant,
)# 导入DFlash Draft模型与自适应选层函数

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
        default=os.environ.get(
            "VLA_PATH",
            "/mnt/storage/cgh/hf_files/openvla-7b-finetuned-libero-goal"
            if Path("/mnt/storage/cgh").exists()
            else "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal",
        ),
        help="OpenVLA 目标模型权重路径（用于加载 embedding 和 lm_head 权重）",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=os.environ.get(
            "DATAPATH",
            "/mnt/storage/cgh/specvla-data/dflash_goal_dataset"
            if Path("/mnt/storage/cgh").exists()
            else "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset",
        ),
        help="离线数据路径（预计算的 .ckpt 文件）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get(
            "OUTPUT_DIR",
            "/mnt/storage/cgh/specvla-data/ckpt_goal_dflash_specvla_anchor_hidden"
            if Path("/mnt/storage/cgh").exists()
            else "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_specvla_anchor_hidden",
        ),
        help="输出目录（用于保存模型权重）",
    )
    # ---- 训练规模与优化器 ----
    parser.add_argument("--batch_size", type=int, default=8, help="每张卡的 micro batch size；DDP 下全局 batch = batch_size * 卡数 * gradient_accumulation_steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数；显存不够时可增大它来保持全局 batch")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--num_epochs", type=int, default=200, help="最大训练 epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="AdamW 学习率；当前 4 卡 batch=64 时推荐 5e-5 起步")
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="AdamW weight decay；用于抑制 draft 过度记忆训练集")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="AdamW beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="学习率 warmup 步数；<=0 时退回 warmup_ratio")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="当 warmup_steps<=0 时使用")
    parser.add_argument("--save_every", type=int, default=5, help="按 epoch 保存 checkpoint 的间隔；例如 10 表示每 10 个 epoch 保存一次")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")

    # ---- Draft 结构与输入 hidden 组织方式 ----
    parser.add_argument("--block_size", type=int, default=7, help="一次投机生成的 action token 块大小；OpenVLA action 默认 7 维")
    parser.add_argument("--num_draft_layers", type=int, default=3, help="DFlash Draft Transformer 层数；当前轻量实验通常设为 1")
    parser.add_argument("--target_layer_ids", type=int, nargs="*", default=[1, 8, 15, 22, 29], help="离线数据保存的 OpenVLA 多层 hidden；当前数据为 [1,8,15,22,29]")
    parser.add_argument(
        "--selected_hidden_variant",
        type=str,
        choices=SELECTED_HIDDEN_VARIANTS,
        default="target_layers",
        help="DFlash context hidden 组装方式：target_layers=直接用保存的目标层；replace_22_with_final=把 [1,8,15,22,29] 替换为 [1,8,15,29,final]",
    )
    parser.add_argument("--mask_token_id", type=int, default=None, help="加噪声的 token ID，不指定也会自适应取pad_token_id")
    parser.add_argument(
        "--include_anchor_hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SpecVLA式注入：context包含当前anchor的目标模型hidden；draft从anchor+1开始并行预测后续token",
    )
    # ---- 基础蒸馏损失 ----
    parser.add_argument("--hidden_w", type=float, default=1.0, help="基础 hidden 蒸馏权重；让 draft hidden 追目标模型最后层 hidden，是当前最核心的监督")
    parser.add_argument("--soft_w", type=float, default=0, help="teacher soft distribution 蒸馏权重；用目标模型 logits 的软分布辅助 token 层对齐")
    parser.add_argument("--soft_temperature", type=float, default=2.0, help="soft distribution 蒸馏温度；越大分布越平滑")
    parser.add_argument("--cos_w", type=float, default=0.05, help="hidden cosine 辅助约束权重；强调方向一致性，通常小权重即可")
    parser.add_argument("--slot_decay", type=float, default=0.85, help="块内位置衰减权重；1.0 表示 p1-p6 不衰减，<1 时更重视靠前 slot")

    # ---- 旧版跨 anchor hidden 一致性，当前主实验关闭 ----
    parser.add_argument("--anchor_consistency_w", type=float, default=0.0, help="旧版跨 anchor 一致性 loss 权重；0 表示关闭，主实验改用 Markov-ACD")
    parser.add_argument(
        "--anchor_consistency_type",
        type=str,
        default="cosine",
        choices=["cosine", "smooth_l1", "norm_mse"],
        help="跨 anchor 一致性距离；默认 cosine，作为温和的结构正则",
    )
    parser.add_argument(
        "--anchor_consistency_warmup_steps",
        type=int,
        default=0,
        help="跨 anchor 一致性权重线性 warmup 步数；0 表示不 warmup",
    )
    parser.add_argument("--position_balance", action=argparse.BooleanOptionalAction, default=True, help="是否平衡多 anchor 中不同 action 位置的重复监督次数，避免靠后位置因出现次数更多而主导训练")

    # ---- Markov-ACD hidden 残差头 ----
    parser.add_argument(
        "--causal_residual_type",
        type=str,
        default="none",
        choices=["none", "hidden"],
        help="前序 token hidden 残差头；hidden=根据前序 token 信息给 p2-p5 等远 slot 补一段因果残差",
    )
    parser.add_argument("--causal_residual_rank", type=int, default=256, help="hidden 残差头中间维度；越大表达力越强但参数和过拟合风险也更高")
    parser.add_argument("--causal_residual_scale", type=float, default=1.0, help="hidden 残差幅度系数；0 等价于禁用残差效果")
    parser.add_argument("--causal_residual_start_index", type=int, default=0, help="从第几个 draft slot 开始启用残差；0 表示连第一跳 slot0/p1 也使用前序 token 条件")
    parser.add_argument(
        "--causal_residual_cad_w",
        type=float,
        default=0.0,
        help="残差版跨 anchor 因果蒸馏权重；弱路径经残差修正后追同一目标位置的一步强路径",
    )
    parser.add_argument(
        "--causal_residual_cad_type",
        type=str,
        default="smooth_l1",
        choices=["cosine", "smooth_l1", "norm_mse"],
        help="残差 CAD hidden 距离类型；cosine 更关注方向，smooth_l1 更关注数值接近",
    )
    parser.add_argument(
        "--causal_residual_cad_warmup_steps",
        type=int,
        default=0,
        help="残差 CAD 权重线性 warmup 步数；用于避免训练早期 strong path 尚不稳定时过早约束 weak path",
    )
    parser.add_argument(
        "--causal_residual_min_position",
        type=int,
        default=2,
        help="残差 CAD 作用的最小目标 token 位置（1-based，默认 p2）",
    )
    parser.add_argument(
        "--causal_residual_max_position",
        type=int,
        default=5,
        help="残差 CAD 作用的最大目标 token 位置（1-based，默认 p5）",
    )
    parser.add_argument(
        "--causal_residual_cad_correct_teacher_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="只在一步强路径预测正确时使用该 hidden 作为残差 CAD teacher，减少错误 teacher 传播",
    )

    # ---- 残差修正后的 weak-path 直接监督 ----
    parser.add_argument(
        "--refined_hidden_w",
        type=float,
        default=0.0,
        help="refined weak hidden 直接追目标 hidden 的损失权重；配合 CAD，让残差头不仅靠相对蒸馏，也看到真实目标 hidden",
    )
    parser.add_argument(
        "--refined_hidden_loss_type",
        type=str,
        default="smooth_l1",
        choices=["cosine", "smooth_l1", "norm_mse", "raw_mse"],
        help="refined hidden 直接监督的距离类型；smooth_l1 稳定，cosine 更偏方向约束",
    )
    parser.add_argument(
        "--refined_hidden_min_position",
        type=int,
        default=2,
        help="refined hidden 直接监督的最小目标 token 位置（1-based）",
    )
    parser.add_argument(
        "--refined_hidden_max_position",
        type=int,
        default=5,
        help="refined hidden 直接监督的最大目标 token 位置（1-based）",
    )
    # ---- 残差修正后的 token 层监督 ----
    parser.add_argument("--residual_token_ce_w", type=float, default=0.0, help="残差修正后 logits 的 hard token CE 权重；只作用于 weak slot，用于直接提高 p2-p5 命中率")
    parser.add_argument("--residual_token_ce_min_position", type=int, default=2, help="residual token CE 作用的最小目标 token 位置；默认 p2")
    parser.add_argument("--residual_token_ce_max_position", type=int, default=5, help="residual token CE 作用的最大目标 token 位置；默认 p5")
    parser.add_argument("--residual_token_ce_label_smoothing", type=float, default=0.0, help="residual token CE 的 label smoothing；0 表示使用硬标签")

    # ---- logits 级 Markov bias 与跨 anchor logits 蒸馏 ----
    parser.add_argument(
        "--logit_markov_type",
        type=str,
        default="none",
        choices=["none", "bias"],
        help="logits 级前序 token 修正头；bias=在 lm_head 后给远 slot logits 加轻量 Markov 偏置",
    )
    parser.add_argument("--logit_markov_rank", type=int, default=256, help="logits Markov bias 头中间维度；只影响 bias 头表达力")
    parser.add_argument("--logit_markov_scale", type=float, default=1.0, help="logits Markov bias 幅度系数；调小可减弱 token 层修正")
    parser.add_argument("--anchor_logit_distill_w", type=float, default=0.0, help="跨 anchor 强弱路径 logits 蒸馏权重；让弱前缀预测追更强前缀预测")
    parser.add_argument("--anchor_logit_distill_temperature", type=float, default=2.0, help="跨 anchor logits KL 温度；越大越强调暗知识而非单个 hard token")
    parser.add_argument("--anchor_logit_distill_min_position", type=int, default=2, help="anchor logits 蒸馏作用的最小目标 token 位置；默认 p2")
    parser.add_argument("--anchor_logit_distill_max_position", type=int, default=5, help="anchor logits 蒸馏作用的最大目标 token 位置；默认 p5")
    parser.add_argument(
        "--anchor_logit_distill_correct_teacher_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="只用一步强路径预测正确的样本作为跨 anchor logits teacher，减少错误分布蒸馏",
    )
    parser.add_argument("--kl_w", type=float, dest="soft_w", help=argparse.SUPPRESS)
    parser.add_argument("--kl_temperature", type=float, dest="soft_temperature", help=argparse.SUPPRESS)
    parser.add_argument(
        "--hidden_loss_type",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "cosine", "norm_mse", "raw_mse"],
        help="hidden 蒸馏损失类型；默认 smooth_l1，参考 SpecVLA 以 hidden 回归为主",
    )
    parser.add_argument("--action_dim", type=int, default=7, help="OpenVLA action token 维度数，用于 action-dimension embedding")
    parser.add_argument("--hidden_noise", type=float, default=0.03, help="训练时 context hidden 加噪标准差（0=不加，推荐 0.03）")
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
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值（epoch 数）；验证 loss 不下降多少个 epoch 后停止")
    parser.add_argument("--eval_every", type=int, default=1, help="每隔多少个 epoch 进行一次验证")
    return parser.parse_args()

# 固定随机种子
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_if_needed() -> Dict[str, Any]:
    """启用 torchrun/DDP 时初始化分布式；普通单卡启动时保持原行为。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_available():
            raise RuntimeError("DDP requested but torch.distributed is not available.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def rank0_print(is_main: bool, *args, **kwargs) -> None:
    if is_main:
        print(*args, **kwargs)


def distributed_log_metrics(metrics: Dict[str, torch.Tensor], distributed: bool) -> Dict[str, torch.Tensor]:
    """把各 rank 的日志指标聚合成全局指标；loss/acc 取均值，count 类指标取和。"""
    if not distributed:
        return metrics
    sum_keys = {
        "anchor_correct",
        "anchor_total",
        "position_correct",
        "position_total",
        "anchor_position_correct",
        "anchor_position_total",
        "anchor_consistency_pairs",
        "causal_residual_cad_pairs",
    }
    reduced: Dict[str, torch.Tensor] = {}
    world_size = dist.get_world_size()
    for key, value in metrics.items():
        if not torch.is_tensor(value):
            reduced[key] = value
            continue
        tensor = value.detach().float().clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if key not in sum_keys:
            tensor /= world_size
        reduced[key] = tensor
    return reduced

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
    def __init__(
        self,
        datapath: List[str],
        expected_selected_layers: Optional[int] = None,
        target_layer_ids: Optional[List[int]] = None,
        selected_hidden_variant: str = "target_layers",
    ):
        self.data = datapath# 所有 .ckpt 文件的路径列表
        self.expected_selected_layers = expected_selected_layers# 期望的 selected layers 数量（用于校验数据兼容性）
        self.target_layer_ids = target_layer_ids
        self.selected_hidden_variant = normalize_selected_hidden_variant(selected_hidden_variant)

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
        if self.selected_hidden_variant == "replace_22_with_final":
            required_keys.add("prompt_last")
        if not required_keys.issubset(hidden_state):
            raise ValueError(
                f"{self.data[index]} is missing one of {sorted(required_keys)} in hidden_state. "
                "Please regenerate the offline DFlash dataset."
            )

        action_last_hidden = self._collapse_step_hidden(hidden_state["action_last"], "action_last", self.data[index])
        action_selected_hidden = self._collapse_step_hidden(
            hidden_state["action_selected"], "action_selected", self.data[index]
        )
        data_layer_ids = hidden_state.get("layer_ids", self.target_layer_ids)
        if data_layer_ids is None:
            raise ValueError(f"{self.data[index]} is missing hidden_state['layer_ids']; cannot validate selected hidden layout.")
        data_layer_ids = [int(layer_id) for layer_id in data_layer_ids]
        if self.target_layer_ids is not None and data_layer_ids != list(self.target_layer_ids):
            raise ValueError(
                f"{self.data[index]} layer_ids={data_layer_ids} != configured target_layer_ids={self.target_layer_ids}. "
                "Please use matching --target_layer_ids or regenerate data."
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
        prompt_selected = apply_selected_hidden_variant(
            hidden_state["prompt_selected"],
            hidden_state.get("prompt_last"),
            data_layer_ids,
            self.selected_hidden_variant,
            self.data[index],
        )
        action_selected_hidden = apply_selected_hidden_variant(
            action_selected_hidden,
            action_last_hidden,
            data_layer_ids,
            self.selected_hidden_variant,
            self.data[index],
        )
        # 返回单个样本的dict
        return {
            "prompt_selected": prompt_selected,# 完整 prefill/prefix 目标层 [prefix_len, L*hidden]
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
        "selected_hidden_variant": args.selected_hidden_variant,
        "mask_token_id": args.mask_token_id,
        "loss_design": (
            "specvla_anchor_hidden_main+teacher_soft_ce"
            + ("+causal_residual_cad" if args.causal_residual_cad_w > 0 else "")
            + ("+logit_markov_bias" if args.logit_markov_type != "none" else "")
            + ("+anchor_logit_distill" if args.anchor_logit_distill_w > 0 else "")
            + ("+weak_path_refined_hidden" if args.refined_hidden_w > 0 else "")
            + ("+residual_token_ce" if args.residual_token_ce_w > 0 else "")
        )
        if args.include_anchor_hidden
        else "hidden_smooth_l1_main_plus_teacher_soft_ce",
        "include_anchor_hidden": args.include_anchor_hidden,
        "hidden_w": args.hidden_w,
        "soft_w": args.soft_w,
        "soft_temperature": args.soft_temperature,
        "cos_w": args.cos_w,
        "anchor_consistency_w": args.anchor_consistency_w,
        "anchor_consistency_type": args.anchor_consistency_type,
        "anchor_consistency_warmup_steps": args.anchor_consistency_warmup_steps,
        "causal_residual_type": args.causal_residual_type,
        "causal_residual_rank": args.causal_residual_rank,
        "causal_residual_scale": args.causal_residual_scale,
        "causal_residual_start_index": args.causal_residual_start_index,
        "causal_residual_cad_w": args.causal_residual_cad_w,
        "causal_residual_cad_type": args.causal_residual_cad_type,
        "causal_residual_cad_warmup_steps": args.causal_residual_cad_warmup_steps,
        "causal_residual_min_position": args.causal_residual_min_position,
        "causal_residual_max_position": args.causal_residual_max_position,
        "causal_residual_cad_correct_teacher_only": args.causal_residual_cad_correct_teacher_only,
        "refined_hidden_w": args.refined_hidden_w,
        "refined_hidden_loss_type": args.refined_hidden_loss_type,
        "refined_hidden_min_position": args.refined_hidden_min_position,
        "refined_hidden_max_position": args.refined_hidden_max_position,
        "residual_token_ce_w": args.residual_token_ce_w,
        "residual_token_ce_min_position": args.residual_token_ce_min_position,
        "residual_token_ce_max_position": args.residual_token_ce_max_position,
        "residual_token_ce_label_smoothing": args.residual_token_ce_label_smoothing,
        "logit_markov_type": args.logit_markov_type,
        "logit_markov_rank": args.logit_markov_rank,
        "logit_markov_scale": args.logit_markov_scale,
        "anchor_logit_distill_w": args.anchor_logit_distill_w,
        "anchor_logit_distill_temperature": args.anchor_logit_distill_temperature,
        "anchor_logit_distill_min_position": args.anchor_logit_distill_min_position,
        "anchor_logit_distill_max_position": args.anchor_logit_distill_max_position,
        "anchor_logit_distill_correct_teacher_only": args.anchor_logit_distill_correct_teacher_only,
        "slot_decay": args.slot_decay,
        "position_balance": args.position_balance,
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
    raw_model = unwrap_model(model)
    # 保存模型权重
    torch.save(raw_model.state_dict(), save_dir / "pytorch_model.bin")
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
    torch.save(raw_model.state_dict(), Path(output_dir) / "pytorch_model.bin")
    with open(Path(output_dir) / "dflash_config.json", "w") as f:
        json.dump(config_payload, f, indent=2)
    with open(Path(output_dir) / "latest_checkpoint.txt", "w") as f:
        f.write(str(save_dir))


def save_best_checkpoint(output_dir: str, epoch: int, global_step: int, model: DFlashDraftModel, args) -> None:
    """保存当前最优权重，自动覆盖上一个最优权重。"""
    best_dir = Path(output_dir) / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), best_dir / "pytorch_model.bin")
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
    unwrap_model(model).load_state_dict(torch.load(checkpoint_dir / "pytorch_model.bin", map_location=device))
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


def hidden_distance(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, loss_type: str) -> torch.Tensor:
    """返回逐样本 hidden 距离 [B]，用于主 hidden loss 之外的结构约束。"""
    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(student_hidden, teacher_hidden, reduction="none").mean(dim=-1)
    if loss_type == "norm_mse":
        return F.mse_loss(
            F.normalize(student_hidden, dim=-1),
            F.normalize(teacher_hidden, dim=-1),
            reduction="none",
        ).sum(dim=-1)
    if loss_type == "raw_mse":
        return F.mse_loss(student_hidden, teacher_hidden, reduction="none").mean(dim=-1)
    raise ValueError(f"Unsupported hidden distance type: {loss_type}")


# 损失函数与准确计算函数
def compute_loss_and_accuracy(
    model: DFlashDraftModel,# Draft Model
    embed_tokens: nn.Module,
    lm_head: nn.Module,# 目标模型的 lm head 层（冻结）
    batch: Dict[str, torch.Tensor],# 一个 batch 数据
    args,# 训练参数
    device: torch.device,
    consistency_scale: float = 1.0,
    causal_residual_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    draft_model = model.module if hasattr(model, "module") else model
    prompt_selected = batch["prompt_selected"].to(device=device, dtype=torch.bfloat16)# 完整 prefill/prefix 上下文 [B, P, L*hidden]
    prompt_position_ids = batch["prompt_position_ids"].to(device=device)# 完整 prefill/prefix 的绝对 position [B, P]
    prompt_attention_mask = batch["prompt_attention_mask"].to(device=device)# prefix padding mask [B, P]
    prompt_lengths = batch["prompt_lengths"].to(device=device)# 每条样本真实 prefix 长度 [B]
    action_selected = batch["action_selected"].to(device=device, dtype=torch.bfloat16)# action历史上下文[B, seq, L*hidden]
    target_hidden = batch["target_hidden"].to(device=device, dtype=torch.bfloat16)# token0..token5最后层[B, seq, hidden]
    tokens = batch["tokens"].to(device=device)# 目标模型生成的 token 序列[B, seq]
    lengths = batch["lengths"].to(device=device)# 目标模型生成的 序列 的长度[B, ]

    batch_size, seq_len, _ = target_hidden.shape
    soft_sum = torch.zeros((), device=device, dtype=torch.float32)
    hidden_sum = torch.zeros((), device=device, dtype=torch.float32)
    cos_sum = torch.zeros((), device=device, dtype=torch.float32)
    anchor_consistency_sum = torch.zeros((), device=device, dtype=torch.float32)
    anchor_consistency_weight_sum = torch.zeros((), device=device, dtype=torch.float32)
    anchor_consistency_pairs = torch.zeros((), device=device, dtype=torch.float32)
    causal_residual_cad_sum = torch.zeros((), device=device, dtype=torch.float32)
    causal_residual_cad_weight_sum = torch.zeros((), device=device, dtype=torch.float32)
    causal_residual_cad_pairs = torch.zeros((), device=device, dtype=torch.float32)
    refined_hidden_sum = torch.zeros((), device=device, dtype=torch.float32)
    refined_hidden_weight_sum = torch.zeros((), device=device, dtype=torch.float32)
    residual_token_ce_sum = torch.zeros((), device=device, dtype=torch.float32)
    residual_token_ce_weight_sum = torch.zeros((), device=device, dtype=torch.float32)
    residual_token_ce_correct = torch.zeros((), device=device, dtype=torch.float32)
    residual_token_ce_total = torch.zeros((), device=device, dtype=torch.float32)
    anchor_logit_distill_sum = torch.zeros((), device=device, dtype=torch.float32)
    anchor_logit_distill_weight_sum = torch.zeros((), device=device, dtype=torch.float32)
    anchor_logit_distill_pairs = torch.zeros((), device=device, dtype=torch.float32)
    weight_sum = torch.zeros((), device=device, dtype=torch.float32)
    total_positions = torch.zeros((), device=device, dtype=torch.float32)
    total_correct = torch.zeros((), device=device, dtype=torch.float32)
    base_total_correct = torch.zeros((), device=device, dtype=torch.float32)
    anchor_correct = torch.zeros(seq_len, device=device, dtype=torch.float32)
    anchor_total = torch.zeros(seq_len, device=device, dtype=torch.float32)
    position_correct = torch.zeros(seq_len, device=device, dtype=torch.float32)
    position_total = torch.zeros(seq_len, device=device, dtype=torch.float32)
    anchor_position_correct = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    anchor_position_total = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    causal_slot0_hidden: Dict[int, torch.Tensor] = {}
    causal_slot0_mask: Dict[int, torch.Tensor] = {}
    far_slot_entries = []
    causal_residual_teacher_hidden: Dict[int, torch.Tensor] = {}
    causal_residual_teacher_mask: Dict[int, torch.Tensor] = {}
    causal_residual_student_entries = []
    anchor_logit_teacher_logits: Dict[int, torch.Tensor] = {}
    anchor_logit_teacher_mask: Dict[int, torch.Tensor] = {}
    anchor_logit_student_entries = []
    # 主循环：
    # - 原始 DFLASH: context=P+A[:anchor]，block=[t_anchor, MASK...]，预测 t_{anchor+1}...
    # - SpecVLA式注入: context=P+A[:anchor+1]，block=[t_anchor, MASK...]，预测 t_{anchor+1}...
    # 后者让 anchor=0 时真正注入 A0；训练/推理都由 target 先走过 anchor token。
    for anchor in range(seq_len):# anchor：一个滑动窗口的起始位置
        if args.include_anchor_hidden:
            # 需要 A_anchor 已存在；slot0 仍然对齐 H_anchor，经 lm_head 预测 t_{anchor+1}。
            if int((lengths > anchor).sum().item()) == 0:
                continue
            max_block_len = min(args.block_size - 1, seq_len - anchor)
            known_token_index = anchor
            ctx_action_count = anchor + 1
            teacher_start = anchor
            target_token_start = anchor + 1
        else:
            if int((lengths > anchor).sum().item()) == 0:
                continue
            max_block_len = min(args.block_size - 1, seq_len - anchor)
            known_token_index = anchor
            ctx_action_count = anchor
            teacher_start = anchor
            target_token_start = anchor + 1
        if max_block_len <= 0:
            continue
        # 构造输入序列
        block_ids = torch.full(
            (batch_size, max_block_len),
            args.mask_token_id,# 指定填充噪声的位置（token id）
            dtype=tokens.dtype,
            device=device,
        )
        anchor_active = lengths > teacher_start
        block_ids[anchor_active, 0] = tokens[anchor_active, known_token_index]# 给输入序列第0位放当前已知 token
        noise_embedding = embed_tokens(block_ids)# 加噪

        ctx_hidden = torch.cat(
            [prompt_selected, action_selected[:, :ctx_action_count, :]],
            dim=1,
        )# [B, prefix_len+ctx_action_count, L*hidden] 上下文：完整 prefill/prefix + 已注入action hidden
        if ctx_action_count > 0:
            action_ctx_mask = (
                torch.arange(ctx_action_count, device=device, dtype=torch.long).unsqueeze(0)
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
        if ctx_action_count > 0:
            action_ctx_positions = (
                action_base_positions.unsqueeze(1)
                + torch.arange(ctx_action_count, device=device, dtype=torch.long).unsqueeze(0)
            )
            ctx_position_ids = torch.cat([prompt_position_ids, action_ctx_positions], dim=1)
        else:
            ctx_position_ids = prompt_position_ids
        noise_position_ids = (
            action_base_positions.unsqueeze(1)
            + known_token_index
            + torch.arange(max_block_len, device=device, dtype=torch.long).unsqueeze(0)
        )
        action_position_ids = (
            known_token_index + torch.arange(max_block_len, device=device, dtype=torch.long).unsqueeze(0)
        ).expand(batch_size, -1)
        prev_token_ids = tokens[:, known_token_index : known_token_index + max_block_len].contiguous()

        # Draft 模型推理
        pred_hidden_output = model(# 输出去噪序列[B, block_size, hidden]
            noise_embedding=noise_embedding, # [B, max_block_len, hidden]
            target_hidden=ctx_hidden,# [B, prefix_len+anchor, L*hidden]
            ctx_position_ids=ctx_position_ids,# [B, prefix_len+anchor]
            noise_position_ids=noise_position_ids,# [B, max_block_len]
            ctx_attention_mask=ctx_attention_mask,# [B, prefix_len+anchor]
            action_position_ids=action_position_ids,# [B, max_block_len] action维度/槽位位置
            prev_token_ids=prev_token_ids if args.causal_residual_type != "none" else None,
            causal_residual_start_index=args.causal_residual_start_index,
            return_base_hidden=args.causal_residual_type != "none",
        )
        if args.causal_residual_type != "none":
            pred_hidden, refined_pred_hidden = pred_hidden_output
        else:
            pred_hidden = pred_hidden_output
            refined_pred_hidden = pred_hidden

        student_hidden = pred_hidden[:, :max_block_len, :].float()# 草稿预测的最终层 hidden [B, block, hidden]
        refined_student_hidden = refined_pred_hidden[:, :max_block_len, :].float()
        teacher_hidden = target_hidden[:, teacher_start : teacher_start + max_block_len, :].float()# H(已知token..)
        # 草稿预测最终层hidden过lm头取logits
        base_student_logits = lm_head(student_hidden.to(torch.bfloat16)).float()
        student_logits = lm_head(refined_student_hidden.to(torch.bfloat16)).float()
        if args.logit_markov_type != "none":
            student_logits = draft_model.apply_logit_markov_bias(
                student_logits,
                refined_pred_hidden[:, :max_block_len, :],
                prev_token_ids=prev_token_ids,
                start_index=args.causal_residual_start_index,
            ).float()
        target_tokens = tokens[:, target_token_start : target_token_start + max_block_len]

        valid_mask = (
            torch.arange(max_block_len, device=device, dtype=torch.long).unsqueeze(0)
            < (lengths - teacher_start).unsqueeze(1)
        ).float()
        valid_count = valid_mask.sum()
        if valid_count.item() == 0:
            continue
        slot_weights = args.slot_decay ** torch.arange(
            max_block_len,
            device=device,
            dtype=torch.float32,
        )
        target_positions = teacher_start + torch.arange(
            max_block_len,
            device=device,
            dtype=torch.float32,
        )
        if args.position_balance:
            position_weights = 1.0 / (target_positions + 1.0)
        else:
            position_weights = torch.ones_like(target_positions)
        local_indices = torch.arange(max_block_len, device=device, dtype=torch.long)
        target_token_positions = teacher_start + local_indices + 1
        loss_weight = (
            valid_mask
            * slot_weights.unsqueeze(0)
            * position_weights.unsqueeze(0)
        )
        current_weight_sum = loss_weight.sum()
        if current_weight_sum.item() == 0:
            continue
        weight_sum += current_weight_sum

        # SpecVLA 风格的 soft distribution 交叉熵：学习 teacher 的相对偏好，而不是 hard token id。
        if args.soft_w > 0:
            with torch.no_grad():
                teacher_logits = lm_head(teacher_hidden.to(torch.bfloat16)).float()
                teacher_probs = F.softmax(teacher_logits / args.soft_temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / args.soft_temperature, dim=-1)
            soft_ce = -(teacher_probs * student_log_probs).sum(dim=-1) * (args.soft_temperature ** 2)
            soft_sum += (soft_ce * loss_weight).sum()

        # hidden 蒸馏是主任务；默认 SmoothL1，参考 SpecVLA 的 hidden/value regression 经验。
        if args.hidden_loss_type == "smooth_l1":
            hidden_reg = F.smooth_l1_loss(student_hidden, teacher_hidden, reduction="none").mean(dim=-1)
        elif args.hidden_loss_type == "cosine":
            hidden_reg = 1.0 - F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)
        elif args.hidden_loss_type == "norm_mse":
            hidden_reg = F.mse_loss(
                F.normalize(student_hidden, dim=-1),
                F.normalize(teacher_hidden, dim=-1),
                reduction="none",
            ).sum(dim=-1)
        else:
            hidden_reg = F.mse_loss(student_hidden, teacher_hidden, reduction="none").mean(dim=-1)
        hidden_sum += (hidden_reg * loss_weight).sum()
        cos_reg = 1.0 - F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)
        cos_sum += (cos_reg * loss_weight).sum()

        base_pred_tokens = base_student_logits.argmax(dim=-1)
        base_correct_mask = (base_pred_tokens == target_tokens) & valid_mask.bool()
        pred_tokens = student_logits.argmax(dim=-1)# 草稿模型走贪婪解码
        correct_mask = (pred_tokens == target_tokens) & valid_mask.bool()

        if args.refined_hidden_w > 0 and args.causal_residual_type != "none":
            refined_pos_mask = (
                (local_indices >= args.causal_residual_start_index)
                & (target_token_positions >= args.refined_hidden_min_position)
                & (target_token_positions <= args.refined_hidden_max_position)
            )
            refined_weight = loss_weight * refined_pos_mask.unsqueeze(0).float()
            current_refined_weight = refined_weight.sum()
            if current_refined_weight.item() > 0:
                refined_hidden_reg = hidden_distance(
                    refined_student_hidden.float(),
                    teacher_hidden.float(),
                    args.refined_hidden_loss_type,
                )
                refined_hidden_sum += (refined_hidden_reg * refined_weight).sum()
                refined_hidden_weight_sum += current_refined_weight

        if args.residual_token_ce_w > 0 and args.causal_residual_type != "none":
            residual_ce_pos_mask = (
                (local_indices >= args.causal_residual_start_index)
                & (target_token_positions >= args.residual_token_ce_min_position)
                & (target_token_positions <= args.residual_token_ce_max_position)
            )
            residual_ce_weight = loss_weight * residual_ce_pos_mask.unsqueeze(0).float()
            current_residual_ce_weight = residual_ce_weight.sum()
            if current_residual_ce_weight.item() > 0:
                residual_ce = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.shape[-1]),
                    target_tokens.reshape(-1),
                    reduction="none",
                    label_smoothing=args.residual_token_ce_label_smoothing,
                ).view_as(target_tokens)
                residual_token_ce_sum += (residual_ce * residual_ce_weight).sum()
                residual_token_ce_weight_sum += current_residual_ce_weight
                residual_ce_valid_mask = valid_mask.bool() & residual_ce_pos_mask.unsqueeze(0)
                residual_token_ce_correct += (correct_mask & residual_ce_valid_mask).sum().float()
                residual_token_ce_total += residual_ce_valid_mask.sum().float()

        if args.anchor_consistency_w > 0:
            for local_pos in range(max_block_len):
                target_pos = teacher_start + local_pos
                pos_mask = valid_mask[:, local_pos].bool()
                if local_pos == 0:
                    causal_slot0_hidden[target_pos] = student_hidden[:, local_pos, :]
                    causal_slot0_mask[target_pos] = pos_mask
                else:
                    far_slot_entries.append(
                        (
                            target_pos,
                            student_hidden[:, local_pos, :],
                            pos_mask,
                            loss_weight[:, local_pos].detach(),
                        )
                    )

        if args.causal_residual_cad_w > 0:
            for local_pos in range(max_block_len):
                target_pos = teacher_start + local_pos
                target_token_position = target_pos + 1
                pos_mask = valid_mask[:, local_pos].bool()
                if local_pos == 0:
                    teacher_mask = pos_mask
                    if args.causal_residual_cad_correct_teacher_only:
                        teacher_mask = teacher_mask & correct_mask[:, local_pos]
                    causal_residual_teacher_hidden[target_pos] = student_hidden[:, local_pos, :]
                    causal_residual_teacher_mask[target_pos] = teacher_mask
                elif (
                    local_pos >= args.causal_residual_start_index
                    and args.causal_residual_min_position <= target_token_position <= args.causal_residual_max_position
                ):
                    causal_residual_student_entries.append(
                        (
                            target_pos,
                            refined_student_hidden[:, local_pos, :],
                            pos_mask,
                            loss_weight[:, local_pos].detach(),
                        )
                    )

        if args.anchor_logit_distill_w > 0:
            for local_pos in range(max_block_len):
                target_pos = teacher_start + local_pos
                target_token_position = target_pos + 1
                pos_mask = valid_mask[:, local_pos].bool()
                if local_pos == 0:
                    teacher_mask = pos_mask
                    if args.anchor_logit_distill_correct_teacher_only:
                        teacher_mask = teacher_mask & correct_mask[:, local_pos]
                    anchor_logit_teacher_logits[target_pos] = student_logits[:, local_pos, :]
                    anchor_logit_teacher_mask[target_pos] = teacher_mask
                elif (
                    local_pos >= args.causal_residual_start_index
                    and args.anchor_logit_distill_min_position <= target_token_position <= args.anchor_logit_distill_max_position
                ):
                    anchor_logit_student_entries.append(
                        (
                            target_pos,
                            student_logits[:, local_pos, :],
                            pos_mask,
                            loss_weight[:, local_pos].detach(),
                        )
                    )

        base_total_correct += base_correct_mask.sum().float()
        total_correct += correct_mask.sum().float()
        total_positions += valid_count
        anchor_total[anchor] += valid_count
        anchor_correct[anchor] += correct_mask.sum().float()
        for local_pos in range(max_block_len):
            target_pos = teacher_start + local_pos
            pos_mask = valid_mask[:, local_pos].bool()
            pos_count = pos_mask.sum().float()
            if pos_count.item() == 0:
                continue
            pos_correct = correct_mask[:, local_pos].sum().float()
            position_total[target_pos] += pos_count
            position_correct[target_pos] += pos_correct
            anchor_position_total[anchor, target_pos] += pos_count
            anchor_position_correct[anchor, target_pos] += pos_correct

    if args.anchor_consistency_w > 0:
        for target_pos, far_hidden, far_mask, far_weight in far_slot_entries:
            ref_hidden = causal_slot0_hidden.get(target_pos)
            ref_mask = causal_slot0_mask.get(target_pos)
            if ref_hidden is None or ref_mask is None:
                continue
            pair_mask = far_mask & ref_mask
            if int(pair_mask.sum().item()) == 0:
                continue
            pair_weight = far_weight * pair_mask.float()
            current_weight = pair_weight.sum()
            if current_weight.item() == 0:
                continue
            consistency_reg = hidden_distance(
                far_hidden.float(),
                ref_hidden.detach().float(),
                args.anchor_consistency_type,
            )
            anchor_consistency_sum += (consistency_reg * pair_weight).sum()
            anchor_consistency_weight_sum += current_weight
            anchor_consistency_pairs += pair_mask.sum().float()

    if args.causal_residual_cad_w > 0:
        for target_pos, weak_refined_hidden, weak_mask, weak_weight in causal_residual_student_entries:
            strong_hidden = causal_residual_teacher_hidden.get(target_pos)
            strong_mask = causal_residual_teacher_mask.get(target_pos)
            if strong_hidden is None or strong_mask is None:
                continue
            pair_mask = weak_mask & strong_mask
            if int(pair_mask.sum().item()) == 0:
                continue
            pair_weight = weak_weight * pair_mask.float()
            current_weight = pair_weight.sum()
            if current_weight.item() == 0:
                continue
            residual_cad_reg = hidden_distance(
                weak_refined_hidden.float(),
                strong_hidden.detach().float(),
                args.causal_residual_cad_type,
            )
            causal_residual_cad_sum += (residual_cad_reg * pair_weight).sum()
            causal_residual_cad_weight_sum += current_weight
            causal_residual_cad_pairs += pair_mask.sum().float()

    if args.anchor_logit_distill_w > 0:
        temperature = float(args.anchor_logit_distill_temperature)
        for target_pos, weak_logits, weak_mask, weak_weight in anchor_logit_student_entries:
            strong_logits = anchor_logit_teacher_logits.get(target_pos)
            strong_mask = anchor_logit_teacher_mask.get(target_pos)
            if strong_logits is None or strong_mask is None:
                continue
            pair_mask = weak_mask & strong_mask
            if int(pair_mask.sum().item()) == 0:
                continue
            pair_weight = weak_weight * pair_mask.float()
            current_weight = pair_weight.sum()
            if current_weight.item() == 0:
                continue
            strong_probs = F.softmax(strong_logits.detach() / temperature, dim=-1)
            weak_log_probs = F.log_softmax(weak_logits / temperature, dim=-1)
            logit_kl = F.kl_div(
                weak_log_probs,
                strong_probs,
                reduction="none",
            ).sum(dim=-1) * (temperature ** 2)
            anchor_logit_distill_sum += (logit_kl * pair_weight).sum()
            anchor_logit_distill_weight_sum += current_weight
            anchor_logit_distill_pairs += pair_mask.sum().float()

    loss_denom = weight_sum.clamp_min(1.0)
    metric_denom = total_positions.clamp_min(1.0)
    consistency_denom = anchor_consistency_weight_sum.clamp_min(1.0)
    residual_cad_denom = causal_residual_cad_weight_sum.clamp_min(1.0)
    refined_hidden_denom = refined_hidden_weight_sum.clamp_min(1.0)
    residual_token_ce_denom = residual_token_ce_weight_sum.clamp_min(1.0)
    anchor_logit_distill_denom = anchor_logit_distill_weight_sum.clamp_min(1.0)
    soft_loss = soft_sum / loss_denom
    hidden_loss = hidden_sum / loss_denom
    cos_loss = cos_sum / loss_denom
    anchor_consistency_loss = anchor_consistency_sum / consistency_denom
    causal_residual_cad_loss = causal_residual_cad_sum / residual_cad_denom
    refined_hidden_loss = refined_hidden_sum / refined_hidden_denom
    residual_token_ce_loss = residual_token_ce_sum / residual_token_ce_denom
    anchor_logit_distill_loss = anchor_logit_distill_sum / anchor_logit_distill_denom
    hidden_component = args.hidden_w * hidden_loss
    soft_component = args.soft_w * soft_loss
    cos_component = args.cos_w * cos_loss
    anchor_consistency_scale = torch.tensor(float(consistency_scale), device=device, dtype=torch.float32)
    anchor_consistency_component = (
        args.anchor_consistency_w * anchor_consistency_scale * anchor_consistency_loss
    )
    causal_residual_cad_scale = torch.tensor(
        float(causal_residual_scale),
        device=device,
        dtype=torch.float32,
    )
    causal_residual_cad_component = (
        args.causal_residual_cad_w * causal_residual_cad_scale * causal_residual_cad_loss
    )
    refined_hidden_component = args.refined_hidden_w * refined_hidden_loss
    residual_token_ce_component = args.residual_token_ce_w * residual_token_ce_loss
    anchor_logit_distill_component = args.anchor_logit_distill_w * anchor_logit_distill_loss
    total_loss = (
        hidden_component
        + soft_component
        + cos_component
        + anchor_consistency_component
        + causal_residual_cad_component
        + refined_hidden_component
        + residual_token_ce_component
        + anchor_logit_distill_component
    )
    accuracy = total_correct / metric_denom
    base_accuracy = base_total_correct / metric_denom
    residual_token_ce_accuracy = residual_token_ce_correct / residual_token_ce_total.clamp_min(1.0)

    return {
        "loss": total_loss,
        "soft_loss": soft_loss,
        "hidden_loss": hidden_loss,
        "cos_loss": cos_loss,
        "anchor_consistency_loss": anchor_consistency_loss,
        "causal_residual_cad_loss": causal_residual_cad_loss,
        "refined_hidden_loss": refined_hidden_loss,
        "residual_token_ce_loss": residual_token_ce_loss,
        "anchor_logit_distill_loss": anchor_logit_distill_loss,
        "soft_component": soft_component,
        "hidden_component": hidden_component,
        "cos_component": cos_component,
        "anchor_consistency_component": anchor_consistency_component,
        "causal_residual_cad_component": causal_residual_cad_component,
        "refined_hidden_component": refined_hidden_component,
        "residual_token_ce_component": residual_token_ce_component,
        "anchor_logit_distill_component": anchor_logit_distill_component,
        "anchor_consistency_scale": anchor_consistency_scale.detach(),
        "causal_residual_cad_scale": causal_residual_cad_scale.detach(),
        "anchor_consistency_pairs": anchor_consistency_pairs.detach(),
        "causal_residual_cad_pairs": causal_residual_cad_pairs.detach(),
        "anchor_logit_distill_pairs": anchor_logit_distill_pairs.detach(),
        "refined_hidden_weight_sum": refined_hidden_weight_sum.detach(),
        "accuracy": accuracy,
        "base_accuracy": base_accuracy,
        "residual_token_ce_accuracy": residual_token_ce_accuracy,
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
    total_soft = 0.0
    total_hidden = 0.0
    total_cos = 0.0
    total_anchor_consistency = 0.0
    total_causal_residual_cad = 0.0
    total_refined_hidden = 0.0
    total_anchor_logit_distill = 0.0
    total_soft_component = 0.0
    total_hidden_component = 0.0
    total_cos_component = 0.0
    total_anchor_consistency_component = 0.0
    total_causal_residual_cad_component = 0.0
    total_refined_hidden_component = 0.0
    total_anchor_logit_distill_component = 0.0
    total_anchor_consistency_scale = 0.0
    total_causal_residual_cad_scale = 0.0
    total_anchor_consistency_pairs = 0.0
    total_causal_residual_cad_pairs = 0.0
    total_anchor_logit_distill_pairs = 0.0
    total_acc = 0.0
    total_base_acc = 0.0
    total_samples = 0
    detail_accumulator = None

    for batch in val_loader:
        metrics = compute_loss_and_accuracy(model, embed_tokens, lm_head, batch, args, device)
        bs = batch["lengths"].shape[0]
        total_loss += metrics["loss"].item() * bs
        total_soft += metrics["soft_loss"].item() * bs
        total_hidden += metrics["hidden_loss"].item() * bs
        total_cos += metrics["cos_loss"].item() * bs
        total_anchor_consistency += metrics["anchor_consistency_loss"].item() * bs
        total_causal_residual_cad += metrics["causal_residual_cad_loss"].item() * bs
        total_refined_hidden += metrics["refined_hidden_loss"].item() * bs
        total_anchor_logit_distill += metrics["anchor_logit_distill_loss"].item() * bs
        total_soft_component += metrics["soft_component"].item() * bs
        total_hidden_component += metrics["hidden_component"].item() * bs
        total_cos_component += metrics["cos_component"].item() * bs
        total_anchor_consistency_component += metrics["anchor_consistency_component"].item() * bs
        total_causal_residual_cad_component += metrics["causal_residual_cad_component"].item() * bs
        total_refined_hidden_component += metrics["refined_hidden_component"].item() * bs
        total_anchor_logit_distill_component += metrics["anchor_logit_distill_component"].item() * bs
        total_anchor_consistency_scale += metrics["anchor_consistency_scale"].item() * bs
        total_causal_residual_cad_scale += metrics["causal_residual_cad_scale"].item() * bs
        total_anchor_consistency_pairs += metrics["anchor_consistency_pairs"].item()
        total_causal_residual_cad_pairs += metrics["causal_residual_cad_pairs"].item()
        total_anchor_logit_distill_pairs += metrics["anchor_logit_distill_pairs"].item()
        total_acc += metrics["accuracy"].item() * bs
        total_base_acc += metrics["base_accuracy"].item() * bs
        total_samples += bs
        detail_accumulator = accumulate_detail_metrics(detail_accumulator, metrics)

    model.train()
    denom = max(total_samples, 1)
    result = {
        "val/loss": total_loss / denom,
        "val/soft_loss": total_soft / denom,
        "val/hidden_loss": total_hidden / denom,
        "val/cos_loss": total_cos / denom,
        "val/anchor_consistency_loss": total_anchor_consistency / denom,
        "val/causal_residual_cad_loss": total_causal_residual_cad / denom,
        "val/refined_hidden_loss": total_refined_hidden / denom,
        "val/anchor_logit_distill_loss": total_anchor_logit_distill / denom,
        "val/soft_component": total_soft_component / denom,
        "val/hidden_component": total_hidden_component / denom,
        "val/cos_component": total_cos_component / denom,
        "val/anchor_consistency_component": total_anchor_consistency_component / denom,
        "val/causal_residual_cad_component": total_causal_residual_cad_component / denom,
        "val/refined_hidden_component": total_refined_hidden_component / denom,
        "val/anchor_logit_distill_component": total_anchor_logit_distill_component / denom,
        "val/anchor_consistency_scale": total_anchor_consistency_scale / denom,
        "val/causal_residual_cad_scale": total_causal_residual_cad_scale / denom,
        "val/anchor_consistency_pairs_per_sample": total_anchor_consistency_pairs / denom,
        "val/causal_residual_cad_pairs_per_sample": total_causal_residual_cad_pairs / denom,
        "val/anchor_logit_distill_pairs_per_sample": total_anchor_logit_distill_pairs / denom,
        "val/accuracy": total_acc / denom,
        "val/base_accuracy": total_base_acc / denom,
    }
    result.update(detail_metrics_to_log("val", detail_accumulator))
    return result


def main():
    args = parse_args()
    ddp_info = init_distributed_if_needed()
    distributed = ddp_info["distributed"]
    world_size = ddp_info["world_size"]
    rank = ddp_info["rank"]
    local_rank = ddp_info["local_rank"]
    is_main = ddp_info["is_main"]
    if args.soft_temperature <= 0:
        raise ValueError("--soft_temperature must be > 0.")
    if args.slot_decay <= 0 or args.slot_decay > 1:
        raise ValueError("--slot_decay must be in (0, 1].")
    for loss_name in (
        "hidden_w",
        "soft_w",
        "cos_w",
        "anchor_consistency_w",
        "causal_residual_cad_w",
        "refined_hidden_w",
        "residual_token_ce_w",
        "anchor_logit_distill_w",
    ):
        if getattr(args, loss_name) < 0:
            raise ValueError(f"--{loss_name} must be >= 0.")
    if args.anchor_consistency_warmup_steps < 0:
        raise ValueError("--anchor_consistency_warmup_steps must be >= 0.")
    if args.causal_residual_cad_warmup_steps < 0:
        raise ValueError("--causal_residual_cad_warmup_steps must be >= 0.")
    if args.anchor_logit_distill_temperature <= 0:
        raise ValueError("--anchor_logit_distill_temperature must be > 0.")
    if args.anchor_logit_distill_min_position < 1:
        raise ValueError("--anchor_logit_distill_min_position must be >= 1.")
    if args.anchor_logit_distill_max_position < args.anchor_logit_distill_min_position:
        raise ValueError("--anchor_logit_distill_max_position must be >= --anchor_logit_distill_min_position.")
    if args.logit_markov_rank <= 0:
        raise ValueError("--logit_markov_rank must be > 0.")
    if args.logit_markov_scale < 0:
        raise ValueError("--logit_markov_scale must be >= 0.")
    if args.causal_residual_rank <= 0:
        raise ValueError("--causal_residual_rank must be > 0.")
    if args.causal_residual_scale < 0:
        raise ValueError("--causal_residual_scale must be >= 0.")
    if args.causal_residual_start_index < 0:
        raise ValueError("--causal_residual_start_index must be >= 0.")
    if args.causal_residual_min_position < 1:
        raise ValueError("--causal_residual_min_position must be >= 1.")
    if args.causal_residual_max_position < args.causal_residual_min_position:
        raise ValueError("--causal_residual_max_position must be >= --causal_residual_min_position.")
    if args.causal_residual_cad_w > 0 and args.causal_residual_type == "none":
        raise ValueError("--causal_residual_cad_w > 0 requires --causal_residual_type hidden.")
    if args.refined_hidden_w > 0 and args.causal_residual_type == "none":
        raise ValueError("--refined_hidden_w > 0 requires --causal_residual_type hidden.")
    if args.anchor_logit_distill_w > 0 and args.logit_markov_type == "none":
        rank0_print(
            is_main,
            "WARNING: --anchor_logit_distill_w > 0 is most useful with --logit_markov_type bias; "
            "otherwise it distills raw draft logits only.",
        )
    if args.action_dim <= 0:
        raise ValueError("--action_dim must be > 0.")
    if args.include_anchor_hidden and args.block_size < 3:
        raise ValueError("--block_size must be >= 3 when --include_anchor_hidden is enabled.")
    if distributed and args.val_split > 0:
        raise ValueError("DDP mode currently supports pure training only. Please set --val_split 0.")
    if args.lr > 1e-3:
        rank0_print(
            is_main,
            f"WARNING: lr={args.lr:g} is very high for DFLASH AdamW training. "
            "This can quickly improve early metrics and then destabilize the draft model."
        )
    set_seed(args.seed)
    device = torch.device(f"cuda:{local_rank}" if distributed else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.run_name is None:
        args.run_name = f"dflash-libero-goal-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    if distributed:
        dist.barrier()
    metrics_log_path = Path(args.output_dir) / "metrics.jsonl"

    # transformer库内置的模型、配置、分词器加载器
    AutoConfig.register("openvla", OpenVLAConfig)# 遇到"openvla"时，使用本地写好的OpenVLAConfig加载配置
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)# 使用PrismaticImageProcessor图像处理器
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)# 使用PrismaticProcessor分词器
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)# 使用 OpenVLAForActionPrediction 来实例化VLA模型

    rank0_print(is_main, "正在加载 OpenVLA ...")
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
    draft_config.dflash_selected_hidden_variant = args.selected_hidden_variant
    draft_config.dflash_block_size = args.block_size# 7
    draft_config.dflash_action_dim = args.action_dim# action token维度数，用于action-dimension embedding
    draft_config.dflash_causal_residual_type = args.causal_residual_type
    draft_config.dflash_causal_residual_rank = args.causal_residual_rank
    draft_config.dflash_causal_residual_scale = args.causal_residual_scale
    draft_config.dflash_logit_markov_type = args.logit_markov_type
    draft_config.dflash_logit_markov_rank = args.logit_markov_rank
    draft_config.dflash_logit_markov_scale = args.logit_markov_scale
    model = DFlashDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)# 实例化草稿模型
    trainable_params = count_trainable_parameters(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    cache_path = get_dataset_file_cache_path(args.datapath)
    if is_main:
        if args.refresh_file_cache:
            print(f"正在刷新数据文件缓存: {cache_path}")
        elif cache_path.exists():
            print(f"正在加载数据文件缓存: {cache_path}")
        else:
            print(f"首次扫描数据目录并建立缓存: {args.datapath}")
        datapath = list_files(args.datapath, refresh_cache=args.refresh_file_cache)
    if distributed:
        dist.barrier()
        datapath = list_files(args.datapath, refresh_cache=False)
    elif not is_main:
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

    train_dataset = OfflineDFlashDataset(
        train_files,
        expected_selected_layers=len(args.target_layer_ids),
        target_layer_ids=args.target_layer_ids,
        selected_hidden_variant=args.selected_hidden_variant,
    )
    collator = DataCollatorForOfflineDFlash()
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    ) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    if val_files:
        val_dataset = OfflineDFlashDataset(
            val_files,
            expected_selected_layers=len(args.target_layer_ids),
            target_layer_ids=args.target_layer_ids,
            selected_hidden_variant=args.selected_hidden_variant,
        )
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
        rank0_print(
            is_main,
            f"已从 {resume_checkpoint_dir} 恢复：epoch={start_epoch} global_step={global_step} "
            f"best_val_loss={best_val_loss} best_val_acc={best_val_acc} patience_counter={patience_counter}",
        )

    config_payload = build_dflash_config_dict(args)
    config_payload.update(
        {
            "train_files": len(train_files),
            "val_files": len(val_files),
            "trainable_params": trainable_params,
            "trainable_params_m": round(trainable_params / 1e6, 2),
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
            "distributed": distributed,
            "world_size": world_size,
            "per_device_batch": args.batch_size,
            "global_effective_batch": args.batch_size * args.gradient_accumulation_steps * world_size,
        }
    )
    if is_main:
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
                "distributed": distributed,
                "world_size": world_size,
            },
        )
        print(
            f"训练集={len(train_files)} "
            f"验证集={len(val_files)} "
            f"Draft参数={trainable_params/1e6:.2f}M "
            f"effective_batch={args.batch_size * args.gradient_accumulation_steps * world_size} "
            f"steps_per_epoch={steps_per_epoch} "
            f"world_size={world_size}"
        )

    swanlab_run = init_swanlab_run(args, config_payload, args.output_dir, resume_run_id=resume_run_id) if is_main else None
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
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            train_soft_sum = 0.0
            train_hidden_sum = 0.0
            train_cos_sum = 0.0
            train_anchor_consistency_sum = 0.0
            train_causal_residual_cad_sum = 0.0
            train_refined_hidden_sum = 0.0
            train_residual_token_ce_sum = 0.0
            train_anchor_logit_distill_sum = 0.0
            train_soft_component_sum = 0.0
            train_hidden_component_sum = 0.0
            train_cos_component_sum = 0.0
            train_anchor_consistency_component_sum = 0.0
            train_causal_residual_cad_component_sum = 0.0
            train_refined_hidden_component_sum = 0.0
            train_residual_token_ce_component_sum = 0.0
            train_anchor_logit_distill_component_sum = 0.0
            train_anchor_consistency_scale_sum = 0.0
            train_causal_residual_cad_scale_sum = 0.0
            train_anchor_consistency_pairs_sum = 0.0
            train_causal_residual_cad_pairs_sum = 0.0
            train_anchor_logit_distill_pairs_sum = 0.0
            train_acc_sum = 0.0
            train_base_acc_sum = 0.0
            train_residual_token_ce_acc_sum = 0.0
            train_detail_accumulator = None
            train_log_steps = 0
            pbar = tqdm(train_loader, desc=f"train {epoch}/{args.num_epochs}", dynamic_ncols=True) if is_main else train_loader
            for batch_idx, batch in enumerate(pbar, start=1):
                if args.anchor_consistency_w > 0 and args.anchor_consistency_warmup_steps > 0:
                    consistency_scale = min(
                        1.0,
                        float(global_step + 1) / float(max(1, args.anchor_consistency_warmup_steps)),
                    )
                else:
                    consistency_scale = 1.0
                if args.causal_residual_cad_w > 0 and args.causal_residual_cad_warmup_steps > 0:
                    causal_residual_scale = min(
                        1.0,
                        float(global_step + 1) / float(max(1, args.causal_residual_cad_warmup_steps)),
                    )
                else:
                    causal_residual_scale = 1.0
                metrics = compute_loss_and_accuracy(
                    model,
                    embed_tokens,
                    lm_head,
                    batch,
                    args,
                    device,
                    consistency_scale=consistency_scale,
                    causal_residual_scale=causal_residual_scale,
                )
                (metrics["loss"] / args.gradient_accumulation_steps).backward()
                log_metrics = distributed_log_metrics(metrics, distributed)

                train_loss_sum += log_metrics["loss"].item()
                train_soft_sum += log_metrics["soft_loss"].item()
                train_hidden_sum += log_metrics["hidden_loss"].item()
                train_cos_sum += log_metrics["cos_loss"].item()
                train_anchor_consistency_sum += log_metrics["anchor_consistency_loss"].item()
                train_causal_residual_cad_sum += log_metrics["causal_residual_cad_loss"].item()
                train_refined_hidden_sum += log_metrics["refined_hidden_loss"].item()
                train_residual_token_ce_sum += log_metrics["residual_token_ce_loss"].item()
                train_anchor_logit_distill_sum += log_metrics["anchor_logit_distill_loss"].item()
                train_soft_component_sum += log_metrics["soft_component"].item()
                train_hidden_component_sum += log_metrics["hidden_component"].item()
                train_cos_component_sum += log_metrics["cos_component"].item()
                train_anchor_consistency_component_sum += log_metrics["anchor_consistency_component"].item()
                train_causal_residual_cad_component_sum += log_metrics["causal_residual_cad_component"].item()
                train_refined_hidden_component_sum += log_metrics["refined_hidden_component"].item()
                train_residual_token_ce_component_sum += log_metrics["residual_token_ce_component"].item()
                train_anchor_logit_distill_component_sum += log_metrics["anchor_logit_distill_component"].item()
                train_anchor_consistency_scale_sum += log_metrics["anchor_consistency_scale"].item()
                train_causal_residual_cad_scale_sum += log_metrics["causal_residual_cad_scale"].item()
                train_anchor_consistency_pairs_sum += log_metrics["anchor_consistency_pairs"].item()
                train_causal_residual_cad_pairs_sum += log_metrics["causal_residual_cad_pairs"].item()
                train_anchor_logit_distill_pairs_sum += log_metrics["anchor_logit_distill_pairs"].item()
                train_acc_sum += log_metrics["accuracy"].item()
                train_base_acc_sum += log_metrics["base_accuracy"].item()
                train_residual_token_ce_acc_sum += log_metrics["residual_token_ce_accuracy"].item()
                train_detail_accumulator = accumulate_detail_metrics(train_detail_accumulator, log_metrics)
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

                    if is_main and global_step % args.log_every_steps == 0:
                        denom_log_steps = max(1, train_log_steps)
                        train_payload = {
                            "event": "train_step",
                            "timestamp": datetime.now().isoformat(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "train/loss": train_loss_sum / denom_log_steps,
                            "train/soft_loss": train_soft_sum / denom_log_steps,
                            "train/hidden_loss": train_hidden_sum / denom_log_steps,
                            "train/cos_loss": train_cos_sum / denom_log_steps,
                            "train/anchor_consistency_loss": train_anchor_consistency_sum / denom_log_steps,
                            "train/causal_residual_cad_loss": train_causal_residual_cad_sum / denom_log_steps,
                            "train/refined_hidden_loss": train_refined_hidden_sum / denom_log_steps,
                            "train/residual_token_ce_loss": train_residual_token_ce_sum / denom_log_steps,
                            "train/anchor_logit_distill_loss": train_anchor_logit_distill_sum / denom_log_steps,
                            "train/soft_component": train_soft_component_sum / denom_log_steps,
                            "train/hidden_component": train_hidden_component_sum / denom_log_steps,
                            "train/cos_component": train_cos_component_sum / denom_log_steps,
                            "train/anchor_consistency_component": train_anchor_consistency_component_sum / denom_log_steps,
                            "train/causal_residual_cad_component": train_causal_residual_cad_component_sum / denom_log_steps,
                            "train/refined_hidden_component": train_refined_hidden_component_sum / denom_log_steps,
                            "train/residual_token_ce_component": train_residual_token_ce_component_sum / denom_log_steps,
                            "train/anchor_logit_distill_component": train_anchor_logit_distill_component_sum / denom_log_steps,
                            "train/anchor_consistency_scale": train_anchor_consistency_scale_sum / denom_log_steps,
                            "train/causal_residual_cad_scale": train_causal_residual_cad_scale_sum / denom_log_steps,
                            "train/anchor_consistency_pairs_per_batch": train_anchor_consistency_pairs_sum / denom_log_steps,
                            "train/causal_residual_cad_pairs_per_batch": train_causal_residual_cad_pairs_sum / denom_log_steps,
                            "train/anchor_logit_distill_pairs_per_batch": train_anchor_logit_distill_pairs_sum / denom_log_steps,
                            "train/accuracy": train_acc_sum / denom_log_steps,
                            "train/base_accuracy": train_base_acc_sum / denom_log_steps,
                            "train/residual_token_ce_accuracy": train_residual_token_ce_acc_sum / denom_log_steps,
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
                        print(
                            f"train step={global_step} epoch={epoch} "
                            f"loss={train_payload['train/loss']:.4f} "
                            f"soft={train_payload['train/soft_loss']:.4f} "
                            f"soft*={train_payload['train/soft_component']:.4f} "
                            f"h={train_payload['train/hidden_loss']:.4f} "
                            f"h*={train_payload['train/hidden_component']:.4f} "
                            f"cos={train_payload['train/cos_loss']:.4f} "
                            f"cos*={train_payload['train/cos_component']:.4f} "
                            f"anc={train_payload['train/anchor_consistency_loss']:.4f} "
                            f"anc*={train_payload['train/anchor_consistency_component']:.4f} "
                            f"rescad={train_payload['train/causal_residual_cad_loss']:.4f} "
                            f"rescad*={train_payload['train/causal_residual_cad_component']:.4f} "
                            f"alogit={train_payload['train/anchor_logit_distill_loss']:.4f} "
                            f"alogit*={train_payload['train/anchor_logit_distill_component']:.4f} "
                            f"refh={train_payload['train/refined_hidden_loss']:.4f} "
                            f"refh*={train_payload['train/refined_hidden_component']:.4f} "
                            f"acc={train_payload['train/accuracy']:.3f} "
                            f"base_acc={train_payload['train/base_accuracy']:.3f} "
                            f"lr={train_payload['train/lr']:.2e}",
                            flush=True,
                        )
                        train_loss_sum = 0.0
                        train_soft_sum = 0.0
                        train_hidden_sum = 0.0
                        train_cos_sum = 0.0
                        train_anchor_consistency_sum = 0.0
                        train_causal_residual_cad_sum = 0.0
                        train_refined_hidden_sum = 0.0
                        train_residual_token_ce_sum = 0.0
                        train_anchor_logit_distill_sum = 0.0
                        train_soft_component_sum = 0.0
                        train_hidden_component_sum = 0.0
                        train_cos_component_sum = 0.0
                        train_anchor_consistency_component_sum = 0.0
                        train_causal_residual_cad_component_sum = 0.0
                        train_refined_hidden_component_sum = 0.0
                        train_residual_token_ce_component_sum = 0.0
                        train_anchor_logit_distill_component_sum = 0.0
                        train_anchor_consistency_scale_sum = 0.0
                        train_causal_residual_cad_scale_sum = 0.0
                        train_anchor_consistency_pairs_sum = 0.0
                        train_causal_residual_cad_pairs_sum = 0.0
                        train_anchor_logit_distill_pairs_sum = 0.0
                        train_acc_sum = 0.0
                        train_base_acc_sum = 0.0
                        train_residual_token_ce_acc_sum = 0.0
                        train_detail_accumulator = None
                        train_log_steps = 0

                if is_main:
                    pbar.set_postfix(
                        s=f"{log_metrics['soft_loss'].item():.3f}",
                        h=f"{log_metrics['hidden_loss'].item():.3f}",
                        c=f"{log_metrics['cos_loss'].item():.3f}",
                        anc=f"{log_metrics['anchor_consistency_loss'].item():.3f}",
                        rescad=f"{log_metrics['causal_residual_cad_loss'].item():.3f}",
                        alogit=f"{log_metrics['anchor_logit_distill_loss'].item():.3f}",
                        refh=f"{log_metrics['refined_hidden_loss'].item():.3f}",
                        L=f"{log_metrics['loss'].item():.3f}",
                        acc=f"{log_metrics['accuracy'].item():.3f}",
                        bacc=f"{log_metrics['base_accuracy'].item():.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        step=global_step,
                    )

            # ── 验证 + 早停 + 最优权重 ──
            do_eval = (
                val_loader is not None
                and epoch % args.eval_every == 0
            )
            if do_eval:
                if distributed:
                    raise RuntimeError("Validation is disabled in DDP mode; use --val_split 0.")
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
                    f"验证 epoch={epoch} | loss={current_val_loss:.4f} "
                    f"soft={val_metrics['val/soft_loss']:.4f} "
                    f"soft*={val_metrics['val/soft_component']:.4f} "
                    f"hidden={val_metrics['val/hidden_loss']:.4f} "
                    f"hidden*={val_metrics['val/hidden_component']:.4f} "
                    f"cos={val_metrics['val/cos_loss']:.4f} "
                    f"cos*={val_metrics['val/cos_component']:.4f} "
                    f"anc={val_metrics['val/anchor_consistency_loss']:.4f} "
                    f"anc*={val_metrics['val/anchor_consistency_component']:.4f} "
                    f"rescad={val_metrics['val/causal_residual_cad_loss']:.4f} "
                    f"rescad*={val_metrics['val/causal_residual_cad_component']:.4f} "
                    f"alogit={val_metrics['val/anchor_logit_distill_loss']:.4f} "
                    f"alogit*={val_metrics['val/anchor_logit_distill_component']:.4f} "
                    f"refh={val_metrics['val/refined_hidden_loss']:.4f} "
                    f"refh*={val_metrics['val/refined_hidden_component']:.4f} "
                    f"acc={current_val_acc:.3f} "
                    f"base_acc={val_metrics['val/base_accuracy']:.3f} "
                    f"| best_loss={best_val_loss:.4f} best_acc={best_val_acc:.3f} "
                    f"patience={patience_counter}/{args.patience}"
                )

                if patience_counter >= args.patience:
                    print(f"早停触发：验证 loss 连续 {args.patience} 个 epoch 未下降，停止训练。")
                    break

            # 按 epoch 保存 checkpoint
            if epoch % args.save_every == 0:
                if is_main:
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
                if distributed:
                    dist.barrier()
    except KeyboardInterrupt:
        rank0_print(is_main, "检测到手动中断，正在保存 latest checkpoint 以便续训 ...")
        if is_main:
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
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
