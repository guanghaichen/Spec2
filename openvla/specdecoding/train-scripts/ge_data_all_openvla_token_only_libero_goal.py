"""关键：调用modeling_prismatic.py的OpenVLAForActionPrediction来生成动作和hidden states"""
import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def resolve_default_path(env_names, candidates, fallback):
    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value)
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    return Path(fallback)


DEFAULT_VLA_PATH = resolve_default_path(
    ("VLA_PATH", "OPENVLA_MODEL_PATH"),
    (
        "/data/wulin/models/openvla-7b-finetuned-libero-goal",
        "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal",
    ),
    "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal",
)
DEFAULT_RLDS_ROOT = resolve_default_path(
    ("LIBERO_RLDS_ROOT", "RLDS_ROOT", "DATA_ROOT_DIR"),
    (
        "/data/wulin/datasets/modified_libero_rlds",
        "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/dataset/modified_libero_rlds",
    ),
    "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/dataset/modified_libero_rlds",
)
if os.environ.get("DFLASH_DATA_OUTDIR") or os.environ.get("OUTDIR"):
    DEFAULT_OUTDIR = Path(os.environ.get("DFLASH_DATA_OUTDIR") or os.environ["OUTDIR"])
elif Path("/data/wulin").exists():
    DEFAULT_OUTDIR = Path("/data/wulin/specvla-data/dflash_goal_dataset")
else:
    DEFAULT_OUTDIR = Path("/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset")

#参数：开始/结束的数据idx，gpu-idx，用于多线程并行生成
parser = argparse.ArgumentParser(description='sp')

# 这俩起始结束的idx，后面主循环里实际没用到，属于预留接口
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])# 指定用哪几张 GPU（nargs='+' 表示可接收多个，如 --gpu_index 0 1）
parser.add_argument('--outdir', type=str, default=None)# 输出目录；默认按机器自动选择
parser.add_argument('--vla_path', type=str, default=None)
parser.add_argument('--data_root_dir', type=str, default=None)
parser.add_argument('--dataset_name', type=str, default='libero_goal_no_noops')
parser.add_argument('--shuffle_buffer_size', type=int, default=100_000)
parser.add_argument('--image_aug', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

#Config

class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # 模型家族，这里是 OpenVLA
    pretrained_checkpoint: Union[str, Path] = str(args.vla_path or DEFAULT_VLA_PATH)     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # 仿真环境初始化后等多少步让物体稳定
    num_trials_per_task: int = 50                    # 每个任务试多少次

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    use_spec: bool = False# 是否使用 SpecVLA 相关功能（预留开关）
    save_all_hidden_states: bool = False             # 调试用：保存每步所有层 hidden
    hidden_layer_ids: Optional[list[int]] = [1, 8, 15, 22, 29]     # DFlash 用：只保存指定层（层号按 decoder layer，从0开始）

gen_model_cfg=GenerateConfig()

class DataGenerationConfig:
    # fmt: off
    vla_path: str = str(args.vla_path or DEFAULT_VLA_PATH)                            # Path to OpenVLA model (on HuggingFace Hub)
    shuffle_buffer_size: int = args.shuffle_buffer_size                              # Dataloader shuffle buffer size (can reduce if OOM)
    image_aug: bool = args.image_aug                                          # Whether to train with image augmentations
    # Directory Paths
    data_root_dir: Path = Path(args.data_root_dir or DEFAULT_RLDS_ROOT)        # RLDS 格式数据集的根目录和子集名称
    dataset_name: str = args.dataset_name                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    batch_size: int = 1                                          # 生成时一次处理一条样本（因为需要把每条样本的 hidden state 精确存下来）
#暂时粘贴过来
os.chdir(REPO_ROOT)# 切到 SpecVLA 项目根目录，确保后面能 import 到 openvla 等自定义包
os.environ.setdefault('RANK', '1')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '23456')
#倒入需要的modules
#import openvla
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from openvla.prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset# 读取 RLDS（TensorFlow 序列化格式）数据集，并把原始数据转成模型输入
from openvla.prismatic.vla.action_tokenizer import ActionTokenizer# 把连续的动作向量（夹爪位置、旋转、开合等）离散化成 token，方便语言模型预测。
from transformers import AutoConfig, AutoImageProcessor
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig# HuggingFace Transformers 的自动加载接口
from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig# 模型配置（vision backbone + LLM + action head）
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction# 真正的模型类，继承自 AutoModelForVision2Seq，但输出动作 token
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor# 图像+文本的预处理（resize、normalize、tokenize）
from openvla.prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder# 构造输入给 LLM 的 prompt 模板
from openvla.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import get_processor
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from accelerate import PartialState

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
distributed_state = PartialState()
torch.cuda.set_device(device_id := distributed_state.local_process_index)
torch.cuda.empty_cache()

cfg=DataGenerationConfig()
outdir = str(args.outdir or DEFAULT_OUTDIR)
gen_model_cfg.pretrained_checkpoint = cfg.vla_path
print(f'repo root: {REPO_ROOT}')
print(f'vla path: {cfg.vla_path}')
print(f'rlds root: {cfg.data_root_dir}')
print(f'output dir: {outdir}')
# 加载投影器和动作分词器
processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=False)# 使用已注册的本地 PrismaticProcessor，避免联网拉 HF dynamic module
action_tokenizer = ActionTokenizer(processor.tokenizer)
#Load大模型
quantization_config = None# 不量化
print('正在加载 vla模型')
model = get_model(gen_model_cfg)# 根据配置加载 OpenVLAForActionPrediction 到 GPU
processor = get_processor(gen_model_cfg)# 再次获取 processor（和前面重复，但这里用的是 openvla_utils 里的封装版本，可能带额外配置）
print('正在加载 data')
batch_transform = RLDSBatchTransform(# 把 RLDS 原始数据（图像、动作、指令）转换成模型输入格式
        action_tokenizer,# 用 action_tokenizer 把动作向量变成 token
        processor.tokenizer,# 用 processor.tokenizer 把语言指令变成 input_ids
        image_transform=processor.image_processor.apply_transform,# 对图像进行 resize、 normalize 等变换
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,# 用 prompt_builder_fn 拼成最终 prompt（如果是 v01 版本模型，用 Vicuna 对话模板）
    )
vla_dataset = RLDSDataset(# 底层用 TensorFlow 读取 RLDS 数据流
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple([224, 224]),# 图像最终分辨率（OpenVLA 默认 224×224）
        shuffle_buffer_size=cfg.shuffle_buffer_size,# 数据流里的 shuffle 缓冲区大小，越大随机性越强，但占内存
        image_aug=cfg.image_aug,
    )
collator = PaddedCollatorForActionPrediction(# 构建 DataLoader，把样本按 batch_size 分组，填充到最大长度
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
dataloader = DataLoader(# 构建 DataLoader，把样本按 batch_size 分组，填充到最大长度
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
print('start enumerating')

write_idx = 0
def writedata(name,data_point):
    global write_idx
    if not os.path.exists(name):
        os.makedirs(name)
    torch.save(data_point, f'{name}/data_{write_idx}.ckpt')# 把 PyTorch tensor 字典直接序列化保存。这不是标准 checkpoint，而是单个数据样本的 tensor 包
    write_idx += 1

#from transformers.modeling_outputs import CausalLMOutputWithPast
gen_model_cfg.unnorm_key = gen_model_cfg.task_suite_name# 反归一化动作时用的 key，告诉模型这是 libero_goal 任务集的动作统计量（均值/方差）
total_samples = len(vla_dataset)
sample_num = 0# 处理总数
write_sample_num = 0# 实际写入数
print(f'数据集总样本数: {total_samples}')
# 开始遍历dataloader（tqdm 显示进度条，含已处理数/总数、百分比、速率、预估剩余时间）
pbar = tqdm(dataloader, total=total_samples, desc='生成 DFlash 数据', unit='samples')
for batch_idx, batch in enumerate(pbar):
        # action：最终反归一化后的连续动作向量（实际机器人控制指令）
        # token：模型预测的离散动作 token
        # hidden：隐藏状态
        action,token,hidden = model.predict_action(# OpenVLA 的自定义前向接口OpenVLAForActionPrediction
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),# 转成 bfloat16，占用显存更小
                    unnorm_key=gen_model_cfg.unnorm_key,# 反归一化动作，把模型输出的归一化动作转回真实物理动作范围
                    return_hidden_states=True,# 关键！ 要求模型返回中间层隐藏状态，用于后续训练
                    return_all_hidden_states=gen_model_cfg.save_all_hidden_states,# 是否返回所有层
                    hidden_layer_ids=gen_model_cfg.hidden_layer_ids,# 指定要返回的隐藏状态的层索引
                    do_sample=False# 不采样，而是贪婪解码，走greedy_search，直接取概率最高的动作 token
                )
        # 把需要保存的数据打包成字典 tensor dictionary
        td={
            "input_ids": batch["input_ids"].cpu()[0],
            "pixel_values": batch["pixel_values"],
            "hidden_state": hidden,
            "loss_mask": batch["attention_mask"].cpu()[0],
            "predicted_tokens": token,
            "dflash_data_format": "full_prefix_plus_action_hidden_v4",
        }
        if isinstance(td["hidden_state"], dict) and "action_last" in td["hidden_state"]:
            # v3 格式保存的是 prefill 最后位置 hidden + action token0..token5 hidden。
            # 7 个 action token 只需要 6 个 action hidden，因为 H(token_i) 用于预测 token_{i+1}。
            saved_steps = len(td["hidden_state"]["action_last"]) + 1
        else:
            # 老的 SpecVLA 路径：hidden_state=(first_layer_hidden, last_layer_hidden)，两个 list 的长度都应等于解码步数。
            saved_steps = len(td["hidden_state"][1])
        if saved_steps == len(token):# 只保留"hidden 步数 == 动作 token 数"的样本
            writedata(outdir,td)# 保存张量字典，一个字典对应一次vlm推理
            write_sample_num += 1
        sample_num += 1
        pbar.set_postfix({'valid': write_sample_num, 'drop': sample_num - write_sample_num})
        if sample_num >= total_samples:
            break

pbar.close()
print('generation ended')
print('sample num',sample_num)
print('valid sample num',write_sample_num)
