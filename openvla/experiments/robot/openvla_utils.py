"""OpenVLA的推理工具函数"""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig,SpecVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction# 基本 VLM 模型
from openvla.prismatic.extern.hf.modeling_speculation import (
    SpecVLAforActionPrediction,
    normalize_dflash_tree_mode,
)# 投机采样包装模型 （包裹基本模型 + 草稿模型）
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor# 图像处理器 + 通用处理器

#import the speculative decoding dependency
from openvla.specdecoding.model.cnets import MMModel

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """从checkpoint中加载vla模型，分两大分支：投机模式与标准模式"""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading target model in BF16 (attention backend follows the checkpoint/runtime config)")

    # 注册vla模型的配置类、图像处理器、通用处理器
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoConfig.register("specvla", SpecVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    
    # 使用transformers的from_pretrained加载模型
    print("[*] 使用本地OpenVLAForActionPrediction类并从预训练检查点加载")
    if cfg.use_spec:# 推理时是否用投机解码
        cfg.dflash_tree_mode = normalize_dflash_tree_mode(
            getattr(cfg, "dflash_tree_mode", "off")
        )
        print('load the vla model')
        vla = OpenVLAForActionPrediction.from_pretrained(# 加载vla模型OpenVLAForActionPrediction
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_spec = True# 用投机解码的llm
        )
        # 投机采样参数
        spec_kwargs = dict(
            base_model=vla,# 加载基础模型
            base_model_name_or_path=cfg.pretrained_checkpoint,# 基础模型路径
            ea_model_path=cfg.spec_checkpoint,# 草稿模型路径
            parallel_draft=getattr(cfg, "parallel_draft", False),# 是否并行草稿
            accept_threshold=getattr(cfg, "accept_threshold", None),# 接受阈值
            draft_backend=getattr(cfg, "draft_backend", "eagle"),# 草稿类型
            dflash_block_size=getattr(cfg, "dflash_block_size", 4),# block 大小
            dflash_num_draft_layers=getattr(cfg, "dflash_num_draft_layers", 1),# 草稿层数
            dflash_target_layer_ids=getattr(cfg, "dflash_target_layer_ids", None),# 目标层id
            dflash_mask_token_id=getattr(cfg, "dflash_mask_token_id", None),# 掩码tokenid
            dflash_action_dim=getattr(cfg, "dflash_action_dim", ACTION_DIM),# action维度数
            dflash_use_causal_residual_sampling=getattr(cfg, "dflash_use_causal_residual_sampling", False),
            dflash_confidence_threshold=getattr(cfg, "dflash_confidence_threshold", 0.0),
            dflash_confidence_min_tokens=getattr(cfg, "dflash_confidence_min_tokens", 1),
            dflash_acceptance_mode=getattr(cfg, "dflash_acceptance_mode", "token"),
            dflash_tree_mode=getattr(cfg, "dflash_tree_mode", "off"),
            dflash_tree_branch_position=getattr(cfg, "dflash_tree_branch_position", 0),
            dflash_tree_first_anchor_only=getattr(cfg, "dflash_tree_first_anchor_only", True),
        )
        # 投机采样外层包装（先让基础模型 prefill 处理输入序列，再用草稿模型快速生成候选 token，最后用基础模型验证并接受/拒绝）
        vla = SpecVLAforActionPrediction(**spec_kwargs)

    else:# 不用投机解码，llm是AutoModelForCausalLM
        vla = OpenVLAForActionPrediction.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    
    # Move model to device if not already
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # 加载训练数据集的动作统计量
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    图像增强函数（训练时做了随机裁剪增强，推理时必须做 一致的裁剪 ，避免分布偏移）
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, return_hidden_states=False,return_time=False,center_crop=False,generate_mode=None,accept_threshold=None,return_topk_index=False,token=None,return_dflash_stats=False,return_generation_stats=False,sync_cuda_timing=False):
    """Generates an action with the VLA policy.
    generate_mode控制是否用草稿
    """
    
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # print(f"使用的提示语: {prompt}")
    
    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # 根据不同模式调用推理
    # 投机推理 - 调用 SpecVLAforActionPrediction.predict_action()
    if return_hidden_states:
        action,token,hidden = vla.predict_action(**inputs, unnorm_key=unnorm_key, return_hidden_states=return_hidden_states,do_sample=False)
        return action,token,hidden
    if return_topk_index:
        action,token,hidden = vla.eval_topk(**inputs, unnorm_key=unnorm_key, return_hidden_states=return_hidden_states,do_sample=False)
        return action
    # SpecVLA 上游使用无同步的 paper-style 计时；需要真实 GPU 完成时间时再显式打开。
    if sync_cuda_timing and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    # 标准推理 推理动作
    predict_kwargs = dict(
        **inputs,
        unnorm_key=unnorm_key,
        return_hidden_states=return_hidden_states,
        do_sample=False,
    )
    if generate_mode in {"speculative", "dflash"}:
        predict_kwargs["generate_mode"] = generate_mode
    if return_dflash_stats:
        predict_kwargs["return_dflash_stats"] = True
    action = vla.predict_action(**predict_kwargs)# 调用 OpenVLAForActionPrediction.predict_action()
    if sync_cuda_timing and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    if return_time:
        if return_generation_stats:
            stats_getter = getattr(vla, "get_generation_stats", None)
            generation_stats = stats_getter() if stats_getter is not None else getattr(vla, "last_generation_stats", None)
            return action, (end_time, start_time), generation_stats
        if return_dflash_stats:
            return action[0], (end_time, start_time), action[1]
        return action,(end_time,start_time)
    return action
