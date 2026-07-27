"""
modeling_prismatic.py

Prismatic VLM（视觉-语言模型） 的 HuggingFace 风格实现，以及它的机器人动作预测扩展（OpenVLA）

PrismaticVisionBackbone（视觉编码器 timm ViT, 可选双编码器融合）输出: patch 特征，PrismaticProjector(MLP 投影: vision_dim → llm_dim)，多模态 Embedding 拼接（[BOS] + [视觉Patch] + [文本Tokens]）
Language Model (LLaMA 标准模式: AutoModelForCausalLM 投机采样: LlamaSpecForCausalLM) 输出: logits + KV cache
OpenVLAForActionPrediction token → bin index → action value（可选返回 hidden states）

spec在原本的代码上做了什么：在基类里指定了本地的SpeculativeGenerationMixin（指定generate函数，以免库不一致）；语言模型换了本地的LlamaSpecForCausalLM；在VLM中增加了树验证的分支；增加了保存首尾层hidden states的功能，生成数据
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import os
import numpy as np
import timm# 用于加载ViT
import tokenizers
import torch
import torch.nn as nn
import transformers# HF 标准库，提供 PreTrainedModel , PretrainedConfig 等基类
from timm.models.vision_transformer import LayerScale# ViT 中的一个模块，包含名为 gamma 的参数
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

# 从本地模块导入GenerationMixin及其generate方法，防止transformer库更新导致获取hidden states的方法变动
from local_transformers import SpeculativeGenerationMixin as GenerationMixin
from local_transformers.generation_utils import SpeculativeGenerationMixin as LocalGenerationMixin

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig,SpecVLAConfig

from .modeling_speculation import LlamaSpecForCausalLM# 是支持投机采样的 LLaMA 模型

# Get Logger
logger = logging.getLogger(__name__)


# === 标准的忽略索引值，用于 CrossEntropyLoss 中屏蔽掉不需要计算 loss 的位置
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === 视觉骨干网络 ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
            self,
            use_fused_vision_backbone: bool,# 是否使用双编码器融合
            image_sizes: List[int],# 对应每个编码器的输入图像尺寸
            timm_model_ids: List[str],# timm 模型标识列表，如 "vit_base_patch16_224"
            timm_override_act_layers: List[Optional[str]],
        ) -> None:
            super().__init__()
            self.use_fused_vision_backbone = use_fused_vision_backbone

            # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
            #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
            #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
            assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
            self.featurizer = timm.create_model(# 创建第一个视觉编码器（"alpha"）
                timm_model_ids[0],
                pretrained=False,
                num_classes=0,# 去掉分类头
                img_size=image_sizes[0],
                act_layer=timm_override_act_layers[0],
            )
            self.featurizer.forward = unpack_tuple(# 替换 forward 为 get_intermediate_layers ，输出 倒数第二层 的 patch 特征
                partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
            )
            self.embed_dim = self.featurizer.embed_dim

            # 如果使用融合骨干，再创建第二个编码器（"beta"）
            if self.use_fused_vision_backbone:
                self.fused_featurizer = timm.create_model(# 创建第二个视觉编码器（"beta"）
                    timm_model_ids[1],
                    pretrained=False,
                    num_classes=0,# 去掉分类头
                    img_size=image_sizes[1],
                    act_layer=timm_override_act_layers[1],
                )
                self.fused_featurizer.forward = unpack_tuple(
                    partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
                )
                self.embed_dim += self.fused_featurizer.embed_dim# 合并两个编码器的特征维度

            # 遍历所有子模块，对 LayerScale 应用 monkey-patch
            for module in self.featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

            if self.use_fused_vision_backbone:
                for module in self.fused_featurizer.modules():
                    if isinstance(module, LayerScale):
                        ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:# 非融合模式直接过featurizer
            return self.featurizer(pixel_values)
        # 融合模式下，将输入图像分为两个部分，分别通过两个编码器进行特征提取，最后合并特征
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === 多模态投影器 将视觉特征映射到 LLM 的 embedding 空间===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === 定义输出数据结构 ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):# 继承 HF 的 ModelOutput
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None# # 模型预测的logits
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None# # KV cache
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None# # 各层隐藏状态
    attentions: Optional[Tuple[torch.FloatTensor]] = None# # 注意力权重

    # 扩展 for VLMs
    projector_features: Optional[torch.FloatTensor] = None# 视觉投影后的特征还没进 LLM 之前的样子）
    # 扩展 for Speculations
    attention_mask : Optional[torch.BoolTensor] = None# 投机采样（Speculative Decoding）的后续步骤需要知道 mask 的结构
    multimodal_labels : Optional[torch.BoolTensor] = None

# === 基类 提供保存加载模型状态字典、权重初始化等的方法 ===
class PrismaticPreTrainedModel(PreTrainedModel, LocalGenerationMixin):# 继承标准基类以及本地写的 LocalGenerationMixin
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]# # FSDP 时不拆分
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True# # 支持 Flash 
    
    # 显式覆盖generate方法，确保使用本地GenerationMixin中的实现
    generate = LocalGenerationMixin.generate

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa

# === 核心 VLM 模型 组装 ===
class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # 检查 use_fused_vision_backbone 是否存在
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")
        # 检查 timm 版本是否在 {0.9.10, 0.9.11, 0.9.12, 0.9.16} 中
        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )
        # 检查 transformers==4.40.1 和 tokenizers==0.19.1 ，不匹配时只打警告
        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # 视觉编码器实例化
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # 投影层实例化
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # LLM 组件实例化
        # 如果 config.use_spec 为 True ，则加载自定义的 speculative LLaMA 模型（ LlamaSpecForCausalLM ），否则加载标准 HF 模型
        if config.use_spec:
            print('加载本地投机用语言模型LlamaSpecForCausalLM')
            self.language_model = LlamaSpecForCausalLM(config=config.text_config,attn_implementation=config._attn_implementation)
        else:
            print('加载标准HF语言模型AutoModelForCausalLM')
            self.language_model = AutoModelForCausalLM.from_config(
                config.text_config, attn_implementation=config._attn_implementation
            )

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF 标准调用，会触发 _init_weights 并设置 gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` 样板方法 ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === VLM 模型 `forward()` 逻辑 ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        multimodal_tree_position_ids: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        logit_token_range: Optional[Tuple[int, int]] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        return_multimodal_labels = None

        # === 投机前向 ===
        # === 进入 tree decoding 模式 ===
        """
        投机需要大模型验证
        要在一棵树的多个分支上并行做 attention，普通的 causal mask （三角形 mask）不行，因为树的拓扑结构不是简单的直线。
        所以需要:
        自定义 tree_mask ：定义树中每个 token 能看到哪些祖先 token
        自定义 position_ids ：树中不同分支上的 token 可能处在相同的位置（position），需要手动指定
        """
        if position_ids is not None:
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"
            # === 核心是LlamaSpecForCausalLM
            # === 传入 tree 结构的 inputs_embeds + position_ids + 自定义 attention_mask ，让 LLM 一次 forward 验证整棵树，输出所有候选位置的 logits ===
            # === 后续的逻辑（在 eagenerate 、 tree_decoding 等函数中）会根据这些 logits 决定接受/拒绝哪些草稿 token ===
            language_model_output = self.language_model(
                input_ids=None,# 不传 input_ids ，只传 inputs_embeds （草稿 token 的 embedding 是手动算好的）
                attention_mask=attention_mask,# attention_mask 是树结构的 mask，不是简单的 causal mask
                position_ids=position_ids,# 传 position_ids （手动指定位置编码）
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                num_logits_to_keep=num_logits_to_keep,
                logit_token_range=logit_token_range,
            )

            return_attention_mask = attention_mask
        # === 标准自回归缓存生成 ===
        # === 每步只输入最新的 1 个 token，复用之前所有 token 的 KV cache ===
        elif input_ids.shape[1] == 1:# 触发条件 ：只有一个 token 输入 + past_key_values 非空
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"


            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                num_logits_to_keep=num_logits_to_keep,
                logit_token_range=logit_token_range,
            )
            return_attention_mask = attention_mask
        # === 纯文本前向 ===
        # === 标准的语言模型前向 ===
        elif pixel_values is None:# 触发条件 ：没有图像输入
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                num_logits_to_keep=num_logits_to_keep,
                logit_token_range=logit_token_range,
            )
            return_attention_mask = attention_mask
        # === 多模态前向（有图像输入） ===
        # === 视觉提取 → 投影 → 拼接 embeddings → 送入 LLM ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):# 触发条件 ：有图像 + batch size 匹配
            # 视觉特征提取
            patch_features = self.vision_backbone(pixel_values)

            # 投影到 LLM 空间
            projected_patch_embeddings = self.projector(patch_features)
            # 构建投影区域的 attention mask
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # 获取文本 embeddings
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            # 拼接多模态 embeddings [BOS] + [视觉Patch] + [文本Tokens]
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], # # <BOS> token
                projected_patch_embeddings, # 图像 patch 特征
                input_embeddings[:, 1:, :]], # 剩余文本 token
                dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            multimodal_position_ids = None
            if multimodal_tree_position_ids is not None:
                tree_len = int(multimodal_tree_position_ids.shape[1])
                prefix_len = int(multimodal_embeddings.shape[1]) - tree_len
                if tree_len <= 0 or prefix_len <= 0:
                    raise ValueError(
                        "multimodal_tree_position_ids must describe a non-empty suffix "
                        "after a non-empty multimodal prefix."
                    )
                prefix_position_ids = torch.arange(
                    prefix_len,
                    device=multimodal_embeddings.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                multimodal_position_ids = torch.cat(
                    [
                        prefix_position_ids,
                        prefix_len
                        + multimodal_tree_position_ids.to(
                            device=multimodal_embeddings.device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=1,
                )

            # 构建 labels（训练时）
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,# 图像 patch 对应的 label 全部设为 IGNORE_INDEX ，即模型只对文本部分计算 loss
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
            # 送入语言模型
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=multimodal_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=multimodal_embeddings,# 传入 inputs_embeds （而非 input_ids ），因为输入已经是拼接好的 embedding
                labels=multimodal_labels,# 传入 labels （训练时）
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                num_logits_to_keep=num_logits_to_keep,
                logit_token_range=logit_token_range,
            )
            return_attention_mask = multimodal_attention_mask
            return_multimodal_labels = multimodal_labels
        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output
        # 返回结果
        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
            attention_mask = return_attention_mask
            #multimodal_labels = return_multimodal_labels
        )

    # === GenerationMixin 方法， HF GenerationMixin 在生成时会调用此方法 ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # 如果有 past_key_values ，则只取 input_ids 的最后一个 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 如果 inputs_embeds 存在且是第一步，则优先使用它
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 保证 pixel_values 被传递到下一步
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)

# === 机器人动作预测 ===
class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):# 继承核心VLM组合模型，扩展为机器人动作预测模型
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        """初始化机器人动作预测模型，将连续的动作值 离散化 为 token。 bins 在 [-1, 1] 之间均匀划分， bin_centers 是每个区间的中点"""
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)# 划分动作区间
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0# 每个区间的中心值

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of# 实际词汇表大小

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        unnorm_key: Optional[str] = None,
        return_hidden_states: bool = False,
        return_all_hidden_states: bool = False,
        hidden_layer_ids: Optional[List[int]] = None,# 指定Dflash需要的hidden states层索引
        legacy_output_hidden: Optional[bool] = None,
        **kwargs: str
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[torch.FloatTensor]]]:
        """Wrapper around .generate() that decodes predicted actions and can return hidden states.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalizing actions
            return_hidden_states: Whether to return the last hidden state
            legacy_output_hidden: Legacy parameter, equivalent to return_hidden_states
            **kwargs: Additional arguments for generate

        Returns:
            If return_hidden_states=False:
                unnormalized actions as numpy array
            Otherwise:
                Tuple of (unnormalized_actions, hidden_states)
        """
        # 处理参数，支持旧的参数命名方式
        if legacy_output_hidden is not None:
            return_hidden_states = legacy_output_hidden

        # 设置generate方法的参数
        if return_hidden_states or return_all_hidden_states:
            kwargs['output_hidden_states'] = True
        
        # 如果特殊的空标记不在提示末尾，则添加它
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            kwargs['attention_mask']=torch.cat(
                (kwargs['attention_mask'], torch.unsqueeze(torch.Tensor([1]), dim=0).to(input_ids.device)), dim=1
            ).to(int)

        # 运行模型生成
        outputs = self.generate(# 调用VLM模型的 generate 方法，生成动作 token
            input_ids=input_ids,
            max_new_tokens=self.get_action_dim(unnorm_key),# 生成 max_new_tokens 个 token，每个 token 对应动作空间的一个维度
            return_dict=True,
            return_dict_in_generate=True,
            **kwargs
        )

        # 获取生成的token IDs
        if hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences
        else:
            generated_ids = outputs
            
        # 从生成的tokens转换为动作值
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):].cpu().numpy()

        discretized_actions = self.vocab_size - predicted_action_token_ids

        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
   
        # 反归一化动作
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # 如果需要返回隐藏状态
        if return_hidden_states or return_all_hidden_states:
            if hidden_layer_ids is not None:
                # DFlash v4 数据格式：
                #   step 0 是完整 prefill/prefix forward，保存整段 prefix 的多层 hidden；
                #   step 1..N-1 是已生成 action token 的 cached forward hidden，用于预测后续 token。
                # 因此 action_hidden 的长度应为 action_token_len - 1。
                prompt_step_hidden = outputs.hidden_states[0]
                prompt_selected = torch.cat(
                    [prompt_step_hidden[layer_id + 1].cpu()[0] for layer_id in hidden_layer_ids],
                    dim=-1,
                )
                prompt_last = prompt_step_hidden[-1].cpu()[0]
                prompt_length = prompt_step_hidden[-1].shape[1]
                prompt_position_ids = torch.arange(prompt_length, dtype=torch.long)

                action_selected_hidden = []
                action_last_hidden = []
                for step_hidden in outputs.hidden_states[1:]:
                    selected = [step_hidden[layer_id + 1].cpu()[0, -1] for layer_id in hidden_layer_ids]
                    action_selected_hidden.append(torch.cat(selected, dim=-1))
                    action_last_hidden.append(step_hidden[-1].cpu()[0, -1])
                return actions, predicted_action_token_ids, {
                    "prompt_selected": prompt_selected,
                    "prompt_last": prompt_last,
                    "prompt_position_ids": prompt_position_ids,
                    "prompt_length": prompt_length,
                    "action_selected": action_selected_hidden,
                    "action_last": action_last_hidden,
                    "layer_ids": hidden_layer_ids,# 所选层索引，用于训练和推理标识
                }

            if return_all_hidden_states:
                # 全量层路径
                all_layer_hidden = []
                for step_hidden in outputs.hidden_states:
                    all_layer_hidden.append(tuple(layer.cpu()[0] for layer in step_hidden))
                return actions, predicted_action_token_ids, {
                    "all_layers": all_layer_hidden,
                }

            # 传统 SpecVLA 路径：只取首尾层
            first_layer_hidden = []
            last_layer_hidden = []
            for step_hidden in outputs.hidden_states:
                first_layer_hidden.append(step_hidden[0].cpu()[0])
                last_layer_hidden.append(step_hidden[-1].cpu()[0])
            return actions, predicted_action_token_ids, (first_layer_hidden, last_layer_hidden)
        
        # 否则只返回动作
        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
    
            
    #copy the eagenerate and other methods here
    #overwrite the predict_action
