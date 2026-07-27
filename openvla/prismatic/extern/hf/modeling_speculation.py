"""包装了LlamaForCausalLM以及SpecVLAforActionPrediction（包在VLA上），添加了dflash的generate方法（草稿、验证）"""
import copy
import heapq
import json
import time

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from openvla.specdecoding.model.cnets import MMModel,PMMModel
from openvla.specdecoding.model.dflash import (
    DFlashDraftModel,
    extract_context_feature,
    normalize_selected_hidden_variant,
    sample as dflash_sample,
)# 导入dflash模型、从目标模型提取hidden的函数、从logits采样token的函数
from openvla.specdecoding.model.cnets import EConfig
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
import safetensors

from .utils import *
from .kv_cache import initialize_past_key_values

import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache

logger = logging.get_logger(__name__)


def normalize_dflash_tree_mode(value) -> str:
    """Normalize CLI/YAML aliases to off or the dynamic DDTree mode."""
    if value is False or value is None:
        return "off"
    normalized = str(value).strip().lower()
    if normalized in {"off", "false", "0", "none", "disabled"}:
        return "off"
    if normalized in {"ddtree", "single_fork"}:
        return "ddtree"
    raise ValueError("dflash_tree_mode must be 'off' or 'ddtree'.")


def normalize_dflash_target_logits_mode(value) -> str:
    """Normalize target output-projection modes used by DFlash."""
    normalized = str(value or "full").strip().lower()
    aliases = {
        "full": "full",
        "shadow": "shadow",
        "action_only": "action_only",
        "action-only": "action_only",
    }
    if normalized not in aliases:
        raise ValueError(
            "dflash_target_logits_mode must be 'full', 'shadow', or 'action_only'."
        )
    return aliases[normalized]


def normalize_dflash_verify_skip_mode(value) -> str:
    """Normalize verification-skip modes without silently enabling unsafe skips."""
    normalized = str(value or "off").strip().lower()
    if normalized in {"off", "false", "0", "none", "disabled"}:
        return "off"
    if normalized in {"shadow", "route", "active"}:
        return normalized
    raise ValueError(
        "dflash_verify_skip_mode must be 'off', 'shadow', 'route', or 'active'."
    )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
):
    """
    Create a causal mask for bi-directional self-attention.

    Args:
        input_ids_shape (torch.Size): The shape of input_ids tensor, typically (batch_size, tgt_len).
        dtype (torch.dtype): The data type of the mask.
        device (torch.device): The device on which the mask will be placed.
        past_key_values_length (int, optional): The length of past key values. Default is 0.

    Returns:
        torch.Tensor: The causal mask tensor.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    Args:
        mask (torch.Tensor): The attention mask tensor of shape `[bsz, seq_len]`.
        dtype (torch.dtype): The data type of the mask.
        tgt_len (Optional[int], optional): The target sequence length. If None, it defaults to the source sequence length.

    Returns:
        torch.Tensor: The expanded mask tensor.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )

class LlamaSpecForCausalLM(LlamaForCausalLM):
    def __init__(self,config:LlamaConfig,attn_implementation):
        super().__init__(config=config)
        self.tree_mask = None
        return
    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )


        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = combined_attention_mask.min()

        return combined_attention_mask
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        logit_token_range: Optional[Tuple[int, int]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #print('customized tree mask')
        #print(attention_mask.shape)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if num_logits_to_keep < 0:
            raise ValueError("num_logits_to_keep must be >= 0.")
        logits_hidden_states = (
            hidden_states[:, -num_logits_to_keep:, :]
            if num_logits_to_keep > 0
            else hidden_states
        )
        if logit_token_range is not None:
            token_start, token_end = (int(value) for value in logit_token_range)
            if not 0 <= token_start < token_end <= self.vocab_size:
                raise ValueError(
                    "logit_token_range must be inside the language-model vocabulary: "
                    f"got [{token_start}, {token_end}) for vocab_size={self.vocab_size}."
                )
            if labels is not None:
                raise ValueError("Reduced-vocabulary logits cannot be used with labels.")
            logits = F.linear(
                logits_hidden_states,
                self.lm_head.weight[token_start:token_end],
                None if self.lm_head.bias is None else self.lm_head.bias[token_start:token_end],
            )
        elif self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(logits_hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(logits_hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        #print('past kv type',type(outputs.past_key_values))
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        #labels: torch.LongTensor = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        #print('customized forward!!!!!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()
        #print('past seen tokens',past_seen_tokens)
        #print(past_seen_tokens)
        #if hasattr(self, "tree_mask") and self.tree_mask is not None:
        #    cache_position = position_ids
        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        #print(cache_position)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        #else:
        #    cache_position = position_ids
        #print('cache position',cache_position)
        #print('past seen tokens',past_seen_tokens)
        #TODO:Update this function to fullfill the requirements.
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
        #print('causal mack',causal_mask[0][0][-20:][:,-20:])
        #print(self.tree_mask)
        #print(position_ids)
        # embed positions
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        #print(hidden_states.shape)
        #print(causal_mask.shape)
        #print('verify position ids',position_ids)
        #print()
        #print('model forward use cache')
        #print(use_cache)

        for decoder_layer in self.model.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.model.gradient_checkpointing and self.model.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            #loss = None,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        #disable this feature
        #to specify the attention mask.
        '''if self.config._attn_implementation == "sdpa":
            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument,
            # in order to dispatch on Flash Attention 2.
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
            ):
                #print('return None')
                return None'''
        
        dtype, device = input_tensor.dtype, input_tensor.device
        #print(attention_mask.shape)
        #print(input_tensor.shape)
        #print('cache_position',cache_position.shape)
        #print('past see tokens',past_seen_tokens)
        min_dtype = torch.finfo(dtype).min
        #max_dtype = torch.finfo(dtype).max
        sequence_length = input_tensor.shape[1]
        #print('sequence length',sequence_length)
        #print('attention mask',attention_mask)
        if hasattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
            #print('static cache')
        else:  # dynamic cache
            if hasattr(self, "tree_mask") and self.tree_mask is not None:
                target_length=past_seen_tokens + sequence_length
            elif isinstance(attention_mask, torch.Tensor):
                target_length = attention_mask.shape[-1]
            else:
                target_length = past_seen_tokens + sequence_length + 1
            '''target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )'''
        #print('target length',target_length)

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if hasattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value"):
            causal_mask = torch.triu(causal_mask, diagonal=1+past_seen_tokens)
        elif sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        #print('causal mask',causal_mask[:, -2])
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        #print('causal mask shape',causal_mask.shape)
        #print('causal mask num',causal_mask[0][0][-1])
        #print('attention mask shape',attention_mask.shape)
        #print('causal mask',causal_mask[:, :, -2])
        #print('causal mask',causal_mask[:, :, -2])
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            #print('update based on attention mask')
            if attention_mask.dim() == 2:
                #print('dim = 2')
                mask_length = attention_mask.shape[-1]
                #print('mask length',mask_length)
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                #print(padding_mask)
                padding_mask = padding_mask == 0
                #print('padding mask',padding_mask)
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                #print('dim = 4')
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice
        #print('no tree mask')
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            #print('tree_mask',tree_mask)
            tree_len = tree_mask.size(-1)
            causal_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = min_dtype
        #else:
        #    print('no tree mask')
        #print('final tree mask')
        #print(causal_mask.shape)
        #print('causal mask',causal_mask[:, :, -2])
        #print(causal_mask.shape)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class SpecVLAforActionPrediction(nn.Module):
    '''def __init__(self,openvla=None,head=None):
        self.base_model = openvla
        self.ea_layer = head'''
    def __init__(
            self,
            base_model,# 基础 VLM（OpenVLAForActionPrediction）
            base_model_name_or_path,# 基础模型路径（用于加载 tokenizer）
            ea_model_path,# 草稿模型 checkpoint 路径
            parallel_draft=False,
            total_token=None,
            depth=None,
            top_k=None,
            threshold=None,
            accept_threshold=None,
            draft_backend="eagle",# 后端类型，设为 "dflash" 启用 DFlash
            dflash_block_size=4,# DFlash 块大小
            dflash_num_draft_layers=1,# 草稿模型层数    
            dflash_target_layer_ids=None,# 目标层id
            dflash_mask_token_id=None,# 掩码tokenid
            dflash_action_dim=7,# action token维度数，用于DFlash action-dimension embedding
            dflash_include_anchor_hidden=False,# 是否启用SpecVLA式当前anchor hidden注入
            dflash_selected_hidden_variant="target_layers",# DFlash条件hidden层选择，默认保持原始target layers
            dflash_use_causal_residual_sampling=False,# 是否在推理采样时启用前序token残差修正
            dflash_confidence_threshold=0.0,# 动作顺序头低于该置信度时缩短草稿块；0 表示关闭
            dflash_confidence_min_tokens=1,# 置信截断前至少生成多少个草稿 token
            dflash_acceptance_mode="token",# token | action_group；后者用于动作组级 relaxed 接受
            dflash_tree_mode="off",# off | ddtree；固定预算动态候选树
            dflash_tree_budget=0,# 0=每块使用与线性校验相同的 q-1 个目标节点
            dflash_target_logits_mode="full",# full | shadow | action_only；shadow只检查等价性
            dflash_verify_skip_mode="off",# off | shadow | route | active；仅active会免校验
            dflash_verify_skip_min_top1_prob=1.0,
            dflash_verify_skip_min_margin=1.0,
            dflash_verify_skip_min_base_agreement=1.0,
            dflash_temporal_route_min_cosine=1.0,
            dflash_temporal_route_stop_on_reject=True,
            dflash_temporal_fuse_verify=True,
            dflash_temporal_prefill_fusion=False,
            dflash_temporal_prefill_min_stable_actions=3,
            dflash_temporal_prefill_tree=False,
            dflash_temporal_prefill_tree_max_candidates=3,
            dflash_temporal_prefill_tree_min_history=2,
            dflash_verify_skip_min_temporal_cosine=1.0,
            dflash_verify_skip_min_stable_actions=4,
            dflash_verify_skip_max_consecutive=1,
            dflash_profile_stages=False,# 仅诊断使用；会在阶段边界同步CUDA
            dflash_debug_compare_target_ar=False,# 仅诊断：额外串行解码一条目标AR参考链
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.language_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.language_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.draft_backend = draft_backend
        self.dflash_include_anchor_hidden = dflash_include_anchor_hidden
        self.dflash_selected_hidden_variant = normalize_selected_hidden_variant(dflash_selected_hidden_variant)
        self.dflash_use_causal_residual_sampling = dflash_use_causal_residual_sampling
        self.dflash_confidence_threshold = float(dflash_confidence_threshold)
        self.dflash_confidence_min_tokens = max(1, int(dflash_confidence_min_tokens))
        if not 0.0 <= self.dflash_confidence_threshold < 1.0:
            raise ValueError("dflash_confidence_threshold must be in [0, 1).")
        self.dflash_acceptance_mode = str(dflash_acceptance_mode)
        if self.dflash_acceptance_mode not in {"token", "action_group"}:
            raise ValueError("dflash_acceptance_mode must be 'token' or 'action_group'.")
        self.dflash_tree_mode = normalize_dflash_tree_mode(dflash_tree_mode)
        self.dflash_tree_budget = int(dflash_tree_budget)
        if self.dflash_tree_budget < 0:
            raise ValueError("dflash_tree_budget must be >= 0; 0 selects the cost-neutral q-1 budget.")
        self.dflash_target_logits_mode = normalize_dflash_target_logits_mode(
            dflash_target_logits_mode
        )
        self.dflash_verify_skip_mode = normalize_dflash_verify_skip_mode(
            dflash_verify_skip_mode
        )
        self.dflash_verify_skip_min_top1_prob = float(
            dflash_verify_skip_min_top1_prob
        )
        self.dflash_verify_skip_min_margin = float(dflash_verify_skip_min_margin)
        self.dflash_verify_skip_min_base_agreement = float(
            dflash_verify_skip_min_base_agreement
        )
        self.dflash_temporal_route_min_cosine = float(
            dflash_temporal_route_min_cosine
        )
        self.dflash_temporal_route_stop_on_reject = bool(
            dflash_temporal_route_stop_on_reject
        )
        self.dflash_temporal_fuse_verify = bool(dflash_temporal_fuse_verify)
        self.dflash_temporal_prefill_fusion = bool(
            dflash_temporal_prefill_fusion
        )
        self.dflash_temporal_prefill_min_stable_actions = int(
            dflash_temporal_prefill_min_stable_actions
        )
        if self.dflash_temporal_prefill_min_stable_actions < 1:
            raise ValueError(
                "dflash_temporal_prefill_min_stable_actions must be >= 1."
            )
        self.dflash_temporal_prefill_tree = bool(dflash_temporal_prefill_tree)
        self.dflash_temporal_prefill_tree_max_candidates = int(
            dflash_temporal_prefill_tree_max_candidates
        )
        self.dflash_temporal_prefill_tree_min_history = int(
            dflash_temporal_prefill_tree_min_history
        )
        if self.dflash_temporal_prefill_tree_max_candidates < 1:
            raise ValueError(
                "dflash_temporal_prefill_tree_max_candidates must be >= 1."
            )
        if self.dflash_temporal_prefill_tree_min_history not in (1, 2):
            raise ValueError(
                "dflash_temporal_prefill_tree_min_history must be 1 or 2."
            )
        self.dflash_verify_skip_min_temporal_cosine = float(
            dflash_verify_skip_min_temporal_cosine
        )
        self.dflash_verify_skip_min_stable_actions = int(
            dflash_verify_skip_min_stable_actions
        )
        self.dflash_verify_skip_max_consecutive = int(
            dflash_verify_skip_max_consecutive
        )
        for name, value in (
            ("dflash_verify_skip_min_top1_prob", self.dflash_verify_skip_min_top1_prob),
            ("dflash_verify_skip_min_margin", self.dflash_verify_skip_min_margin),
            (
                "dflash_verify_skip_min_base_agreement",
                self.dflash_verify_skip_min_base_agreement,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        self.dflash_profile_stages = bool(dflash_profile_stages)
        self.dflash_debug_compare_target_ar = bool(
            dflash_debug_compare_target_ar
        )
        if not -1.0 <= self.dflash_verify_skip_min_temporal_cosine <= 1.0:
            raise ValueError("dflash_verify_skip_min_temporal_cosine must be in [-1, 1].")
        if not -1.0 <= self.dflash_temporal_route_min_cosine <= 1.0:
            raise ValueError("dflash_temporal_route_min_cosine must be in [-1, 1].")
        if self.dflash_verify_skip_max_consecutive < 1:
            raise ValueError("dflash_verify_skip_max_consecutive must be >= 1.")
        if self.dflash_verify_skip_min_stable_actions < 1:
            raise ValueError("dflash_verify_skip_min_stable_actions must be >= 1.")
        if self.dflash_verify_skip_mode != "off" and self.dflash_tree_mode != "off":
            raise ValueError(
                "Verification-skip calibration currently requires dflash_tree_mode='off' "
                "so every shadow label refers to the same linear draft path."
            )
        self.dflash_causal_residual_type = "none"
        self.dflash_causal_residual_rank = 256
        self.dflash_causal_residual_scale = 1.0
        self.dflash_causal_residual_start_index = 1
        self.dflash_logit_markov_type = "none"
        self.dflash_logit_markov_rank = 256
        self.dflash_logit_markov_scale = 1.0
        self.dflash_action_head_type = "none"
        self.dflash_action_head_rank = 256
        self.dflash_action_token_start = 0
        self.dflash_action_vocab_size = 256
        self.dflash_action_confidence_enabled = False
        self.dflash_mask_token_id = (
            dflash_mask_token_id
            if dflash_mask_token_id is not None
            else self.tokenizer.pad_token_id# ← 自动取 pad_token_id
        )
        dflash_config_path = os.path.join(ea_model_path, "dflash_config.json")# 训练时保存的 dflash_config.json 是 真正的训练配置，以训练配置为准，保证推理时和训练时的 block_size 、 target_layer_ids 一致
        if self.draft_backend == "dflash" and os.path.exists(dflash_config_path):
            with open(dflash_config_path, "r") as f:
                saved_dflash_cfg = json.load(f)
            dflash_block_size = saved_dflash_cfg.get("block_size", dflash_block_size)
            dflash_num_draft_layers = saved_dflash_cfg.get("num_draft_layers", dflash_num_draft_layers)
            dflash_action_dim = saved_dflash_cfg.get("action_dim", dflash_action_dim)
            self.dflash_include_anchor_hidden = saved_dflash_cfg.get(
                "include_anchor_hidden", self.dflash_include_anchor_hidden
            )
            self.dflash_selected_hidden_variant = normalize_selected_hidden_variant(
                saved_dflash_cfg.get("selected_hidden_variant", self.dflash_selected_hidden_variant)
            )
            self.dflash_causal_residual_type = saved_dflash_cfg.get(
                "causal_residual_type", self.dflash_causal_residual_type
            )
            self.dflash_causal_residual_rank = saved_dflash_cfg.get(
                "causal_residual_rank", self.dflash_causal_residual_rank
            )
            self.dflash_causal_residual_scale = saved_dflash_cfg.get(
                "causal_residual_scale", self.dflash_causal_residual_scale
            )
            self.dflash_causal_residual_start_index = saved_dflash_cfg.get(
                "causal_residual_start_index", self.dflash_causal_residual_start_index
            )
            self.dflash_logit_markov_type = saved_dflash_cfg.get(
                "logit_markov_type", self.dflash_logit_markov_type
            )
            self.dflash_logit_markov_rank = saved_dflash_cfg.get(
                "logit_markov_rank", self.dflash_logit_markov_rank
            )
            self.dflash_logit_markov_scale = saved_dflash_cfg.get(
                "logit_markov_scale", self.dflash_logit_markov_scale
            )
            self.dflash_action_head_type = saved_dflash_cfg.get(
                "action_head_type", self.dflash_action_head_type
            )
            self.dflash_action_head_rank = saved_dflash_cfg.get(
                "action_head_rank", self.dflash_action_head_rank
            )
            self.dflash_action_token_start = saved_dflash_cfg.get(
                "action_token_start", self.dflash_action_token_start
            )
            self.dflash_action_vocab_size = saved_dflash_cfg.get(
                "action_vocab_size", self.dflash_action_vocab_size
            )
            self.dflash_action_confidence_enabled = saved_dflash_cfg.get(
                "action_confidence_enabled", self.dflash_action_confidence_enabled
            )
            if dflash_target_layer_ids is None:
                dflash_target_layer_ids = saved_dflash_cfg.get("target_layer_ids")
            if dflash_mask_token_id is None:
                self.dflash_mask_token_id = saved_dflash_cfg.get("mask_token_id", self.dflash_mask_token_id)
        #if not parallel_draft:
            #model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}
        config = None if self.draft_backend == "dflash" else EConfig.from_pretrained(ea_model_path)
        #else:
        #    config = EConfig.from_pretrained(ea_model_path)

        self.accept_threshold=accept_threshold
        #print('init accept threshold',accept_threshold)
        self.norm_stats = base_model.norm_stats
        self.last_dflash_stats = None
        self.last_generation_stats = None
        # EAGLE's per-block values stay on CUDA until the caller has recorded
        # inference time. Materializing them inside eagenerate would mix
        # metric collection into the paper-style timing path.
        self._pending_generation_stats = None
        self._target_logit_shadow_checks = 0
        self._target_logit_shadow_mismatches = 0
        self._dflash_previous_anchor_signature = None
        self._dflash_previous_prompt_signature = None
        self._dflash_previous_prompt_pooled_signature = None
        self._dflash_previous_first_action_probs = None
        self._dflash_previous_pixel_signature = None
        self._dflash_previous_action_tokens = None
        self._dflash_action_history_cpu = []
        self._dflash_previous_verified_action_tokens = None
        self._dflash_verified_action_run_length = 0
        self._dflash_consecutive_verify_skips = 0

        # Compute action bins
        self.bins = base_model.bins
        self.bin_centers = base_model.bin_centers

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = base_model.vocab_size
        self.get_action_stats = base_model.get_action_stats
        # 构造 DFlash 草稿模型的 config
        if self.draft_backend == "dflash":
            target_config = copy.deepcopy(self.base_model.language_model.config)
            target_config.num_hidden_layers = dflash_num_draft_layers
            target_config.num_target_layers = self.base_model.language_model.config.num_hidden_layers
            target_config.dflash_target_layer_ids = dflash_target_layer_ids
            target_config.dflash_block_size = dflash_block_size
            target_config.dflash_action_dim = dflash_action_dim
            target_config.dflash_selected_hidden_variant = self.dflash_selected_hidden_variant
            target_config.dflash_causal_residual_type = self.dflash_causal_residual_type
            target_config.dflash_causal_residual_rank = self.dflash_causal_residual_rank
            target_config.dflash_causal_residual_scale = self.dflash_causal_residual_scale
            target_config.dflash_logit_markov_type = self.dflash_logit_markov_type
            target_config.dflash_logit_markov_rank = self.dflash_logit_markov_rank
            target_config.dflash_logit_markov_scale = self.dflash_logit_markov_scale
            target_config.dflash_action_head_type = self.dflash_action_head_type
            target_config.dflash_action_head_rank = self.dflash_action_head_rank
            target_config.dflash_action_token_start = self.dflash_action_token_start
            target_config.dflash_action_vocab_size = self.dflash_action_vocab_size
            target_config.dflash_action_confidence_enabled = self.dflash_action_confidence_enabled
            # # 实例化草稿模型
            self.ea_layer = DFlashDraftModel(target_config)
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                raise FileNotFoundError(
                    f"DFlash checkpoint not found: {load_model_path}. "
                    "Please point spec_checkpoint to a saved DFlash checkpoint directory."
                )
            ea_layer_state_dict = torch.load(load_model_path, map_location="cpu")
        elif parallel_draft:
            #model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}
            with open(ea_model_path+'/model.safetensors', "rb") as f:
                safetensors_model = f.read()
                #pytorch_model = safetensors.torch.deserialize(safetensors_model)
                #print(state_dict.keys())
                self.ea_layer = PMMModel(config, path=base_model_name_or_path,load_emb=True)
                ea_layer_state_dict = safetensors.torch.load(safetensors_model)
        else:
            self.ea_layer = MMModel(config, path=base_model_name_or_path,load_emb=True)
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path)
        #self.ea_layer.init_tree()
        self.tree_mask = None
        low_memory = False

        device = base_model.language_model.model.layers[-1].self_attn.q_proj.weight.device
        #load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        if ea_layer_state_dict is not None:
            self.ea_layer.load_state_dict(ea_layer_state_dict, strict=self.draft_backend != "dflash")
        if hasattr(self.ea_layer, "embed_tokens"):
            self.ea_layer.embed_tokens = self.base_model.language_model.model.embed_tokens
        self.ea_layer.to(self.base_model.dtype).to(device)
        if hasattr(self.ea_layer, "init_tree"):
            self.ea_layer.init_tree()
            self.ea_layer.tree_mask = None
            self.ea_layer.tree_mode = None
        #print(self.ea_layer.fc.weight)
        #exit()
        #self.base_model.language_model = LlamaSpecForCausalLM(language_model)

    def get_generation_stats(self):
        """Return generation metrics without adding CUDA scalar reads to decoding."""
        pending = self._pending_generation_stats
        if pending is None:
            return self.last_generation_stats

        if pending.get("backend") != "eagle":
            self._pending_generation_stats = None
            return self.last_generation_stats

        def as_int(value):
            if torch.is_tensor(value):
                return int(value.detach().item())
            return int(value)

        num_blocks = max(int(pending["num_blocks"]), 0)
        progressed_tokens = min(
            int(pending["generated_tokens"]), as_int(pending["progressed_tokens"])
        )
        # Each tree verification block advances one correction token plus its
        # accepted prefix, so accepted = progressed - verification blocks.
        accepted_tokens = max(progressed_tokens - num_blocks, 0)
        progress_lengths = [progressed_tokens] if num_blocks > 0 else []
        accept_lengths = [accepted_tokens] if num_blocks > 0 else []
        self.last_generation_stats = {
            "backend": "eagle",
            "generated_tokens": pending["generated_tokens"],
            "num_blocks": num_blocks,
            "progressed_tokens": progressed_tokens,
            "progress_lengths": progress_lengths,
            "length": (progressed_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "table1_length": (progressed_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "avg_progress_length": (progressed_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "accept_lengths": accept_lengths,
            "raw_accept_lengths": accept_lengths,
            "avg_accept_length": (accepted_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "accepted_tokens": accepted_tokens,
            "compared_tokens": 0,
            "overall_hit_rate": None,
            "per_position": [],
        }
        self._pending_generation_stats = None
        return self.last_generation_stats

    def _dflash_target_logit_kwargs(self, *, num_logits_to_keep: int = 0) -> dict:
        """Return target-projection arguments for the selected DFlash mode."""
        result = {"num_logits_to_keep": int(num_logits_to_keep)}
        if self.dflash_target_logits_mode == "action_only":
            result["logit_token_range"] = (
                self.dflash_action_token_start,
                self.dflash_action_token_start + self.dflash_action_vocab_size,
            )
        return result

    def _sample_dflash_target_logits(self, logits: torch.Tensor) -> torch.LongTensor:
        """Sample target logits and account for a reduced action-only vocabulary."""
        if self.dflash_target_logits_mode == "action_only":
            return (
                dflash_sample(logits, temperature=0.0)
                + self.dflash_action_token_start
            )

        target_tokens = dflash_sample(logits, temperature=0.0)
        if self.dflash_target_logits_mode == "shadow":
            action_logits = logits[
                ...,
                self.dflash_action_token_start : (
                    self.dflash_action_token_start + self.dflash_action_vocab_size
                ),
            ]
            action_tokens = (
                dflash_sample(action_logits, temperature=0.0)
                + self.dflash_action_token_start
            )
            self._target_logit_shadow_checks += int(target_tokens.numel())
            self._target_logit_shadow_mismatches += int(
                (target_tokens != action_tokens).sum().item()
            )
        return target_tokens

    def _dflash_action_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the 256 action-token logits regardless of projection mode."""
        if self.dflash_target_logits_mode == "action_only":
            return logits
        return logits[
            ...,
            self.dflash_action_token_start : (
                self.dflash_action_token_start + self.dflash_action_vocab_size
            ),
        ]

    @staticmethod
    def _dflash_skip_features(
        base_action_logits: torch.Tensor,
        final_action_logits: torch.Tensor,
        proposed_tokens: torch.LongTensor,
        target_next_token: torch.LongTensor,
        *,
        action_token_start: int,
    ) -> dict[str, torch.Tensor]:
        """Compute no-retrain gate signals while keeping values on CUDA."""
        log_probs = F.log_softmax(final_action_logits.float(), dim=-1)
        top_log_probs, _ = torch.topk(log_probs, k=2, dim=-1)
        top_probs = top_log_probs.exp()
        probabilities = log_probs.exp()
        normalized_entropy = -(
            probabilities * log_probs
        ).sum(dim=-1) / float(np.log(final_action_logits.shape[-1]))
        base_tokens = (
            torch.argmax(base_action_logits, dim=-1) + int(action_token_start)
        )
        return {
            "sentinel_match": proposed_tokens[:, 0].eq(target_next_token[:, 0]),
            "min_top1_prob": top_probs[..., 0].amin(dim=1),
            "mean_top1_prob": top_probs[..., 0].mean(dim=1),
            "min_margin": (top_probs[..., 0] - top_probs[..., 1]).amin(dim=1),
            "max_normalized_entropy": normalized_entropy.amax(dim=1),
            "base_agreement": base_tokens.eq(proposed_tokens).float().mean(dim=1),
        }

    def _dflash_skip_gate(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Conservative gate; active mode remains opt-in and threshold-driven."""
        return (
            features["sentinel_match"]
            & (features["min_top1_prob"] >= self.dflash_verify_skip_min_top1_prob)
            & (features["min_margin"] >= self.dflash_verify_skip_min_margin)
            & (
                features["base_agreement"]
                >= self.dflash_verify_skip_min_base_agreement
            )
        )

    def reset_dflash_runtime_state(self):
        """Reset temporal calibration state at every environment episode boundary."""
        self._dflash_previous_anchor_signature = None
        self._dflash_previous_prompt_signature = None
        self._dflash_previous_prompt_pooled_signature = None
        self._dflash_previous_first_action_probs = None
        self._dflash_previous_pixel_signature = None
        self._dflash_previous_action_tokens = None
        self._dflash_action_history_cpu = []
        self._dflash_previous_verified_action_tokens = None
        self._dflash_verified_action_run_length = 0
        self._dflash_consecutive_verify_skips = 0

    def _record_dflash_action_history(self, action_tokens: torch.LongTensor) -> None:
        """Keep two tiny CPU action records for the next temporal prefill."""
        action_cpu = action_tokens[0].detach().to(device="cpu", dtype=torch.long)
        self._dflash_action_history_cpu.append(action_cpu)
        if len(self._dflash_action_history_cpu) > 2:
            self._dflash_action_history_cpu.pop(0)

    @staticmethod
    def _dflash_pixel_signature(pixel_values) -> Optional[torch.Tensor]:
        """Build a small pre-target visual signature for shadow calibration."""
        if pixel_values is None:
            return None
        tensors = []
        values = pixel_values.values() if isinstance(pixel_values, dict) else (pixel_values,)
        for value in values:
            if not torch.is_tensor(value):
                continue
            tensor = value.detach().float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 4:
                tensor = F.adaptive_avg_pool2d(tensor, output_size=(16, 16))
            tensors.append(tensor.reshape(tensor.shape[0], -1))
        if not tensors:
            return None
        return torch.cat(tensors, dim=-1)

    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )


        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = combined_attention_mask.min()

        return combined_attention_mask
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        return self.base_model.get_action_dim(unnorm_key)
    def _crop_past_key_values(self, past_key_values, length: int):
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "crop"):
            past_key_values.crop(length)
            return past_key_values

        cropped = []
        for layer in past_key_values:
            new_layer = []
            for tensor in layer:
                if torch.is_tensor(tensor) and tensor.ndim >= 3:
                    new_layer.append(tensor[..., :length, :].contiguous())
                else:
                    new_layer.append(tensor)
            cropped.append(tuple(new_layer))
        return tuple(cropped)

    @staticmethod
    def _past_key_values_length(past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        first_layer = past_key_values[0]
        first_tensor = first_layer[0]
        return int(first_tensor.shape[-2])

    def _select_tree_past_key_values(
        self,
        past_key_values,
        *,
        base_length: int,
        tree_node_indices: torch.LongTensor,
    ):
        """Commit one verified tree path as a contiguous target-model cache."""
        if past_key_values is None:
            return None
        tree_node_indices = tree_node_indices.to(dtype=torch.long)
        absolute_indices = tree_node_indices + int(base_length)

        def select_tensor(tensor: torch.Tensor) -> torch.Tensor:
            prefix = tensor[..., :base_length, :]
            if absolute_indices.numel() == 0:
                return prefix.contiguous()
            selected = tensor.index_select(-2, absolute_indices.to(tensor.device))
            return torch.cat([prefix, selected], dim=-2).contiguous()

        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            selected_cache = DynamicCache()
            for layer_idx, (key, value) in enumerate(
                zip(past_key_values.key_cache, past_key_values.value_cache)
            ):
                selected_cache.update(
                    select_tensor(key),
                    select_tensor(value),
                    layer_idx,
                )
            if hasattr(selected_cache, "_seen_tokens"):
                selected_cache._seen_tokens = base_length + int(tree_node_indices.numel())
            return selected_cache

        selected_layers = []
        for layer in past_key_values:
            selected_layer = []
            for tensor in layer:
                if torch.is_tensor(tensor) and tensor.ndim >= 3:
                    selected_layer.append(select_tensor(tensor))
                else:
                    selected_layer.append(tensor)
            selected_layers.append(tuple(selected_layer))
        return tuple(selected_layers)

    def _build_temporal_prefill_candidates(
        self,
        max_new_tokens: int,
    ) -> tuple[Optional[torch.LongTensor], list[str]]:
        """Build unique whole-action proposals from already executed actions.

        Candidate 0 repeats the latest action. With two history entries, the
        remaining candidates are a constant-velocity extrapolation in OpenVLA's
        normalized action space and the second-latest action. Gripper remains a
        categorical hold value during extrapolation.
        """
        history = self._dflash_action_history_cpu
        if len(history) < self.dflash_temporal_prefill_tree_min_history:
            return None, []
        latest = history[-1][:max_new_tokens].clone().long()
        if latest.numel() < max_new_tokens:
            return None, []

        raw_candidates: list[tuple[str, torch.LongTensor]] = [("hold", latest)]
        if len(history) >= 2:
            previous = history[-2][:max_new_tokens].clone().long()
            if previous.numel() >= max_new_tokens:
                centers = torch.as_tensor(self.bin_centers, dtype=torch.float32)

                def decode(tokens: torch.LongTensor) -> torch.Tensor:
                    indices = (int(self.vocab_size) - tokens - 1).clamp(
                        min=0,
                        max=centers.numel() - 1,
                    )
                    return centers.index_select(0, indices)

                latest_values = decode(latest)
                previous_values = decode(previous)
                extrapolated_values = (
                    latest_values + (latest_values - previous_values)
                ).clamp(min=-1.0, max=1.0)
                if max_new_tokens > 6:
                    extrapolated_values[6] = latest_values[6]
                nearest_indices = torch.argmin(
                    torch.abs(extrapolated_values.unsqueeze(-1) - centers),
                    dim=-1,
                )
                extrapolated_tokens = (
                    int(self.vocab_size) - nearest_indices - 1
                ).long()
                raw_candidates.extend(
                    [
                        ("constant_velocity", extrapolated_tokens),
                        ("recent", previous),
                    ]
                )

        unique_candidates = []
        sources = []
        seen = set()
        for source, candidate in raw_candidates:
            key = tuple(int(token) for token in candidate.tolist())
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
            sources.append(source)
            if len(unique_candidates) >= self.dflash_temporal_prefill_tree_max_candidates:
                break
        if not unique_candidates:
            return None, []
        return torch.stack(unique_candidates, dim=0), sources

    @staticmethod
    def _build_temporal_prefill_trie(
        candidate_tokens: torch.LongTensor,
    ) -> tuple[
        torch.LongTensor,
        list[dict[int, int]],
        torch.Tensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        """Compile whole-action proposals into a shared-prefix verification trie."""
        if candidate_tokens.ndim != 2 or candidate_tokens.shape[1] < 2:
            raise ValueError(
                "Temporal prefill candidates must have shape [candidates, action_dim>=2]."
            )
        input_depth = int(candidate_tokens.shape[1]) - 1
        parents = [-1]
        child_maps: list[dict[int, int]] = [dict()]
        node_tokens: list[int] = []
        node_depths: list[int] = []
        candidate_paths: list[list[int]] = []

        for candidate in candidate_tokens:
            parent_index = 0
            path = []
            for depth, token in enumerate(candidate[:input_depth].tolist()):
                token = int(token)
                child_index = child_maps[parent_index].get(token)
                if child_index is None:
                    child_index = len(child_maps)
                    child_maps[parent_index][token] = child_index
                    child_maps.append(dict())
                    parents.append(parent_index)
                    node_tokens.append(token)
                    node_depths.append(depth)
                path.append(child_index - 1)
                parent_index = child_index
            candidate_paths.append(path)

        node_count = len(node_tokens)
        tree_mask = torch.zeros((node_count, node_count), dtype=torch.bool)
        for node_index in range(1, node_count + 1):
            ancestor_index = node_index
            while ancestor_index > 0:
                tree_mask[node_index - 1, ancestor_index - 1] = True
                ancestor_index = parents[ancestor_index]

        return (
            torch.tensor(node_tokens, dtype=torch.long).unsqueeze(0),
            child_maps,
            tree_mask.unsqueeze(0).unsqueeze(0),
            torch.tensor(node_depths, dtype=torch.long).unsqueeze(0),
            torch.tensor(candidate_paths, dtype=torch.long),
        )

    def _select_temporal_prefill_path(
        self,
        *,
        candidate_tokens: torch.LongTensor,
        candidate_paths: torch.LongTensor,
        root_posterior_token: torch.LongTensor,
        node_posterior_tokens: torch.LongTensor,
        accept_threshold: Optional[int],
    ) -> dict:
        """Choose the longest valid temporal proposal, then the most exact one."""
        action_dim = int(candidate_tokens.shape[1])
        action_positions = torch.arange(
            action_dim,
            device=candidate_tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        best = None
        candidate_accept_lengths = []
        candidate_exact_lengths = []
        for candidate_index in range(candidate_tokens.shape[0]):
            path_nodes = candidate_paths[candidate_index]
            path_posteriors = node_posterior_tokens.index_select(1, path_nodes)
            posterior_tokens = torch.cat(
                [root_posterior_token, path_posteriors],
                dim=1,
            )
            proposed = candidate_tokens[candidate_index : candidate_index + 1]
            accept_mask = self._compute_dflash_accept_mask(
                proposed,
                posterior_tokens,
                accept_threshold=accept_threshold,
                action_position_ids=action_positions,
                acceptance_mode=self.dflash_acceptance_mode,
            )
            exact_mask = proposed.eq(posterior_tokens)
            accept_length = int(accept_mask.cumprod(dim=1).sum().item())
            exact_length = int(exact_mask.int().cumprod(dim=1).sum().item())
            accepted_distance = int(
                torch.abs(proposed[:, :accept_length] - posterior_tokens[:, :accept_length])
                .sum()
                .item()
            )
            candidate_accept_lengths.append(accept_length)
            candidate_exact_lengths.append(exact_length)
            score = (
                accept_length,
                exact_length,
                -accepted_distance,
                -candidate_index,
            )
            if best is None or score > best["score"]:
                correction = (
                    posterior_tokens[:, accept_length : accept_length + 1]
                    if accept_length < action_dim
                    else None
                )
                best = {
                    "score": score,
                    "candidate_index": candidate_index,
                    "accept_length": accept_length,
                    "exact_accept_length": exact_length,
                    "proposed_tokens": proposed,
                    "posterior_tokens": posterior_tokens,
                    "path_nodes": path_nodes,
                    "correction_token": correction,
                }
        if best is None:
            raise AssertionError("Temporal prefill path selection received no candidates.")
        best["candidate_accept_lengths"] = candidate_accept_lengths
        best["candidate_exact_lengths"] = candidate_exact_lengths
        best.pop("score")
        return best

    @staticmethod
    def _build_ddtree_from_logits(
        draft_logits: torch.Tensor,
        *,
        node_budget: int,
        token_id_offset: int,
    ) -> tuple[
        torch.LongTensor,
        list[dict[int, int]],
        torch.Tensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        """Build the highest-joint-probability DDTree under a fixed node budget.

        This is the DDTree best-first heap construction adapted to OpenVLA's
        contiguous action-token vocabulary. Index 0 in ``child_maps`` is the
        already-decoded anchor (the virtual root); candidate nodes are one-based
        in ``child_maps`` and zero-based in returned tensors.
        """
        if draft_logits.ndim != 2:
            raise ValueError("DDTree draft_logits must have shape [depth, action_vocab].")
        depth_limit, vocab_size = draft_logits.shape
        node_budget = max(0, min(int(node_budget), int(depth_limit * vocab_size)))
        if depth_limit == 0 or vocab_size == 0 or node_budget == 0:
            raise ValueError("DDTree requires non-empty logits and a positive node budget.")

        top_k = min(node_budget, int(vocab_size))
        logits = draft_logits.float()
        top_logits, top_local_ids = torch.topk(logits, k=top_k, dim=-1)
        top_log_probs = (
            top_logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        ).to(device="cpu", dtype=torch.float32).numpy()
        top_local_ids_cpu = top_local_ids.to(device="cpu", dtype=torch.long).numpy()

        first_log_weight = float(top_log_probs[0, 0])
        heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
            (-first_log_weight, (0,), 0, 0, 0, first_log_weight)
        ]
        node_token_ids: list[int] = []
        node_depths: list[int] = []
        parents = [-1]
        child_maps: list[dict[int, int]] = [dict()]

        while heap and len(node_token_ids) < node_budget:
            _, ranks, parent_index, depth, rank, log_weight = heapq.heappop(heap)
            token_id = int(top_local_ids_cpu[depth, rank]) + int(token_id_offset)
            current_index = len(child_maps)
            child_maps[parent_index][token_id] = current_index
            child_maps.append(dict())
            parents.append(parent_index)
            node_token_ids.append(token_id)
            node_depths.append(depth)

            if rank + 1 < top_k:
                sibling_ranks = ranks[:-1] + (rank + 1,)
                sibling_log_weight = (
                    log_weight
                    - float(top_log_probs[depth, rank])
                    + float(top_log_probs[depth, rank + 1])
                )
                heapq.heappush(
                    heap,
                    (
                        -sibling_log_weight,
                        sibling_ranks,
                        parent_index,
                        depth,
                        rank + 1,
                        sibling_log_weight,
                    ),
                )

            if depth + 1 < depth_limit:
                child_ranks = ranks + (0,)
                child_log_weight = log_weight + float(top_log_probs[depth + 1, 0])
                heapq.heappush(
                    heap,
                    (
                        -child_log_weight,
                        child_ranks,
                        current_index,
                        depth + 1,
                        0,
                        child_log_weight,
                    ),
                )

        device = draft_logits.device
        flat_tokens = torch.tensor(node_token_ids, device=device, dtype=torch.long).unsqueeze(0)
        relative_position_ids = torch.tensor(
            node_depths, device=device, dtype=torch.long
        ).unsqueeze(0)
        greedy_tokens = torch.argmax(draft_logits, dim=-1).to(dtype=torch.long)
        greedy_tokens = (greedy_tokens + int(token_id_offset)).unsqueeze(0)

        node_count = len(node_token_ids)
        tree_mask = torch.zeros(
            (node_count, node_count), device=device, dtype=torch.bool
        )
        for node_index in range(1, node_count + 1):
            ancestor_index = node_index
            while ancestor_index > 0:
                tree_mask[node_index - 1, ancestor_index - 1] = True
                ancestor_index = parents[ancestor_index]
        return (
            flat_tokens,
            child_maps,
            tree_mask.unsqueeze(0).unsqueeze(0),
            relative_position_ids,
            greedy_tokens,
        )

    @staticmethod
    def _follow_ddtree_target_path(
        child_maps: list[dict[int, int]],
        root_posterior_token: torch.LongTensor,
        node_posterior_tokens: torch.LongTensor,
        max_accept_length: int,
    ) -> tuple[torch.LongTensor, int]:
        """Follow only the child selected by the target model at each node."""
        if root_posterior_token.numel() != 1:
            raise ValueError("DDTree traversal expects one target token at the virtual root.")
        if node_posterior_tokens.ndim != 2 or node_posterior_tokens.shape[0] != 1:
            raise ValueError("DDTree node posterior tokens must have shape [1, nodes].")

        posterior_tokens = torch.cat(
            [root_posterior_token.reshape(-1), node_posterior_tokens.reshape(-1)]
        ).tolist()
        current_index = 0
        next_token = int(posterior_tokens[0])
        accepted_node_indices: list[int] = []
        while len(accepted_node_indices) < int(max_accept_length):
            child_index = child_maps[current_index].get(next_token)
            if child_index is None:
                break
            accepted_node_indices.append(child_index - 1)
            current_index = child_index
            next_token = int(posterior_tokens[child_index])

        return (
            torch.tensor(
                accepted_node_indices,
                device=node_posterior_tokens.device,
                dtype=torch.long,
            ),
            next_token,
        )

    @staticmethod
    def _enumerate_ddtree_leaf_paths(
        child_maps: list[dict[int, int]],
    ) -> list[list[int]]:
        """Return root-to-leaf paths as one-based DDTree node indices."""
        paths: list[list[int]] = []
        stack: list[tuple[int, list[int]]] = [(0, [])]
        while stack:
            node_index, prefix = stack.pop()
            children = list(child_maps[node_index].values())
            if not children:
                if prefix:
                    paths.append(prefix)
                continue
            for child_index in reversed(children):
                stack.append((child_index, prefix + [child_index]))
        return paths

    def _select_relaxed_ddtree_path(
        self,
        *,
        child_maps: list[dict[int, int]],
        verify_input_ids: torch.LongTensor,
        root_posterior_token: torch.LongTensor,
        node_posterior_tokens: torch.LongTensor,
        action_start_position: int,
        max_action_tokens: int,
        accept_threshold: int,
    ) -> tuple[torch.LongTensor, int, int]:
        """Select the longest action-group-valid DDTree prefix.

        Relaxed verification is intentionally approximate and therefore cannot
        use the exact-token child traversal. It evaluates each tiny DDTree leaf
        path under the existing joint action-group budget and returns committed
        node indices plus the target correction token.
        """
        leaf_paths = self._enumerate_ddtree_leaf_paths(child_maps)
        if not leaf_paths:
            return (
                torch.empty(0, device=verify_input_ids.device, dtype=torch.long),
                int(root_posterior_token.item()),
                0,
            )

        best_nodes: list[int] = []
        best_accept_length = -1
        best_correction = int(root_posterior_token.item())
        for path in leaf_paths:
            tensor_indices = torch.tensor(
                [node_index - 1 for node_index in path],
                device=verify_input_ids.device,
                dtype=torch.long,
            )
            proposed = verify_input_ids.index_select(1, tensor_indices)
            parent_indices = tensor_indices[:-1]
            posterior_tail = node_posterior_tokens.index_select(1, parent_indices)
            posterior = torch.cat([root_posterior_token, posterior_tail], dim=1)
            positions = torch.arange(
                action_start_position,
                action_start_position + proposed.shape[1],
                device=verify_input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            accept_mask = self._compute_dflash_accept_mask(
                proposed,
                posterior,
                accept_threshold=accept_threshold,
                action_position_ids=positions,
                acceptance_mode="action_group",
            )
            accept_length = min(
                int(accept_mask.cumprod(dim=1).sum().item()),
                int(max_action_tokens),
            )
            if accept_length <= best_accept_length:
                continue
            best_accept_length = accept_length
            best_nodes = [node_index - 1 for node_index in path[:accept_length]]
            if accept_length < len(path):
                best_correction = int(posterior[0, accept_length].item())
            else:
                best_correction = int(
                    node_posterior_tokens[0, path[-1] - 1].item()
                )

        return (
            torch.tensor(
                best_nodes,
                device=verify_input_ids.device,
                dtype=torch.long,
            ),
            best_correction,
            max(best_accept_length, 0),
        )

    def _compute_dflash_accept_mask(
        self,
        proposed_tokens: torch.Tensor,
        posterior_tokens: torch.Tensor,
        accept_threshold: Optional[int] = None,
        action_position_ids: Optional[torch.LongTensor] = None,
        acceptance_mode: Optional[str] = None,
    ) -> torch.Tensor:
        exact_mask = proposed_tokens == posterior_tokens
        if accept_threshold is None or int(accept_threshold) <= 0:
            return exact_mask.int()

        token_distance = torch.abs(proposed_tokens.long() - posterior_tokens.long())
        token_mask = token_distance <= int(accept_threshold)
        acceptance_mode = acceptance_mode or self.dflash_acceptance_mode
        if acceptance_mode == "token":
            return token_mask.int()
        if acceptance_mode != "action_group":
            raise ValueError(f"Unsupported DFlash acceptance mode: {acceptance_mode!r}")
        if action_position_ids is None:
            raise ValueError("action_group acceptance requires absolute action_position_ids.")

        positions = action_position_ids.to(device=proposed_tokens.device, dtype=torch.long)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.shape[0] == 1 and proposed_tokens.shape[0] > 1:
            positions = positions.expand(proposed_tokens.shape[0], -1)
        if positions.shape != proposed_tokens.shape:
            raise ValueError(
                "action_position_ids must match proposed paths: "
                f"positions={tuple(positions.shape)} proposals={tuple(proposed_tokens.shape)}"
            )

        # The old per-token rule accepts a cube with side radius r. For each
        # physical group we preserve the same worst-corner squared-error budget:
        # sum(delta_bin^2) <= number_of_visible_dimensions * r^2. This only adds
        # group-consistent candidates; if a group fails, the old token mask is kept.
        group_mask = token_mask.clone()
        threshold_sq = int(accept_threshold) ** 2
        for group_start, group_end in ((0, 2), (3, 5)):
            in_group = (positions >= group_start) & (positions <= group_end)
            visible_dims = in_group.sum(dim=1)
            group_error_sq = (token_distance.square() * in_group.long()).sum(dim=1)
            group_budget = visible_dims * threshold_sq
            group_accepted = (visible_dims > 0) & (group_error_sq <= group_budget)
            group_mask = torch.where(
                in_group & group_accepted.unsqueeze(1),
                torch.ones_like(group_mask),
                group_mask,
            )

        # Gripper is categorical. Nearby token ids do not mean a partially open
        # gripper, so its relaxed criterion remains exact.
        gripper_mask = positions == 6
        group_mask = torch.where(gripper_mask, exact_mask, group_mask)
        return group_mask.int()

    def _compute_dflash_accept_length(
        self,
        proposed_tokens: torch.Tensor,
        posterior_tokens: torch.Tensor,
        accept_threshold: Optional[int] = None,
        action_position_ids: Optional[torch.LongTensor] = None,
        acceptance_mode: Optional[str] = None,
    ) -> int:
        accept_mask = self._compute_dflash_accept_mask(
            proposed_tokens,
            posterior_tokens,
            accept_threshold=accept_threshold,
            action_position_ids=action_position_ids,
            acceptance_mode=acceptance_mode,
        )
        return int(accept_mask.cumprod(dim=1).sum(dim=1)[0].item())# 一旦有一个 False，后面全部变 0

    @torch.no_grad()
    def dflash_generate(
        self,
        input_ids,
        max_new_tokens,
        accept_threshold=None,
        **kwargs
    ):
        if self.dflash_include_anchor_hidden:
            return self._dflash_generate_with_anchor_hidden(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                accept_threshold=accept_threshold,
                **kwargs,
            )
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_ids = input_ids.clone()

        model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)# 完整走一遍 VLM 的 prefill
        outputs, orig, _, _ = self(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
            output_orig=True,
            **self._dflash_target_logit_kwargs(num_logits_to_keep=1),
        )

        token_prefix_len = input_ids.shape[1]
        block_size = self.ea_layer.block_size
        max_length = token_prefix_len + max_new_tokens
        output_ids = torch.full(
            (1, max_length + block_size),
            self.dflash_mask_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        output_ids[:, :token_prefix_len] = input_ids

        first_token = self._sample_dflash_target_logits(orig[:, -1:, :])
        output_ids[:, token_prefix_len : token_prefix_len + 1] = first_token

        past_key_values = outputs.past_key_values
        prefill_hidden = extract_context_feature(
            outputs.hidden_states,
            self.ea_layer.target_layer_ids,
            self.ea_layer.selected_hidden_variant,
        )
        prompt_context = prefill_hidden
        prompt_position_ids = torch.arange(
            prefill_hidden.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        action_base_position = prefill_hidden.shape[1]
        action_context = prefill_hidden[:, :0, :]

        max_draft_tokens = block_size - 1
        position_hits = [0 for _ in range(max_draft_tokens)]
        position_counts = [0 for _ in range(max_draft_tokens)]
        accept_lengths = []
        progress_lengths = []
        total_accepted = 0
        total_compared = 0
        confidence_truncated_blocks = 0

        anchor_idx = 0
        while anchor_idx < max_new_tokens - 1:
            remaining = max_new_tokens - anchor_idx - 1
            q_len = min(max_draft_tokens, remaining)
            token_anchor_pos = token_prefix_len + anchor_idx

            block_input_ids = torch.full(
                (1, q_len),
                self.dflash_mask_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            block_input_ids[:, 0] = output_ids[:, token_anchor_pos]
            noise_embedding = self.base_model.language_model.model.embed_tokens(block_input_ids)

            target_hidden = torch.cat([prompt_context, action_context], dim=1)
            if action_context.shape[1] > 0:
                action_ctx_position_ids = torch.arange(
                    action_base_position,
                    action_base_position + action_context.shape[1],
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                ctx_position_ids = torch.cat(
                    [
                        prompt_position_ids,
                        action_ctx_position_ids,
                    ],
                    dim=1,
                )
            else:
                ctx_position_ids = prompt_position_ids
            noise_position_ids = torch.arange(
                action_base_position + anchor_idx,
                action_base_position + anchor_idx + q_len,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            action_position_ids = torch.arange(
                anchor_idx,
                anchor_idx + q_len,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)

            draft_hidden = self.ea_layer(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                ctx_position_ids=ctx_position_ids,
                noise_position_ids=noise_position_ids,
                ctx_attention_mask=None,
                action_position_ids=action_position_ids,
            )
            if (
                self.dflash_use_causal_residual_sampling
                and getattr(self.ea_layer, "action_sequential_enabled", False)
            ):
                base_draft_logits = self.ea_layer.project_action_logits(
                    draft_hidden,
                    self.base_model.language_model.lm_head,
                )
                proposed_tokens, draft_logits, _ = self.ea_layer.sample_action_block(
                    base_logits=base_draft_logits,
                    hidden_states=draft_hidden,
                    first_prev_token_ids=block_input_ids[:, :1],
                    action_position_ids=action_position_ids,
                    temperature=0.0,
                    confidence_threshold=self.dflash_confidence_threshold,
                    confidence_min_tokens=self.dflash_confidence_min_tokens,
                )
                if proposed_tokens.shape[1] < q_len:
                    confidence_truncated_blocks += 1
                    q_len = proposed_tokens.shape[1]
                    block_input_ids = block_input_ids[:, :q_len]
                    noise_position_ids = noise_position_ids[:, :q_len]
            elif (
                self.dflash_use_causal_residual_sampling
                and (
                    getattr(self.ea_layer, "causal_residual_enabled", False)
                    or getattr(self.ea_layer, "logit_markov_enabled", False)
                )
            ):
                proposed_tokens, draft_logits = self.ea_layer.sample_with_causal_residual(
                    hidden_states=draft_hidden,
                    first_prev_token_ids=block_input_ids[:, :1],
                    lm_head=self.base_model.language_model.lm_head,
                    temperature=0.0,
                    start_index=self.dflash_causal_residual_start_index,
                )
            else:
                if getattr(self.ea_layer, "action_projection_enabled", False):
                    base_draft_logits = self.ea_layer.project_action_logits(
                        draft_hidden,
                        self.base_model.language_model.lm_head,
                    )
                    draft_logits = base_draft_logits
                    proposed_tokens = (
                        dflash_sample(draft_logits, temperature=0.0)
                        + self.ea_layer.action_token_start
                    )
                else:
                    base_draft_logits = self.base_model.language_model.lm_head(draft_hidden)
                    draft_logits = base_draft_logits
                    proposed_tokens = dflash_sample(draft_logits, temperature=0.0)
            
            # 目标模型验证
            verify_input_ids = torch.cat([block_input_ids[:, :1], proposed_tokens[:, :-1]], dim=1)
            verify_embeds = self.base_model.language_model.model.embed_tokens(verify_input_ids)
            verify_outputs, verify_logits, _, _ = self(
                input_embeds=verify_embeds,
                output_orig=True,
                attention_mask=None,
                past_key_values=past_key_values,
                return_dict=True,
                position_ids=noise_position_ids,
                use_cache=True,
                **self._dflash_target_logit_kwargs(),
            )
            posterior = self._sample_dflash_target_logits(verify_logits)
            
            # 接受/拒绝
            predicted_action_positions = torch.arange(
                anchor_idx + 1,
                anchor_idx + q_len + 1,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            accept_mask = self._compute_dflash_accept_mask(
                proposed_tokens,
                posterior,
                accept_threshold=accept_threshold,
                action_position_ids=predicted_action_positions,
            )
            accept_length = self._compute_dflash_accept_length(
                proposed_tokens,
                posterior,
                accept_threshold=accept_threshold,
                action_position_ids=predicted_action_positions,
            )
            effective_accept_length = min(accept_length, q_len)
            if q_len > 0:
                current_hits = accept_mask[0, :q_len].tolist()
                for idx, hit in enumerate(current_hits):
                    position_counts[idx] += 1
                    position_hits[idx] += int(hit)
                total_compared += q_len
            accept_lengths.append(effective_accept_length)
            total_accepted += effective_accept_length

            if effective_accept_length > 0:
                output_ids[
                    :,
                    token_anchor_pos + 1 : token_anchor_pos + 1 + effective_accept_length,
                ] = proposed_tokens[:, :effective_accept_length]

            all_accepted = effective_accept_length == q_len
            if not all_accepted:
                output_ids[:, token_anchor_pos + 1 + effective_accept_length] = posterior[
                    :, effective_accept_length
                ]

            append_count = effective_accept_length if all_accepted else effective_accept_length + 1
            progress_lengths.append(int(append_count))
            new_anchor_idx = anchor_idx + append_count
            new_cache_length = action_base_position + new_anchor_idx
            past_key_values = self._crop_past_key_values(verify_outputs.past_key_values, new_cache_length)
            verified_hidden = extract_context_feature(
                verify_outputs.hidden_states,
                self.ea_layer.target_layer_ids,
                self.ea_layer.selected_hidden_variant,
            )[:, :append_count, :]
            action_context = torch.cat([action_context, verified_hidden], dim=1)
            anchor_idx = new_anchor_idx
        per_position_stats = []
        for idx, (hit_count, compare_count) in enumerate(zip(position_hits, position_counts), start=1):
            reject_count = compare_count - hit_count
            per_position_stats.append(
                {
                    "position": idx,
                    "count": compare_count,
                    "hit_count": hit_count,
                    "reject_count": reject_count,
                    "hit_rate": (hit_count / compare_count) if compare_count > 0 else None,
                    "reject_rate": (reject_count / compare_count) if compare_count > 0 else None,
                }
            )
        num_blocks = len(accept_lengths)
        progressed_tokens = min(
            max_new_tokens, bootstrapped_tokens + sum(progress_lengths)
        )
        generation_stats = {
            "backend": "dflash",
            "block_size": block_size,
            "generated_tokens": max_new_tokens,
            "use_causal_residual_sampling": bool(self.dflash_use_causal_residual_sampling),
            "action_head_type": getattr(self.ea_layer, "action_head_type", "none"),
            "acceptance_mode": self.dflash_acceptance_mode,
            "tree_mode": self.dflash_tree_mode,
            "tree_budget": self.dflash_tree_budget,
            "target_logits_mode": self.dflash_target_logits_mode,
            "target_logit_shadow_checks": self._target_logit_shadow_checks,
            "target_logit_shadow_mismatches": self._target_logit_shadow_mismatches,
            "confidence_threshold": self.dflash_confidence_threshold,
            "confidence_min_tokens": self.dflash_confidence_min_tokens,
            "confidence_truncated_blocks": confidence_truncated_blocks,
            "num_blocks": num_blocks,
            "bootstrapped_tokens": 1,
            "progressed_tokens": progressed_tokens,
            "progress_lengths": progress_lengths,
            "length": (progressed_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "table1_length": (progressed_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "avg_progress_length": (progressed_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "avg_tail_progress_length": (sum(progress_lengths) / num_blocks) if num_blocks > 0 else 0.0,
            "accept_lengths": [int(x) for x in accept_lengths],
            "avg_accept_length": (sum(accept_lengths) / num_blocks) if num_blocks > 0 else 0.0,
            "accepted_tokens": total_accepted,
            "compared_tokens": total_compared,
            "overall_hit_rate": (total_accepted / total_compared) if total_compared > 0 else None,
            "per_position": per_position_stats,
        }
        self.last_dflash_stats = generation_stats
        self.last_generation_stats = generation_stats
        return output_ids[:, token_prefix_len:max_length]

    @torch.no_grad()
    def _dflash_generate_with_anchor_hidden(
        self,
        input_ids,
        max_new_tokens,
        accept_threshold=None,
        **kwargs
    ):
        """SpecVLA-style anchor hidden injection for DFLASH.

        Invariant:
        - Before draft, target has already decoded the current anchor token.
        - action_context therefore contains A0..A_anchor.
        - The draft block still starts from t_anchor and predicts t_{anchor+1}...
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_ids = input_ids.clone()

        profile_totals_ms: dict[str, float] = {}
        profile_call_counts: dict[str, int] = {}

        def profile_start():
            if not self.dflash_profile_stages:
                return None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        def profile_end(stage_name: str, started_at):
            if started_at is None:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            profile_totals_ms[stage_name] = (
                profile_totals_ms.get(stage_name, 0.0) + elapsed_ms
            )
            profile_call_counts[stage_name] = profile_call_counts.get(stage_name, 0) + 1

        token_prefix_len = input_ids.shape[1]
        block_size = self.ea_layer.block_size
        max_draft_tokens = max(block_size - 1, 0)
        max_length = token_prefix_len + max_new_tokens

        model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        prefill_fusion_candidate = None
        prefill_tree_candidates = None
        prefill_tree_candidate_sources = []
        prefill_tree_candidate_paths = None
        prefill_tree_mask = None
        prefill_tree_relative_positions = None
        prefill_fusion_input_count = 0
        if (
            self.dflash_temporal_prefill_tree
            and self.dflash_tree_mode == "off"
            and max_new_tokens > 1
            and model_inputs.get("input_ids") is not None
        ):
            candidate_tokens_cpu, prefill_tree_candidate_sources = (
                self._build_temporal_prefill_candidates(max_new_tokens)
            )
            if candidate_tokens_cpu is not None:
                (
                    prefill_verify_inputs,
                    _,
                    prefill_tree_mask,
                    prefill_tree_relative_positions,
                    prefill_tree_candidate_paths,
                ) = self._build_temporal_prefill_trie(candidate_tokens_cpu)
                prefill_tree_candidates = candidate_tokens_cpu.to(input_ids.device)
                prefill_verify_inputs = prefill_verify_inputs.to(input_ids.device)
                prefill_tree_mask = prefill_tree_mask.to(input_ids.device)
                prefill_tree_relative_positions = prefill_tree_relative_positions.to(
                    input_ids.device
                )
                prefill_tree_candidate_paths = prefill_tree_candidate_paths.to(
                    input_ids.device
                )
                prefill_fusion_input_count = int(prefill_verify_inputs.shape[1])
                model_inputs["input_ids"] = torch.cat(
                    [model_inputs["input_ids"], prefill_verify_inputs], dim=1
                )
        elif (
            self.dflash_temporal_prefill_fusion
            and self.dflash_tree_mode == "off"
            and max_new_tokens > 1
            and self._dflash_verified_action_run_length
            >= self.dflash_temporal_prefill_min_stable_actions
            and self._dflash_previous_action_tokens is not None
            and self._dflash_previous_action_tokens.shape[1] >= max_new_tokens
            and model_inputs.get("input_ids") is not None
        ):
            # The previous verified action is known before the current image
            # prefill. Appending c0..c5 makes the mandatory multimodal prefill
            # produce target logits for c0..c6 in one causal forward.
            prefill_fusion_candidate = self._dflash_previous_action_tokens[
                :, :max_new_tokens
            ].detach()
            prefill_verify_inputs = prefill_fusion_candidate[:, :-1]
            prefill_fusion_input_count = int(prefill_verify_inputs.shape[1])
            model_inputs["input_ids"] = torch.cat(
                [model_inputs["input_ids"], prefill_verify_inputs], dim=1
            )
            if model_inputs.get("attention_mask") is not None:
                attention_extension = torch.ones(
                    (
                        model_inputs["attention_mask"].shape[0],
                        prefill_fusion_input_count,
                    ),
                    dtype=model_inputs["attention_mask"].dtype,
                    device=model_inputs["attention_mask"].device,
                )
                model_inputs["attention_mask"] = torch.cat(
                    [model_inputs["attention_mask"], attention_extension], dim=1
                )
        if (
            prefill_tree_candidates is not None
            and model_inputs.get("attention_mask") is not None
        ):
            attention_extension = torch.ones(
                (
                    model_inputs["attention_mask"].shape[0],
                    prefill_fusion_input_count,
                ),
                dtype=model_inputs["attention_mask"].dtype,
                device=model_inputs["attention_mask"].device,
            )
            model_inputs["attention_mask"] = torch.cat(
                [model_inputs["attention_mask"], attention_extension], dim=1
            )
        current_pixel_signature = (
            self._dflash_pixel_signature(model_inputs.get("pixel_values"))
            if self.dflash_verify_skip_mode == "shadow"
            else None
        )
        pixel_temporal_cosine = None
        pixel_temporal_relative_l2 = None
        if (
            current_pixel_signature is not None
            and self._dflash_previous_pixel_signature is not None
            and current_pixel_signature.shape
            == self._dflash_previous_pixel_signature.shape
        ):
            previous_pixel_signature = self._dflash_previous_pixel_signature
            pixel_temporal_cosine = float(
                F.cosine_similarity(
                    current_pixel_signature,
                    previous_pixel_signature,
                    dim=-1,
                )[0].item()
            )
            pixel_temporal_relative_l2 = float(
                (
                    torch.linalg.vector_norm(
                        current_pixel_signature - previous_pixel_signature,
                        dim=-1,
                    )
                    / torch.linalg.vector_norm(
                        previous_pixel_signature, dim=-1
                    ).clamp_min(1e-6)
                )[0].item()
            )

        stage_started = profile_start()
        prefill_active = (
            prefill_fusion_candidate is not None
            or prefill_tree_candidates is not None
        )
        prefill_logit_count = prefill_fusion_input_count + 1 if prefill_active else 1
        language_model = self.base_model.language_model
        if (
            prefill_tree_candidates is not None
            and getattr(language_model.config, "_attn_implementation", None)
            == "flash_attention_2"
        ):
            raise RuntimeError(
                "Temporal prefill trees require an explicit ancestor mask and are "
                "not compatible with flash_attention_2. Use eager/SDPA."
            )
        previous_tree_mask = getattr(language_model, "tree_mask", None)
        language_model.tree_mask = prefill_tree_mask
        try:
            outputs, orig, _, _ = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=True,
                output_orig=True,
                multimodal_tree_position_ids=prefill_tree_relative_positions,
                **self._dflash_target_logit_kwargs(
                    num_logits_to_keep=prefill_logit_count
                ),
            )
        finally:
            language_model.tree_mask = previous_tree_mask
        profile_end(
            (
                "target_prefill_temporal_tree"
                if prefill_tree_candidates is not None
                else "target_prefill_fused"
                if prefill_fusion_candidate is not None
                else "target_prefill"
            ),
            stage_started,
        )

        output_ids = torch.full(
            (1, max_length + block_size),
            self.dflash_mask_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        output_ids[:, :token_prefix_len] = input_ids

        full_prefill_hidden = extract_context_feature(
            outputs.hidden_states,
            self.ea_layer.target_layer_ids,
            self.ea_layer.selected_hidden_variant,
        )
        action_base_position = (
            full_prefill_hidden.shape[1] - prefill_fusion_input_count
        )
        prompt_context = full_prefill_hidden[:, :action_base_position, :]
        prompt_position_ids = torch.arange(
            action_base_position,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        action_context = prompt_context[:, :0, :]

        prefill_fusion_record = None
        prefill_selected_hidden = None
        initial_anchor_idx = 0
        if prefill_fusion_candidate is None and prefill_tree_candidates is None:
            first_token_logits = orig[:, -1:, :]
            first_token = self._sample_dflash_target_logits(first_token_logits)
            output_ids[:, token_prefix_len : token_prefix_len + 1] = first_token
            past_key_values = outputs.past_key_values
        elif prefill_tree_candidates is not None:
            root_posterior_token = self._sample_dflash_target_logits(orig[:, :1, :])
            node_posterior_tokens = self._sample_dflash_target_logits(orig[:, 1:, :])
            selected_prefill = self._select_temporal_prefill_path(
                candidate_tokens=prefill_tree_candidates,
                candidate_paths=prefill_tree_candidate_paths,
                root_posterior_token=root_posterior_token,
                node_posterior_tokens=node_posterior_tokens,
                accept_threshold=accept_threshold,
            )
            selected_candidate_index = int(selected_prefill["candidate_index"])
            selected_candidate = selected_prefill["proposed_tokens"]
            selected_path_nodes = selected_prefill["path_nodes"]
            prefill_accept_length = min(
                int(selected_prefill["accept_length"]),
                max_new_tokens,
            )
            if prefill_accept_length > 0:
                output_ids[
                    :,
                    token_prefix_len : token_prefix_len + prefill_accept_length,
                ] = selected_candidate[:, :prefill_accept_length]
            if prefill_accept_length < max_new_tokens:
                output_ids[:, token_prefix_len + prefill_accept_length] = (
                    selected_prefill["correction_token"][:, 0]
                )
            prefill_progress_length = (
                max_new_tokens
                if prefill_accept_length == max_new_tokens
                else prefill_accept_length + 1
            )
            initial_anchor_idx = prefill_progress_length - 1
            accepted_cached_inputs = min(
                prefill_accept_length,
                max_new_tokens - 1,
            )
            if prefill_progress_length < max_new_tokens:
                past_key_values = self._select_tree_past_key_values(
                    outputs.past_key_values,
                    base_length=action_base_position,
                    tree_node_indices=selected_path_nodes[:accepted_cached_inputs],
                )
            else:
                # A complete action returns immediately; avoid gathering 32-layer KV.
                past_key_values = None
            if accepted_cached_inputs > 0:
                selected_hidden_indices = (
                    action_base_position
                    + selected_path_nodes[:accepted_cached_inputs]
                )
                action_context = full_prefill_hidden.index_select(
                    1,
                    selected_hidden_indices,
                )
                prefill_selected_hidden = action_context
            first_token_logits = orig[:, :1, :]
            first_token = output_ids[
                :, token_prefix_len : token_prefix_len + 1
            ]
            hold_accept_length = int(
                selected_prefill["candidate_accept_lengths"][0]
            )
            prefill_fusion_record = {
                "mode": "temporal_tree",
                "accept_length": prefill_accept_length,
                "exact_accept_length": int(
                    selected_prefill["exact_accept_length"]
                ),
                "progress_length": prefill_progress_length,
                "full_match": prefill_accept_length == max_new_tokens,
                "full_exact_match": int(selected_prefill["exact_accept_length"])
                == max_new_tokens,
                "candidate_count": int(prefill_tree_candidates.shape[0]),
                "verified_node_count": prefill_fusion_input_count,
                "candidate_sources": prefill_tree_candidate_sources,
                "candidate_accept_lengths": selected_prefill[
                    "candidate_accept_lengths"
                ],
                "candidate_exact_lengths": selected_prefill[
                    "candidate_exact_lengths"
                ],
                "selected_candidate_index": selected_candidate_index,
                "selected_candidate_source": prefill_tree_candidate_sources[
                    selected_candidate_index
                ],
                "selected_alternate": selected_candidate_index != 0,
                "hold_accept_length": hold_accept_length,
                "extra_accepted_over_hold": max(
                    prefill_accept_length - hold_accept_length,
                    0,
                ),
            }
        else:
            prefill_posterior_tokens = self._sample_dflash_target_logits(orig)
            prefill_action_positions = torch.arange(
                max_new_tokens,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            prefill_accept_mask = self._compute_dflash_accept_mask(
                prefill_fusion_candidate,
                prefill_posterior_tokens,
                accept_threshold=accept_threshold,
                action_position_ids=prefill_action_positions,
                acceptance_mode=self.dflash_acceptance_mode,
            )
            prefill_accept_length = min(
                int(prefill_accept_mask.int().cumprod(dim=1).sum(dim=1)[0].item()),
                max_new_tokens,
            )
            if prefill_accept_length > 0:
                output_ids[
                    :,
                    token_prefix_len : token_prefix_len + prefill_accept_length,
                ] = prefill_fusion_candidate[:, :prefill_accept_length]
            if prefill_accept_length < max_new_tokens:
                output_ids[:, token_prefix_len + prefill_accept_length] = (
                    prefill_posterior_tokens[:, prefill_accept_length]
                )
            prefill_progress_length = (
                max_new_tokens
                if prefill_accept_length == max_new_tokens
                else prefill_accept_length + 1
            )
            initial_anchor_idx = prefill_progress_length - 1
            accepted_cached_inputs = min(
                prefill_accept_length, prefill_fusion_input_count
            )
            past_key_values = self._crop_past_key_values(
                outputs.past_key_values,
                action_base_position + accepted_cached_inputs,
            )
            if accepted_cached_inputs > 0:
                action_context = full_prefill_hidden[
                    :,
                    action_base_position : (
                        action_base_position + accepted_cached_inputs
                    ),
                    :,
                ]
                prefill_selected_hidden = action_context
            first_token_logits = orig[:, :1, :]
            first_token = output_ids[
                :, token_prefix_len : token_prefix_len + 1
            ]
            prefill_fusion_record = {
                "mode": "single_hold",
                "accept_length": prefill_accept_length,
                "progress_length": prefill_progress_length,
                "full_match": prefill_accept_length == max_new_tokens,
            }

        target_ar_reference_tokens = None
        if self.dflash_debug_compare_target_ar:
            reference_tokens = [first_token]
            reference_token = first_token
            reference_past_key_values = self._crop_past_key_values(
                outputs.past_key_values, action_base_position
            )
            stage_started = profile_start()
            for reference_idx in range(max_new_tokens - 1):
                reference_embeds = (
                    self.base_model.language_model.model.embed_tokens(
                        reference_token
                    )
                )
                reference_position_ids = torch.full(
                    (1, 1),
                    action_base_position + reference_idx,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                reference_outputs, reference_logits, _, _ = self(
                    input_embeds=reference_embeds,
                    output_orig=True,
                    attention_mask=None,
                    past_key_values=reference_past_key_values,
                    return_dict=True,
                    position_ids=reference_position_ids,
                    use_cache=True,
                    **self._dflash_target_logit_kwargs(),
                )
                reference_token = self._sample_dflash_target_logits(
                    reference_logits[:, -1:, :]
                )
                reference_tokens.append(reference_token)
                reference_past_key_values = reference_outputs.past_key_values
            profile_end("target_ar_reference", stage_started)
            target_ar_reference_tokens = torch.cat(reference_tokens, dim=1)

        # These signatures reuse the target prefill that is required to obtain t0.
        # They add only vector comparisons and never run an extra target forward.
        current_prompt_signature = prompt_context[:, -1, :].detach()
        collect_temporal_diagnostics = self.dflash_verify_skip_mode == "shadow"
        current_prompt_pooled_signature = (
            prompt_context.mean(dim=1).detach()
            if collect_temporal_diagnostics
            else None
        )
        current_first_action_probs = (
            F.softmax(
                self._dflash_action_logits(first_token_logits[:, 0, :]).float(),
                dim=-1,
            ).detach()
            if collect_temporal_diagnostics
            else None
        )
        temporal_action_skip_record = None
        temporal_action_skip_active = False
        temporal_proposal_route_active = False
        if self.dflash_verify_skip_mode != "off":
            previous_prompt_signature = self._dflash_previous_prompt_signature
            previous_prompt_pooled_signature = (
                self._dflash_previous_prompt_pooled_signature
            )
            previous_first_action_probs = self._dflash_previous_first_action_probs
            previous_action_tokens = self._dflash_previous_action_tokens
            previous_verified_action_run_length = int(
                self._dflash_verified_action_run_length
            )
            prompt_temporal_cosine = None
            prompt_temporal_min_layer_cosine = None
            prompt_temporal_mean_layer_cosine = None
            prompt_temporal_relative_l2 = None
            prompt_pooled_min_layer_cosine = None
            prompt_pooled_mean_layer_cosine = None
            prompt_pooled_relative_l2 = None
            first_action_prob_cosine = None
            first_action_prob_total_variation = None
            first_token_matches_previous = False
            needs_temporal_gate = prefill_fusion_record is None
            if (
                previous_prompt_signature is not None
                and previous_action_tokens is not None
                and (needs_temporal_gate or collect_temporal_diagnostics)
            ):
                num_selected_layers = len(self.ea_layer.target_layer_ids)
                current_prompt_layers = current_prompt_signature.float().reshape(
                    1, num_selected_layers, self.hidden_size
                )
                previous_prompt_layers = previous_prompt_signature.float().reshape(
                    1, num_selected_layers, self.hidden_size
                )
                prompt_layer_cosines = F.cosine_similarity(
                    current_prompt_layers, previous_prompt_layers, dim=-1
                )
                gate_values = torch.stack(
                    [
                        prompt_layer_cosines[0, -1],
                        first_token[0, 0].eq(previous_action_tokens[0, 0]).to(
                            prompt_layer_cosines.dtype
                        ),
                    ]
                )
                gate_cosine, gate_token_match = gate_values.detach().cpu().tolist()
                prompt_temporal_cosine = float(gate_cosine)
                first_token_matches_previous = bool(gate_token_match)

                if collect_temporal_diagnostics:
                    prompt_temporal_min_layer_cosine = float(
                        prompt_layer_cosines.amin().item()
                    )
                    prompt_temporal_mean_layer_cosine = float(
                        prompt_layer_cosines.mean().item()
                    )
                    prompt_temporal_relative_l2 = float(
                        (
                            torch.linalg.vector_norm(
                                current_prompt_signature.float()
                                - previous_prompt_signature.float(),
                                dim=-1,
                            )
                            / torch.linalg.vector_norm(
                                previous_prompt_signature.float(), dim=-1
                            ).clamp_min(1e-6)
                        )[0].item()
                    )

                if (
                    collect_temporal_diagnostics
                    and previous_prompt_pooled_signature is not None
                ):
                    current_pooled_layers = (
                        current_prompt_pooled_signature.float().reshape(
                            1, num_selected_layers, self.hidden_size
                        )
                    )
                    previous_pooled_layers = (
                        previous_prompt_pooled_signature.float().reshape(
                            1, num_selected_layers, self.hidden_size
                        )
                    )
                    pooled_layer_cosines = F.cosine_similarity(
                        current_pooled_layers, previous_pooled_layers, dim=-1
                    )
                    prompt_pooled_min_layer_cosine = float(
                        pooled_layer_cosines.amin().item()
                    )
                    prompt_pooled_mean_layer_cosine = float(
                        pooled_layer_cosines.mean().item()
                    )
                    prompt_pooled_relative_l2 = float(
                        (
                            torch.linalg.vector_norm(
                                current_prompt_pooled_signature.float()
                                - previous_prompt_pooled_signature.float(),
                                dim=-1,
                            )
                            / torch.linalg.vector_norm(
                                previous_prompt_pooled_signature.float(), dim=-1
                            ).clamp_min(1e-6)
                        )[0].item()
                    )

                if (
                    collect_temporal_diagnostics
                    and previous_first_action_probs is not None
                ):
                    first_action_prob_cosine = float(
                        F.cosine_similarity(
                            current_first_action_probs,
                            previous_first_action_probs,
                            dim=-1,
                        )[0].item()
                    )
                    first_action_prob_total_variation = float(
                        (
                            0.5
                            * torch.abs(
                                current_first_action_probs
                                - previous_first_action_probs
                            ).sum(dim=-1)
                        )[0].item()
                    )
            temporal_gate_selected = bool(
                first_token_matches_previous
                and prompt_temporal_cosine is not None
                and prompt_temporal_cosine
                >= self.dflash_verify_skip_min_temporal_cosine
                and previous_verified_action_run_length
                >= self.dflash_verify_skip_min_stable_actions
            )
            temporal_route_gate_selected = bool(
                first_token_matches_previous
                and prompt_temporal_cosine is not None
                and prompt_temporal_cosine
                >= self.dflash_temporal_route_min_cosine
            )
            temporal_proposal_route_active = bool(
                self.dflash_verify_skip_mode in {"route", "active"}
                and temporal_route_gate_selected
                and prefill_fusion_record is None
            )
            temporal_action_skip_active = bool(
                self.dflash_verify_skip_mode == "active"
                and temporal_gate_selected
                and prefill_fusion_record is None
                and self._dflash_consecutive_verify_skips
                < self.dflash_verify_skip_max_consecutive
            )
            temporal_action_skip_record = {
                "eligible": previous_action_tokens is not None,
                "gate_selected": temporal_gate_selected,
                "actually_skipped": temporal_action_skip_active,
                "route_gate_selected": temporal_route_gate_selected,
                "actually_routed": (
                    temporal_proposal_route_active
                    and not temporal_action_skip_active
                ),
                "first_token_matches_previous": first_token_matches_previous,
                "previous_verified_action_run_length": (
                    previous_verified_action_run_length
                ),
                "prompt_temporal_cosine": prompt_temporal_cosine,
                "prompt_temporal_min_layer_cosine": prompt_temporal_min_layer_cosine,
                "prompt_temporal_mean_layer_cosine": prompt_temporal_mean_layer_cosine,
                "prompt_temporal_relative_l2": prompt_temporal_relative_l2,
                "prompt_pooled_min_layer_cosine": prompt_pooled_min_layer_cosine,
                "prompt_pooled_mean_layer_cosine": prompt_pooled_mean_layer_cosine,
                "prompt_pooled_relative_l2": prompt_pooled_relative_l2,
                "first_action_prob_cosine": first_action_prob_cosine,
                "first_action_prob_total_variation": first_action_prob_total_variation,
                "pixel_temporal_cosine": pixel_temporal_cosine,
                "pixel_temporal_relative_l2": pixel_temporal_relative_l2,
                "full_exact_match": None,
                "full_action_group_match_r3": None,
                "full_action_group_match_r5": None,
                "full_action_group_match_r9": None,
            }

        if temporal_action_skip_active:
            output_ids[
                :, token_prefix_len + 1 : token_prefix_len + max_new_tokens
            ] = self._dflash_previous_action_tokens[:, 1:max_new_tokens]
            if target_ar_reference_tokens is not None:
                skipped_action_tokens = output_ids[
                    :, token_prefix_len : token_prefix_len + max_new_tokens
                ]
                temporal_action_skip_record["full_exact_match"] = bool(
                    skipped_action_tokens.eq(target_ar_reference_tokens).all().item()
                )
                exact_tail_mask = skipped_action_tokens[:, 1:].eq(
                    target_ar_reference_tokens[:, 1:]
                )
                temporal_action_skip_record["tail_exact_accept_length"] = int(
                    exact_tail_mask.int().cumprod(dim=1).sum(dim=1)[0].item()
                )
                full_action_positions = torch.arange(
                    max_new_tokens,
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                for threshold in (3, 5, 9):
                    relaxed_mask = self._compute_dflash_accept_mask(
                        skipped_action_tokens,
                        target_ar_reference_tokens,
                        accept_threshold=threshold,
                        action_position_ids=full_action_positions,
                        acceptance_mode="action_group",
                    )
                    temporal_action_skip_record[
                        f"full_action_group_match_r{threshold}"
                    ] = bool(relaxed_mask.bool().all().item())
                    temporal_action_skip_record[
                        f"tail_action_group_accept_length_r{threshold}"
                    ] = int(
                        relaxed_mask[:, 1:]
                        .int()
                        .cumprod(dim=1)
                        .sum(dim=1)[0]
                        .item()
                    )
            self._dflash_previous_prompt_signature = current_prompt_signature
            self._dflash_previous_prompt_pooled_signature = (
                current_prompt_pooled_signature
            )
            self._dflash_previous_first_action_probs = current_first_action_probs
            self._dflash_previous_pixel_signature = current_pixel_signature
            self._dflash_consecutive_verify_skips += 1
            stage_profile = {
                name: {
                    "total_ms": profile_totals_ms[name],
                    "calls": profile_call_counts[name],
                    "mean_ms": profile_totals_ms[name] / profile_call_counts[name],
                }
                for name in profile_totals_ms
            }
            generation_stats = {
                "backend": "dflash",
                "block_size": block_size,
                "generated_tokens": max_new_tokens,
                "include_anchor_hidden": True,
                "use_causal_residual_sampling": bool(
                    self.dflash_use_causal_residual_sampling
                ),
                "action_head_type": getattr(self.ea_layer, "action_head_type", "none"),
                "acceptance_mode": self.dflash_acceptance_mode,
                "tree_mode": self.dflash_tree_mode,
                "tree_budget": self.dflash_tree_budget,
                "target_logits_mode": self.dflash_target_logits_mode,
                "target_logit_shadow_checks": self._target_logit_shadow_checks,
                "target_logit_shadow_mismatches": self._target_logit_shadow_mismatches,
                "verify_skip_mode": self.dflash_verify_skip_mode,
                "debug_compare_target_ar": self.dflash_debug_compare_target_ar,
                "temporal_route_min_cosine": self.dflash_temporal_route_min_cosine,
                "temporal_route_stop_on_reject": self.dflash_temporal_route_stop_on_reject,
                "temporal_fuse_verify": self.dflash_temporal_fuse_verify,
                "temporal_prefill_fusion": self.dflash_temporal_prefill_fusion,
                "temporal_prefill_min_stable_actions": (
                    self.dflash_temporal_prefill_min_stable_actions
                ),
                "temporal_prefill_tree": self.dflash_temporal_prefill_tree,
                "temporal_prefill_tree_max_candidates": (
                    self.dflash_temporal_prefill_tree_max_candidates
                ),
                "temporal_prefill_tree_min_history": (
                    self.dflash_temporal_prefill_tree_min_history
                ),
                "temporal_prefill_fusion_record": None,
                "verify_skip_min_temporal_cosine": self.dflash_verify_skip_min_temporal_cosine,
                "verify_skip_min_stable_actions": self.dflash_verify_skip_min_stable_actions,
                "verify_skip_max_consecutive": self.dflash_verify_skip_max_consecutive,
                "verify_skip_records": [],
                "temporal_action_skip_record": temporal_action_skip_record,
                "verify_skipped_actions": 1,
                "temporal_proposal_routed_actions": 0,
                "temporal_proposal_routed_blocks": 0,
                "temporal_proposal_rejected_blocks": 0,
                "temporal_fallback_draft_blocks": 0,
                "temporal_fused_verify_blocks": 0,
                "verify_skipped_blocks": 1,
                "verify_skipped_tokens": max_new_tokens - 1,
                "first_token_early_reject_blocks": 0,
                "confidence_truncated_blocks": 0,
                "num_blocks": 1,
                "bootstrapped_tokens": 1,
                "progressed_tokens": max_new_tokens,
                "progress_lengths": [max_new_tokens],
                "length": float(max_new_tokens),
                "table1_length": float(max_new_tokens),
                "avg_progress_length": float(max_new_tokens),
                "avg_tail_progress_length": float(max_new_tokens - 1),
                "anchor_decode_steps": 0,
                "target_bootstrap_tokens": 0,
                "accept_lengths": [max_new_tokens - 1],
                "main_path_accept_lengths": [max_new_tokens - 1],
                "avg_accept_length": float(max_new_tokens - 1),
                "accepted_tokens": max_new_tokens - 1,
                "verified_accepted_tokens": 0,
                "compared_tokens": 0,
                "overall_hit_rate": None,
                "action_group_rescued_blocks": 0,
                "action_group_extra_accepted": 0,
                "tree_triggered_blocks": 0,
                "tree_selected_alternate_blocks": 0,
                "tree_extra_verified_nodes": 0,
                "tree_extra_accepted": 0,
                "tree_average_verified_nodes": None,
                "tree_average_max_depth": None,
                "stage_profile": stage_profile,
                "per_position": [],
                "final_action_tokens": (
                    output_ids[
                        0, token_prefix_len : token_prefix_len + max_new_tokens
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                    if self.dflash_profile_stages
                    else None
                ),
                "target_ar_reference_tokens": (
                    target_ar_reference_tokens[0].detach().cpu().tolist()
                    if target_ar_reference_tokens is not None
                    else None
                ),
            }
            self._record_dflash_action_history(
                output_ids[:, token_prefix_len : token_prefix_len + max_new_tokens]
            )
            self.last_dflash_stats = generation_stats
            self.last_generation_stats = generation_stats
            return output_ids[:, token_prefix_len:max_length]

        position_hits = [0 for _ in range(max_draft_tokens)]
        position_counts = [0 for _ in range(max_draft_tokens)]
        if prefill_fusion_record is None:
            accept_lengths = []
            main_path_accept_lengths = []
            progress_lengths = []
            total_accepted = 0
            total_compared = 0
            total_verified_accepted = 0
            bootstrapped_tokens = 1
        else:
            prefill_accept_length = int(prefill_fusion_record["accept_length"])
            prefill_progress_length = int(prefill_fusion_record["progress_length"])
            accept_lengths = [prefill_accept_length]
            main_path_accept_lengths = [
                int(
                    prefill_fusion_record.get(
                        "hold_accept_length",
                        prefill_accept_length,
                    )
                )
            ]
            progress_lengths = [prefill_progress_length]
            total_accepted = prefill_accept_length
            total_compared = max_new_tokens
            total_verified_accepted = prefill_accept_length
            bootstrapped_tokens = 0
        anchor_decode_steps = 0
        confidence_truncated_blocks = 0
        action_group_extra_accepted = 0
        action_group_rescued_blocks = 0
        tree_triggered_blocks = 0
        tree_selected_alternate_blocks = 0
        tree_extra_verified_nodes = 0
        tree_extra_accepted = 0
        tree_total_verified_nodes = 0
        tree_max_depth_sum = 0
        verify_skip_records = []
        verify_skipped_blocks = 0
        verify_skipped_tokens = 0
        first_token_early_reject_blocks = 0
        temporal_proposal_routed_blocks = 0
        temporal_proposal_rejected_blocks = 0
        temporal_fallback_draft_blocks = 0
        temporal_fused_verify_blocks = 0
        temporal_route_available = temporal_proposal_route_active
        current_anchor_signature = None
        temporal_hidden_cosine = None

        if (
            prefill_fusion_record is not None
            and int(prefill_fusion_record["accept_length"]) > 0
            and prefill_selected_hidden is not None
        ):
            current_anchor_signature = prefill_selected_hidden[
                :, 0, -self.hidden_size :
            ].detach()
            if self._dflash_previous_anchor_signature is not None:
                temporal_hidden_cosine = F.cosine_similarity(
                    current_anchor_signature.float(),
                    self._dflash_previous_anchor_signature.float(),
                    dim=-1,
                )

        anchor_idx = initial_anchor_idx
        while anchor_idx < max_new_tokens - 1:
            cache_length_before_block = self._past_key_values_length(
                past_key_values
            )
            token_anchor_pos = token_prefix_len + anchor_idx
            anchor_input_ids = output_ids[:, token_anchor_pos : token_anchor_pos + 1]
            remaining = max_new_tokens - anchor_idx - 1
            q_len = min(max_draft_tokens, remaining)
            if q_len <= 0:
                break

            use_temporal_proposal = bool(
                temporal_route_available
                and self._dflash_previous_action_tokens is not None
                and self._dflash_previous_action_tokens.shape[1]
                >= anchor_idx + 1 + q_len
            )
            fused_temporal_verify = bool(
                use_temporal_proposal
                and self.dflash_temporal_fuse_verify
                and self.dflash_tree_mode == "off"
            )
            fused_target_hidden = None
            verify_outputs = None
            if use_temporal_proposal:
                proposed_tokens = self._dflash_previous_action_tokens[
                    :, anchor_idx + 1 : anchor_idx + 1 + q_len
                ]
                temporal_proposal_routed_blocks += 1

            if fused_temporal_verify:
                # The historical proposal is available before anchor decoding,
                # so one causal target call can verify the complete block.
                fused_input_ids = torch.cat(
                    [anchor_input_ids, proposed_tokens[:, : q_len - 1]], dim=1
                )
                fused_position_ids = torch.arange(
                    action_base_position + anchor_idx,
                    action_base_position + anchor_idx + q_len,
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                fused_embeds = self.base_model.language_model.model.embed_tokens(
                    fused_input_ids
                )
                stage_started = profile_start()
                verify_outputs, fused_logits, _, _ = self(
                    input_embeds=fused_embeds,
                    output_orig=True,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    return_dict=True,
                    position_ids=fused_position_ids,
                    use_cache=True,
                    **self._dflash_target_logit_kwargs(),
                )
                profile_end("target_fused_verify", stage_started)
                anchor_decode_steps += 1
                temporal_fused_verify_blocks += 1
                fused_target_hidden = extract_context_feature(
                    verify_outputs.hidden_states,
                    self.ea_layer.target_layer_ids,
                    self.ea_layer.selected_hidden_variant,
                )
                anchor_hidden = fused_target_hidden[:, :1, :]
                posterior_tokens = self._sample_dflash_target_logits(fused_logits)
                target_next_token = posterior_tokens[:, :1]
                anchor_outputs = verify_outputs
            else:
                # DFlash needs the target anchor hidden before it can propose.
                stage_started = profile_start()
                anchor_embeds = self.base_model.language_model.model.embed_tokens(
                    anchor_input_ids
                )
                anchor_position_ids = torch.full(
                    (1, 1),
                    action_base_position + anchor_idx,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                anchor_outputs, anchor_logits, _, _ = self(
                    input_embeds=anchor_embeds,
                    output_orig=True,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    return_dict=True,
                    position_ids=anchor_position_ids,
                    use_cache=True,
                    **self._dflash_target_logit_kwargs(),
                )
                profile_end("target_anchor", stage_started)
                anchor_decode_steps += 1
                anchor_hidden = extract_context_feature(
                    anchor_outputs.hidden_states,
                    self.ea_layer.target_layer_ids,
                    self.ea_layer.selected_hidden_variant,
                )[:, :1, :]
                target_next_token = self._sample_dflash_target_logits(
                    anchor_logits[:, -1:, :]
                )

            if anchor_idx == 0:
                current_anchor_signature = anchor_hidden[
                    :, 0, -self.hidden_size :
                ].detach()
                if self._dflash_previous_anchor_signature is not None:
                    temporal_hidden_cosine = F.cosine_similarity(
                        current_anchor_signature.float(),
                        self._dflash_previous_anchor_signature.float(),
                        dim=-1,
                    )
            action_context_with_anchor = torch.cat([action_context, anchor_hidden], dim=1)

            action_position_ids = torch.arange(
                anchor_idx,
                anchor_idx + q_len,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            base_draft_logits = None
            draft_logits = None
            if not use_temporal_proposal:
                if temporal_proposal_route_active and not temporal_route_available:
                    temporal_fallback_draft_blocks += 1
                block_input_ids = torch.full(
                    (1, q_len),
                    self.dflash_mask_token_id,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                block_input_ids[:, 0] = anchor_input_ids.squeeze(1)
                noise_embedding = self.base_model.language_model.model.embed_tokens(
                    block_input_ids
                )

                target_hidden = torch.cat(
                    [prompt_context, action_context_with_anchor], dim=1
                )
                action_ctx_position_ids = torch.arange(
                    action_base_position,
                    action_base_position + action_context_with_anchor.shape[1],
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                ctx_position_ids = torch.cat(
                    [prompt_position_ids, action_ctx_position_ids], dim=1
                )
                noise_position_ids = torch.arange(
                    action_base_position + anchor_idx,
                    action_base_position + anchor_idx + q_len,
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0)

                stage_started = profile_start()
                draft_hidden = self.ea_layer(
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    ctx_position_ids=ctx_position_ids,
                    noise_position_ids=noise_position_ids,
                    ctx_attention_mask=None,
                    action_position_ids=action_position_ids,
                )
                profile_end("draft_backbone", stage_started)
                stage_started = profile_start()
                if (
                    self.dflash_use_causal_residual_sampling
                    and getattr(self.ea_layer, "action_sequential_enabled", False)
                ):
                    base_draft_logits = self.ea_layer.project_action_logits(
                        draft_hidden,
                        self.base_model.language_model.lm_head,
                    )
                    proposed_tokens, draft_logits, _ = self.ea_layer.sample_action_block(
                        base_logits=base_draft_logits,
                        hidden_states=draft_hidden,
                        first_prev_token_ids=anchor_input_ids,
                        action_position_ids=action_position_ids,
                        temperature=0.0,
                        confidence_threshold=self.dflash_confidence_threshold,
                        confidence_min_tokens=self.dflash_confidence_min_tokens,
                    )
                    if proposed_tokens.shape[1] < q_len:
                        confidence_truncated_blocks += 1
                        q_len = proposed_tokens.shape[1]
                elif (
                    self.dflash_use_causal_residual_sampling
                    and (
                        getattr(self.ea_layer, "causal_residual_enabled", False)
                        or getattr(self.ea_layer, "logit_markov_enabled", False)
                    )
                ):
                    proposed_tokens, draft_logits = (
                        self.ea_layer.sample_with_causal_residual(
                            hidden_states=draft_hidden,
                            first_prev_token_ids=anchor_input_ids,
                            lm_head=self.base_model.language_model.lm_head,
                            temperature=0.0,
                            start_index=self.dflash_causal_residual_start_index,
                        )
                    )
                else:
                    if getattr(self.ea_layer, "action_projection_enabled", False):
                        base_draft_logits = self.ea_layer.project_action_logits(
                            draft_hidden,
                            self.base_model.language_model.lm_head,
                        )
                        draft_logits = base_draft_logits
                        proposed_tokens = (
                            dflash_sample(draft_logits, temperature=0.0)
                            + self.ea_layer.action_token_start
                        )
                    else:
                        base_draft_logits = self.base_model.language_model.lm_head(
                            draft_hidden
                        )
                        draft_logits = base_draft_logits
                        proposed_tokens = dflash_sample(draft_logits, temperature=0.0)
                profile_end("action_head", stage_started)

            skip_features = None
            skip_gate_selected = False
            if self.dflash_verify_skip_mode == "shadow" and not use_temporal_proposal:
                if not getattr(self.ea_layer, "action_sequential_enabled", False):
                    raise RuntimeError(
                        "Verification-skip diagnostics require the Action-RNN checkpoint."
                    )
                base_gate_logits = self.ea_layer.action_logits_from_full(
                    base_draft_logits[:, :q_len]
                )
                final_gate_logits = self.ea_layer.action_logits_from_full(
                    draft_logits[:, :q_len]
                )
                skip_features = self._dflash_skip_features(
                    base_gate_logits,
                    final_gate_logits,
                    proposed_tokens[:, :q_len],
                    target_next_token,
                    action_token_start=self.dflash_action_token_start,
                )
                skip_gate_selected = bool(
                    self._dflash_skip_gate(skip_features)[0].item()
                )
            # Block-level confidence remains diagnostic only. Long-block shadow
            # precision is too low to permit an unverified commit.
            skip_active = False

            # The anchor forward already gives the exact target token for the
            # first proposal. If even the configured acceptance rule rejects
            # that token, prefix acceptance is necessarily zero and verifying
            # the remaining proposal suffix cannot change the committed result.
            first_action_position = torch.full(
                (1, 1),
                anchor_idx + 1,
                device=input_ids.device,
                dtype=torch.long,
            )
            first_proposal_accepts = bool(
                self._compute_dflash_accept_mask(
                    proposed_tokens[:, :1],
                    target_next_token,
                    accept_threshold=accept_threshold,
                    action_position_ids=first_action_position,
                    acceptance_mode=self.dflash_acceptance_mode,
                )[0, 0].item()
            )

            # 当前 anchor 已在目标模型 cache 中，因此它等价于 DDTree 的虚拟根；
            # DDTree 在与线性校验相同的节点预算内动态分配深度和分支。
            if not fused_temporal_verify:
                verify_outputs = None
            tree_used = self.dflash_tree_mode == "ddtree" and q_len > 1
            first_token_early_reject = bool(
                q_len > 1
                and not tree_used
                and not skip_active
                and not fused_temporal_verify
                and not first_proposal_accepts
            )
            tree_child_maps = None
            target_selected_tree_nodes = None
            verify_node_count = 0
            anchor_cache_length = (
                cache_length_before_block + 1
                if fused_temporal_verify
                else self._past_key_values_length(anchor_outputs.past_key_values)
            )
            if not fused_temporal_verify:
                stage_started = profile_start()
            if (
                not fused_temporal_verify
                and q_len > 1
                and not skip_active
                and not first_token_early_reject
            ):
                if tree_used:
                    action_draft_logits = self.ea_layer.action_logits_from_full(draft_logits)[
                        0, :q_len
                    ]
                    node_budget = (
                        self.dflash_tree_budget
                        if self.dflash_tree_budget > 0
                        else q_len - 1
                    )
                    (
                        verify_input_ids,
                        tree_child_maps,
                        verify_tree_mask,
                        relative_verify_positions,
                        greedy_tree_tokens,
                    ) = self._build_ddtree_from_logits(
                        action_draft_logits,
                        node_budget=node_budget,
                        token_id_offset=self.ea_layer.action_token_start,
                    )
                    verify_position_ids = (
                        action_base_position + anchor_idx + 1 + relative_verify_positions
                    )
                else:
                    verify_input_ids = proposed_tokens[:1, :-1]
                    verify_tree_mask = None
                    verify_position_ids = torch.arange(
                        action_base_position + anchor_idx + 1,
                        action_base_position + anchor_idx + q_len,
                        device=input_ids.device,
                        dtype=torch.long,
                    ).unsqueeze(0)
                verify_node_count = int(verify_input_ids.shape[1])
                verify_embeds = self.base_model.language_model.model.embed_tokens(verify_input_ids)
                language_model = self.base_model.language_model
                if tree_used and getattr(
                    language_model.config,
                    "_attn_implementation",
                    None,
                ) == "flash_attention_2":
                    raise RuntimeError(
                        "DFlash DDTree verification needs an explicit 4D tree mask and "
                        "is not compatible with flash_attention_2. Use eager/SDPA or disable the tree."
                    )
                previous_tree_mask = getattr(language_model, "tree_mask", None)
                language_model.tree_mask = verify_tree_mask
                profile_end("verify_prepare", stage_started)
                stage_started = profile_start()
                try:
                    verify_outputs, verify_logits, _, _ = self(
                        input_embeds=verify_embeds,
                        output_orig=True,
                        attention_mask=None,
                        past_key_values=anchor_outputs.past_key_values,
                        return_dict=True,
                        position_ids=verify_position_ids,
                        use_cache=True,
                        **self._dflash_target_logit_kwargs(),
                    )
                finally:
                    language_model.tree_mask = previous_tree_mask

                if tree_used:
                    node_posterior_tokens = self._sample_dflash_target_logits(
                        verify_logits
                    )
                else:
                    posterior_tail = self._sample_dflash_target_logits(verify_logits)
                    posterior_tokens = torch.cat([target_next_token, posterior_tail], dim=1)
                profile_end("target_verify", stage_started)
            elif not fused_temporal_verify:
                posterior_tokens = target_next_token
                profile_end("verify_prepare", stage_started)

            stage_started = profile_start()
            selected_correction_token = None
            if skip_active:
                effective_accept_length = q_len
                token_best_accept_length = 0
                main_path_accept_length = q_len
                selected_proposed_tokens = proposed_tokens
                selected_alternate_path = False
            elif first_token_early_reject:
                effective_accept_length = 0
                token_best_accept_length = 0
                main_path_accept_length = 0
                selected_proposed_tokens = proposed_tokens
                selected_alternate_path = False
                selected_correction_token = target_next_token
                first_token_early_reject_blocks += 1
            elif tree_used:
                if tree_child_maps is None or verify_outputs is None:
                    raise AssertionError("DDTree traversal requires a compiled and verified node graph.")
                exact_tree_nodes, exact_correction_token = self._follow_ddtree_target_path(
                    tree_child_maps,
                    target_next_token,
                    node_posterior_tokens,
                    q_len,
                )
                token_best_accept_length = int(exact_tree_nodes.numel())
                if accept_threshold is None or int(accept_threshold) <= 0:
                    target_selected_tree_nodes = exact_tree_nodes
                    effective_accept_length = token_best_accept_length
                    correction_token_id = exact_correction_token
                else:
                    (
                        target_selected_tree_nodes,
                        correction_token_id,
                        effective_accept_length,
                    ) = self._select_relaxed_ddtree_path(
                        child_maps=tree_child_maps,
                        verify_input_ids=verify_input_ids,
                        root_posterior_token=target_next_token,
                        node_posterior_tokens=node_posterior_tokens,
                        action_start_position=anchor_idx + 1,
                        max_action_tokens=q_len,
                        accept_threshold=int(accept_threshold),
                    )
                selected_proposed_tokens = verify_input_ids.index_select(
                    1, target_selected_tree_nodes
                )
                selected_correction_token = torch.tensor(
                    [[correction_token_id]],
                    device=input_ids.device,
                    dtype=torch.long,
                )

                target_prefix = torch.cat(
                    [selected_proposed_tokens, selected_correction_token], dim=1
                )
                compare_length = min(
                    effective_accept_length, int(target_prefix.shape[1])
                )
                main_matches = (
                    greedy_tree_tokens[:, :compare_length]
                    == target_prefix[:, :compare_length]
                ).int()
                main_path_accept_length = int(
                    main_matches.cumprod(dim=1).sum(dim=1)[0].item()
                )
                selected_alternate_path = (
                    effective_accept_length > main_path_accept_length
                )
            else:
                predicted_action_positions = torch.arange(
                    anchor_idx + 1,
                    anchor_idx + q_len + 1,
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                token_accept_mask = self._compute_dflash_accept_mask(
                    proposed_tokens,
                    posterior_tokens,
                    accept_threshold=accept_threshold,
                    action_position_ids=predicted_action_positions,
                    acceptance_mode="token",
                )
                accept_mask = self._compute_dflash_accept_mask(
                    proposed_tokens,
                    posterior_tokens,
                    accept_threshold=accept_threshold,
                    action_position_ids=predicted_action_positions,
                    acceptance_mode=self.dflash_acceptance_mode,
                )
                effective_accept_length = min(
                    int(accept_mask.cumprod(dim=1).sum(dim=1)[0].item()), q_len
                )
                token_best_accept_length = min(
                    int(token_accept_mask.cumprod(dim=1).sum(dim=1)[0].item()), q_len
                )
                main_path_accept_length = effective_accept_length
                selected_proposed_tokens = proposed_tokens
                selected_alternate_path = False
                if effective_accept_length < q_len:
                    selected_correction_token = posterior_tokens[
                        :, effective_accept_length : effective_accept_length + 1
                    ]
            profile_end("accept_select", stage_started)

            # A target rejection proves that the cached action has diverged at
            # this anchor. Reusing its later tail is usually a sequence of cheap
            # proposals but expensive target failures. Switch the remaining
            # positions to DFlash, which now receives the corrected true prefix.
            if use_temporal_proposal and effective_accept_length < q_len:
                temporal_proposal_rejected_blocks += 1
                if self.dflash_temporal_route_stop_on_reject:
                    temporal_route_available = False

            stage_started = profile_start()
            if q_len > 0 and not skip_active:
                current_hits = [1] * effective_accept_length + [0] * (
                    q_len - effective_accept_length
                )
                for idx, hit in enumerate(current_hits):
                    position_counts[idx] += 1
                    position_hits[idx] += int(hit)
                total_compared += q_len
                total_verified_accepted += effective_accept_length
            accept_lengths.append(effective_accept_length)
            main_path_accept_lengths.append(main_path_accept_length)
            total_accepted += effective_accept_length
            group_gain = (
                max(effective_accept_length - token_best_accept_length, 0)
                if not skip_active
                else 0
            )
            action_group_extra_accepted += group_gain
            action_group_rescued_blocks += int(group_gain > 0)
            if tree_used:
                tree_triggered_blocks += 1
                tree_selected_alternate_blocks += int(selected_alternate_path)
                tree_extra_verified_nodes += max(verify_node_count - (q_len - 1), 0)
                tree_extra_accepted += max(effective_accept_length - main_path_accept_length, 0)
                tree_total_verified_nodes += verify_node_count
                tree_max_depth_sum += int(relative_verify_positions.max().item()) + 1

            if skip_features is not None:
                candidate_action_tokens = torch.cat(
                    [
                        output_ids[
                            :,
                            token_prefix_len : token_prefix_len + anchor_idx + 1,
                        ],
                        proposed_tokens[:, :q_len],
                    ],
                    dim=1,
                )
                previous_action_max_delta = None
                previous_action_mean_delta = None
                if (
                    self._dflash_previous_action_tokens is not None
                    and candidate_action_tokens.shape
                    == self._dflash_previous_action_tokens.shape
                ):
                    previous_delta = torch.abs(
                        candidate_action_tokens.long()
                        - self._dflash_previous_action_tokens.long()
                    ).float()
                    previous_action_max_delta = float(previous_delta.max().item())
                    previous_action_mean_delta = float(previous_delta.mean().item())
                verify_skip_records.append(
                    {
                        "anchor_index": anchor_idx,
                        "q_len": q_len,
                        "gate_selected": skip_gate_selected,
                        "actually_skipped": skip_active,
                        "sentinel_match": bool(
                            skip_features["sentinel_match"][0].item()
                        ),
                        "min_top1_prob": float(
                            skip_features["min_top1_prob"][0].item()
                        ),
                        "mean_top1_prob": float(
                            skip_features["mean_top1_prob"][0].item()
                        ),
                        "min_margin": float(skip_features["min_margin"][0].item()),
                        "max_normalized_entropy": float(
                            skip_features["max_normalized_entropy"][0].item()
                        ),
                        "base_agreement": float(
                            skip_features["base_agreement"][0].item()
                        ),
                        "temporal_hidden_cosine": (
                            float(temporal_hidden_cosine[0].item())
                            if temporal_hidden_cosine is not None
                            else None
                        ),
                        "previous_action_max_token_delta": previous_action_max_delta,
                        "previous_action_mean_token_delta": previous_action_mean_delta,
                        "candidate_tokens": proposed_tokens[0, :q_len]
                        .detach()
                        .cpu()
                        .tolist(),
                        "exact_accept_length": (
                            None if skip_active else int(token_best_accept_length)
                        ),
                        "full_exact_match": (
                            None
                            if skip_active
                            else bool(token_best_accept_length == q_len)
                        ),
                    }
                )
            if skip_active:
                verify_skipped_blocks += 1
                verify_skipped_tokens += q_len

            proposed_start_pos = token_prefix_len + anchor_idx + 1
            if effective_accept_length > 0:
                output_ids[
                    :,
                    proposed_start_pos : proposed_start_pos + effective_accept_length,
                ] = selected_proposed_tokens[:, :effective_accept_length]

            all_accepted = effective_accept_length == q_len
            if not all_accepted:
                if selected_correction_token is None:
                    raise AssertionError("A rejected proposal must be corrected by the target token.")
                output_ids[:, proposed_start_pos + effective_accept_length] = (
                    selected_correction_token[:, 0]
                )

            new_anchor_idx = anchor_idx + (q_len if all_accepted else effective_accept_length + 1)
            progress_lengths.append(int(new_anchor_idx - anchor_idx))
            if skip_active:
                anchor_idx = new_anchor_idx
                profile_end("cache_commit", stage_started)
                continue
            if fused_temporal_verify:
                new_cache_length = action_base_position + new_anchor_idx
                past_key_values = self._crop_past_key_values(
                    verify_outputs.past_key_values, new_cache_length
                )
                # The fused call contains the anchor followed by q-1 proposal
                # inputs. Keep only target hiddens that precede the next anchor.
                fused_hidden_count = 1 + min(
                    effective_accept_length, max(q_len - 1, 0)
                )
                action_context = torch.cat(
                    [action_context, fused_target_hidden[:, :fused_hidden_count, :]],
                    dim=1,
                )
            else:
                source_past_key_values = (
                    verify_outputs.past_key_values
                    if verify_outputs is not None
                    else anchor_outputs.past_key_values
                )
                tail_hidden_count = (
                    effective_accept_length
                    if tree_used
                    else min(effective_accept_length, max(q_len - 1, 0))
                )
                if tree_used and verify_outputs is not None:
                    if target_selected_tree_nodes is None:
                        raise AssertionError(
                            "Tree verification did not produce committed node indices."
                        )
                    selected_tree_nodes = target_selected_tree_nodes[:tail_hidden_count]
                    past_key_values = self._select_tree_past_key_values(
                        source_past_key_values,
                        base_length=anchor_cache_length,
                        tree_node_indices=selected_tree_nodes,
                    )
                    new_cache_length = anchor_cache_length + tail_hidden_count
                    if self._past_key_values_length(past_key_values) != new_cache_length:
                        raise AssertionError(
                            "Committed tree KV length does not match the accepted action prefix: "
                            f"cache={self._past_key_values_length(past_key_values)} "
                            f"expected={new_cache_length}."
                        )
                else:
                    new_cache_length = action_base_position + new_anchor_idx
                    past_key_values = self._crop_past_key_values(
                        source_past_key_values, new_cache_length
                    )

                if verify_outputs is not None and tail_hidden_count > 0:
                    all_verified_hidden = extract_context_feature(
                        verify_outputs.hidden_states,
                        self.ea_layer.target_layer_ids,
                        self.ea_layer.selected_hidden_variant,
                    )
                    if tree_used:
                        verified_hidden = all_verified_hidden.index_select(
                            1,
                            target_selected_tree_nodes[:tail_hidden_count].to(
                                all_verified_hidden.device
                            ),
                        )
                    else:
                        verified_hidden = all_verified_hidden[:, :tail_hidden_count, :]
                    action_context = torch.cat(
                        [action_context_with_anchor, verified_hidden], dim=1
                    )
                else:
                    action_context = action_context_with_anchor
            anchor_idx = new_anchor_idx
            profile_end("cache_commit", stage_started)

        final_action_tokens = output_ids[
            :, token_prefix_len : token_prefix_len + max_new_tokens
        ]
        for record in verify_skip_records:
            candidate_tokens = torch.tensor(
                record.pop("candidate_tokens"),
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            if record["actually_skipped"]:
                for threshold in (3, 5, 9):
                    record[f"full_action_group_match_r{threshold}"] = None
                record["target_tail_max_token_delta"] = None
                record["target_tail_mean_token_delta"] = None
                continue
            start_position = int(record["anchor_index"]) + 1
            q_len = int(record["q_len"])
            target_tail = final_action_tokens[
                :, start_position : start_position + q_len
            ]
            target_delta = torch.abs(
                candidate_tokens.long() - target_tail.long()
            ).float()
            record["target_tail_max_token_delta"] = float(target_delta.max().item())
            record["target_tail_mean_token_delta"] = float(target_delta.mean().item())
            action_positions = torch.arange(
                start_position,
                start_position + q_len,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            for threshold in (3, 5, 9):
                relaxed_mask = self._compute_dflash_accept_mask(
                    candidate_tokens,
                    target_tail,
                    accept_threshold=threshold,
                    action_position_ids=action_positions,
                    acceptance_mode="action_group",
                )
                record[f"full_action_group_match_r{threshold}"] = bool(
                    relaxed_mask.bool().all().item()
                )

        if current_anchor_signature is not None:
            self._dflash_previous_anchor_signature = current_anchor_signature
        if temporal_action_skip_record is not None and self._dflash_previous_action_tokens is not None:
            previous_action_tokens = self._dflash_previous_action_tokens
            temporal_action_skip_record["full_exact_match"] = bool(
                previous_action_tokens.eq(final_action_tokens).all().item()
            )
            exact_tail_mask = previous_action_tokens[:, 1:max_new_tokens].eq(
                final_action_tokens[:, 1:max_new_tokens]
            )
            temporal_action_skip_record["tail_exact_accept_length"] = int(
                exact_tail_mask.int().cumprod(dim=1).sum(dim=1)[0].item()
            )
            full_action_positions = torch.arange(
                max_new_tokens,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            for threshold in (3, 5, 9):
                relaxed_mask = self._compute_dflash_accept_mask(
                    previous_action_tokens,
                    final_action_tokens,
                    accept_threshold=threshold,
                    action_position_ids=full_action_positions,
                    acceptance_mode="action_group",
                )
                temporal_action_skip_record[
                    f"full_action_group_match_r{threshold}"
                ] = bool(relaxed_mask.bool().all().item())
                relaxed_tail_mask = relaxed_mask[:, 1:max_new_tokens]
                temporal_action_skip_record[
                    f"tail_action_group_accept_length_r{threshold}"
                ] = int(
                    relaxed_tail_mask.int().cumprod(dim=1).sum(dim=1)[0].item()
                )
        self._dflash_previous_prompt_signature = current_prompt_signature
        self._dflash_previous_prompt_pooled_signature = (
            current_prompt_pooled_signature
        )
        self._dflash_previous_first_action_probs = current_first_action_probs
        self._dflash_previous_pixel_signature = current_pixel_signature
        if (
            self._dflash_previous_verified_action_tokens is not None
            and self._dflash_previous_verified_action_tokens.eq(
                final_action_tokens
            ).all()
        ):
            self._dflash_verified_action_run_length += 1
        else:
            self._dflash_verified_action_run_length = 1
        self._dflash_previous_verified_action_tokens = (
            final_action_tokens.detach().clone()
        )
        self._dflash_previous_action_tokens = final_action_tokens.detach().clone()
        self._record_dflash_action_history(final_action_tokens)
        self._dflash_consecutive_verify_skips = 0

        per_position_stats = []
        for idx, (hit_count, compare_count) in enumerate(zip(position_hits, position_counts), start=1):
            reject_count = compare_count - hit_count
            per_position_stats.append(
                {
                    "position": idx,
                    "count": compare_count,
                    "hit_count": hit_count,
                    "reject_count": reject_count,
                    "hit_rate": (hit_count / compare_count) if compare_count > 0 else None,
                    "reject_rate": (reject_count / compare_count) if compare_count > 0 else None,
                }
            )
        num_blocks = len(accept_lengths)
        progressed_tokens = min(max_new_tokens, 1 + sum(progress_lengths))
        generation_stats = {
            "backend": "dflash",
            "block_size": block_size,
            "generated_tokens": max_new_tokens,
            "include_anchor_hidden": True,
            "causal_residual_type": getattr(self.ea_layer, "causal_residual_type", "none"),
            "causal_residual_start_index": self.dflash_causal_residual_start_index,
            "use_causal_residual_sampling": bool(self.dflash_use_causal_residual_sampling),
            "action_head_type": getattr(self.ea_layer, "action_head_type", "none"),
            "acceptance_mode": self.dflash_acceptance_mode,
            "tree_mode": self.dflash_tree_mode,
            "tree_budget": self.dflash_tree_budget,
            "target_logits_mode": self.dflash_target_logits_mode,
            "target_logit_shadow_checks": self._target_logit_shadow_checks,
            "target_logit_shadow_mismatches": self._target_logit_shadow_mismatches,
            "verify_skip_mode": self.dflash_verify_skip_mode,
            "debug_compare_target_ar": self.dflash_debug_compare_target_ar,
            "verify_skip_min_top1_prob": self.dflash_verify_skip_min_top1_prob,
            "verify_skip_min_margin": self.dflash_verify_skip_min_margin,
            "verify_skip_min_base_agreement": self.dflash_verify_skip_min_base_agreement,
            "temporal_route_min_cosine": self.dflash_temporal_route_min_cosine,
            "temporal_route_stop_on_reject": self.dflash_temporal_route_stop_on_reject,
            "temporal_fuse_verify": self.dflash_temporal_fuse_verify,
            "temporal_prefill_fusion": self.dflash_temporal_prefill_fusion,
            "temporal_prefill_min_stable_actions": (
                self.dflash_temporal_prefill_min_stable_actions
            ),
            "temporal_prefill_tree": self.dflash_temporal_prefill_tree,
            "temporal_prefill_tree_max_candidates": (
                self.dflash_temporal_prefill_tree_max_candidates
            ),
            "temporal_prefill_tree_min_history": (
                self.dflash_temporal_prefill_tree_min_history
            ),
            "temporal_prefill_fusion_record": prefill_fusion_record,
            "verify_skip_min_temporal_cosine": self.dflash_verify_skip_min_temporal_cosine,
            "verify_skip_min_stable_actions": self.dflash_verify_skip_min_stable_actions,
            "verify_skip_max_consecutive": self.dflash_verify_skip_max_consecutive,
            "verify_skip_records": verify_skip_records,
            "temporal_action_skip_record": temporal_action_skip_record,
            "verify_skipped_actions": 0,
            "temporal_proposal_routed_actions": int(
                temporal_proposal_routed_blocks > 0
            ),
            "temporal_proposal_routed_blocks": temporal_proposal_routed_blocks,
            "temporal_proposal_rejected_blocks": temporal_proposal_rejected_blocks,
            "temporal_fallback_draft_blocks": temporal_fallback_draft_blocks,
            "temporal_fused_verify_blocks": temporal_fused_verify_blocks,
            "temporal_prefill_fused_actions": int(
                prefill_fusion_record is not None
            ),
            "verify_skipped_blocks": verify_skipped_blocks,
            "verify_skipped_tokens": verify_skipped_tokens,
            "first_token_early_reject_blocks": first_token_early_reject_blocks,
            "confidence_threshold": self.dflash_confidence_threshold,
            "confidence_min_tokens": self.dflash_confidence_min_tokens,
            "confidence_truncated_blocks": confidence_truncated_blocks,
            "num_blocks": num_blocks,
            "bootstrapped_tokens": bootstrapped_tokens,
            "progressed_tokens": progressed_tokens,
            "progress_lengths": progress_lengths,
            "length": (progressed_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "table1_length": (progressed_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "avg_progress_length": (progressed_tokens / num_blocks) if num_blocks > 0 else 0.0,
            "avg_tail_progress_length": (sum(progress_lengths) / num_blocks) if num_blocks > 0 else 0.0,
            "anchor_decode_steps": anchor_decode_steps,
            "target_bootstrap_tokens": anchor_decode_steps,
            "accept_lengths": [int(x) for x in accept_lengths],
            "main_path_accept_lengths": [int(x) for x in main_path_accept_lengths],
            "avg_accept_length": (sum(accept_lengths) / num_blocks) if num_blocks > 0 else 0.0,
            "accepted_tokens": total_accepted,
            "verified_accepted_tokens": total_verified_accepted,
            "compared_tokens": total_compared,
            "overall_hit_rate": (
                total_verified_accepted / total_compared if total_compared > 0 else None
            ),
            "action_group_rescued_blocks": action_group_rescued_blocks,
            "action_group_extra_accepted": action_group_extra_accepted,
            "tree_triggered_blocks": tree_triggered_blocks,
            "tree_selected_alternate_blocks": tree_selected_alternate_blocks,
            "tree_extra_verified_nodes": tree_extra_verified_nodes,
            "tree_extra_accepted": tree_extra_accepted,
            "tree_average_verified_nodes": (
                tree_total_verified_nodes / tree_triggered_blocks
                if tree_triggered_blocks > 0
                else None
            ),
            "tree_average_max_depth": (
                tree_max_depth_sum / tree_triggered_blocks
                if tree_triggered_blocks > 0
                else None
            ),
            "stage_profile": {
                name: {
                    "total_ms": profile_totals_ms[name],
                    "calls": profile_call_counts[name],
                    "mean_ms": profile_totals_ms[name] / profile_call_counts[name],
                }
                for name in profile_totals_ms
            },
            "per_position": per_position_stats,
            "final_action_tokens": (
                final_action_tokens[0].detach().cpu().tolist()
                if self.dflash_profile_stages
                else None
            ),
            "target_ar_reference_tokens": (
                target_ar_reference_tokens[0].detach().cpu().tolist()
                if target_ar_reference_tokens is not None
                else None
            ),
        }
        self.last_dflash_stats = generation_stats
        self.last_generation_stats = generation_stats
        return output_ids[:, token_prefix_len:max_length]
    def forward(
        self,
        output_orig=False,
        input_embeds = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        #inputs_embeds: Optional[torch.FloatTensor] = None,
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
    ):
         #先这样，后面看结合需求怎么改，我的判断是得根据需要的数据模态把需要的内容放进去
         with torch.inference_mode():
            #reorganize the embeddings
            #print('forward not tested.')
            #print(output_hidden_states)
            # Pass input through the base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values = pixel_values,
                labels = labels,
                inputs_embeds = input_embeds,
                past_key_values=past_key_values,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states=True,
                output_projector_features=output_projector_features,
                return_dict=return_dict,
                position_ids=position_ids,
                multimodal_tree_position_ids=multimodal_tree_position_ids,
                num_logits_to_keep=num_logits_to_keep,
                logit_token_range=logit_token_range,
            )
            #print(outputs.keys())
            if output_orig:
                orig = outputs.logits
            #print(len(outputs.hidden_states))
            hidden_states = outputs.hidden_states[-1]
            input_embeddings = outputs.hidden_states[0]
            #print(len(hidden_states))
            #print(torch.cat(hidden_states).shape)
            if output_orig:
                return outputs, orig, hidden_states,input_embeddings
            else:
                return outputs, hidden_states
    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        unnorm_key: Optional[str] = None,
        return_hidden_states: bool = False,
        legacy_output_hidden: Optional[bool] = None,
        return_dflash_stats: bool = False,
        return_generation_stats: bool = False,
        generate_mode = 'Speculative',
        #accept_threshold=None,
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
        self.last_dflash_stats = None
        self.last_generation_stats = None
        self._pending_generation_stats = None
        self._target_logit_shadow_checks = 0
        self._target_logit_shadow_mismatches = 0

        # 设置generate方法的参数
        if return_hidden_states:
            kwargs['output_hidden_states'] = True
        
        # 如果特殊的空标记不在提示末尾，则添加它
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            #print('add special token')
            #print(kwargs['attention_mask'])
            kwargs['attention_mask']=torch.cat(
                (kwargs['attention_mask'], torch.unsqueeze(torch.Tensor([1]), dim=0).to(input_ids.device)), dim=1
            ).to(int)
            #print(kwargs['attention_mask'])
        #print(kwargs)
        #exit()
        # 运行模型生成
        #print('base model generate')
        '''outputs = self.ea_forward(
            input_ids=input_ids,
            max_new_tokens=self.get_action_dim(unnorm_key),
            #return_dict=True,
            #return_dict_in_generate=True,
            **kwargs
        )'''
        if return_hidden_states and generate_mode == 'dflash':
            # DFlash 推理当前只返回 token；如需 teacher hidden 做数据收集，退回到 AR 路径以保留逐步 hidden。
            outputs = self.ea_forward(# 回退自回归
                input_ids=input_ids,
                max_new_tokens=self.get_action_dim(unnorm_key),
                output_hidden_states=True,
                **kwargs
            )
        elif generate_mode == 'speculative':# Eagle / tree decoding
            outputs = self.eagenerate(
                input_ids=input_ids,
                max_new_tokens=self.get_action_dim(unnorm_key),
                #return_dict=True,
                #return_dict_in_generate=True,
                accept_threshold=self.accept_threshold,
                **kwargs
            )
            #print(outputs)
        elif generate_mode == 'dflash':
            outputs = self.dflash_generate(# dflash的generate方法（草稿、验证）
                input_ids=input_ids,
                max_new_tokens=self.get_action_dim(unnorm_key),
                accept_threshold=self.accept_threshold,
                **kwargs
            )
        else:

            outputs = self.ea_forward(# 自回归
                input_ids=input_ids,
                max_new_tokens=self.get_action_dim(unnorm_key),

                **kwargs
                )

        if hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences

        elif len(outputs)==2:
            generated_ids = outputs[0]
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
        if return_hidden_states:
            if (len(outputs)==2):
                tmp_hidden = outputs[1]
            else:
                tmp_hidden=outputs.hidden_states
            # 使用前向传播获取隐藏状态
            first_layer_hidden = []
            last_layer_hidden = []
            #print(len(outputs.hidden_states))
            for i in range(len(tmp_hidden)):
                last_layer_hidden.append(tmp_hidden[i][-1].cpu()[0])
                first_layer_hidden.append(tmp_hidden[i][0].cpu()[0])
            # 返回二元组: (动作, 隐藏状态)
            #print(last_layer_hidden[0].shape)
            return actions, predicted_action_token_ids,(first_layer_hidden,last_layer_hidden)
        if return_generation_stats:
            return actions, self.get_generation_stats()
        if return_dflash_stats:
            return actions, self.last_dflash_stats

        # 否则只返回动作
        return actions
    @torch.no_grad()
    def _extract_past_from_model_output(self, outputs, standardize_cache_format: bool = False):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        return past_key_values
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder
    ):
        #print(model_kwargs.keys())
        #print(outputs.keys())
        #exit()
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs
        )
        #print(model_kwargs["past_key_values"][0][0].shape)
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                #print('update attention mask')
                #print(attention_mask)
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
            #else:
            #    print('no attention mask to update')
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs
    @torch.no_grad()
    def ea_forward(self,input_ids,max_new_tokens, logits_processor=None,output_hidden_states=False,**kwargs):
        #prefill the past kv embeddings
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        #use the openvla.forward to initilaize kv
        model = self
        '''if hasattr(model.base_model.language_model, "past_key_values"):
            past_key_values = model.base_model.language_model.past_key_values
            past_key_values_data = model.base_model.language_model.past_key_values_data
            current_length_data = model.base_model.language_model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model)
            model.base_model.language_model.past_key_values = past_key_values
            model.base_model.language_model.past_key_values_data = past_key_values_data
            model.base_model.language_model.current_length_data = current_length_data
        #print(len(model.base_model.language_model.past_key_values))'''
        #print(model.base_model.language_model.past_key_values[0][1].data.shape)
        #exit()
        input_len = input_ids.shape[1]
        #reset_tree_mode(model.ea_layer)
        tokenizer = self.get_tokenizer()
        max_steps = max_new_tokens
        model_inputs = model.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        #print('model inputs:')
        #print(model_inputs['input_ids'])
        #print(model_inputs['attention_mask'])
        #print(model_inputs['pixel_values'])
        #exit()
        #print('start forwarding')
        #print(model_inputs)
        if output_hidden_states:
            hidden_states = []
        outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=output_hidden_states
            )
        if output_hidden_states:
            hidden_states.append(outputs.hidden_states)
        #print('outputs')
        #print('loss',outputs.loss)
        #print('logits',outputs.logits)
        #print('past key values',outputs.past_key_values)
        #print('hidden states',outputs.hidden_states)
        #print('attentions',outputs.attentions)
        #print('projector',outputs.projector_features)
        #exit()
        input_len = input_ids.shape[1]-1
        input_embed_len = outputs['past_key_values'][0][0].shape[2]-1
        #print(type(outputs['past_key_values']))
        new_token = 0
        model_inputs["cache_position"] = torch.arange(input_embed_len, device=input_ids.device)
        model_inputs['use_cache']=True
        model_inputs['attention_mask']=outputs.attention_mask
        for idx in range(max_steps):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(input_ids, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            #print(input_id)
            #exit()
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            model_inputs = self._update_model_kwargs_for_generation(
                outputs,
                model_inputs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            model_inputs['input_ids']=input_ids
            #print(model_inputs)
            model_inputs = model.base_model.prepare_inputs_for_generation(**model_inputs)
            outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=output_hidden_states,
                #use_cache = True
            )
            if output_hidden_states:
                hidden_states.append(outputs.hidden_states)
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
        #print('ea forward',output_hidden_states)
        if output_hidden_states:
            #print(outputs.hidden_states)
            return input_ids[:,input_len+1:],hidden_states[:-1]
        return input_ids[:,input_len+1:]
    def ea_forward_embed(self,input_ids,max_new_tokens, logits_processor=None,**kwargs):
        #prefill the past kv embeddings
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        #use the openvla.forward to initilaize kv
        model = self
        '''if hasattr(model.base_model.language_model, "past_key_values"):
            past_key_values = model.base_model.language_model.past_key_values
            past_key_values_data = model.base_model.language_model.past_key_values_data
            current_length_data = model.base_model.language_model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model)
            model.base_model.language_model.past_key_values = past_key_values
            model.base_model.language_model.past_key_values_data = past_key_values_data
            model.base_model.language_model.current_length_data = current_length_data
        #print(len(model.base_model.language_model.past_key_values))'''
        #print(model.base_model.language_model.past_key_values[0][1].data.shape)
        #exit()
        input_len = input_ids.shape[1]
        #reset_tree_mode(model.ea_layer)
        tokenizer = self.get_tokenizer()
        max_steps = max_new_tokens
        model_inputs = model.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        #print('model inputs:')
        #print(model_inputs['input_ids'])
        #print(model_inputs['pixel_values'])
        #exit()
        #print('start forwarding')
        #print(model_inputs)
        outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        #print('outputs')
        #print('loss',outputs.loss)
        #print('logits',outputs.logits)
        #print('past key values',outputs.past_key_values)
        #print('hidden states',outputs.hidden_states)
        #print('attentions',outputs.attentions)
        #print('projector',outputs.projector_features)
        #exit()
        input_len = input_ids.shape[1]-1
        input_embed_len = outputs['past_key_values'][0][0].shape[2]-1
        new_token = 0
        model_inputs["cache_position"] = torch.arange(input_embed_len, device=input_ids.device)
        model_inputs['use_cache']=True
        for idx in range(max_steps):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(input_ids, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            #print(input_id)
            #exit()
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            model_inputs = self._update_model_kwargs_for_generation(
                outputs,
                model_inputs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            model_inputs['input_ids']=input_ids
            model_inputs = model.base_model.prepare_inputs_for_generation(**model_inputs)
            outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                #use_cache = True
            )
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
        return input_ids[:,input_len:]
    @torch.no_grad()
    def eagenerate(
        self,
        input_ids,
        max_new_tokens,
        #return_dict=True,
        #return_dict_in_generate=True,
        log = False,
        accept_threshold=None,
        **kwargs
    ):
        temperature=0.0
        top_p=0.0
        top_k=0.0
        self.tree_mask=None
        self.base_model.language_model.tree_mask=None
        #input_len = input_ids.shape[1]-1
        max_length=2048
        logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        tokenizer = self.get_tokenizer()
        max_steps = max_new_tokens
        model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        reset_tree_mode(self.ea_layer)
        time_0 = time.time()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, prompt_hidden_states, sample_token, past_key_value_data,prompt_embeds,attention_mask = initialize_tree(model_inputs, self, logits_processor)
        input_len = input_ids.shape[1]-1
        max_length = max_length - self.ea_layer.total_tokens - 10
        new_token = 0
        idx = -1
        for idx in range(max_length):
            # with Timer("all"):
            cycle_begin_time = time.time()
            self.base_model.language_model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            logits, hidden_state_new,hidden_embedding_new,past_kv_data_new,outputs= tree_decoding(
                self,
                prompt_embeds,
                draft_tokens,
                attention_mask,
                past_key_value_data,
                tree_position_ids,
                #input_ids,
                retrieve_indices,
                #draft_logit=draft_logit
            )
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor,accept_threshold=accept_threshold
            )
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token,prompt_embeds,past_key_value_data,attention_mask = update_inference_inputs(
                prompt_embeds,
                #prompt_hidden_states,
                input_ids,
                input_len,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_kv_data_new,
                #current_length_data,
                self,
                hidden_state_new,
                #hidden_embedding_new,
                sample_p,
                attention_mask
            )
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        #print('end loop')
        #print('check stop tokens')
        stop_token_ids_index = [
                    i
                    for i, id in enumerate(input_ids[0])
                    if (id == self.tokenizer.eos_token_id or id == self.tokenizer.pad_token_id)
                ]
        if len(stop_token_ids_index) > 0:
                    input_ids = input_ids[:,:stop_token_ids_index[0]]

        self._pending_generation_stats = {
            "backend": "eagle",
            "generated_tokens": max_new_tokens,
            "num_blocks": idx + 1,
            "progressed_tokens": new_token,
        }
        if not log:
            return input_ids[:,input_len+1:]
        else:
            return input_ids, new_token, idx
    def eval_topk(self,input_ids, logits_processor=None,**kwargs):
        #token = torch.tensor(token).to(input_ids.device)
        temperature=0.0
        top_p=0.0
        top_k=0.0
        self.tree_mask=None
        self.base_model.language_model.tree_mask=None
        #input_len = input_ids.shape[1]-1
        max_length=2048
        logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        #self.ea_layer.reset_kv()

        # Initialize the past key and value states
        tokenizer = self.get_tokenizer()
        max_steps = 6
        #model = self
        #print('base model')
        model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        #这里直接用ea_forward那最后一个位置的hidden state
        #print(kwargs)
        kwargs['return_hidden_states']=True
        #print(kwargs)
        #exit()
        action,tokens,hidden = self.predict_action(
               input_ids, logits_processor=None,**kwargs
            )
        token = torch.tensor(tokens).to(input_ids.device)
        input_embeds = hidden[0]
        hidden_states = hidden[1]
        hidden_states = torch.cat([item for item in hidden[1]],dim=0).to(input_ids.device)
        input_embeds = torch.cat([item for item in hidden[0]],dim=0).to(input_ids.device)
        #print(input_embeds.device)
        input_token_embeds = self.ea_layer.embed_tokens(torch.tensor([2]).to(input_ids.device))
        ea_layer_input_embeds = torch.cat((input_embeds,input_token_embeds),dim=0)
        self.ea_layer._eval_top_k(hidden_states,token, ea_layer_input_embeds,self.base_model.language_model.lm_head, 0,logits_processor)
        return action,None,None
