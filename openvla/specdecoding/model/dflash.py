from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

"""
导入Llama中的基础组件，LlamaMLP即前馈层，LlamaRMSNorm即RMS层归一化，LlamaRotaryEmbedding即RoPE位置编码，repeat_kv即GQA (Grouped Query Attention) 中扩展 KV head，rotate_half是RoPE的辅助函数
"""
from .cnets import LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding, repeat_kv, rotate_half


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """
    从 logits 中采样 token
    根据logits采样，返回一个token序列
    temperature=0 时退化为 greedy
    """
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """
    决定草稿模型的每一层从目标模型的 哪一层 取特征作为条件输入
    这是没指定层时的默认策略，即均匀间隔取特征
    输出层索引列表
    """
    if num_draft_layers <= 0:# Draft 层数必须大于0
        raise ValueError("num_draft_layers must be > 0")
    if num_draft_layers == 1:# Draft 层数为1时，从目标模型的中间层取特征
        return [num_target_layers // 2]
    start = 1
    end = max(1, num_target_layers - 3)# 最大只取到倒数第3层
    span = end - start
    return [# 均匀间隔采
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def build_evenly_spaced_target_layer_ids(
    num_target_layers: int,
    num_feature_layers: int = 5,
    first_layer_id: int = 1,
) -> list[int]:
    """在首个指定层和最终层之间做算法化的近似等间隔取样。

    OpenVLA/Llama-2 有 32 个 decoder layers，0-based 的最终层是 31。
    从 1 到 31 取 5 点时跨度 30 不能被 4 整除，因此不存在整数层上的绝对等间隔解；
    使用四舍五入后的 linspace 得到 [1, 9, 16, 24, 31]，相邻间隔只相差 1。
    layer_id=31 经 hidden_states[layer_id + 1] 正好对应最终归一化 hidden。
    """
    if num_target_layers <= 0:
        raise ValueError("num_target_layers must be > 0")
    if num_feature_layers < 2:
        raise ValueError("num_feature_layers must be >= 2 so both first and final layers are present")
    final_layer_id = num_target_layers - 1
    if not 0 <= first_layer_id < final_layer_id:
        raise ValueError(
            f"first_layer_id must be in [0, {final_layer_id - 1}], got {first_layer_id}"
        )
    if num_feature_layers > final_layer_id - first_layer_id + 1:
        raise ValueError("num_feature_layers exceeds the number of available integer layer ids")

    span = final_layer_id - first_layer_id
    layer_ids = [
        int(math.floor(first_layer_id + i * span / (num_feature_layers - 1) + 0.5))
        for i in range(num_feature_layers)
    ]
    if layer_ids[0] != first_layer_id or layer_ids[-1] != final_layer_id:
        raise AssertionError(f"Even layer selection lost an endpoint: {layer_ids}")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"Even layer selection produced duplicate ids: {layer_ids}")
    return layer_ids


SELECTED_HIDDEN_VARIANTS = ("target_layers", "replace_22_with_final")


def normalize_selected_hidden_variant(variant: str | None) -> str:
    variant = variant or "target_layers"
    if variant not in SELECTED_HIDDEN_VARIANTS:
        raise ValueError(
            f"Unsupported selected hidden variant: {variant!r}. "
            f"Expected one of {SELECTED_HIDDEN_VARIANTS}."
        )
    return variant


def _drop_22_and_append_final(
    selected_states: list[torch.Tensor],
    layer_ids: list[int],
    final_state: torch.Tensor,
) -> list[torch.Tensor]:
    if 22 not in layer_ids:
        raise ValueError(
            "selected_hidden_variant='replace_22_with_final' requires source layer_ids to contain layer 22. "
            f"Got layer_ids={layer_ids}."
        )
    kept_states = [state for state, layer_id in zip(selected_states, layer_ids) if layer_id != 22]
    if len(kept_states) != len(selected_states) - 1:
        raise ValueError(f"Expected exactly one layer 22 in layer_ids={layer_ids}.")
    kept_states.append(final_state)
    return kept_states


def apply_selected_hidden_variant(
    selected_hidden: torch.Tensor,
    final_hidden: torch.Tensor | None,
    layer_ids: list[int],
    variant: str | None = None,
    file_path: str | None = None,
) -> torch.Tensor:
    """Optionally replace source layer 22 with final-layer hidden while preserving feature width.

    The offline exporter stores selected hidden as concat([1, 8, 15, 22, 29]) and stores final
    hidden separately as prompt_last/action_last. For replace_22_with_final we rebuild the concat
    as concat([1, 8, 15, 29, final]) without regenerating data.
    """
    variant = normalize_selected_hidden_variant(variant)
    if variant == "target_layers":
        return selected_hidden
    if final_hidden is None:
        location = f" in {file_path}" if file_path else ""
        raise ValueError(f"Missing final hidden{location} for selected_hidden_variant={variant!r}.")
    if selected_hidden.shape[:-1] != final_hidden.shape[:-1]:
        location = f" in {file_path}" if file_path else ""
        raise ValueError(
            f"selected/final hidden shape mismatch{location}: "
            f"selected={tuple(selected_hidden.shape)}, final={tuple(final_hidden.shape)}."
        )
    hidden_size = final_hidden.shape[-1]
    expected_width = len(layer_ids) * hidden_size
    if selected_hidden.shape[-1] != expected_width:
        location = f" in {file_path}" if file_path else ""
        raise ValueError(
            f"selected hidden width mismatch{location}: got {selected_hidden.shape[-1]}, "
            f"expected len(layer_ids) * hidden={expected_width} for layer_ids={layer_ids}."
        )
    chunks = list(selected_hidden.split(hidden_size, dim=-1))
    final_hidden = final_hidden.to(device=selected_hidden.device, dtype=selected_hidden.dtype)
    return torch.cat(_drop_22_and_append_final(chunks, layer_ids, final_hidden), dim=-1)


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int],
    selected_hidden_variant: str | None = None,
) -> torch.Tensor:
    """
    这是给在线推理用的函数，用于从目标模型的 hidden states 中提取特征，作为草稿模型的条件输入
    从目标模型的 hidden states 中（包含所有层）提取特征
    按照层索引取出所需层的hidden states
    """
    selected_hidden_variant = normalize_selected_hidden_variant(selected_hidden_variant)
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    if selected_hidden_variant == "replace_22_with_final":
        selected_states = _drop_22_and_append_final(selected_states, layer_ids, hidden_states[-1])
    return torch.cat(selected_states, dim=-1)# 提取指定层的输出，拼接成一个大的特征向量。这个拼接特征会作为草稿模型的"上下文"条件输入


def _apply_rotary_pos_emb_dflash(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    ctx_position_ids: torch.LongTensor,# 上下文（目标模型）的位置编码
    noise_position_ids: torch.LongTensor,# 草稿 token 的位置编码
) -> tuple[torch.Tensor, torch.Tensor]:

    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)

    q_pos = noise_position_ids# query 只看草稿自己的位置
    k_pos = torch.cat([ctx_position_ids, noise_position_ids], dim=-1)# key 看上下文+草稿的位置，用 [ctx_position_ids, noise_position_ids] 拼接

    cos_q = cos[q_pos].unsqueeze(1)
    sin_q = sin[q_pos].unsqueeze(1)
    cos_k = cos[k_pos].unsqueeze(1)
    sin_k = sin[k_pos].unsqueeze(1)

    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    # 这意味着草稿 token 在做 attention 时，能正确区分"这是上下文的位置"和"这是草稿的位置"
    return q_embed, k_embed

# === Draft模型的注意力层 ===
class LlamaDFlashAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,# 草稿模型的隐藏状态
        target_hidden: torch.Tensor,# 目标模型提取的上下文特征
        cos: torch.Tensor,# RoPE 预计算值
        sin: torch.Tensor,
        ctx_position_ids: torch.LongTensor,# 上下文的位置 IDs
        noise_position_ids: torch.LongTensor,# 草稿 token 的位置 IDs
        ctx_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]
        # Step 1: 对草稿 token 做 Q 投影
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Step 2: 对上下文特征做 K/V 投影（只做一次，复用）
        k_ctx = self.k_proj(target_hidden)
        v_ctx = self.v_proj(target_hidden)
        # Step 3: 对草稿 token 做 K/V 投影
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)
        # Step 4: 拼接上下文 KV + 草稿 KV
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # Step 5: 应用 RoPE（使用上面讲的分段位置编码）
        q, k = _apply_rotary_pos_emb_dflash(q, k, cos, sin, ctx_position_ids, noise_position_ids)
        # Step 6: 重复 KV head（GQA）
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # Step 7: 计算注意力输出；完整 prefix batch padding 位置需要 mask 掉。
        attn_mask = None
        if ctx_attention_mask is not None:
            noise_attention_mask = torch.ones(
                bsz,
                q_len,
                device=hidden_states.device,
                dtype=torch.bool,
            )
            key_attention_mask = torch.cat([ctx_attention_mask.bool(), noise_attention_mask], dim=1)
            attn_mask = key_attention_mask[:, None, None, :]
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,# 必须False（块内全可见），块内非因果并行生成，即无内置 mask，block 内所有 token 互相可见；如果是True，则上三角 mask（token i 只能看到 0~i）
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output

# === Decoder 层 ===
class LlamaDFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LlamaDFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx_position_ids: torch.LongTensor,
        noise_position_ids: torch.LongTensor,
        ctx_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """与Llama一致：Norm → Attention → Residual → Norm → MLP → Residual"""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            cos=cos,
            sin=sin,
            ctx_position_ids=ctx_position_ids,
            noise_position_ids=noise_position_ids,
            ctx_attention_mask=ctx_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config# 外部配置文件
        self.layers = nn.ModuleList([LlamaDFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = LlamaRotaryEmbedding(# RoPE
            head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 2048),
            base=getattr(config, "rope_theta", 10000),
        )
        self.block_size = getattr(config, "dflash_block_size", None)
        self.action_dim = getattr(config, "dflash_action_dim", self.block_size)
        if self.action_dim is None or self.action_dim <= 0:
            raise ValueError("Missing or invalid DFlash config field `dflash_action_dim`.")
        num_target_layers = getattr(config, "num_target_layers", None)
        configured_target_layer_ids = getattr(config, "dflash_target_layer_ids", None)
        if self.block_size is None:
            raise ValueError("Missing DFlash config field `dflash_block_size`.")
        # 决定取目标模型的哪些层
        if configured_target_layer_ids is None:
            if num_target_layers is None:
                raise ValueError("Missing DFlash config field `num_target_layers`.")
            self.target_layer_ids = build_evenly_spaced_target_layer_ids(
                num_target_layers,
                int(getattr(config, "dflash_num_target_feature_layers", 5)),
            )
        else:
            self.target_layer_ids = configured_target_layer_ids# 按配置指定的层取索引
        self.selected_hidden_variant = normalize_selected_hidden_variant(
            getattr(config, "dflash_selected_hidden_variant", "target_layers")
        )
        if self.selected_hidden_variant == "replace_22_with_final" and 22 not in self.target_layer_ids:
            raise ValueError(
                "DFlash replace_22_with_final requires dflash_target_layer_ids to contain layer 22. "
                f"Got {self.target_layer_ids}."
            )
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.action_dim_embed = nn.Embedding(self.action_dim, config.hidden_size)
        nn.init.normal_(self.action_dim_embed.weight, mean=0.0, std=0.02)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.causal_residual_type = getattr(config, "dflash_causal_residual_type", "none")
        self.causal_residual_rank = int(getattr(config, "dflash_causal_residual_rank", 256))
        self.causal_residual_scale = float(getattr(config, "dflash_causal_residual_scale", 1.0))
        if self.causal_residual_type not in ("none", "hidden"):
            raise ValueError(
                f"Unsupported dflash_causal_residual_type={self.causal_residual_type!r}; "
                "expected 'none' or 'hidden'."
            )
        if self.causal_residual_type == "hidden":
            vocab_size = getattr(config, "vocab_size", None)
            if vocab_size is None or vocab_size <= 0:
                raise ValueError("DFlash causal residual head requires config.vocab_size > 0.")
            if self.causal_residual_rank <= 0:
                raise ValueError("dflash_causal_residual_rank must be > 0.")
            self.causal_residual_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.causal_residual_token = nn.Embedding(vocab_size, self.causal_residual_rank)
            self.causal_residual_hidden = nn.Linear(config.hidden_size, self.causal_residual_rank, bias=False)
            self.causal_residual_out = nn.Linear(self.causal_residual_rank, config.hidden_size, bias=False)
            nn.init.normal_(self.causal_residual_token.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.causal_residual_hidden.weight, mean=0.0, std=0.02)
            # Near-zero rather than exact zero keeps DDP gradient hooks active from step 1.
            nn.init.normal_(self.causal_residual_out.weight, mean=0.0, std=1e-4)
        else:
            self.causal_residual_norm = None
            self.causal_residual_token = None
            self.causal_residual_hidden = None
            self.causal_residual_out = None

        self.logit_markov_type = getattr(config, "dflash_logit_markov_type", "none")
        self.logit_markov_rank = int(getattr(config, "dflash_logit_markov_rank", 256))
        self.logit_markov_scale = float(getattr(config, "dflash_logit_markov_scale", 1.0))
        if self.logit_markov_type not in ("none", "bias"):
            raise ValueError(
                f"Unsupported dflash_logit_markov_type={self.logit_markov_type!r}; "
                "expected 'none' or 'bias'."
            )
        if self.logit_markov_type == "bias":
            vocab_size = getattr(config, "vocab_size", None)
            if vocab_size is None or vocab_size <= 0:
                raise ValueError("DFlash logit Markov head requires config.vocab_size > 0.")
            if self.logit_markov_rank <= 0:
                raise ValueError("dflash_logit_markov_rank must be > 0.")
            self.logit_markov_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.logit_markov_token = nn.Embedding(vocab_size, self.logit_markov_rank)
            self.logit_markov_hidden = nn.Linear(config.hidden_size, self.logit_markov_rank, bias=False)
            self.logit_markov_out = nn.Linear(self.logit_markov_rank, vocab_size, bias=False)
            nn.init.normal_(self.logit_markov_token.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.logit_markov_hidden.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.logit_markov_out.weight, mean=0.0, std=1e-4)
        else:
            self.logit_markov_norm = None
            self.logit_markov_token = None
            self.logit_markov_hidden = None
            self.logit_markov_out = None

        # 动作专用顺序残差头：先用 frozen lm_head 的 256 个动作行得到基准 logits，
        # 再递推一个很小的状态并只输出 residual bias；它不取代 lm_head。
        self.action_head_type = getattr(config, "dflash_action_head_type", "none")
        self.action_head_rank = int(getattr(config, "dflash_action_head_rank", 256))
        self.action_token_start = int(getattr(config, "dflash_action_token_start", 0))
        self.action_vocab_size = int(getattr(config, "dflash_action_vocab_size", 256))
        self.action_token_end = self.action_token_start + self.action_vocab_size
        self.action_confidence_enabled = bool(
            getattr(config, "dflash_action_confidence_enabled", False)
        )
        if self.action_head_type not in ("none", "slot_rnn"):
            raise ValueError(
                f"Unsupported dflash_action_head_type={self.action_head_type!r}; "
                "expected 'none' or 'slot_rnn'."
            )
        if self.action_head_type == "slot_rnn":
            vocab_size = int(getattr(config, "vocab_size", 0))
            if self.action_head_rank <= 0:
                raise ValueError("dflash_action_head_rank must be > 0.")
            if self.action_vocab_size <= 0:
                raise ValueError("dflash_action_vocab_size must be > 0.")
            if self.action_token_start < 0 or self.action_token_end > vocab_size:
                raise ValueError(
                    "DFlash action-token range must lie inside the target vocabulary: "
                    f"[{self.action_token_start}, {self.action_token_end}) vs vocab_size={vocab_size}."
                )
            rank = self.action_head_rank
            self.action_head_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.action_head_hidden = nn.Linear(config.hidden_size, rank, bias=False)
            self.action_head_token = nn.Embedding(self.action_vocab_size, rank)
            self.action_head_slot = nn.Embedding(self.action_dim, rank)
            self.action_head_state = nn.Linear(4 * rank, 2 * rank)
            self.action_head_out = nn.Linear(rank, self.action_vocab_size, bias=False)
            self.action_confidence_head = (
                nn.Linear(2 * rank, 1) if self.action_confidence_enabled else None
            )
            nn.init.normal_(self.action_head_hidden.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.action_head_token.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.action_head_slot.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.action_head_state.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.action_head_state.bias)
            # 初始行为严格等于 frozen lm_head；训练只学习必要的前缀条件残差。
            nn.init.zeros_(self.action_head_out.weight)
            if self.action_confidence_head is not None:
                nn.init.normal_(self.action_confidence_head.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.action_confidence_head.bias)
        else:
            self.action_head_norm = None
            self.action_head_hidden = None
            self.action_head_token = None
            self.action_head_slot = None
            self.action_head_state = None
            self.action_head_out = None
            self.action_confidence_head = None

    @property
    def causal_residual_enabled(self) -> bool:
        return self.causal_residual_type == "hidden"

    @property
    def logit_markov_enabled(self) -> bool:
        return self.logit_markov_type == "bias"

    @property
    def action_sequential_enabled(self) -> bool:
        return self.action_head_type == "slot_rnn"

    def action_logits_from_full(self, logits: torch.Tensor) -> torch.Tensor:
        """只取 OpenVLA 动作 token 对应的连续词表区间。"""
        if logits.shape[-1] == self.action_vocab_size:
            return logits
        if logits.shape[-1] < self.action_token_end:
            raise ValueError(
                f"logits vocab={logits.shape[-1]} is smaller than action_token_end={self.action_token_end}."
            )
        return logits[..., self.action_token_start : self.action_token_end]

    def project_action_logits(self, hidden_states: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        """只使用 frozen lm_head 的动作行，避免计算 3 万多个无关语言 token。"""
        action_weight = lm_head.weight[self.action_token_start : self.action_token_end]
        action_bias = None
        if getattr(lm_head, "bias", None) is not None:
            action_bias = lm_head.bias[self.action_token_start : self.action_token_end]
        return F.linear(hidden_states, action_weight, action_bias)

    def action_token_ids_to_local(self, token_ids: torch.LongTensor) -> torch.LongTensor:
        """把全词表 token id 映射到动作子词表 [0, action_vocab_size)。"""
        local_ids = token_ids.long() - self.action_token_start
        # 合法训练/推理路径中的 token 都来自动作区间；clamp 仅用于避免坏输入引发越界崩溃。
        return local_ids.clamp(min=0, max=self.action_vocab_size - 1)

    def action_local_ids_to_token(self, local_ids: torch.LongTensor) -> torch.LongTensor:
        return local_ids.long() + self.action_token_start

    def _action_head_step(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_token_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """执行一个低秩 slot-aware 前缀状态更新，返回新状态、动作 bias 和置信度。"""
        if not self.action_sequential_enabled:
            raise RuntimeError("_action_head_step requires dflash_action_head_type='slot_rnn'.")
        hidden_feat = self.action_head_hidden(self.action_head_norm(hidden_state))
        token_feat = self.action_head_token(self.action_token_ids_to_local(prev_token_ids))
        slot_feat = self.action_head_slot(action_position_ids.long())
        return self._action_head_step_from_features(state, hidden_feat, token_feat, slot_feat)

    def _action_head_step_from_features(
        self,
        state: torch.Tensor,
        hidden_feat: torch.Tensor,
        token_feat: torch.Tensor,
        slot_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        gate_raw, candidate_raw = self.action_head_state(
            torch.cat([state, hidden_feat, token_feat, slot_feat], dim=-1)
        ).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        action_bias = self.action_head_out(torch.tanh(new_state))
        confidence_logits = None
        if self.action_confidence_head is not None:
            confidence_logits = self.action_confidence_head(
                torch.cat([new_state, hidden_feat], dim=-1)
            ).squeeze(-1)
        return new_state, action_bias, confidence_logits

    def apply_action_sequential_head(
        self,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """训练时用 teacher 前缀递推，在 frozen lm_head 动作 logits 上叠加残差。"""
        base_action_logits = self.action_logits_from_full(base_logits)
        if not self.action_sequential_enabled:
            return base_action_logits, None
        if hidden_states.shape[:2] != prev_token_ids.shape:
            raise ValueError("hidden_states and prev_token_ids must share [B, T].")
        if hidden_states.shape[:2] != action_position_ids.shape:
            raise ValueError("hidden_states and action_position_ids must share [B, T].")

        state = hidden_states.new_zeros(hidden_states.shape[0], self.action_head_rank)
        hidden_features = self.action_head_hidden(self.action_head_norm(hidden_states))
        token_features = self.action_head_token(self.action_token_ids_to_local(prev_token_ids))
        slot_features = self.action_head_slot(action_position_ids.long())
        corrected_logits = []
        confidence_logits = []
        for slot_idx in range(hidden_states.shape[1]):
            state, bias, confidence = self._action_head_step_from_features(
                state,
                hidden_features[:, slot_idx, :],
                token_features[:, slot_idx, :],
                slot_features[:, slot_idx, :],
            )
            corrected_logits.append(base_action_logits[:, slot_idx, :] + bias)
            if confidence is not None:
                confidence_logits.append(confidence)
        stacked_confidence = (
            torch.stack(confidence_logits, dim=1) if confidence_logits else None
        )
        return torch.stack(corrected_logits, dim=1), stacked_confidence

    def sample_action_block(
        self,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        first_prev_token_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        confidence_min_tokens: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """推理时使用刚采样出的动作前缀递推，并始终保留 frozen lm_head 基准 logits。"""
        if not self.action_sequential_enabled:
            raise RuntimeError("sample_action_block requires dflash_action_head_type='slot_rnn'.")
        if first_prev_token_ids.ndim == 2:
            first_prev_token_ids = first_prev_token_ids[:, 0]
        if first_prev_token_ids.ndim != 1:
            raise ValueError("first_prev_token_ids must be [B] or [B, 1].")
        if confidence_threshold < 0 or confidence_threshold >= 1:
            raise ValueError("confidence_threshold must be in [0, 1).")
        confidence_min_tokens = max(1, int(confidence_min_tokens))

        base_action_logits = self.action_logits_from_full(base_logits)
        state = hidden_states.new_zeros(hidden_states.shape[0], self.action_head_rank)
        hidden_features = self.action_head_hidden(self.action_head_norm(hidden_states))
        slot_features = self.action_head_slot(action_position_ids.long())
        prev_token_ids = first_prev_token_ids.long()
        sampled_tokens = []
        corrected_logits = []
        confidence_probs = []
        for slot_idx in range(hidden_states.shape[1]):
            token_feat = self.action_head_token(self.action_token_ids_to_local(prev_token_ids))
            state, bias, confidence = self._action_head_step_from_features(
                state,
                hidden_features[:, slot_idx, :],
                token_feat,
                slot_features[:, slot_idx, :],
            )
            if confidence is not None:
                confidence_prob = confidence.sigmoid()
                confidence_probs.append(confidence_prob)
                if (
                    confidence_threshold > 0
                    and len(sampled_tokens) >= confidence_min_tokens
                    and bool((confidence_prob < confidence_threshold).all().item())
                ):
                    break
            step_logits = base_action_logits[:, slot_idx, :] + bias
            local_token = sample(step_logits.unsqueeze(1), temperature=temperature)[:, 0]
            token_ids = self.action_local_ids_to_token(local_token)
            sampled_tokens.append(token_ids)
            corrected_logits.append(step_logits)
            prev_token_ids = token_ids

        if not sampled_tokens:
            raise RuntimeError("DFlash action head produced an empty proposal.")
        return (
            torch.stack(sampled_tokens, dim=1),
            torch.stack(corrected_logits, dim=1),
            torch.stack(confidence_probs[: len(sampled_tokens)], dim=1)
            if confidence_probs
            else None,
        )

    def apply_causal_residual(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.LongTensor,
        start_index: int = 1,
    ) -> torch.Tensor:
        """Refine hidden states using the previous token available for each slot.

        start_index=1 leaves slot0 unchanged. For the first-step-boost branch we set
        start_index=0, because the anchor token is available for slot0 as a valid
        previous-token condition and the one-step path also needs direct refinement.
        """
        if not self.causal_residual_enabled:
            return hidden_states
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}.")
        if prev_token_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"prev_token_ids shape {tuple(prev_token_ids.shape)} must match hidden prefix "
                f"{tuple(hidden_states.shape[:2])}."
            )
        seq_len = hidden_states.shape[1]
        if start_index >= seq_len:
            return hidden_states
        if start_index < 0:
            raise ValueError("start_index must be >= 0.")

        tail_hidden = hidden_states[:, start_index:, :]
        tail_prev_tokens = prev_token_ids[:, start_index:]
        residual_input = self.causal_residual_norm(tail_hidden)
        hidden_feat = self.causal_residual_hidden(residual_input)
        token_feat = self.causal_residual_token(tail_prev_tokens).to(dtype=hidden_feat.dtype)
        residual = self.causal_residual_out(F.silu(hidden_feat + token_feat))
        refined_tail = tail_hidden + residual.to(dtype=tail_hidden.dtype) * self.causal_residual_scale
        if start_index == 0:
            return refined_tail
        return torch.cat([hidden_states[:, :start_index, :], refined_tail], dim=1)

    def apply_logit_markov_bias(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.LongTensor,
        start_index: int = 1,
    ) -> torch.Tensor:
        """Add a lightweight previous-token-conditioned bias on logits.

        The hidden block is still produced in one DFlash forward. This head corrects
        the decision boundary with the previous token that is available in training,
        or the anchor / just-sampled draft token during inference.
        """
        if not self.logit_markov_enabled:
            return logits
        if logits.ndim != 3:
            raise ValueError(f"logits must be [B, T, V], got {tuple(logits.shape)}.")
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}.")
        if logits.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"logits prefix shape {tuple(logits.shape[:2])} must match hidden prefix "
                f"{tuple(hidden_states.shape[:2])}."
            )
        if prev_token_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"prev_token_ids shape {tuple(prev_token_ids.shape)} must match hidden prefix "
                f"{tuple(hidden_states.shape[:2])}."
            )
        seq_len = hidden_states.shape[1]
        if start_index >= seq_len:
            return logits
        if start_index < 0:
            raise ValueError("start_index must be >= 0.")

        tail_hidden = hidden_states[:, start_index:, :]
        tail_prev_tokens = prev_token_ids[:, start_index:]
        hidden_feat = self.logit_markov_hidden(self.logit_markov_norm(tail_hidden))
        token_feat = self.logit_markov_token(tail_prev_tokens).to(dtype=hidden_feat.dtype)
        bias = self.logit_markov_out(F.silu(hidden_feat + token_feat))
        bias = bias.to(dtype=logits.dtype) * self.logit_markov_scale
        if start_index == 0:
            return logits + bias
        return torch.cat([logits[:, :start_index, :], logits[:, start_index:, :] + bias], dim=1)

    def sample_with_causal_residual(
        self,
        hidden_states: torch.Tensor,
        first_prev_token_ids: torch.LongTensor,
        lm_head: nn.Module,
        temperature: float = 0.0,
        start_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequentially sample a block after optional residual refinement.

        The expensive DFlash transformer has already produced all hidden states in parallel.
        This loop is only the light residual head + frozen lm_head, using each just-sampled
        draft token as the next slot's previous-token condition.
        """
        if first_prev_token_ids.ndim == 2:
            first_prev_token_ids = first_prev_token_ids[:, 0]
        if first_prev_token_ids.ndim != 1:
            raise ValueError("first_prev_token_ids must be [B] or [B, 1].")

        sampled_tokens = []
        refined_logits = []
        prev_token = first_prev_token_ids
        for slot_idx in range(hidden_states.shape[1]):
            step_hidden = hidden_states[:, slot_idx : slot_idx + 1, :]
            if self.causal_residual_enabled and slot_idx >= start_index:
                step_hidden = self.apply_causal_residual(
                    step_hidden,
                    prev_token.view(-1, 1),
                    start_index=0,
                )
            step_logits = lm_head(step_hidden)
            if self.logit_markov_enabled and slot_idx >= start_index:
                step_logits = self.apply_logit_markov_bias(
                    step_logits,
                    step_hidden,
                    prev_token.view(-1, 1),
                    start_index=0,
                )
            step_token = sample(step_logits, temperature=temperature)
            refined_logits.append(step_logits)
            sampled_tokens.append(step_token)
            prev_token = step_token[:, 0]
        return torch.cat(sampled_tokens, dim=1), torch.cat(refined_logits, dim=1)

    def forward(
        self,
        noise_embedding: torch.Tensor,# diffusion的噪声输入
        target_hidden: torch.Tensor,# 目标模型上下文hidden
        ctx_position_ids: torch.LongTensor,
        noise_position_ids: torch.LongTensor,
        ctx_attention_mask: torch.Tensor | None = None,
        action_position_ids: torch.LongTensor | None = None,
        prev_token_ids: torch.LongTensor | None = None,
        causal_residual_start_index: int = 1,
        return_base_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden_states = noise_embedding
        if action_position_ids is not None:
            if int(action_position_ids.max().item()) >= self.action_dim:
                raise ValueError(
                    f"action_position_ids max={int(action_position_ids.max().item())} exceeds "
                    f"dflash_action_dim={self.action_dim}."
                )
            action_dim_embedding = self.action_dim_embed(action_position_ids.to(noise_embedding.device))
            hidden_states = hidden_states + action_dim_embedding.to(dtype=hidden_states.dtype)
        # Step 1: 将上下文特征压缩到 hidden_size
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        # Step 2: 计算 RoPE
        max_position = torch.cat([ctx_position_ids, noise_position_ids], dim=-1).max().item() + 1
        dummy = hidden_states.new_zeros(
            (
                hidden_states.shape[0],
                self.config.num_attention_heads,
                max_position,
                hidden_states.shape[-1] // self.config.num_attention_heads,
            )
        )
        cos, sin = self.rotary_emb(dummy, seq_len=max_position)
        # Step 3: 逐层 forward
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                cos=cos,
                sin=sin,
                ctx_position_ids=ctx_position_ids,
                noise_position_ids=noise_position_ids,
                ctx_attention_mask=ctx_attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        if prev_token_ids is not None and self.causal_residual_enabled:
            refined_hidden_states = self.apply_causal_residual(
                hidden_states,
                prev_token_ids=prev_token_ids,
                start_index=causal_residual_start_index,
            )
        else:
            refined_hidden_states = hidden_states
        if return_base_hidden:
            return hidden_states, refined_hidden_states
        return refined_hidden_states
