from __future__ import annotations

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


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int],
) -> torch.Tensor:
    """
    这是给在线推理用的函数，用于从目标模型的 hidden states 中提取特征，作为草稿模型的条件输入
    从目标模型的 hidden states 中（包含所有层）提取特征
    按照层索引取出所需层的hidden states
    """
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
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
            self.target_layer_ids = build_target_layer_ids(num_target_layers, config.num_hidden_layers)
        else:
            self.target_layer_ids = configured_target_layer_ids# 按配置指定的层取索引
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.action_dim_embed = nn.Embedding(self.action_dim, config.hidden_size)
        nn.init.normal_(self.action_dim_embed.weight, mean=0.0, std=0.02)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        noise_embedding: torch.Tensor,# diffusion的噪声输入
        target_hidden: torch.Tensor,# 目标模型上下文hidden
        ctx_position_ids: torch.LongTensor,
        noise_position_ids: torch.LongTensor,
        ctx_attention_mask: torch.Tensor | None = None,
        action_position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
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
        return hidden_states
