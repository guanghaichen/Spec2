"""Lightweight residual adapter for OpenVLA-OFT layer early exit.

The OFT backbone predicts a full action chunk with 56 placeholder action
tokens (8 future actions x 7 action dimensions).  This adapter maps the
hidden states available after an early LLaMA layer to the final-layer action
representation expected by OFT's frozen continuous action head.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class LayerExitAdapterConfig:
    hidden_size: int = 4096
    bottleneck_size: int = 512
    num_action_tokens: int = 56
    num_mixer_layers: int = 2
    num_attention_heads: int = 8
    dropout: float = 0.0
    early_exit_layer: int = 16


class ChunkMixerBlock(nn.Module):
    """Mix action-token information in a compact bottleneck space."""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(hidden_states)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        hidden_states = hidden_states + attn_output
        return hidden_states + self.ffn(self.ffn_norm(hidden_states))


class LayerExitResidualAdapter(nn.Module):
    """Predict a final-layer residual from early OFT action-token hidden states.

    The learned residual is token-aware and chunk-aware: position embeddings
    distinguish the 56 action placeholders, while compact self-attention lets
    related future actions exchange information before projecting back to the
    LLaMA hidden dimension.
    """

    def __init__(self, config: LayerExitAdapterConfig) -> None:
        super().__init__()
        if config.bottleneck_size % config.num_attention_heads != 0:
            raise ValueError("bottleneck_size must be divisible by num_attention_heads")
        self.config = config
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.down_proj = nn.Linear(config.hidden_size, config.bottleneck_size, bias=False)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_action_tokens, config.bottleneck_size)
        )
        self.mixer = nn.ModuleList(
            [
                ChunkMixerBlock(
                    config.bottleneck_size,
                    config.num_attention_heads,
                    config.dropout,
                )
                for _ in range(config.num_mixer_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.bottleneck_size)
        self.up_proj = nn.Linear(config.bottleneck_size, config.hidden_size, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        # Identity initialization makes the first optimization steps stable:
        # the adapter starts from the early hidden state and learns a residual.
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, early_hidden_states: torch.Tensor) -> torch.Tensor:
        if early_hidden_states.ndim != 3:
            raise ValueError("Expected [batch, action_tokens, hidden_size] hidden states")
        if early_hidden_states.shape[1] != self.config.num_action_tokens:
            raise ValueError(
                f"Expected {self.config.num_action_tokens} action tokens, got {early_hidden_states.shape[1]}"
            )
        if early_hidden_states.shape[2] != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.config.hidden_size}, got {early_hidden_states.shape[2]}"
            )

        mixed = self.down_proj(self.input_norm(early_hidden_states)) + self.position_embedding
        for block in self.mixer:
            mixed = block(mixed)
        residual = self.up_proj(self.output_norm(mixed))
        return early_hidden_states + self.residual_scale * residual

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "layer_exit_adapter_config.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n"
        )
        torch.save(self.state_dict(), output_dir / "layer_exit_adapter.pt")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "LayerExitResidualAdapter":
        checkpoint_dir = Path(checkpoint_dir)
        config = LayerExitAdapterConfig(
            **json.loads((checkpoint_dir / "layer_exit_adapter_config.json").read_text())
        )
        adapter = cls(config)
        state_dict = torch.load(checkpoint_dir / "layer_exit_adapter.pt", map_location=map_location)
        adapter.load_state_dict(state_dict, strict=True)
        return adapter
