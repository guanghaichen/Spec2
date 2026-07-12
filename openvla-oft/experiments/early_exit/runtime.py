"""Runtime helpers that stop OpenVLA-OFT after a chosen LLaMA layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class _LayerExitSignal(Exception):
    """Private control-flow exception raised by a decoder-layer forward hook."""


@dataclass
class EarlyExitResult:
    hidden_states: torch.Tensor
    executed_layers: int


def _get_decoder_layers(language_model: torch.nn.Module) -> torch.nn.ModuleList:
    model = getattr(language_model, "model", None)
    layers = getattr(model, "layers", None)
    if layers is None:
        raise AttributeError(
            "The OFT language model must expose decoder layers as language_model.model.layers"
        )
    return layers


def run_language_model_until_layer(
    language_model: torch.nn.Module,
    *,
    early_exit_layer: int,
    **forward_kwargs: Any,
) -> EarlyExitResult:
    """Execute exactly the first ``early_exit_layer`` decoder blocks.

    A hook interrupts the ordinary Hugging Face forward directly after the
    selected decoder layer. This retains OFT's original embedding, RoPE,
    attention-mask and bidirectional action-token semantics while avoiding all
    later decoder blocks and the unused vocabulary projection.
    """

    layers = _get_decoder_layers(language_model)
    if not 1 <= early_exit_layer <= len(layers):
        raise ValueError(f"early_exit_layer must be in [1, {len(layers)}], got {early_exit_layer}")

    captured: dict[str, torch.Tensor] = {}

    def capture_and_stop(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden_states = output[0] if isinstance(output, tuple) else output
        captured["hidden_states"] = hidden_states
        raise _LayerExitSignal

    handle = layers[early_exit_layer - 1].register_forward_hook(capture_and_stop)
    try:
        language_model(**forward_kwargs)
    except _LayerExitSignal:
        if "hidden_states" not in captured:
            raise RuntimeError("Early-exit hook fired without a decoder hidden state")
    finally:
        handle.remove()

    return EarlyExitResult(hidden_states=captured["hidden_states"], executed_layers=early_exit_layer)


def slice_action_hidden_states(
    hidden_states: torch.Tensor,
    *,
    action_start: int,
    num_action_tokens: int,
) -> torch.Tensor:
    """Extract the OFT action-placeholder region from a sequence hidden state."""

    action_end = action_start + num_action_tokens
    if hidden_states.shape[1] < action_end:
        raise ValueError(
            f"Action range [{action_start}, {action_end}) exceeds sequence length {hidden_states.shape[1]}"
        )
    return hidden_states[:, action_start:action_end, :]
