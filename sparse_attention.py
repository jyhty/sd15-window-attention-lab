"""Windowed SDPA for the single SD 1.5 self-attention layer in Appendix E."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers.models.attention_processor import Attention
    from torch import Tensor


TARGET = "down_blocks.0.attentions.0.transformer_blocks.0.attn1"
RESOLUTIONS = (512, 768, 1024)
LATENT_SCALE = 8
WINDOW = 16
HEADS = 8
HEAD_DIM = 40
CHANNELS = HEADS * HEAD_DIM
ERROR_LIMIT = 0.005


def validate_layer(attn: Attention) -> None:
    """Check the layer contract used by both the sparse and reference paths."""
    if (
        attn.heads != HEADS
        or attn.inner_dim != CHANNELS
        or attn.is_cross_attention
        or not math.isclose(attn.scale, HEAD_DIM**-0.5)
        or any(x is not None for x in (attn.spatial_norm, attn.group_norm, attn.norm_q, attn.norm_k))
        or attn.residual_connection
        or attn.rescale_output_factor != 1.0
        or attn.to_out[1].p != 0.0
        or any(
            layer.weight.shape != (CHANNELS, CHANNELS) or (layer.bias is not None) != has_bias
            for layer, has_bias in (
                (attn.to_q, False), (attn.to_k, False), (attn.to_v, False), (attn.to_out[0], True)
            )
        )
    ):
        raise ValueError("The attention layer does not match the supported SD 1.5 target.")


class WindowAttentionProcessor:
    """Reuse the trained projections and apply SDPA independently in each 2D window."""

    def __init__(self, height: int, width: int):
        from torch.nn.functional import scaled_dot_product_attention

        if height <= 0 or width <= 0 or height % WINDOW or width % WINDOW:
            raise ValueError(f"Both feature dimensions must be positive multiples of {WINDOW}.")
        self.height = height
        self.width = width
        self.window_rows = height // WINDOW
        self.window_cols = width // WINDOW
        self.sdpa = scaled_dot_product_attention

    def pack(self, tensor: Tensor) -> Tensor:
        """[batch, tokens, channels] to [batch * windows, heads, 256, head_dim]."""
        batch = tensor.shape[0]
        return (
            tensor.reshape(batch, self.window_rows, WINDOW, self.window_cols, WINDOW, HEADS, HEAD_DIM)
            .permute(0, 1, 3, 5, 2, 4, 6)
            .reshape(batch * self.window_rows * self.window_cols, HEADS, WINDOW**2, HEAD_DIM)
        )

    def merge(self, tensor: Tensor, batch: int) -> Tensor:
        return (
            tensor.reshape(batch, self.window_rows, self.window_cols, HEADS, WINDOW, WINDOW, HEAD_DIM)
            .permute(0, 1, 4, 2, 5, 3, 6)
            .reshape(batch, self.height * self.width, CHANNELS)
        )

    def attend(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        result = self.sdpa(
            self.pack(query), self.pack(key), self.pack(value),
            attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        return self.merge(result, query.shape[0])

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("This processor expects unmasked self-attention.")
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (self.height * self.width, CHANNELS):
            raise ValueError(f"Unexpected attention input shape: {tuple(hidden_states.shape)}.")
        local = self.attend(
            attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        )
        return attn.to_out[1](attn.to_out[0](local)) / attn.rescale_output_factor


def reference_attention(attn: Attention, hidden: Tensor, height: int, width: int) -> Tensor:
    """Independent FP32 reference using spatial slices and explicit softmax."""
    import torch

    batch = hidden.shape[0]
    query = attn.to_q(hidden).reshape(batch, height, width, HEADS, HEAD_DIM)
    key = attn.to_k(hidden).reshape_as(query)
    value = attn.to_v(hidden).reshape_as(query)
    merged = torch.empty_like(query)
    for row in range(0, height, WINDOW):
        for col in range(0, width, WINDOW):
            region = (slice(None), slice(row, row + WINDOW), slice(col, col + WINDOW))
            q, k, v = (
                x[region].reshape(batch, WINDOW**2, HEADS, HEAD_DIM).transpose(1, 2)
                for x in (query, key, value)
            )
            weights = torch.softmax((q @ k.transpose(-1, -2)) * attn.scale, dim=-1)
            merged[region] = (weights @ v).transpose(1, 2).reshape(
                batch, WINDOW, WINDOW, HEADS, HEAD_DIM
            )
    ordered = merged.reshape(batch, height * width, CHANNELS)
    return attn.to_out[1](attn.to_out[0](ordered)) / attn.rescale_output_factor


def check_window_layout() -> None:
    """Check spatial order and isolation with a minimal two-by-two window grid."""
    import torch

    side = WINDOW * 2
    processor = WindowAttentionProcessor(side, side)
    markers = torch.arange(2 * side * side * CHANNELS).reshape(2, side * side, CHANNELS)
    grid = markers.reshape(2, side, side, CHANNELS)
    expected = torch.stack([
        grid[batch, row:row + WINDOW, col:col + WINDOW].reshape(WINDOW**2, HEADS, HEAD_DIM)
        .transpose(0, 1)
        for batch in range(2)
        for row in (0, WINDOW)
        for col in (0, WINDOW)
    ])
    packed = processor.pack(markers)
    if not torch.equal(packed, expected) or not torch.equal(processor.merge(packed, 2), markers):
        raise RuntimeError("Window packing or spatial restoration is incorrect.")

    zeros = torch.zeros(markers.shape, dtype=torch.float32)
    values = zeros.clone()
    values[0, WINDOW, 0] = WINDOW**2  # Token (0, 16), top-right window of batch 0.
    expected_output = zeros.clone().reshape(2, side, side, CHANNELS)
    expected_output[0, :WINDOW, WINDOW:, 0] = 1
    output = processor.attend(zeros, zeros, values)
    if not torch.equal(output, expected_output.reshape_as(output)):
        raise RuntimeError("Attention did not stay inside its window and batch.")
