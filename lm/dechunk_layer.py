from typing import cast

from einops import rearrange, repeat
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from torch import (
  Tensor,
  arange,
  argsort,
  bfloat16,
  clamp,
  cumsum,
  float32,
  gather,
  log,
  ones,
  ones_like,
  zeros,
)
from torch.nn import Module

from lm.dechunk_state import DeChunkState


class DeChunkLayer(Module):
  def __init__(
    self,
    d_model,
    dtype=bfloat16,
    block_size=256,
    headdim=32,
  ):
    super().__init__()
    self.d_model = d_model

    # Just for Mamba2 kernel.
    self.dtype = dtype
    self.block_size = block_size
    self.headdim = headdim
    assert d_model % self.headdim == 0
    self.nheads = d_model // self.headdim

  def allocate_inference_cache(
    self, batch_size, max_seqlen, device, dtype=None
  ):
    return DeChunkState(
      last_value=zeros(batch_size, self.d_model, device=device, dtype=dtype),
    )

  def forward(
    self,
    hidden_states: Tensor,
    boundary_mask: Tensor,
    boundary_prob: Tensor,
    mask: Tensor,
    inference_params: DeChunkState,
  ):
    if inference_params is None:
      assert mask is not None, (
        "Mask must be provided if inference_params is not provided"
      )
      assert boundary_mask[:, 0].all(), (
        "First token must be a boundary if running prefill"
      )

    p = clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

    B, L = boundary_mask.shape
    seq_idx = None

    token_idx = (
      arange(L, device=hidden_states.device)[None, :]
      + (~boundary_mask).long() * L
    )
    seq_sorted_indices = argsort(token_idx, dim=1)

    p = gather(
      p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
    )  # (B, M)

    original_dtype = hidden_states.dtype
    # Reuse Mamba2 kernel for EMA Deaggregator.
    dt = log(1 / (1 - p)).to(self.dtype)
    x = (hidden_states / dt[..., None]).to(self.dtype)
    A = -ones((self.nheads,), device=hidden_states.device, dtype=float32)
    b = p.to(self.dtype)
    c = ones_like(b)

    out = cast(
      Tensor,
      mamba_chunk_scan_combined(
        rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
        repeat(dt, "b l -> b l h", h=self.nheads),
        A,
        rearrange(b, "b l -> b l 1 1"),
        rearrange(c, "b l -> b l 1 1"),
        chunk_size=self.block_size,
        seq_idx=seq_idx,
      ),
    )
    out = rearrange(out, "b l h p -> b l (h p)")

    plug_back_idx = cumsum(boundary_mask, dim=1) - 1  # (B, L)
    out = gather(
      out,
      dim=1,
      index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
    )

    if inference_params is not None:
      inference_params.last_value.copy_(out[:, -1])

    return out.to(original_dtype)

  def step(
    self,
    hidden_states: Tensor,
    boundary_mask: Tensor,
    boundary_prob: Tensor,
    inference_params,
  ):
    # hidden_states is (B', 1, D), where B' = boundary_mask.sum()
    # boundary_mask is (B,) and boundary_prob is (B, 2)

    B = boundary_mask.shape[0]
    # B_selected = hidden_states.shape[0]
    D = hidden_states.shape[-1]

    p = zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
    p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(
      min=1e-4, max=1 - (1e-4)
    )

    current_hidden_states = zeros(
      B, D, device=hidden_states.device, dtype=hidden_states.dtype
    )
    current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

    result = p * current_hidden_states + (1 - p) * inference_params.last_value
    inference_params.last_value.copy_(result)

    return result.unsqueeze(1)
