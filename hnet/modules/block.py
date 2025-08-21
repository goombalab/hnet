from typing import Optional

from flash_attn.ops.triton.layer_norm import RMSNorm
from mamba_ssm.modules.mamba2 import Mamba2
from torch import Tensor, dtype
from torch._prims_common import DeviceLikeType
from torch.nn import Module

from lm.swiglu import Swiglu

from .mha import CausalMHA


class Mamba2Wrapper(Mamba2):
  """
  Mamba2 wrapper class that has the same inference interface as the CausalMHA class.
  """

  def next_step(self, hidden_states, inference_params):
    # Don't use _get_states_from_cache because we want to assert that they exist
    conv_state, ssm_state = inference_params.key_value_memory_dict[
      self.layer_idx
    ]  # init class of Mamba2 accepts layer_idx
    result, conv_state, ssm_state = super().step(
      hidden_states,
      conv_state,
      ssm_state,
    )

    # Update the state cache in-place
    inference_params.key_value_memory_dict[self.layer_idx][0].copy_(conv_state)
    inference_params.key_value_memory_dict[self.layer_idx][1].copy_(ssm_state)
    return result


class Block(Module):
  mixer: CausalMHA | Mamba2Wrapper
  mlp: Swiglu | None
  norm1: RMSNorm
  norm2: RMSNorm | None
  residual_in_fp32: bool

  def __init__(
    self,
    mixer: CausalMHA | Mamba2Wrapper,
    mlp: Swiglu | None,
    norm1: RMSNorm,
    norm2: RMSNorm | None = None,
    residual_in_fp32: bool = True,
  ):
    super().__init__()

    self.residual_in_fp32 = residual_in_fp32
    self.norm1 = norm1
    self.mixer = mixer
    self.mlp = mlp
    self.norm2 = norm2

  def forward(
    self,
    hidden_states: Tensor,
    residual: Optional[Tensor] = None,
    inference_params=None,
    mixer_kwargs=None,
  ):
    hidden_states, residual = self.norm1(
      hidden_states,
      residual=residual,
      prenorm=True,
      residual_in_fp32=self.residual_in_fp32,
    )

    if mixer_kwargs is None:
      mixer_kwargs = {}
    hidden_states = self.mixer(
      hidden_states,
      inference_params=inference_params,
      **mixer_kwargs,
    )

    if self.mlp is not None:
      from typing import Tuple, cast

      assert self.norm2 is not None
      hidden_states, residual = cast(
        Tuple[Tensor, Tensor],
        self.norm2.forward(
          hidden_states,
          residual=residual,
          prenorm=True,
          residual_in_fp32=self.residual_in_fp32,
        ),
      )
      hidden_states = self.mlp(hidden_states)

    return hidden_states, residual

  def allocate_inference_cache(
    self,
    batch_size,
    max_seqlen,
    dtype=None,
  ):
    return self.mixer.allocate_inference_cache(
      batch_size,
      max_seqlen,
      dtype=dtype,
    )

  def step(self, hidden_states, inference_params, residual=None):
    hidden_states, residual = self.norm1(
      hidden_states,
      residual=residual,
      prenorm=True,
      residual_in_fp32=self.residual_in_fp32,
    )
    hidden_states = self.mixer.next_step(hidden_states, inference_params)
    if self.mlp is not None:
      from typing import Tuple, cast

      assert self.norm2 is not None
      hidden_states, residual = cast(
        Tuple[Tensor, Tensor],
        self.norm2.forward(
          hidden_states,
          residual=residual,
          prenorm=True,
          residual_in_fp32=self.residual_in_fp32,
        ),
      )
      hidden_states = self.mlp(hidden_states)

    return hidden_states, residual


def create_block(
  arch: str,
  d_model: int,
  d_intermediate: int | None = None,
  ssm_cfg: dict = dict(),
  attn_cfg: dict = dict(),
  norm_epsilon: float = 1e-5,
  layer_idx: int | None = None,
  residual_in_fp32: bool = True,
  device: DeviceLikeType | None = None,
  dtype: dtype | None = None,
) -> Block:
  if arch in ("t", "T"):
    mixer = CausalMHA(
      d_model,
      **attn_cfg,
      layer_idx=layer_idx,
      device=device,
      dtype=dtype,
    )
  elif arch in ("m", "M"):
    mixer = Mamba2Wrapper(
      d_model,
      **ssm_cfg,
      layer_idx=layer_idx,
      device=device,
      dtype=dtype,
    )
  else:
    raise NotImplementedError

  if arch in ("T", "M"):
    mlp = Swiglu(
      d_model,
      d_intermediate,
      device=device,
      dtype=dtype,
    )
  elif arch in ("t", "m"):
    mlp = None
  else:
    raise NotImplementedError

  norm1 = RMSNorm(d_model, eps=norm_epsilon, device=device, dtype=dtype)
  norm2 = (
    RMSNorm(d_model, eps=norm_epsilon, device=device, dtype=dtype)
    if isinstance(mlp, Swiglu)
    else None
  )

  return Block(
    mixer,
    mlp,
    norm1,
    norm2,
    residual_in_fp32,
  )
