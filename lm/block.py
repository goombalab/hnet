from typing import cast

from flash_attn.ops.triton.layer_norm import RMSNorm
from torch import Tensor, dtype
from torch._prims_common import DeviceLikeType
from torch.nn import Module

from lm.causal_mha import CausalMHA
from lm.isotropic_inference_params import IsotropicInferenceParams
from lm.mamba_2_wrapper import Mamba2Wrapper
from lm.swiglu import Swiglu


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

    self.mixer = mixer
    self.mlp = mlp
    self.norm1 = norm1
    self.norm2 = norm2
    self.residual_in_fp32 = residual_in_fp32

  def forward(
    self,
    hidden_states: Tensor,
    residual: Tensor | None,
    inference_params: IsotropicInferenceParams,
  ):
    from typing import cast

    hidden_states, residual = cast(
      Tensor,
      self.norm1.forward(
        hidden_states,
        residual,
        prenorm=True,
        residual_in_fp32=self.residual_in_fp32,
      ),
    )

    hidden_states = self.mixer.forward(
      hidden_states,
      inference_params=inference_params,
    )

    if self.mlp is not None:
      from typing import Tuple, cast

      assert self.norm2 is not None
      hidden_states, residual = cast(
        Tuple[Tensor, Tensor],
        self.norm2.forward(
          hidden_states,
          residual,
          prenorm=True,
          residual_in_fp32=self.residual_in_fp32,
        ),
      )
      hidden_states = self.mlp.forward(hidden_states)

    return hidden_states, residual

  def allocate_inference_cache(
    self,
    batch_size,
    max_seqlen,
    dtype=None,
  ) -> Tensor:
    return cast(
      Tensor,
      self.mixer.allocate_inference_cache(
        batch_size,
        max_seqlen,
        dtype=dtype,
      ),
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
    norm2 = RMSNorm(d_model, eps=norm_epsilon, device=device, dtype=dtype)
  elif arch in ("t", "m"):
    mlp = None
    norm2 = None
  else:
    raise NotImplementedError

  norm1 = RMSNorm(d_model, eps=norm_epsilon, device=device, dtype=dtype)

  return Block(
    mixer,
    mlp,
    norm1,
    norm2,
    residual_in_fp32,
  )
