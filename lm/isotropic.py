import re
from dataclasses import asdict
from typing import cast

from flash_attn.ops.triton.layer_norm import RMSNorm
from torch import Tensor, dtype
from torch._prims_common import DeviceLikeType
from torch.nn import Module, ModuleList

from lm.block import Block, create_block
from lm.hnet_config import HnetConfig
from lm.isotropic_inference_params import IsotropicInferenceParams


class Isotropic(Module):
  stage_idx: int
  d_model: int
  ssm_cfg: dict
  attn_cfg: dict
  arch_full: list
  layers: ModuleList
  rmsnorm: RMSNorm

  def __init__(
    self,
    config: HnetConfig,
    pos_idx: int,
    stage_idx: int,
    device: DeviceLikeType | None = None,
    dtype=None,
  ):
    super().__init__()

    self.stage_idx = stage_idx
    self.d_model = config.d_model[self.stage_idx]
    self.ssm_cfg = _get_stage_cfg(config.ssm_cfg, stage_idx)
    self.attn_cfg = _get_stage_cfg(config.attn_cfg, stage_idx)

    arch_layout = config.arch_layout
    for _ in range(stage_idx):
      arch_layout = arch_layout[1]
    arch_layout = cast(str, arch_layout[pos_idx])
    layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)

    layers = []
    layer_idx = 0
    self.arch_full = []
    for arch, n_layer in layout_parse:
      assert arch in ("m", "M", "t", "T")
      assert n_layer.isdigit()
      layers += [
        create_block(
          arch,
          self.d_model,
          config.d_intermediate[self.stage_idx],
          self.ssm_cfg,
          self.attn_cfg,
          layer_idx=(layer_idx + i),
          device=device,
          dtype=dtype,
        )
        for i in range(int(n_layer))
      ]
      self.arch_full.extend([arch for _ in range(int(n_layer))])
      layer_idx += int(n_layer)

    self.layers = ModuleList(layers)

    self.rmsnorm = RMSNorm(
      self.d_model,
      eps=1e-5,
      device=device,
      dtype=dtype,
    )

  def allocate_inference_cache(
    self,
    batch_size: int,
    max_seqlen: int,
    dtype: dtype,
  ):
    """
    Allocate the inference cache for the Isotropic module.

    Arguments:
        batch_size: The number of sequences in the batch.
        max_seqlen: The maximum sequence length in the batch, not used for this module.
        dtype: The dtype of the inference cache.

    The inference cache contains a list of inference caches, one for each block.
    """
    key_value_memory_dict: dict[int, Tensor] = {}
    for i, layer in enumerate(self.layers):
      block = cast(Block, layer)
      key_value_memory_dict[i] = block.allocate_inference_cache(
        batch_size,
        max_seqlen,
        dtype=dtype,
      )
    return IsotropicInferenceParams(
      key_value_memory_dict=key_value_memory_dict,
      max_seqlen=max_seqlen,
      max_batch_size=batch_size,
    )

  def forward(
    self,
    hidden_states: Tensor,
    mask: Tensor,
    inference_params: IsotropicInferenceParams,
  ):
    residual = None
    for layer in self.layers:
      block = cast(Block, layer)
      hidden_states, residual = block.forward(
        hidden_states,
        residual,
        inference_params=inference_params,
      )

    # Setting prenorm=False ignores the residual
    hidden_states = cast(
      Tensor,
      self.rmsnorm.forward(
        hidden_states,
        residual,
        prenorm=False,
        residual_in_fp32=True,
      ),
    )

    # here we also explicitly assume the mask is all True
    assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
    inference_params.seqlen_offset += hidden_states.shape[1]

    return hidden_states

  def step(
    self,
    hidden_states: Tensor,
    inference_params: IsotropicInferenceParams,
  ):
    """
    Assumes hidden_states is (B, 1, D). Steps each of the layers in order, and then steps the main model.
    """
    residual = None
    for layer in self.layers:
      block = cast(Block, layer)
      hidden_states, residual = block.step(
        hidden_states,
        inference_params,
        residual=residual,
      )

    hidden_states = cast(
      Tensor,
      self.rmsnorm.forward(
        hidden_states,
        residual=residual,
        prenorm=False,
        residual_in_fp32=True,
      ),
    )
    inference_params.seqlen_offset += 1

    return hidden_states


def _get_stage_cfg(cfg, stage_idx):
  return {
    k: v[stage_idx] if isinstance(v, list) else v
    for k, v in asdict(cfg).items()
  }
