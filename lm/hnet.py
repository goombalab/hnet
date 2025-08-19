from dataclasses import dataclass

from torch import Tensor, autograd, cat, dtype, float32, ones_like, zeros
from torch._prims_common import DeviceLikeType
from torch.nn import Linear, Module, Parameter, init
from typing_extensions import Self

from hnet.modules.dc import (
  ChunkLayer,
  DeChunkLayer,
  DeChunkState,
  RoutingModule,
  RoutingModuleState,
)
from hnet.modules.isotropic import Isotropic, IsotropicInferenceParams
from lm.hnet_config import HnetConfig


class STE(autograd.Function):
  @staticmethod
  def forward(ctx, x: Tensor):
    return ones_like(x)

  @staticmethod
  def backward(ctx, *grad_outputs: list[Tensor]):
    (grad_x,) = grad_outputs
    grad_x = grad_outputs
    return grad_x


def ste_func(x: Tensor):
  return STE.apply(x)


@dataclass
class HnetState:
  main_network_state: Self | IsotropicInferenceParams
  encoder_state: IsotropicInferenceParams | None = None
  routing_module_state: RoutingModuleState | None = None
  dechunk_state: DeChunkState | None = None
  decoder_state: IsotropicInferenceParams | None = None


class Hnet(Module):
  is_innermost: bool
  main_network: "Hnet | Isotropic"
  encoder: Isotropic | None
  decoder: Isotropic | None
  routing_module: RoutingModule | None
  chunk_layer: ChunkLayer | None
  dechunk_layer: DeChunkLayer | None
  residual_proj: Linear | None
  pad_dimension: Parameter | None

  def __init__(
    self,
    config: HnetConfig,
    stage_idx: int,
    device: DeviceLikeType | None = None,
    dtype: dtype | None = None,
  ) -> None:
    super().__init__()

    arch_layout = config.arch_layout
    for _ in range(stage_idx):
      arch_layout = arch_layout[1]
    assert isinstance(arch_layout, list) and len(arch_layout) in (1, 3), (
      f"Wrong arch_layout: {arch_layout}"
    )

    self.is_innermost = len(arch_layout) == 1

    if self.is_innermost:
      self.main_network = Isotropic(
        config,
        stage_idx=stage_idx,
        pos_idx=0,
        device=device,
        dtype=dtype,
      )
    else:
      self.encoder = Isotropic(
        config,
        stage_idx=stage_idx,
        pos_idx=0,
        device=device,
        dtype=dtype,
      )
      self.main_network = Hnet(config, stage_idx + 1, device, dtype)
      self.decoder = Isotropic(
        config,
        stage_idx=stage_idx,
        pos_idx=2,
        device=device,
        dtype=dtype,
      )

    d_model_n = config.d_model[stage_idx]
    if not self.is_innermost:
      self.routing_module = RoutingModule(d_model_n, device, dtype)
      self.chunk_layer = ChunkLayer()
      self.dechunk_layer = DeChunkLayer(d_model_n)

      # do the residual in fp32
      self.residual_proj = Linear(
        d_model_n, d_model_n, device=device, dtype=float32
      )
      init.zeros_(self.residual_proj.weight)
      self.residual_proj.weight._no_reinit = True
      self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

    if stage_idx > 0 and d_model_n - config.d_model[stage_idx - 1] > 0:
      self.pad_dimension = Parameter(
        zeros(
          d_model_n - config.d_model[stage_idx - 1],
          device=device,
          dtype=dtype,
        )
      )
    else:
      self.pad_dimension = None

  def allocate_inference_cache(
    self,
    batch_size: int,
    max_seqlen: int,
    dtype: dtype,
  ):
    """
    Allocate the inference cache for the HNet.

    Arguments:
        batch_size: The number of sequences in the batch.
        max_seqlen: The maximum sequence length in the batch.
        dtype: The dtype of the inference cache.

    The structure of the inference cache is as follows:
        - [encoder state]
        - [routing module state]
        - [main network state]
        - [dechunk state]
        - [decoder state]
    It is thus a list of length 5.
    """
    if self.is_innermost:
      return HnetState(
        main_network_state=self.main_network.allocate_inference_cache(
          batch_size, max_seqlen, dtype=dtype
        )
      )
    else:
      assert (
        self.residual_proj is not None
        and self.encoder is not None
        and self.routing_module is not None
        and self.dechunk_layer is not None
        and self.decoder is not None
      )
      device = self.residual_proj.weight.device
      return HnetState(
        main_network_state=self.main_network.allocate_inference_cache(
          batch_size, max_seqlen, dtype=dtype
        ),
        encoder_state=self.encoder.allocate_inference_cache(
          batch_size, max_seqlen, dtype=dtype
        ),
        routing_module_state=self.routing_module.allocate_inference_cache(
          batch_size, max_seqlen, device, dtype=dtype
        ),
        dechunk_state=self.dechunk_layer.allocate_inference_cache(
          batch_size, max_seqlen, device, dtype=dtype
        ),
        decoder_state=self.decoder.allocate_inference_cache(
          batch_size, max_seqlen, dtype=dtype
        ),
      )

  def forward(
    self,
    hidden_states,
    mask,
    inference_params=None,
    **mixer_kwargs,
  ):
    if inference_params is None:
      inference_params = HnetState(main_network_state=None)

    D = hidden_states.shape[-1]
    EARLY_DIMS = hidden_states.shape[:-1]

    if self.pad_dimension is not None:
      hidden_states = cat(
        (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1
      )

    if self.is_innermost:
      hidden_states = self.main_network.forward(
        hidden_states,
        mask=mask,
        inference_params=inference_params.main_network_state,
        **mixer_kwargs,
      )
      hidden_states = hidden_states[..., :D]
      return hidden_states, []

    assert self.encoder is not None
    hidden_states = self.encoder.forward(
      hidden_states,
      mask=mask,
      inference_params=inference_params.encoder_state,
      **mixer_kwargs,
    )

    hidden_states_for_residual = hidden_states.to(
      dtype=self.residual_proj.weight.dtype
    )
    residual = self.residual_proj(hidden_states_for_residual)

    bpred_output = self.routing_module.forward(
      hidden_states,
      mask=mask,
      inference_params=inference_params.routing_module_state,
    )
    hidden_states, next_cu_seqlens, next_max_seqlen, next_mask = (
      self.chunk_layer.forward(
        hidden_states,
        bpred_output.boundary_mask,
        mask=mask,
      )
    )

    hidden_states, prev_boundary_predictions = self.main_network.forward(
      hidden_states,
      mask=next_mask,
      inference_params=inference_params.main_network_state,
      **mixer_kwargs,
    )

    hidden_states = self.dechunk_layer.forward(
      hidden_states,
      bpred_output.boundary_mask,
      bpred_output.boundary_prob,
      mask=mask,
      inference_params=inference_params.dechunk_state,
    )

    hidden_states = self.residual_func(
      hidden_states.to(dtype=residual.dtype),
      residual,
      bpred_output.selected_probs,
    ).to(hidden_states.dtype)

    hidden_states = self.decoder.forward(
      hidden_states,
      mask=mask,
      inference_params=inference_params.decoder_state,
      **mixer_kwargs,
    )

    hidden_states = hidden_states[..., :D]
    return hidden_states, [bpred_output, *prev_boundary_predictions]

  def step(self, hidden_states, inference_params):
    D = hidden_states.shape[-1]

    if self.pad_dimension is not None:
      hidden_states = cat(
        (
          hidden_states,
          self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,)),
        ),
        dim=-1,
      )

    if self.is_innermost:
      hidden_states = self.main_network.step(
        hidden_states, inference_params.main_network_state
      )
      hidden_states = hidden_states[..., :D]
      return hidden_states, []

    hidden_states = self.encoder.step(
      hidden_states, inference_params.encoder_state
    )
    hidden_states_for_residual = hidden_states.to(
      dtype=self.residual_proj.weight.dtype
    )
    residual = self.residual_proj(hidden_states_for_residual)

    bpred_output = self.routing_module.step(
      hidden_states, inference_params.routing_module_state
    )
    hidden_states_inner = self.chunk_layer.step(
      hidden_states, bpred_output.boundary_mask
    )

    if hidden_states_inner.shape[0] > 0:
      hidden_states_inner, prev_boundary_predictions = self.main_network.step(
        hidden_states_inner, inference_params.main_network_state
      )
    else:
      prev_boundary_predictions = []

    hidden_states = self.dechunk_layer.step(
      hidden_states_inner,
      bpred_output.boundary_mask,
      bpred_output.boundary_prob,
      inference_params.dechunk_state,
    )

    hidden_states = self.residual_func(
      hidden_states.to(dtype=residual.dtype),
      residual,
      bpred_output.selected_probs,
    ).to(hidden_states.dtype)

    hidden_states = self.decoder.step(
      hidden_states, inference_params.decoder_state
    )
    hidden_states = hidden_states[..., :D]

    return hidden_states, [bpred_output, *prev_boundary_predictions]
