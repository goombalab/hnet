from dataclasses import dataclass

from omegaconf import ListConfig
from torch import Tensor, dtype, load, serialization
from torch._prims_common import DeviceLikeType
from torch.nn import Linear, Module
from torch.nn.modules.sparse import Embedding
from typing_extensions import Self

from lm.hnet import Hnet, HnetState
from lm.hnet_config import HnetConfig
from lm.routing_module_output import RoutingModuleOutput


@dataclass
class CausalLmOutput:
  logits: Tensor
  bpred_output: list[RoutingModuleOutput]
  inference_params: HnetState


@dataclass(eq=False)
class HnetForCausalLm(Module):
  embeddings: Embedding
  backbone: Hnet
  lm_head: Linear

  def __init__(
    self,
    config: HnetConfig,
    device: DeviceLikeType | None = None,
    dtype: dtype | None = None,
  ) -> None:
    super().__init__()

    vocab_size = config.vocab_size
    d_model_0 = config.d_model[0]

    # We consider the HNet as a map (B, L, D[0]) -> (B, L, D[0])
    # Thus, the embedding is defined outside of the HNet.
    self.embeddings = Embedding(
      num_embeddings=vocab_size,
      embedding_dim=d_model_0,
      device=device,
      dtype=dtype,
    )

    self.backbone = Hnet(
      config=config,
      # We pass in the stage_idx as an HNet needs to know what
      # depth of the hierarchy it is in.
      stage_idx=0,
      device=device,
      dtype=dtype,
    )

    self.lm_head = Linear(
      in_features=d_model_0,
      out_features=vocab_size,
      bias=False,
      device=device,
      dtype=dtype,
    )

    if config.tie_embeddings:
      self.lm_head.weight = self.embeddings.weight

  def allocate_inference_cache(
    self,
    batch_size: int,
    max_seqlen: int,
    dtype: dtype,
  ):
    return self.backbone.allocate_inference_cache(
      batch_size,
      max_seqlen,
      dtype,
    )

  def forward(
    self,
    tokens: Tensor,
    inference_params: HnetState,
    mask: Tensor,
  ) -> CausalLmOutput:
    hidden_states = self.embeddings.forward(tokens)

    hidden_states, bpred_output = self.backbone.forward(
      hidden_states,
      mask,
      inference_params=inference_params,
    )

    logits = self.lm_head.forward(hidden_states)

    return CausalLmOutput(
      logits,
      bpred_output,
      inference_params,
    )

  def load(self, path: str) -> Self:
    self.eval()

    device = next(self.parameters()).device

    with serialization.safe_globals([ListConfig]):
      state_dict = load(path, map_location=device, weights_only=False)
    self.load_state_dict(state_dict)

    return self

  def step(
    self,
    token: Tensor,
    inference_params: HnetState,
  ) -> CausalLmOutput:
    hidden_states = self.embeddings.forward(token)

    hidden_states, bpred_output = self.backbone.step(
      hidden_states,
      inference_params=inference_params,
    )

    logits = self.lm_head.forward(hidden_states)

    return CausalLmOutput(
      logits,
      bpred_output,
      inference_params,
    )
