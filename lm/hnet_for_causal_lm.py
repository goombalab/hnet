from dataclasses import dataclass

from flash_attn.utils.generation import GenerationMixin
from omegaconf import ListConfig
from torch import Tensor, arange, dtype, int, load, serialization, tensor
from torch._prims_common import DeviceLikeType
from torch.nn import Linear, Module
from torch.nn.modules.sparse import Embedding
from typing_extensions import Self

from hnet.modules.dc import RoutingModuleOutput
from lm.hnet import HNet, HNetState
from lm.hnet_config import HnetConfig


@dataclass
class CausalLmOutput:
  logits: Tensor
  bpred_output: list[RoutingModuleOutput]
  inference_params: HNetState


@dataclass(eq=False)
class HnetForCausalLm(Module, GenerationMixin):
  config: HnetConfig
  embeddings: Embedding
  backbone: HNet
  lm_head: Linear

  def __init__(
    self,
    config: HnetConfig,
    device: DeviceLikeType | None = None,
    dtype: dtype | None = None,
  ) -> None:
    super().__init__()

    self.config = config

    vocab_size = self.config.vocab_size
    d_embed = self.config.d_model[0]

    # We consider the HNet as a map (B, L, D[0]) -> (B, L, D[0])
    # Thus, the embedding is defined outside of the HNet.
    self.embeddings = Embedding(vocab_size, d_embed, device=device, dtype=dtype)

    self.backbone = HNet(
      config=config,
      # We pass in the stage_idx as an HNet needs to know what
      # depth of the hierarchy it is in.
      stage_idx=0,
      device=device,
      dtype=dtype,
    )
    self.lm_head = Linear(
      d_embed, vocab_size, bias=False, device=device, dtype=dtype
    )
    if self.config.tie_embeddings:
      self.lm_head.weight = self.embeddings.weight

  def allocate_inference_cache(
    self, batch_size, max_seqlen, dtype=None, **kwargs
  ):
    return self.backbone.allocate_inference_cache(
      batch_size, max_seqlen, dtype=dtype, **kwargs
    )

  def forward(
    self,
    tokens: Tensor,
    mask: Tensor | None = None,
    inference_params: HNetState | None = None,
    **mixer_kwargs,
  ) -> CausalLmOutput:
    """
    num_last_tokens: if > 0, only return the logits for the last n tokens
    """
    hidden_states = self.embeddings.forward(tokens)

    B, L, D = hidden_states.shape

    if mask is None:
      # Absent a mask, we assume we are running in packed mode
      assert inference_params is None, (
        "Inference params are not supported in packed mode"
      )
      hidden_states = hidden_states.flatten(0, 1)
      cu_seqlens = arange(B + 1, device=hidden_states.device) * L
      max_seqlen = tensor(L, dtype=int, device=hidden_states.device)
    else:
      cu_seqlens = None
      max_seqlen = None

    hidden_states, bpred_output = self.backbone.forward(
      hidden_states,
      cu_seqlens=cu_seqlens,
      max_seqlen=max_seqlen,
      mask=mask,
      inference_params=inference_params,
      **mixer_kwargs,
    )

    hidden_states = hidden_states.view(B, L, D)

    lm_logits = self.lm_head.forward(hidden_states)

    assert inference_params is not None
    return CausalLmOutput(
      logits=lm_logits,
      bpred_output=bpred_output,
      inference_params=inference_params,
    )

  def load(self, path: str) -> Self:
    self.eval()

    device = next(self.parameters()).device

    with serialization.safe_globals([ListConfig]):
      state_dict = load(path, map_location=device, weights_only=False)
    self.load_state_dict(state_dict)

    return self

  def step(self, token: Tensor, inference_params: HNetState) -> CausalLmOutput:
    B = token.shape[0]
    assert B == 1, (
      "HNetForCausalLM step currently only supports batch size 1 -- need to handle different-size lengths for each sample"
    )

    hidden_states = self.embeddings.forward(token)

    hidden_states, bpred_output = self.backbone.step(
      hidden_states,
      inference_params,
    )
    logits = self.lm_head.forward(hidden_states)

    return CausalLmOutput(
      logits=logits,
      bpred_output=bpred_output,
      inference_params=inference_params,
    )
