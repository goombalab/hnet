import torch.nn.functional as F
from torch import (
  Tensor,
  arange,
  argmax,
  bool,
  clamp,
  einsum,
  eye,
  no_grad,
  ones_like,
  stack,
  where,
  zeros,
)
from torch.nn import Linear, Module

from lm.routing_module_output import RoutingModuleOutput
from lm.routing_module_state import RoutingModuleState


class RoutingModule(Module):
  def __init__(self, d_model, device=None, dtype=None):
    self.d_model = d_model
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()
    self.q_proj_layer = Linear(d_model, d_model, bias=False, **factory_kwargs)
    self.k_proj_layer = Linear(d_model, d_model, bias=False, **factory_kwargs)
    with no_grad():
      self.q_proj_layer.weight.copy_(eye(d_model))
      self.k_proj_layer.weight.copy_(eye(d_model))
    self.q_proj_layer.weight._no_reinit = True  # type: ignore
    self.k_proj_layer.weight._no_reinit = True  # type: ignore

  def allocate_inference_cache(
    self, batch_size, max_seqlen, device, dtype=None
  ):
    return RoutingModuleState(
      has_seen_tokens=zeros(batch_size, device=device, dtype=bool),
      last_hidden_state=zeros(
        batch_size, self.d_model, device=device, dtype=dtype
      ),
    )

  def forward(
    self, hidden_states, mask: Tensor | None = None, inference_params=None
  ):
    if inference_params is not None:
      assert mask is not None, (
        "Mask must be provided if inference_params is not provided"
      )
      assert (~inference_params.has_seen_tokens).all(), (
        "Cannot have seen tokens when inference_params is not provided"
      )

    cos_sim = einsum(
      "b l d, b l d -> b l",
      F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
      F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
    )
    # this clamp should no-op as long as no precision issues are encountered
    boundary_prob = clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

    # Force boundary probability of the first element to 1.0
    PAD_PROB = 1.0
    boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

    boundary_prob = stack(((1 - boundary_prob), boundary_prob), dim=-1)

    selected_idx = argmax(boundary_prob, dim=-1)

    boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])
    if mask is not None:
      # No invalid tokens can be selected
      boundary_mask = boundary_mask & mask

    if inference_params is not None:
      assert mask is not None
      has_mask = mask.any(dim=-1)
      inference_params.has_seen_tokens.copy_(
        has_mask | inference_params.has_seen_tokens
      )
      last_mask = clamp(mask.sum(dim=-1) - 1, min=0)
      inference_params.last_hidden_state.copy_(
        where(
          has_mask,
          hidden_states[
            arange(hidden_states.shape[0], device=hidden_states.device),
            last_mask,
          ],
          inference_params.last_hidden_state,
        )
      )

    selected_probs = boundary_prob.gather(
      dim=-1, index=selected_idx.unsqueeze(-1)
    )  # (shape hidden_states.shape[:-1], 1)

    return RoutingModuleOutput(
      boundary_prob=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
      boundary_mask=boundary_mask,  # (shape hidden_states.shape[:-1])
      selected_probs=selected_probs,  # (shape hidden_states.shape[:-1], 1)
    )

  def step(self, hidden_states, inference_params):
    # hidden_states is (B, 1, D)
    hidden_states = hidden_states.squeeze(1)
    cos_sim = einsum(
      "b d, b d -> b",
      F.normalize(
        self.q_proj_layer(inference_params.last_hidden_state), dim=-1
      ),
      F.normalize(self.k_proj_layer(hidden_states), dim=-1),
    )
    boundary_prob = clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
    inference_params.last_hidden_state.copy_(hidden_states)
    boundary_prob = where(
      inference_params.has_seen_tokens,
      boundary_prob,
      ones_like(boundary_prob),
    )
    boundary_prob = stack(((1 - boundary_prob), boundary_prob), dim=-1)

    inference_params.has_seen_tokens.copy_(
      ones_like(inference_params.has_seen_tokens)
    )
    return RoutingModuleOutput(
      boundary_prob=boundary_prob,  # (B, 2)
      boundary_mask=boundary_prob[..., 1] > 0.5,  # (B,)
      selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),  # (B, 1)
    )
