from torch import Tensor, arange, argsort, gather
from torch.nn import Module


class ChunkLayer(Module):
  def forward(
    self,
    hidden_states: Tensor,
    boundary_mask: Tensor,
  ) -> tuple[Tensor, Tensor]:
    num_tokens = boundary_mask.sum(dim=-1)
    next_max_seqlen = int(num_tokens.max())

    device = hidden_states.device
    L = hidden_states.shape[1]
    token_idx = arange(L, device=device)[None, :] + (~boundary_mask).long() * L
    seq_sorted_indices = argsort(token_idx, dim=1)

    next_hidden_states = gather(
      hidden_states,
      dim=1,
      index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
        -1, -1, hidden_states.shape[-1]
      ),
    )

    next_mask = (
      arange(next_max_seqlen, device=device)[None, :] < num_tokens[:, None]
    )
    next_max_seqlen = None

    return next_hidden_states, next_mask

  def step(self, hidden_states: Tensor, boundary_mask: Tensor) -> Tensor:
    return hidden_states[boundary_mask]
