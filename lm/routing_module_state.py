from dataclasses import dataclass

from torch import Tensor


@dataclass
class RoutingModuleState:
  """
  The state of the routing module.

  Contains
      - [has_seen_tokens] (batch_size,) bool tensor. Whether that batch element has processed any tokens yet.
      - [last_hidden_state] (batch_size, d_model) tensor. The last hidden state of the batch element (used for boundary prediction).
  """

  has_seen_tokens: Tensor  # (batch_size,)
  last_hidden_state: Tensor  # (batch_size, d_model)
