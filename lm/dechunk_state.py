from dataclasses import dataclass

from torch import Tensor


@dataclass
class DeChunkState:
  """
  The state of the dechunk.

  Contains
      - [last_value] (batch_size, d_model) tensor. The last value of the batch element (used for the EMA).
  """

  last_value: Tensor  # (batch_size, d_model)
