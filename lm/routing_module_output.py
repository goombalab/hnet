from dataclasses import dataclass

from torch import Tensor


@dataclass
class RoutingModuleOutput:
  boundary_prob: Tensor
  boundary_mask: Tensor
  selected_probs: Tensor
