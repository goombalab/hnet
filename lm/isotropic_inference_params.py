from dataclasses import dataclass, field
from typing import cast

from optree import tree_map
from optree.typing import PyTree
from torch import Tensor


@dataclass
class IsotropicInferenceParams:
  """Inference parameters that are passed to the main model in order
  to efficienly calculate and store the context during inference."""

  max_seqlen: int
  max_batch_size: int
  seqlen_offset: int = 0
  batch_size_offset: int = 0
  key_value_memory_dict: dict[int, Tensor] = field(default_factory=dict)
  lengths_per_sample: Tensor | None = None

  def reset(self, max_seqlen: int, max_batch_size: int):
    self.max_seqlen = max_seqlen
    self.max_batch_size = max_batch_size
    self.seqlen_offset = 0
    if self.lengths_per_sample is not None:
      self.lengths_per_sample.zero_()

    tree_map(
      lambda x: x.zero_() if isinstance(x, Tensor) else x,
      cast(PyTree, self.key_value_memory_dict),
    )
