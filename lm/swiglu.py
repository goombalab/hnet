from flash_attn.ops.activations import swiglu
from torch import Tensor, dtype
from torch._prims_common import DeviceLikeType
from torch.nn import Linear, Module


class Swiglu(Module):
  fc1: Linear
  fc2: Linear

  def __init__(
    self,
    d_model: int,
    d_intermediate: int | None = None,
    bias: bool = False,
    multiple_of: int = 128,
    device: DeviceLikeType | None = None,
    dtype: dtype | None = None,
  ):
    super().__init__()

    d_intermediate = (
      d_intermediate if d_intermediate is not None else int(8 * d_model / 3)
    )
    d_intermediate = (
      (d_intermediate + multiple_of - 1) // multiple_of * multiple_of
    )

    self.fc1 = Linear(
      in_features=d_model,
      out_features=2 * d_intermediate,
      bias=bias,
      device=device,
      dtype=dtype,
    )

    self.fc2 = Linear(
      in_features=d_intermediate,
      out_features=d_model,
      bias=bias,
      device=device,
      dtype=dtype,
    )

  def forward(self, x: Tensor):
    y = self.fc1(x)
    y, gate = y.chunk(2, dim=-1)
    y = swiglu(gate, y)
    y = self.fc2(y)
    return y
