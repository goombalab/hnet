from torch import Tensor, autograd, ones_like


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
