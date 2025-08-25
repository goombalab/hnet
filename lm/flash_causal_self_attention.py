from flash_attn import (
  flash_attn_qkvpacked_func,
  flash_attn_varlen_qkvpacked_func,
)
from torch import bfloat16, float16, int32
from torch.nn import Module


class FlashCausalSelfAttention(Module):
  """Implement the scaled dot product attention with softmax.
  Arguments
  ---------
      softmax_scale: The temperature to use for the softmax attention.
                    (default: 1/sqrt(d_keys) where d_keys is computed at
                    runtime)
  """

  def __init__(
    self,
    softmax_scale=None,
    window_size=(-1, -1),
  ):
    super().__init__()
    assert flash_attn_varlen_qkvpacked_func is not None, (
      "FlashAttention is not installed"
    )
    self.softmax_scale = softmax_scale
    self.window_size = window_size

  def forward(self, qkv, cu_seqlens=None, max_seqlen=None):
    """Implements the multihead softmax attention.
    Arguments
    ---------
        qkv: The tensor containing the query, key, and value.
            If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
            If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
            (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
    Returns:
    --------
        out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
            else (B, S, H, D).
    """
    assert qkv.dtype in [float16, bfloat16]
    assert qkv.is_cuda
    if cu_seqlens is not None:
      assert cu_seqlens.dtype == int32
      assert max_seqlen is not None
      assert isinstance(max_seqlen, int)
      return flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens,
        max_seqlen,
        softmax_scale=self.softmax_scale,
        causal=True,
        window_size=(self.window_size, -1),
      )
    else:
      return flash_attn_qkvpacked_func(
        qkv,
        softmax_scale=self.softmax_scale,
        causal=True,
        window_size=(self.window_size, -1),
      )
