from flash_attn import (
  flash_attn_kvpacked_func,
  flash_attn_qkvpacked_func,
  flash_attn_varlen_kvpacked_func,
  flash_attn_varlen_qkvpacked_func,
)
from torch import Tensor, bfloat16, float16, int32
from torch.nn import Linear, Module


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


class FlashCausalCrossAttention(Module):
  """Implement the scaled dot product attention with softmax.
  Arguments
  ---------
      softmax_scale: The temperature to use for the softmax attention.
                    (default: 1/sqrt(d_keys) where d_keys is computed at
                    runtime)
      window_size: The window size to use for the attention.
  """

  def __init__(
    self,
    softmax_scale=None,
    window_size=(-1, -1),
  ):
    super().__init__()
    assert flash_attn_varlen_kvpacked_func is not None, (
      "FlashAttention is not installed"
    )
    assert flash_attn_kvpacked_func is not None, (
      "FlashAttention is not installed"
    )
    self.softmax_scale = softmax_scale
    self.window_size = window_size

  def forward(
    self,
    q,
    kv,
    cu_seqlens=None,
    max_seqlen=None,
    cu_seqlens_k=None,
    max_seqlen_k=None,
  ):
    """Implements the multihead softmax attention.
    Arguments
    ---------
        q: The tensor containing the query. (B, Sq, H, D)
        kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
        causal: if passed, will override self.causal
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into q.
        max_seqlen: int. Maximum sequence length in the batch of q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into kv.
        max_seqlen_k: int. Maximum sequence length in the batch of k and v.
    """
    assert q.dtype in [float16, bfloat16]
    assert q.is_cuda and kv.is_cuda
    if cu_seqlens is not None:
      assert cu_seqlens.dtype == int32
      assert max_seqlen is not None
      assert isinstance(max_seqlen, int)
      assert cu_seqlens_k is not None
      assert cu_seqlens_k.dtype == int32
      assert max_seqlen_k is not None
      assert isinstance(max_seqlen_k, int)
      return flash_attn_varlen_kvpacked_func(
        q,
        kv,
        cu_seqlens,
        cu_seqlens_k,
        max_seqlen,
        max_seqlen_k,
        softmax_scale=self.softmax_scale,
        causal=True,
        window_size=(self.window_size, -1),
      )
    else:
      assert kv.shape[0] == q.shape[0] and kv.shape[4] == q.shape[3]
      return flash_attn_kvpacked_func(
        q,
        kv,
        softmax_scale=self.softmax_scale,
        causal=True,
        window_size=(self.window_size, -1),
      )


class LinearResidual(Linear):
  """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

  def forward(self, input: Tensor) -> Tensor:
    return super().forward(input), input
