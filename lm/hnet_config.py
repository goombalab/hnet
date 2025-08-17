import json
from dataclasses import dataclass, field
from typing import List

from typing_extensions import Self


@dataclass
class AttnConfig:
  num_heads: List = field(default_factory=list)
  rotary_emb_dim: List = field(default_factory=list)
  window_size: List = field(default_factory=list)


@dataclass(eq=False)
class SsmConfig:
  d_conv: int = 4
  expand: int = 2
  d_state: int = 128
  chunk_size: int = 256


@dataclass
class HnetConfig:
  arch_layout: List[str | List] = field(default_factory=list)
  d_model: List[int] = field(default_factory=list)
  d_intermediate: List[int] = field(default_factory=list)
  vocab_size: int = 256
  ssm_cfg: SsmConfig = field(default_factory=SsmConfig)
  attn_cfg: AttnConfig = field(default_factory=AttnConfig)
  tie_embeddings: bool = False

  @classmethod
  def load(cls, path: str) -> Self:
    """Load configuration from file"""
    with open(path, "r") as f:
      config = json.load(f)

    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SsmConfig(**config.pop("ssm_cfg"))
    hnet_cfg = cls(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    return hnet_cfg
