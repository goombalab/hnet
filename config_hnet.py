from dataclasses import dataclass, field, asdict
import json

def get_stage_cfg(cfg, stage_idx): return {
    k: v[stage_idx] if isinstance(v, list) else v for k, v in asdict(cfg).items()
}

@dataclass(frozen=True)
class AttnConfig:
    num_heads: list = field(default_factory=list)
    rotary_emb_dim: list = field(default_factory=list)
    window_size: list = field(default_factory=list)

@dataclass(frozen=True)
class SSMConfig:
    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256

@dataclass(frozen=True)
class HNetConfig:
    arch_layout: list[str | list] = field(default_factory=list)
    d_model: list[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: list[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False
    N_compress: list[float] = field(default_factory=list) # https://arxiv.org/pdf/2507.07955#page=8
    @property
    def S(self): return len(self.d_model)-1 # paper's definition
    @classmethod
    def load_config(cls, config_path: str, **k) -> 'HNetConfig':
        with open(config_path, "r") as f:
            c = json.load(f)
            attn_cfg = AttnConfig(**c.pop("attn_cfg"))
            ssm_cfg = SSMConfig(**c.pop("ssm_cfg"))
            return cls(**c, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg, **k)


