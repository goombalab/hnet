"""
Run configuration for H-Net training.

A run is described by one YAML file (configs/train/*.yaml) whose keys mirror the
dataclasses below. Any field can be overridden on the command line with dotted
keys, e.g. `parallel.dp_shard=4 seed=1`.

Only fields that are used by existing code live here; each new training piece
(data, optimizer, checkpointing, ...) adds its own section when it lands.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

from omegaconf import OmegaConf


@dataclass
class ParallelConfig:
    """
    Sizes of the device mesh dimensions. Their product must equal the world size.

    - dp_replicate: data parallel with full model replicas (the outer dim of HSDP).
    - dp_shard: data parallel with params/grads/optimizer state sharded (FSDP2).
      -1 means "all ranks not used by the other dims".
    - cp: context parallel (one sequence split across ranks). Not implemented yet.
    - tp: tensor parallel (layer matrices split across ranks). Not implemented yet.
    """

    dp_replicate: int = 1
    dp_shard: int = -1
    cp: int = 1
    tp: int = 1


@dataclass
class TrainConfig:
    run_name: str = "debug"
    seed: int = 0
    parallel: ParallelConfig = field(default_factory=ParallelConfig)


def load_config(
    path: Optional[str] = None, overrides: Sequence[str] = ()
) -> TrainConfig:
    """
    Build a TrainConfig from defaults, then a YAML file, then command-line overrides
    (later sources win). Unknown keys and wrong types raise an error instead of being
    silently ignored.
    """
    cfg = OmegaConf.structured(TrainConfig)
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(cfg)
