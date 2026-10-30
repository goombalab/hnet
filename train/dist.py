"""
Distributed setup for H-Net training: process group, device, and device mesh.

Supported launchers:
- Slurm `srun`, one task per GPU (the Mila way). Rank info comes from
  SLURM_PROCID / SLURM_LOCALID / SLURM_STEP_NUM_TASKS.
- torchrun. Rank info comes from RANK / LOCAL_RANK / WORLD_SIZE.
- Neither: a single process (world size 1).

NOTE: we read SLURM_STEP_NUM_TASKS, not SLURM_NTASKS. In a batch script with
--ntasks=4, a plain `python main.py` (no srun) still sees SLURM_NTASKS=4 and would
wait forever for 3 peers. SLURM_STEP_NUM_TASKS is only set inside an srun step.

The process group is initialized with tcp://MASTER_ADDR:MASTER_PORT, as the Mila
docs recommend for multi-node jobs (env:// can deadlock on the shared filesystem).
The launcher script must export MASTER_ADDR and MASTER_PORT when world size > 1.
"""

import math
import os
import socket
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from train.config import ParallelConfig

# Outer to inner. Ranks that are adjacent in the innermost dims (tp, cp) end up on the
# same node, which is what tensor/context parallelism need (lots of communication).
MESH_DIM_NAMES = ("dp_replicate", "dp_shard", "cp", "tp")


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    mesh: DeviceMesh

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _rank_info() -> tuple[int, int, int]:
    """Returns (rank, local_rank, world_size) from the launcher's environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # torchrun
        return (
            int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )
    if "SLURM_STEP_NUM_TASKS" in os.environ:  # inside an srun step
        return (
            int(os.environ["SLURM_PROCID"]),
            int(os.environ["SLURM_LOCALID"]),
            int(os.environ["SLURM_STEP_NUM_TASKS"]),
        )
    return 0, 0, 1


def _select_device(local_rank: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # With --gpus-per-task, Slurm makes each task see only its own GPU (index 0).
    # With --gres=gpu:N, every task sees all N GPUs and picks one by local rank.
    index = local_rank if torch.cuda.device_count() > 1 else 0
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def resolve_mesh_shape(parallel: ParallelConfig, world_size: int) -> tuple[int, ...]:
    """Mesh sizes in MESH_DIM_NAMES order, with dp_shard=-1 filled in."""
    dp_shard = parallel.dp_shard
    if dp_shard == -1:
        others = parallel.dp_replicate * parallel.cp * parallel.tp
        if world_size % others != 0:
            raise ValueError(
                f"world size {world_size} is not divisible by "
                f"dp_replicate * cp * tp = {others}"
            )
        dp_shard = world_size // others
    shape = (parallel.dp_replicate, dp_shard, parallel.cp, parallel.tp)
    if math.prod(shape) != world_size:
        raise ValueError(
            f"mesh {dict(zip(MESH_DIM_NAMES, shape))} has {math.prod(shape)} ranks, "
            f"but the world size is {world_size}"
        )
    return shape


def init_distributed(parallel: ParallelConfig, timeout_s: int = 600) -> DistInfo:
    """
    Initializes the default process group and builds the device mesh.

    timeout_s bounds how long any collective may wait before erroring out, so a dead
    rank fails the job instead of hanging it until the Slurm time limit.
    """
    rank, local_rank, world_size = _rank_info()
    device = _select_device(local_rank)

    if world_size > 1:
        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    else:
        # A process group is still needed for DeviceMesh / DTensor, even alone.
        init_method = f"tcp://127.0.0.1:{_free_port()}"

    dist.init_process_group(
        backend="nccl" if device.type == "cuda" else "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )

    mesh = init_device_mesh(
        device.type,
        resolve_mesh_shape(parallel, world_size),
        mesh_dim_names=MESH_DIM_NAMES,
    )
    return DistInfo(rank, local_rank, world_size, device, mesh)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
