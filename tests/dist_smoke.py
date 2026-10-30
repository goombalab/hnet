"""
Smoke test for train/dist.py. Run it as 1 process, or as N processes under srun:

    python tests/dist_smoke.py parallel.dp_replicate=2      # inside: srun --ntasks=4 ...

Checks, on every rank:
1. the mesh has the expected shape and dim names;
2. an all-reduce over the whole world sums the ranks correctly;
3. an all-reduce over each mesh dim only talks to that dim's group
   (the sum equals the sum of the ranks in this rank's group).
"""

import sys

import torch
import torch.distributed as dist

from train.config import load_config
from train.dist import MESH_DIM_NAMES, cleanup_distributed, init_distributed, resolve_mesh_shape


def main() -> None:
    cfg = load_config(overrides=sys.argv[1:])
    info = init_distributed(cfg.parallel, timeout_s=60)
    mesh = info.mesh

    assert mesh.mesh_dim_names == MESH_DIM_NAMES
    assert tuple(mesh.shape) == resolve_mesh_shape(cfg.parallel, info.world_size)

    # 2. whole world
    t = torch.tensor([float(info.rank)], device=info.device)
    dist.all_reduce(t)
    expected = info.world_size * (info.world_size - 1) / 2
    assert t.item() == expected, (t.item(), expected)

    # 3. each mesh dim separately
    for name in MESH_DIM_NAMES:
        group_ranks = mesh[name].mesh.tolist()  # global ranks in this rank's group
        t = torch.tensor([float(info.rank)], device=info.device)
        dist.all_reduce(t, group=mesh.get_group(name))
        assert t.item() == sum(group_ranks), (name, t.item(), group_ranks)

    coords = dict(zip(MESH_DIM_NAMES, mesh.get_coordinate()))
    print(f"rank {info.rank}/{info.world_size} local {info.local_rank} on {info.device}: {coords} OK", flush=True)

    dist.barrier()
    if info.is_main:
        print(f"PASSED: mesh {dict(zip(MESH_DIM_NAMES, mesh.shape))}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
