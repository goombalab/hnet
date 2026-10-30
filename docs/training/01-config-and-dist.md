# 01: Run config and distributed setup

Files: `train/config.py`, `train/dist.py`, `tests/dist_smoke.py`.

## Goal

Every later piece (data, loss, optimizer, checkpointing, parallelism) needs two things:

1. **a config**, so a run is reproducible from one YAML file plus a few overrides;
2. **a distributed context**: which rank this process is, which GPU it owns, and a
   **device mesh** describing how ranks are split across parallelism types.

We build the mesh with all four dimensions from day one, even though only data
parallelism exists for now. Adding tensor or context parallelism later then means
changing sizes in the config and writing the sharding code, not rewriting the
training loop.

## `train/config.py`

- `TrainConfig` is a dataclass tree. For now it holds `run_name`, `seed` and `parallel`.
  Later pieces add their own sections (`data`, `optim`, ...) when they land, so the
  config never contains fields that nothing reads.
- `load_config(path, overrides)` merges, in order: dataclass defaults, a YAML file,
  then command-line overrides like `parallel.dp_shard=4`. It uses OmegaConf
  (already an H-Net dependency) in "structured" mode, so **typos and wrong types
  raise an error** instead of being silently ignored.

## `train/dist.py`

### Mesh dimensions

`MESH_DIM_NAMES = ("dp_replicate", "dp_shard", "cp", "tp")`, outer to inner:

| Dim | Meaning | Status |
|---|---|---|
| `dp_replicate` | full model copies, gradients all-reduced (DDP; outer dim of HSDP) | usable |
| `dp_shard` | params, grads and optimizer state sharded (FSDP2) | usable |
| `cp` | one sequence split across ranks | later |
| `tp` | weight matrices split across ranks | later |

The order matters: ranks next to each other in the inner dims (`tp`, `cp`) land on
the same node, and those dims need the fastest links. `dp_shard=-1` means "whatever
is left", so `parallel.dp_replicate=2` on 8 GPUs gives `dp_shard=4`.

### Finding rank and world size

`_rank_info()` supports three launchers:

- **`srun`** (the Mila way, one task per GPU): `SLURM_PROCID`, `SLURM_LOCALID`,
  `SLURM_STEP_NUM_TASKS`.
- **torchrun**: `RANK`, `LOCAL_RANK`, `WORLD_SIZE`.
- **Neither**: a single process.

**Gotcha, tested on Mila:** inside a batch script with `--ntasks=2`, a plain
`python` call (no `srun`) still sees `SLURM_NTASKS=2` but `SLURM_STEP_NUM_TASKS` is
**unset**. Using `SLURM_NTASKS` would make that process wait forever for a peer that
never starts, so we use `SLURM_STEP_NUM_TASKS`, which only `srun` sets.

### Picking the GPU

- `--gpus-per-task`: Slurm shows each task only its own GPU, so use `cuda:0`.
- `--gres=gpu:N`: each task sees all N GPUs and picks `cuda:<local_rank>`.

### Process group

- Backend: NCCL on GPU, gloo on CPU (used by the tests).
- Initialized with `tcp://MASTER_ADDR:MASTER_PORT`, as the Mila docs recommend for
  multi-node jobs (`env://` can deadlock on the shared filesystem). The sbatch
  script must export both variables when there's more than one process.
- A single process still gets a process group (on a free local port), because
  DeviceMesh and DTensor require one.
- `timeout_s` (default 10 min) makes a dead rank crash the job instead of hanging it
  until the Slurm time limit.

### Mesh

`resolve_mesh_shape()` fills in `dp_shard=-1` and checks that the product of the
sizes equals the world size, failing early with a readable message otherwise.
`init_device_mesh()` then creates one process group per mesh dimension, which later
code gets with `mesh.get_group("dp_shard")` etc.

## Test: `tests/dist_smoke.py`

On every rank it checks: the mesh shape and names; an all-reduce over the whole
world; and an all-reduce over each mesh dimension that must reach only that
dimension's group.

Results on Mila CPU nodes (gloo), job 11460295:

- 1 process: mesh `{dp_replicate: 1, dp_shard: 1, cp: 1, tp: 1}`. **Passed.**
- 4 processes on **2 nodes** (`cn-f[002,004]`) with `parallel.dp_replicate=2`: mesh
  `{dp_replicate: 2, dp_shard: 2, cp: 1, tp: 1}`. All 4 ranks **passed**.
- `parallel.tp=3` on 1 process: fails immediately with
  `ValueError: world size 1 is not divisible by dp_replicate * cp * tp = 3`.

Not yet tested: NCCL on GPUs. That happens when the first GPU piece lands.

How to rerun (CPU, inside a job with `MASTER_ADDR`/`MASTER_PORT` exported):

```bash
PYTHONPATH=$PWD srun uv run --no-project --with omegaconf --with torch \
    --index https://download.pytorch.org/whl/cpu python tests/dist_smoke.py parallel.dp_replicate=2
```
