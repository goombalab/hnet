# Ad-hoc ugly training script. Do not use for anything serious.


# argparse first to make --help ASAP. please do not isort
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', type=str, default='./configs/hnet_2stage_small.json')
ap.add_argument('-p', '--pt-ckpt', type=str, default=None, help='e.g. ./hnet_1stage_L.pt')
ap.add_argument('-N', '--n-compression', type=str, default='1-3-9', help='''
compression depth to target with L_ratio. this is a bit different from the paper's notation;
n_compression = [1,3,9] -> N = [3/1, 9/3]
''')
ap.add_argument('-e', '--early-exit', choices=['generate', 'dumpparams'], help='''
Early-exit helpers. Use this to quickly test something && exit the script thereafter.
  generate: attempt generation via loaded ckpt.
  dumpparams: show model & all params
''')
ap.add_argument('-l', '--logger', default='local', choices=['wandb', 'local', 'neptune'])
ap.add_argument('-C', '--compile', choices=['block', 'eager'], help='attempt torch compile')
ap.add_argument('-o', '--optim', default='adamw', choices=['adamw', 'sgd'])
ap.add_argument('--lr', type=float, default=3e-4, help='adamw learning rate')
ap.add_argument('--mbs', type=int, default=1<<14, help='maximum microbatchsize (tokens per gpu)')
ap.add_argument('--steps', type=int, default=1000, help='total train steps')
ap.add_argument('--save-dir', type=str, help='overwriting output path to save checkpoints')
# n.b. about 76k steps required for 16ktok * 8gpu to reach 10Btok
args = ap.parse_args()





import os
import sys
import time
import functools
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from typing import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor as TT, distributed as dist
from torch.distributed import device_mesh as tdm, fsdp, checkpoint as dcp
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.checkpoint import state_dict as dcps

from omegaconf import ListConfig
from termcolor import colored

from hnet_trainable import HNetLM, NJT, HNetConfig
from fineweb import seqlen_sorted_fineweb
from comparison import ByteTokenizer, generate, HNetLM as HNetLMInference, yield_utf8_chunks


###
### distributed 
def parent_codeline():
    parent = __import__("inspect").stack()[2]
    return f'{parent.filename}:{parent.lineno} -> {parent.code_context[0].strip()}'
def leave(): (dist.destroy_process_group() if dist.is_initialized() else None),exit() # noqa
def distdbg(): # `for f in distdbg():f()`
    if dist.get_rank() == 0:
        print('[distdbg]', parent_codeline())
        yield __import__('pdbpp').set_trace
    yield dist.barrier
def printflock(*args, fcntl=__import__('fcntl'), builtins=__import__('builtins'), **kwargs):
    r = dist.get_rank() if dist.is_initialized() else 0
    __import__("time").sleep(r*0.05) 
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try: builtins.print(f'[{r}]', *args, **kwargs)
        finally: fcntl.flock(fh, fcntl.LOCK_UN)
    if dist.is_initialized(): dist.barrier()
def pr0(*a, **k): 0 if dist.get_rank() else print(*a,**k)
def dist_equal(t: TT, g: dist.ProcessGroup):
    x = torch.zeros(g.size(), *t.shape, device=t.device, dtype=t.dtype)
    dist.all_gather_into_tensor(x, t[None], group=g)
    return all(torch.equal(x[i], t) for i in range(g.size()))
def assert_rng_equal(g: dist.ProcessGroup):
    if not dist_equal(torch.cuda.get_rng_state(), g):
        printflock(torch.cuda.get_rng_state())
        printflock('unequal! exiting at ' + parent_codeline())
        leave()
@contextmanager
def summon_full_params(model: FSDPModule):
    handles = [
        m.unshard(async_op=True)
        for m in reversed(list(model.modules()))
        if isinstance(m, FSDPModule)
    ]
    for h in handles: h.wait() if h is not None else 0

    yield

    for m in reversed(list(model.modules())):
        if isinstance(m, FSDPModule): m.reshard()


###
### tokenizer / configs
n_compression = [int(s) for s in args.n_compression.split('-')]
t = ByteTokenizer()
c = HNetConfig.load_config(args.config, N_compress=n_compression)
def load_ckpt(m: nn.Module, path: str | None):
    if path is None: return # do not do anything
    with torch.serialization.safe_globals([ListConfig]):
        d = torch.load(path, mmap=True, weights_only=False)
    m.load_state_dict(d)
def decode_sync(g): return t.decode([t for t,_ in g])

###
### Memory-free training module -> inference module copy
def create_inference_model_clone(m_trn: HNetLM, c: HNetConfig):
    # spawn inf model without storage
    with torch.device('meta'): m_inf = HNetLMInference(c).bfloat16().eval()

    # ensure there is no module naming mismatch
    md_trn = dict(m_trn.named_modules())
    md_inf = dict(m_inf.named_modules())
    missing_from_inf = md_trn.keys()-md_inf.keys()
    missing_from_trn = md_inf.keys()-md_trn.keys()
    missing_from_trn = {k for k in missing_from_trn if not any(s in k for s in ('rotary_emb','inner_cross_attn','inner_attn'))}
    assert not missing_from_inf and not missing_from_trn, (missing_from_inf, missing_from_trn)

    # copy all parameter refs
    from flash_attn.ops.triton.layer_norm import RMSNorm
    for k in md_trn.keys()&md_inf.keys():
        md_inf[k]._parameters = md_trn[k]._parameters
        if isinstance(md_inf[k], RMSNorm): md_inf[k].bias = None
    for n,p in m_inf.named_parameters(): assert not p.is_meta, n
    return m_inf

def test_real_ref_versus_clone(m_trn, ckpt: str | None):
    if ckpt is None: raise RuntimeError('inference test only makes sense with pretrained ckpt')
    # our model: load to training, copy to inference
    load_ckpt(m_trn := m_trn.bfloat16(), ckpt)
    m_inf = create_inference_model_clone(m_trn,c)
    # external original inference model w/ weights directy loaded
    with torch.device('cuda'): m2inf = HNetLMInference(c).bfloat16().eval()
    load_ckpt(m2inf, ckpt)

    from lovely_tensors import lovely
    # ✔️  prefill  
    torch.manual_seed(0)
    iids = torch.randint(0,256,(1,77),dtype=torch.long,device='cuda')
    mask = torch.ones_like(iids, dtype=torch.bool)
    with torch.inference_mode():
        printflock('>>> [0]', y0 := m_trn(NJT([iids[0]]))[0].values()[None])
        ic = m_inf.allocate_inference_cache(1, iids.size(1), dtype=torch.bfloat16)
        printflock('>>> [1]', y1 := m_inf(iids, mask=mask, inference_params=ic))
        ic = m2inf.allocate_inference_cache(1, iids.size(1), dtype=torch.bfloat16)
        printflock('>>> [2]', y2 := m2inf.forward(iids, mask=mask, inference_params=ic))
    assert y1.equal(y2), "prefill not equiv"
    assert str(lovely(y0))==str(lovely(y2)), "y0 should look like y2 in lovely summary"
    assert y0.allclose(y2,atol=.3), "y0 had outliers very far from y2"
    # ✔️  greedy
    print('>>> [1]', res1 := decode_sync(generate(m_inf, 'Hello world!', 512, 0.0001, 0.0001)))
    print('>>> [2]', res2 := decode_sync(generate(m2inf, 'Hello world!', 512, 0.0001, 0.0001)))
    assert res1 == res2, "greedy not equiv"
    # ✔️  random
    torch.manual_seed(0);print('>>> [1]', res1 := decode_sync(generate(m_inf, 'Hello world!', 512)))
    torch.manual_seed(0);print('>>> [2]', res2 := decode_sync(generate(m2inf, 'Hello world!', 512)))
    assert res1 == res2, "random not equiv"


###
### model
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
torch.manual_seed(0)
with torch.device('cuda'): m = HNetLM(c).eval() # <-- always fp32 cuda weights
print('params:', sum(p.numel() for p in m.parameters()) / 1_000_000_000, 'B')
if args.compile == 'block': m.backbone.block_compile()
if args.early_exit == 'generate': test_real_ref_versus_clone(m,args.pt_ckpt); leave()


###
### dist init
ws = int(os.environ.get("WORLD_SIZE", 0))
assert ws, "Always run script with torchrun, even with only 1GPU."
dist.init_process_group("cpu:gloo,cuda:nccl")
mesh = tdm.init_device_mesh('cuda', (ws,), mesh_dim_names=('dp',))
r = dist.get_rank()

def apply_fsdp(m: HNetLM, dp_mesh: tdm.DeviceMesh):
    assert dp_mesh.ndim == 1

    # prepare fsdp helpers
    kw = dict( # default: BF16, ZeRO2, 1D mesh
        mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        reshard_after_forward=False,
        mesh=dp_mesh,
    )
    def shard_isotropic(iso):
        for l in iso.layers: fsdp.fully_shard(l, **kw)

    # recurse to collect all hnets
    hnets = [m.backbone]
    for s in range(c.S): hnets.append(hnets[-1].main_network)

    ## Sharding
    # special case: the lastmost hnet has a .main_network as Isotropic and nothing else.
    shard_isotropic(hnets.pop().main_network)

    # in general, shard the following:
    for hnet in hnets[::-1]:
        shard_isotropic(hnet.encoder)
        shard_isotropic(hnet.decoder)
        # NOTE: you could optimize this by packing routing & residual into 1module which fsdp fetches.
        # but it would fork the module tree away from original hnet, which I don't accept
        fsdp.fully_shard(hnet.routing_module, **kw)
        # I really do not believe this is necessary, but let's match the authors.
        fsdp.fully_shard(hnet.residual_proj, **kw|{'mp_policy':fsdp.MixedPrecisionPolicy(param_dtype=torch.float32)})

    fsdp.fully_shard(m, **kw) # top-level: .embeddings .lm_head

if ws==1:
    # Since FSDP breaks in various places for 1gpu, we can either AMP or hard-cast.
    # It is impossible to make torch.mm(..., out=...) work with AMP, so I hard-cast.
    m = m.bfloat16()
    # Neither produce the same numerical behavior as FSDP, so try to only use 1gpu for testing.
else: apply_fsdp(m,mesh)

if args.early_exit == 'dumpparams':
    pr0(m)
    for n,p in m.named_parameters(): pr0(n,p)
    leave()

###
### optim/lrs
def lr_modulation(n_gpt: float=4.6): # https://arxiv.org/pdf/2507.07955#page=35
    n_prod_ratio = n_compression[::-1]
    d_ratio = [c.d_model[-1]/d for d in c.d_model]
    return [(4.5 * n_frac * d_frac)**.5 for n_frac,d_frac in zip(n_prod_ratio, d_ratio)]
def split_params_by_hierachy(m: HNetLM) -> list[list[nn.Parameter]]:
    # for each param, count the number of times ".main_network" appears in it.
    d = defaultdict(list)
    for n,p in m.named_parameters(): d[n.count('main_network')].append(p)
    # special-case innermost hnet which has redundant .main_network
    max_depth = max(d.keys())
    assert 1 == len(d[max_depth-1]), f"expected single .pad_dimension at {max_depth-1}"
    d[max_depth-1] += d.pop(max_depth)

    return [d[k] for k in range(len(d))]
lambda_s = lr_modulation()
param_groups = [
    dict(params=plist,lr=args.lr*λˢ)
    for (plist,λˢ) in zip(split_params_by_hierachy(m), lambda_s)
]

def wsd(step: int, end: int):
    pct = step/end
    return pct*10 if pct < .1 else (
        1 if pct < .9 else (1-pct)*10
    )
opt_cls = functools.partial(torch.optim.AdamW, betas=(0.9,0.95), weight_decay=.01) if args.optim == 'adamw' else torch.optim.SGD
o = opt_cls(param_groups, lr=args.lr)
get_lr_mult = functools.partial(wsd, end=args.steps)
lrs = torch.optim.lr_scheduler.LambdaLR(o, get_lr_mult)

###
### logger
def get_logger(variant: str='local') -> Callable[[dict],None]:
    match variant:
        case 'wandb':
            import wandb
            wandb.init(project='hnet', config=asdict(c))
            return wandb.log
        case _: return print
log = get_logger(args.logger) if r==0 else lambda *a,**k:0

###
### checkpointing
def save_ckpt(step: int):
    save_dir = (Path(args.save_dir)/f'{step}')
    pr0(f'saving checkpoint to {save_dir}')
    save_dir.mkdir(exist_ok=True, parents=True)
    ckpt_m,ckpt_o = dcps.get_state_dict(m,o)
    dcp.save(dict(model=ckpt_m, optim=ckpt_o), checkpoint_id=save_dir)


def get_submod(m: nn.Module, k: str) -> tuple[nn.Module, str]:
    if '.' not in k: return m,k
    l,r = k.split('.', 1)
    return get_submod(getattr(m, l), r)

@contextmanager
def obscure_torch_wrapper_modules(m: nn.Module, *, names: list[str] = ["_fsdp_wrapped_module", "_orig_mod", "_checkpoint_wrapped_module"]):
    restore = []
    for n,c in m.named_modules():
        potential_inner = [getattr(c,k,None) for k in names]
        inner = next((p for p in potential_inner if p is not None), None)
        if inner is not None:
            parent, tail = get_submod(m,n)
            restore.append([parent, tail, c, inner])

    # m.{n} == parent.{tail} -> child; child.{k} -> inner
    for parent, tail, child, inner in restore:
        parent.register_module(tail, inner)
    yield
    for parent, tail, child, inner in restore:
        parent.register_module(tail, child)


###
### training


def calc_metrics(logits: TT, labels: TT, numel: TT, *, ln2=torch.tensor(2,device='cuda').log()):
    ce_sum = F.cross_entropy(logits.float(), labels, reduction='sum')
    loss = ce_sum / numel # local loss = (local ce / local numel)

    ce_sum = ce_sum.clone().detach() # do not mess with actual loss autograd
    dist.all_reduce(ce_sum)
    dist.all_reduce(numel)

    return loss, ce_sum/ln2/numel
def train_step(iids, lbls, *, alpha=0.03):
    numel = torch.tensor(lbls.numel(), dtype=torch.long).to('cuda', non_blocking=True)
    logits, loss_rt, comp_ratios = m(iids)
    loss_ce, bpb = calc_metrics(logits.values(), lbls.values(), numel)

    loss = loss_ce+alpha*loss_rt
    loss.backward()

    o.step()
    o.zero_grad()
    lrs.step()

    metrics = torch.stack([loss, loss_ce, loss_rt, bpb])
    return metrics.tolist() + [comp_ratios] # <-- cpu sync

def seq2args(ls: list[bytes]) -> tuple[TT,TT]:
    samples = [torch.tensor(bytearray(b),device='cuda',dtype=torch.int) for b in ls]
    iids = NJT([s[:-1] for s in samples])
    lbls = NJT([s[1: ] for s in samples]).long()
    return iids, lbls

# add your own if needed; there are no other gpus in my observable reality
gpuflops = {
    'NVIDIA GeForce RTX 3090':71e12,
    'NVIDIA GeForce RTX 4090':165.15e12,
    'NVIDIA B200': 2250e12,
}[torch.cuda.get_device_name()]


assert_rng_equal(dist.group.WORLD) # <-- assumes node of homogenous GPUs
pr0('start training')
for step,batch in enumerate(seqlen_sorted_fineweb(r, ws, args.mbs)):
    # train step
    t_step = time.perf_counter()
    iids, lbls = seq2args(batch)
    loss, loss_ce, loss_rt, bpb, comp_ratios = train_step(iids, lbls)

    # calc mfu (underestimate since we don't know avg seqlen)
    batch_flops = m.flops(iids.values().shape[0], iids._max_seqlen)
    t_delta = time.perf_counter() - t_step
    mfu = batch_flops / (gpuflops*t_delta)

    # Log (every 10steps)
    lr_list = [mult*args.lr*get_lr_mult(step) for mult in lambda_s]
    log_step = log if step % 10 == 0 else lambda d:0
    log_step({
        'step': step,
        'batch_flops (overestimate)': batch_flops,
        'mfu (underestimate)': mfu,
        'bpb': bpb,
        'loss/cross': loss_ce,
        'loss/ratio': loss_rt,
        'loss/total': loss,
        **{f'Compression L{i+1}/L{i}':ratio for i,ratio in enumerate(comp_ratios)},
        **{f'lr/{i}':lr for i,lr in enumerate(lr_list)},
    })

    # Try sampling (every 100)
    if step % 100 == 0:
        pr0('generating...')
        with summon_full_params(m) if isinstance(m, FSDPModule) else nullcontext():
            with obscure_torch_wrapper_modules(m):
                m_inf = create_inference_model_clone(m,c)
            p = 'Hello world!'
            with torch.autocast('cuda', torch.bfloat16, cache_enabled=False):
                try: res1 = ''.join(
                    colored(c, 'white' if i%2 else 'black', 'on_black' if i%2 else 'on_white')
                    for i,c in enumerate(yield_utf8_chunks(generate(m_inf, p, 512)))
                )
                except UnicodeDecodeError: res1 = colored('[failed to decode UTF-8]', 'red')
        pr0(colored(p, attrs=['underline']) + res1)
        pr0('='*50)


    # Try saving (every 1000)
    if step % 1000 == 0:
        save_ckpt(step) if args.save_dir else pr0(f"not saving checkpoint as {args.save_dir=}")

    if step == args.steps: break

