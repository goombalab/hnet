import re
from functools import partial
from dataclasses import dataclass

import torch
from torch import nn, Tensor as TT, nested
import torch.nn.functional as F

### Borrowed kernels/modules
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.modules import mamba2
from flash_attn.ops.triton.layer_norm import RMSNorm
from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn import flash_attn_varlen_func


# https://github.com/state-spaces/mamba/issues/740
mamba2.mamba_split_conv1d_scan_combined = torch.compiler.disable(mamba2.mamba_split_conv1d_scan_combined)

### ################
### Extended NJT ops
### ################
NJT = lambda ls: nested.nested_tensor(ls, layout=torch.jagged)
from torch.nested._internal.ops import (
    register_jagged_func, normalize_function, raggedness_matches,
    _wrap_jagged_dim, extract_kwargs, NestedTensor
)
@register_jagged_func(
    torch.ops.aten.slice_backward.default,
    "grad_output: jt, input_sizes: any, "
    "dim: any?, start: any?, end: any?, step: any?"
)
def slice_backward_jt(func, *a, **k):
    # push all args to kwargs
    _, kw = normalize_function(func, args=a, kwargs=k, normalize_to_only_use_kwargs=True)

    # ensure the slice was on a static dim
    grad = kw.pop("grad_output")
    isz = kw.pop('input_sizes')
    assert (rdim := grad._ragged_idx) == 1, "need the njt's dynamic dim to be 1"
    assert raggedness_matches(grad, isz), "slice backward required different ragged dim"

    # obtain the size of the input grad
    kw['input_sizes'] = [grad._values.shape[0]] + isz[2:]
    # normalize the provided dim && input_sizes for flattened values()
    kw["dim"] = _wrap_jagged_dim(grad.dim(), kw["dim"], grad._ragged_idx, "slice_backward")
    # run the real dense kernel on the packed buffer
    grad_vals = func(grad._values, **kw)
    # re‑wrap so autograd sees a NestedTensor again
    return NestedTensor(grad_vals, **extract_kwargs(grad))

# override _unsafe_view to fast-return when size is identical
@register_jagged_func(
    [torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default],
    "self: jt_all, size: any",
)
def view_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")
    # https://github.com/pytorch/pytorch/blob/d76323d41742cbc05ec6857319b267d2c7ea8fd9/torch/csrc/autograd/VariableTypeUtils.h#L119
    if tuple(inp._size) == tuple(size): return inp.detach() # if view is same sizes, just return same tensor
    if inp._ragged_idx != 1 and tuple(inp._size) != tuple(size): raise RuntimeError()
    # Ensure specified size still includes batch and ragged dims
    if len(size) < 3 or not raggedness_matches(inp, size):
        raise RuntimeError(f"view(): cannot view shape {inp._size} as {size}")
    def get_inner_size(inner_idx):
        nonlocal inp, size
        return inp._values.size(inner_idx) if inner_idx == inp._ragged_idx - 1 else size[inner_idx + 1]
    inner_size = [get_inner_size(i) for i in range(len(size) - 1)]
    with torch.inference_mode(inp.is_inference()):
        return NestedTensor(func(inp._values, inner_size), **extract_kwargs(inp))


###############################
### Config/dataclass defintions
###############################

from config_hnet import HNetConfig, get_stage_cfg
def get_seq_idx(cu_seqlens: TT, flatlen: int) -> TT:
    seq_idx = torch.zeros(flatlen, dtype=torch.int, device=cu_seqlens.device)
    seq_idx[cu_seqlens[1:-1]] = 1 # TODO: ensure this doesn't cpu sync pls
    return seq_idx.cumsum(0)[None].int() # do NOT promote to long, or kernel will crash

@dataclass(frozen=True)
class SequenceRouting:
    p: TT # (B,j1) probability of selecting byte
    b: TT # (B,j1) boolean label for whether byte was selected
    p_selected: TT # (B,j2) filtered probabilities of selected bytes

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.ones_like(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output
def ste_func(x): return STE.apply(x)


### ##########################
### Simplified Isotropic Block
### ##########################

Lin = partial(nn.Linear, bias=False)

class SwiGLU(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.fc1 = Lin(d,2*h)
        self.fc2 = Lin(h,d)
    def forward(self, x: TT):
        h,g = self.fc1(x).chunk(2,dim=-1)
        return self.fc2(F.silu(g) * h)

class RotaryNeoX:
    @staticmethod
    def rotary_cache(base: float, dim: int, msl: int):
        inv_freq = 1. / base**(torch.arange(0,dim,2,dtype=torch.float32)/dim)
        t = torch.arange(msl, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        return freqs.cos(), freqs.sin()

    @staticmethod
    def apply_rope_unbatched(shd: TT, cos: TT, sin: TT):
        d_rope = cos.shape[1]*2
        s = shd.shape[-3]
        assert shd.ndim == 3 and d_rope <= shd.shape[-1]

        x, unmodified = shd[...,:d_rope], shd[...,d_rope:]
        x1,x2 = x.chunk(2,dim=-1)
        cos,sin = cos[:s,None].repeat(1,1,2), sin[:s,None].repeat(1,1,2)

        return torch.cat([
            x*cos + torch.cat([-x2,x1],-1)*sin,
            unmodified
        ], dim=-1)

    @staticmethod
    def apply_rope_njt(bjhd: TT, cos: TT, sin: TT):
        # NOTE: do NOT try to do this out-of-place && construct nested_tensor_from_jagged(...)
        # you will run into missing `_min_seqlen_tensor` from AOTAutograd meta.attrs
        return apply_rotary_emb(
            bjhd.values(), cos, sin, cu_seqlens=bjhd.offsets(),
            interleaved=False, inplace=True, max_seqlen=cos.shape[0]
        )

class CausalMHA(nn.Module):
    def __init__(self, d: int, num_heads: int, rotary_emb_dim: int, window_size: int = -1):
        super().__init__()
        self.num_heads = num_heads
        self.d_head,_r = divmod(d,num_heads)
        self.window_size = (window_size, 0)
        assert _r == 0
        self.Wqkv = Lin(d, d*3)
        self.out_proj = Lin(d,d)
        rope_cache = RotaryNeoX.rotary_cache(10000.0, rotary_emb_dim, 2048)
        self.register_buffer('rope_cache', torch.stack(rope_cache), persistent=False)
    def forward(self, x: TT, cu_seqlens: None | TT = None, max_seqlen: None | int = None):
        if x.layout == torch.jagged:
            q,k,v = self.Wqkv(x).view(x.shape[0], -1, 3*self.num_heads, self.d_head).chunk(3,-2)
            # inplace:
            RotaryNeoX.apply_rope_njt(q, *self.rope_cache)
            RotaryNeoX.apply_rope_njt(k, *self.rope_cache)
            # broken: https://github.com/pytorch/pytorch/issues/155421
            # q,k,v = map(lambda t: t.transpose(1,2), [q,k,v])
            # o = F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2)
            o = nested.nested_tensor_from_jagged(
                flash_attn_varlen_func(
                    q.values(), k.values(), v.values(),
                    q.offsets().int(), k.offsets().int(),
                    q._get_max_seqlen(), k._get_max_seqlen(), 
                    window_size=self.window_size
                ), q.offsets()
            )
        else:
            assert cu_seqlens is not None and max_seqlen is not None
            qk,v = self.Wqkv(x).unflatten(-1, (-1, self.d_head)).split(2*self.num_heads, dim=-2)
            qk = apply_rotary_emb(qk, *self.rope_cache, cu_seqlens=cu_seqlens, interleaved=False, inplace=False, max_seqlen=max_seqlen)
            o = flash_attn_varlen_func(
                *qk.split(self.num_heads, dim=-2), v,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                window_size=self.window_size
            )
        return self.out_proj(o.view(*x.shape))

class Block(nn.Module):
    def __init__(self, d: int, mixer_cls, mlp_cls):
        super().__init__()
        self.residual_in_fp32 = True
        self.norm1 = RMSNorm(d, eps=1e-5)
        self.mixer = mixer_cls(d)
        self.norm2 = RMSNorm(d, eps=1e-5) if mlp_cls is not nn.Identity else None
        self.mlp = mlp_cls(d)

    def forward(self, x: TT, residual: TT | None, cu_seqlens: TT, max_seqlen: int, seq_idx: TT):
        # x, residual = self.norm1(x, residual=residual, prenorm=True, residual_in_fp32=True)
        x = F.rms_norm(residual:=x if residual is None else x+residual, x.shape[-1:], self.norm1.weight, self.norm1.eps)
        if isinstance(self.mixer, mamba2.Mamba2): x = self.mixer(x[None], seq_idx=seq_idx)[0]
        else: x = self.mixer(x, cu_seqlens.int(), max_seqlen)
        if self.norm2 is None: return x, residual
        x = F.rms_norm(residual:=x+residual, x.shape[-1:], self.norm2.weight, self.norm2.eps)
        return self.mlp(x), residual

    def forward_njt(self, x: TT, residual: TT | None):
        # Original NJT-based impl
        residual = x if residual is None else x+residual
        x = F.rms_norm(residual, x.shape[-1:], self.norm1.weight, self.norm1.eps)
        if isinstance(self.mixer, mamba2.Mamba2):
            seqlens = x.offsets().diff()
            p_pad = nested.to_padded_tensor(x, padding=0.0)
            p_pad = self.mixer(p_pad)
            p_njt = nested.narrow(p_pad, dim=1, start=torch.zeros_like(seqlens), length=seqlens, layout=torch.jagged)
            # we now have the correct njt result. but its backing .values() is still padded, which is problematic downstream
            # what's needed is an njt that has matching seqdim's NestedIntNode && has matching underlying storage stride
            # p_flat = p_njt.contiguous().values() <-- fails to https://github.com/pytorch/pytorch/issues/145837

            # note that contiguous() implicitly just calls a dumb for-loop around jagged_from_list:
            # https://github.com/pytorch/pytorch/blob/9d184bda2f190a3ba72a4a0d95e1fde26d6bfc08/torch/nested/_internal/ops.py#L535-L542
            p_flat = torch.cat(p_njt.unbind()) # so this is already "more efficient than torch"

            # removing min_seqlen causes missing attrs in autograd. removing max_seqlen causes seqlen to explode exponentially
            x = nested.nested_tensor_from_jagged(p_flat,x.offsets(),min_seqlen=x._get_min_seqlen(),max_seqlen=p_pad.shape[-2])
        else: x = self.mixer(x) # njt sdpa is natively supported
        if self.norm2 is None: return x, residual
        x = F.rms_norm(residual := x + residual, x.shape[-1:], self.norm2.weight, self.norm2.eps)
        return self.mlp(x), residual

    @classmethod
    def create(cls, arch: str, d: int, h: int, ssm_cfg: dict, attn_cfg: dict, layer_idx: int):
        mixer_cls = dict(
            t=partial(CausalMHA, **attn_cfg),
            m=partial(mamba2.Mamba2, **ssm_cfg, layer_idx=layer_idx)
        )[arch.lower()]
        mlp_cls = partial(SwiGLU, h=h) if arch.isupper() else nn.Identity
        
        return cls(d, mixer_cls, mlp_cls)

class Isotropic(nn.Module):
    def __init__(self, c: HNetConfig, arch: str, stage_idx: int):
        super().__init__()
        self.d = c.d_model[stage_idx]
        self.h = c.d_intermediate[stage_idx]
        self.ssm_cfg = get_stage_cfg(c.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(c.attn_cfg, stage_idx)
        assert self.d * self.ssm_cfg['expand'] / 64 % 8 == 0, "https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940"

        i = -1
        self.layers = nn.ModuleList([
            Block.create(arch, self.d, self.h, ssm_cfg=self.ssm_cfg, attn_cfg=self.attn_cfg, layer_idx=(i:=i+1))
            for arch, n_layer in re.findall(r"([mMtT])(\d+)", arch)
            for _ in range(int(n_layer))
        ])
        self.rmsnorm = RMSNorm(self.d, eps=1e-5)

    def forward_flat(self, x: TT, *, res=None):
        # We unwrap NJT in each isotropic block to prioritize work with varlen&fused kernels
        x, cu_seqlens, max_seqlen = x.values(), x.offsets(), x._get_max_seqlen()
        seq_idx = get_seq_idx(cu_seqlens, x.shape[0])
        for l in self.layers: x, res = l(x,res,cu_seqlens,max_seqlen,seq_idx)
        x = self.rmsnorm(x, residual=res, prenorm=False, residual_in_fp32=True)
        return nested.nested_tensor_from_jagged(x, cu_seqlens)

    def forward_njt(self, x: TT, *, res=None):
        for l in self.layers: x, res = l.forward_njt(x,res)
        return F.rms_norm(x, x.shape[-1:], self.rmsnorm.weight, self.rmsnorm.eps)

    def forward(self, x: TT):
        return self.forward_flat(x)
        # return self.forward_njt(x) # <-- this path is numerically wrong and full of compile errors

    def flops_per_token(self, msl: int):
        # we exclude scalar flops from this.
        d,c_s = self.d, self.ssm_cfg
        mlp = 2*d*self.h*3
        # https://arxiv.org/pdf/2507.07955#page=35
        d_ssm = c_s['expand']*d 
        ssm_heads = d_ssm // 64
        mamba = (
            2 * d * 2 * c_s['expand'] * d +
            2 * d * (2*c_s['d_state'] + ssm_heads) +
            2 * 3 * d_ssm * c_s['d_state'] +
            2 * d * msl +
            2 * d * d
        )
        attn = 4*d*(msl+2*d)
        return sum(
            (attn if isinstance(l.mixer, CausalMHA) else mamba) +
            (0 if isinstance(l.mlp, nn.Identity) else mlp)
            for l in self.layers
        )

### ################
### H-Net submodules
### ################

class ComputePaddedQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: TT, w: TT, k: TT):
        assert x.layout == torch.jagged == k.layout and x.is_nested and k.is_nested
        x_flat, x_cu = x.values(), x.offsets()
        slen = x_flat.shape[0]

        # compute x@w.T, but padded left by 1seqlen
        q_padded = torch.empty(slen+1, *x_flat.shape[1:], dtype=k.dtype, device=x_flat.device)
        torch.mm(x_flat, w.T, out=q_padded[1:]) # k is now <empty>,q0,q1,...,q{t-1},qt,...
        ctx.save_for_backward(x_flat,w)

        # overwrite all q[:,0] with k[:,0]
        kvec = k.values()[k.offsets()[:-1]] # all 0th vectors in k
        q_flat = q_padded.index_copy_(0, x_cu[:-1], -kvec)[:slen]
        # final q is now k0,q0,q1,...,q{t-1},k0,...
        return nested.nested_tensor_from_jagged(q_flat, x_cu)

    @staticmethod
    def backward(ctx, dq):
        # since k is only used as a proxy to get p_0=1.0, k's grad should be None here,
        # and x.grad/w.grad should only be contributed to by non-x_cu positions
        dq_flat, x_cu = dq.values(), dq.offsets()
        # delete all grad contributions from seq position 0
        zero_grad = torch.zeros(x_cu.shape[0]-1, dq_flat.shape[-1], device=dq_flat.device, dtype=dq_flat.dtype)
        dq_flat = dq_flat.index_copy(0, x_cu[:-1], zero_grad) # do NOT in-place

        x_flat, w = ctx.saved_tensors
        dx_flat = torch.zeros_like(x_flat)
        torch.mm(dq_flat[1:], w, out=dx_flat[:-1])
        dw = dq_flat[1:].mT@x_flat[:-1]

        return nested.nested_tensor_from_jagged(dx_flat, x_cu), dw, None


class RoutingModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.q_proj_layer = Lin(d,d)
        self.k_proj_layer = Lin(d,d)
        # https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py#L49
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d))
            self.k_proj_layer.weight.copy_(torch.eye(d))
        self.q_proj_layer.weight._no_reinit = True

    # reference impl for a normal B S D padded tensor.
    def forward_padded(self, x: TT):
        q,k = self.q_proj_layer(x[:, :-1]), self.k_proj_layer(x[:, 1:])
        cos_sim = F.cosine_similarity(q,k,dim=-1) # [-1,1]
        p = (.5-cos_sim/2).clamp(.0,1.) # rescale to [0,1]
        p = F.pad(p, (1,0), 'constant', 1.) # insert p_0 = 1.0
        b = p >= .5
        return SequenceRouting(p, b, p.masked_select(b))

    # njt impl
    def forward_jagged(self, x: TT):
        k = self.k_proj_layer(x)
        q = ComputePaddedQ.apply(x, self.q_proj_layer.weight, k)

        cos_sim = F.cosine_similarity(q.values(), k.values(), dim=-1)
        p = (.5-cos_sim/2).clamp(.0,1.)
        p = nested.nested_tensor_from_jagged(p, x.offsets())
        b = p >= .5
        # ps = p.masked_select(b) <-- bugged because the bwd for aten.masked_select invokes .size() on an njt
        ps = nested.nested_tensor_from_jagged(
            p.values().masked_select(b.values()),
            F.pad(b.values().cumsum(0),(1,0))[p._offsets]
        ) # https://github.com/pytorch/pytorch/blob/51a708ffc679b13f99e4c7cf19bc00082a3266a6/torch/nested/_internal/ops.py#L2455-L2456
        return SequenceRouting(p, b, ps)

    def forward(self, x: TT):
        return self.forward_jagged(x) if x.is_nested else self.forward_padded(x)


class DeChunkLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # for EMA scan kernel.
        self.block_size = 256
        self.headdim = 32
        self.nheads,_r = divmod(d, self.headdim)
        assert _r == 0
        A = -torch.ones(self.nheads, device='cuda', dtype=torch.float32)
        self.register_buffer('A', A, persistent=False)

    def ema_scan_batched(self, x: TT, p: TT):
        dt = (1/(1-p.float())).log()[...,None]
        x = (x.float()/dt).to(x.dtype)
        c = torch.ones_like(p)

        return mamba_chunk_scan_combined(
            x.unflatten(-1, (self.nheads, self.headdim)),
            dt.expand(-1,-1,self.nheads).to(x.dtype),
            self.A,
            p[...,None,None],
            c[...,None,None],
            chunk_size=self.block_size,
            seq_idx=None,
        ).view(x.shape)

    def ema_scan_njt(self, x: TT, p: TT):
        dt = -torch.log1p(-p.float())[...,None]
        x = (x.float()/dt).type_as(x)
        c = torch.ones_like(p := p.type_as(x).values()[None,:,None,None])
            
        return mamba_chunk_scan_combined(
            x.values().view(1, -1, self.nheads, self.headdim),
            dt.values().expand(-1, self.nheads).to(x.dtype)[None],
            self.A, p, c,
            chunk_size=self.block_size, seq_idx=get_seq_idx(x.offsets(), x.values().shape[0]),
        )[0].view(-1, x.shape[-1])

    # def forward(self, x: TT, bpred: SequenceRouting, *, eps=4e-3):
    def forward(self, x: TT, bpred: SequenceRouting, *, eps=1e-4):
        # log1p(-1) == inf == x/log1p(0); so p must be clamped away from 0/1
        p = bpred.p_selected.float().clamp(eps,1-eps)
        # 1-bf16[0.01111110.1111111] = 1-.99609375 ~= .004, so eps=4e-3

        z_bar_flat = self.ema_scan_njt(x,p) # njt -> flat values tensor
        inner2outer_idx = bpred.b.values().cumsum(0)-1

        return nested.nested_tensor_from_jagged(
            z_bar_flat.index_select(0, inner2outer_idx),
            bpred.b.offsets()
        )

### #################
### Final HNet Module
### #################
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# The original HNet uses fp32, so I follow them. But I also enforce tf32 mm.
# https://github.com/goombalab/hnet/blob/main/hnet/models/hnet.py#L101
class HighPrecLinear(nn.Linear):
    def __init__(self, in_features, out_features, device=None):
        super().__init__(in_features, out_features, True, device, torch.float32)
        nn.init.zeros_(self.weight)
        self.weight._no_reinit = True
    def forward(self, x: TT): return super().forward(x.to(self.weight.dtype))

class HNet(nn.Module):
    def __init__(self, c: HNetConfig, stage_idx: int):
        super().__init__()
        self.stage_idx = stage_idx
        self.d = c.d_model[stage_idx]
        try: self.n = c.N_compress[stage_idx+1]/c.N_compress[stage_idx]
        except IndexError: self.n = None

        arch_layout = c.arch_layout
        for _ in range(stage_idx): arch_layout = arch_layout[1]

        assert len(arch_layout) in [3,1]
        self.is_innermost = len(arch_layout) == 1

        if self.is_innermost:
            self.main_network = Isotropic(c, arch_layout[0], stage_idx=stage_idx) # <-- don't increment
        else:
            self.encoder = Isotropic(c, arch_layout[0], stage_idx=stage_idx)
            self.main_network = HNet(c, stage_idx+1)
            self.decoder = Isotropic(c, arch_layout[2], stage_idx=stage_idx)

            self.routing_module = RoutingModule(self.d)
            self.dechunk_layer = DeChunkLayer(self.d)
            self.residual_proj = HighPrecLinear(self.d,self.d)

        d_gain = self.d - c.d_model[stage_idx-1] if stage_idx else None
        self.pad_dimension = nn.Parameter(torch.zeros(d_gain)) if d_gain else None

    # only compile blocks within a hnet, not the hnet itself
    def block_compile(self):
        def compile_isotropic(m: Isotropic):
            for i,l in enumerate(m.layers):
                m.layers.register_module(str(i), torch.compile(l, backend="inductor"))

        if self.is_innermost: return compile_isotropic(self.main_network)
        compile_isotropic(self.encoder)
        compile_isotropic(self.decoder)
        self.main_network.block_compile()
        self.register_module('routing_module',torch.compile(self.routing_module,backend='inductor'))
        # TODO: fix torch compile for these modules:
        # self.register_module('residual_proj', torch.compile(self.residual_proj, backend='inductor'))
        # self.register_module('dechunk_layer', torch.compile(self.dechunk_layer, backend='inductor'))

    def ratio_loss(self, bpred: SequenceRouting):
        assert self.n, "HNetConfig did not receive valid N_compress; please edit it"
        l = bpred.b.numel()
        f = bpred.b.values().sum().float() / l
        g = bpred.p.values().float().sum() / l

        # interpreting the loss as motivated by MoE LBL,
        # u have N-1 "drop token" experts, and 1 "keep token" expert
        drop_experts = self.n*(1-f)*(1-g) / (self.n-1)
        keep_expert = self.n*f*g
        return keep_expert + drop_experts

    def forward(self, x: TT):
        d_orig = x.shape[-1]
        x = x if self.pad_dimension is None else nested.nested_tensor_from_jagged(
            torch.cat([x.values(), self.pad_dimension.expand(x.values().shape[0], -1)], dim=-1),
            x.offsets()
        )

        if self.is_innermost: return self.main_network(x)[...,:d_orig],0,[]

        r = self.encoder(x).type_as(x)

        # calculate b/p
        bpred = self.routing_module(r)
        # select chunk-bytes from r.
        h = nested.nested_tensor_from_jagged(
            r.values()[bpred.b.values()], bpred.p_selected.offsets()
        )

        # compute main chunks && dechunk to outer seqlen
        h,r_loss,comp_ratio = self.main_network(h)
        comp_ratio.append(bpred.p_selected.numel() / bpred.p.numel())
        x = self.dechunk_layer(h, bpred)

        # residual add, decode to output
        c = torch.where(bpred.b, bpred.p, 1-bpred.p)[...,None]
        x = (self.residual_proj(r) + x.float()*ste_func(c)).type_as(x)
        return self.decoder(x)[...,:d_orig], r_loss+self.ratio_loss(bpred), comp_ratio

class HNetLM(nn.Module):
    def __init__(self, c: HNetConfig):
        super().__init__()
        self.c, v, d = c, c.vocab_size, c.d_model[0]
        self.embeddings = nn.Embedding(v,d)
        self.backbone = HNet(c, stage_idx=0)
        self.lm_head = Lin(d,v)

    def forward(self, iids: TT):
        assert iids.is_nested and iids.ndim == 2
        x = self.embeddings(iids)
        x,*others = self.backbone(x)
        return (self.lm_head(x),*others)

    def flops(self, slen: int, msl: int):
        all_d,v = self.c.d_model,self.c.vocab_size

        # llm
        emb = 2*slen*v*all_d[0]
        lmh = 2*slen*v*all_d[0]
        # outer hnets: residual & routing mod have 3 dxd linears
        aux = 2*slen*sum(3*d*d for d in all_d[:-1])
        # isotropics
        iso = sum(
            slen*m.flops_per_token(msl)
            for m in self.modules()
            if isinstance(m, Isotropic)
        )

        # NOTE: we do not account for heavy scalar costs (including dechunk layer) here
        return emb+lmh+aux+iso

    @staticmethod
    def random_data(msl: int, *, s_min=256, s_max=1024):
        # samples = [
        #     torch.randint(256,(l,), dtype=torch.int, device='cuda')
        #     for l in torch.randint(s_min,s_max,(bsz,))
        # ]
        import random
        samples,total = [],0
        while True:
            i = random.randint(s_min,s_max)
            t = torch.randint(256,(i,), dtype=torch.int, device='cuda')
            if i+total > msl:
                iids = NJT([s[:-1] for s in samples])
                lbls = NJT([s[1: ] for s in samples]).long()
                yield (iids,lbls)
                samples,total = [t],i
            else:
                samples.append(t)
                total += i


if __name__ == "__main__":
    __import__("rich.traceback").traceback.install(show_locals=True)
    from contextlib import contextmanager
    from tqdm import tqdm
    from pathlib import Path
    from torch.profiler import schedule, profile, ProfilerActivity, record_function
    def tqdm_with_step(prof: profile, iters: int, **k):
        for i in tqdm(range(iters)):
            yield i; prof.step()
    @contextmanager
    def profiler_setup(path_ct: Path, iters: int):
        path_ct.mkdir(parents=True, exist_ok=True)
        sched = schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=3)
        activ = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        def handler(p: profile): p.export_chrome_trace(str(path_ct / f"step-{p.step_num}.json"))
        with profile(activities=activ, schedule=sched, on_trace_ready=handler, with_stack=True) as prof:
            yield tqdm_with_step(prof, iters)

    #### e2e
    from config_hnet import AttnConfig
    c = HNetConfig.load_config('./configs/hnet_2stage_small.json', N_compress=[1,3,9])
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('cuda'): m = HNetLM(c).bfloat16()
    torch.set_default_dtype(torch.float32)

    # test in eager first
    dl = m.random_data(8192)
    with torch.autograd.detect_anomaly(False):
        iids, lbls = next(dl)
        logits,r_loss,*_ = m(iids)
        loss = F.cross_entropy(logits.values(), lbls.values()) 
        (loss+r_loss).backward()
    print("Eager worked.")

    # test compile
    m.backbone.block_compile()
    with profiler_setup(Path(f'./chrometrace-2stage_S-compiled'), 30) as g:
        for i in g:
            iids,lbls = next(dl)
            torch.cuda.synchronize()
            with record_function("fwd"):
                logits,r_loss,*_ = m(iids)
                loss = F.cross_entropy(logits.values(),lbls.values())+r_loss
            with record_function("bwd"): loss.backward()
            for p in m.parameters(): p.grad = None
    print('compile also worked (apparently)')
    exit()

    #### dechunk verification
    b,s,d = 8,77,128
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)
    m = DeChunkLayer(d)
    x_bsd = torch.randn(b,s,d)
    probs = torch.rand(b,s)
    y_bsd = m.ema_scan_batched(x_bsd,probs)
    x_njt = nested.nested_tensor_from_jagged(x_bsd.flatten(0,1), torch.arange(0,(b+1)*s,s,dtype=torch.int32))
    p_njt = nested.nested_tensor_from_jagged(probs.flatten(0,1), x_njt.offsets())
    y_sd = m.ema_scan_njt(x_njt,p_njt)
    print(y_bsd.flatten(0,1)-y_sd)
    exit()


    #### mha verification
    D = 1024
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    mha = CausalMHA(D, 8, 64)

    x = nested.nested_tensor([torch.randn(17,D), torch.randn(31,D)], requires_grad=True, layout=torch.jagged)
    mha(x).sum().backward()
    # this will crash on 2.7.1 https://github.com/pytorch/pytorch/issues/155421
    mha = torch.compile(mha)
    x = nested.nested_tensor([torch.randn(17,D), torch.randn(31,D)], requires_grad=True, layout=torch.jagged)
    mha(x).sum().backward()
    x = nested.nested_tensor([torch.randn(33,D), torch.randn(11,D), torch.randn(15,D)], requires_grad=True, layout=torch.jagged)
    print('done')
    exit()

    #### Verification for RoutingModule

    # init
    D = 128
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    rm = RoutingModule(D)

    # try nested fwdbwd
    x = nested.nested_tensor([torch.randn(17,D), torch.randn(31,D)], requires_grad=True, layout=torch.jagged)
    p = rm(x).p
    p.sum().backward()
    print(p.values()) 
    print(x.grad.values(), rm.q_proj_layer.weight.grad, rm.k_proj_layer.weight.grad)
    # tensor[48] bf16 x∈[0.410, 1.000] μ=0.512 σ=0.110 grad NestedGetValuesBackward0 cuda:0
    # tensor[48, 128] bf16 n=6144 (12Kb) x∈[-0.020, 0.021] μ=4.435e-05 σ=0.005 cuda:0 tensor[128, 128] bf16 n=16384 (32Kb) x∈[-0.198, 0.241] μ=0.001 σ=0.046 cuda:0 tensor[128, 128] bf16 n=16384 (32Kb) x∈[-0.181, 0.212] μ=0.000 σ=0.046 cuda:0

    # try padded fwdbwd
    x.grad = rm.q_proj_layer.weight.grad = rm.k_proj_layer.weight.grad = None
    with torch.no_grad(): x = nested.to_padded_tensor(x, padding=.0)
    x = x.requires_grad_() # <-- reconstructed leaf tensor with padding
    p = rm(x).p
    p = torch.cat([p[0,:17], p[1,:31]])
    p.sum().backward()
    x_grad = torch.cat([x.grad[0,:17], x.grad[1,:31]])
    print(p)
    print(x_grad         , rm.q_proj_layer.weight.grad, rm.k_proj_layer.weight.grad)
    # tensor[48] bf16 x∈[0.410, 1.000] μ=0.512 σ=0.110 grad CatBackward0 cuda:0
    # tensor[48, 128] bf16 n=6144 (12Kb) x∈[-0.020, 0.021] μ=4.411e-05 σ=0.005 cuda:0 tensor[128, 128] bf16 n=16384 (32Kb) x∈[-0.198, 0.241] μ=0.001 σ=0.046 cuda:0 tensor[128, 128] bf16 n=16384 (32Kb) x∈[-0.180, 0.212] μ=0.000 σ=0.046 cuda:0

    exit()
    ####
