import re
from functools import partial
from dataclasses import dataclass

import torch
from torch import nn, Tensor as TT
import torch.nn.functional as F

from einops import repeat, rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


######################################################
### Borrowed content (to be simplified in the future)
######################################################

# Config/dataclass defintions
from hnet.modules.utils import get_stage_cfg
from hnet.modules.isotropic import IsotropicInferenceParams
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.models.hnet import RoutingModuleState, DeChunkState, HNetState

# Well-studied/familiar architectures
from hnet.modules.block import CausalMHA, Mamba2Wrapper, SwiGLU, RMSNorm

# extra overrides
class HNetState(HNetState):
    def get_innermost_seqlen(self):
        m = self.main_network_state
        if isinstance(m,HNetState): return m.get_innermost_seqlen()
        return None if m is None else m.seqlen_offset

@dataclass(frozen=True)
class SequenceRouting:
    p: TT # (1,S) probability of selecting byte
    b: TT # (1,S) boolean label for whether byte was selected
    p_selected: TT # (L,) filtered probabilities of selected bytes

### ##########################
### Simplified Isotropic Block
### ##########################

Lin = partial(nn.Linear, bias=False)

class Block(nn.Module):
    def __init__(self, d: int, mixer_cls, mlp_cls, norm_cls):
        super().__init__()
        self.residual_in_fp32 = True
        self.norm1 = norm_cls(d)
        self.mixer = mixer_cls(d)
        self.norm2 = norm_cls(d) if mlp_cls is not nn.Identity else None
        self.mlp = mlp_cls(d)

    def forward(self, x: TT, residual: TT | None, inference_params):
        x, residual = self.norm1(x, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
        x = self.mixer(x, inference_params=inference_params)
        if self.norm2 is not None:
            x, residual = self.norm2(x, residual, True, self.residual_in_fp32)
        return self.mlp(x), residual

    def allocate_inference_cache(self, max_seqlen):
        return self.mixer.allocate_inference_cache(1, max_seqlen, dtype=torch.get_default_dtype())

    def step(self, x: TT, res: TT | None, inference_params):
        x, res = self.norm1(x, res, True, self.residual_in_fp32)
        x = self.mixer.step(x, inference_params)
        if self.norm2 is not None:
            x, res = self.norm2(x, res, True, self.residual_in_fp32)
        return self.mlp(x), res

    @classmethod
    def create(cls, arch: str, d: int, h: int, ssm_cfg: dict, attn_cfg: dict, layer_idx: int):
        factk = dict(device='cuda', dtype=torch.bfloat16) # <-- can be deleted after we reimpl mha/mamba
        mixer_cls = dict(
            t=partial(CausalMHA, **attn_cfg, **factk, layer_idx=layer_idx),
            m=partial(Mamba2Wrapper, **ssm_cfg, **factk, layer_idx=layer_idx)
        )[arch.lower()]
        norm_cls = partial(RMSNorm, eps=1e-5)
        mlp_cls = partial(SwiGLU, d_intermediate=h) if arch.isupper() else nn.Identity
        
        return cls(d, mixer_cls, mlp_cls, norm_cls)

class Isotropic(nn.Module):
    def __init__(self, c: HNetConfig, arch: str, stage_idx: int):
        super().__init__()
        self.d = c.d_model[stage_idx]
        self.h = c.d_intermediate[stage_idx]
        self.ssm_cfg = get_stage_cfg(c.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(c.attn_cfg, stage_idx)

        i = -1
        self.layers = nn.ModuleList([
            Block.create(arch, self.d, self.h, ssm_cfg=self.ssm_cfg, attn_cfg=self.attn_cfg, layer_idx=(i:=i+1))
            for arch, n_layer in re.findall(r"([mMtT])(\d+)", arch)
            for _ in range(int(n_layer))
        ])
        self.rmsnorm = RMSNorm(self.d, eps=1e-5) # this can probably be a torch rmsnorm instead

    def allocate_inference_cache(self, max_seqlen):
        return IsotropicInferenceParams(
            # this is a dict instead of list because mha downstream does a membership check .-.
            key_value_memory_dict={
                i:l.allocate_inference_cache(max_seqlen)
                for i,l in enumerate(self.layers)
            },
            max_seqlen=max_seqlen,
            max_batch_size=1,
        )
    def forward(self, x, inference_params, *, res=None):
        for l in self.layers: x, res = l(x,res,inference_params)
        inference_params.seqlen_offset += x.shape[-2]
        return self.rmsnorm(x, res, prenorm=False, residual_in_fp32=True)

    def step(self, x, inference_params, *, res=None):
        for l in self.layers: x,res = l.step(x, res, inference_params)
        inference_params.seqlen_offset += 1
        return self.rmsnorm(x, res, prenorm=False, residual_in_fp32=True)


### ################
### H-Net submodules
### ################

class RoutingModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.q_proj_layer = Lin(d,d)
        self.k_proj_layer = Lin(d,d)
    def allocate_inference_cache(self):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(1, dtype=torch.bool),
            last_hidden_state=torch.zeros(1, self.d),
        )
    def forward(self, x, inference_params):
        inference_params.has_seen_tokens[:] = 1
        inference_params.last_hidden_state.copy_(x[:,-1])
        q,k = self.q_proj_layer(x[:, :-1]), self.k_proj_layer(x[:, 1:])
        cos_sim = F.cosine_similarity(q,k,dim=-1) # [-1,1]
        p = (.5-cos_sim/2).clamp(.0,1.) # rescale to [0,1]
        p = F.pad(p, (1,0), 'constant', 1.) # insert p_0 = 1.0
        b = p >= .5
        return SequenceRouting(p, b, p.masked_select(b))

    def step(self, x, inference_params):
        q,k = self.q_proj_layer(inference_params.last_hidden_state), self.k_proj_layer(x)
        inference_params.last_hidden_state.copy_(x[:,-1])

        cos_sim = F.cosine_similarity(q,k,dim=-1) # [-1,1]
        p = (.5-cos_sim/2).clamp(.0,1.)
        b = p >= .5
        return SequenceRouting(p, b, p.masked_select(b))

class DeChunkLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        # for EMA scan kernel.
        self.dtype = torch.bfloat16
        self.block_size = 256
        self.headdim = 32
        self.nheads,_r = divmod(d, self.headdim)
        assert _r == 0
        A = -torch.ones(self.nheads, device='cuda', dtype=torch.float32)
        self.register_buffer('A', A, persistent=False)

    def ema_scan(self, x: TT, p: TT):
        dt = torch.log(1/(1-p)).to(self.dtype)
        x = (x/dt[...,None]).to(self.dtype)
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            self.A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
            seq_idx=None,
        )
        return rearrange(out, "b l h p -> b l (h p)")

    def allocate_inference_cache(self):
        return DeChunkState(last_value = torch.zeros(1, self.d))

    def forward(self, x: TT, bpred: SequenceRouting, inference_params, *, eps=1e-4):
        p = bpred.p_selected.float().clamp(eps,1-eps) # clamp p away from 0/1 (idk why)
        z_bar = self.ema_scan(x,p[None])

        plug_back_idx = torch.cumsum(bpred.b, dim=1) - 1 # finds chunk idx of each byte pos
        out = torch.gather(z_bar, dim=1, index=plug_back_idx.unsqueeze(-1).expand(-1,-1,self.d))
        inference_params.last_value.copy_(out[:, -1])

        return out.to(x.dtype)

    def step(self, x: TT, bpred, inference_params):
        # return last_value if no new chunk happened
        prev = inference_params.last_value[:,None]
        if x.shape[-2] == 0: return prev
        # otherwise, compute 1 EMA step
        z_t = bpred.p*x + (1-bpred.p)*prev
        inference_params.last_value.copy_(z_t[:,-1])
        return z_t


### #################
### Final HNet Module
### #################

class HNet(nn.Module):
    def __init__(self, c: HNetConfig, stage_idx: int):
        super().__init__()
        self.stage_idx = stage_idx
        self.d = c.d_model[stage_idx]

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
            self.residual_proj = nn.Linear(self.d,self.d) # NOTE: even though this is fp32 in the source code, I allow this to become lower precision.

        d_gain = self.d - c.d_model[stage_idx-1] if stage_idx else None
        self.pad_dimension = nn.Parameter(torch.zeros(d_gain)) if d_gain else None

    def allocate_inference_cache(self, max_seqlen):
        if self.is_innermost: return HNetState(
            main_network_state=self.main_network.allocate_inference_cache(max_seqlen)
        )
        return HNetState(
            encoder_state=self.encoder.allocate_inference_cache(max_seqlen),
            routing_module_state=self.routing_module.allocate_inference_cache(),
            main_network_state=self.main_network.allocate_inference_cache(max_seqlen),
            dechunk_state=self.dechunk_layer.allocate_inference_cache(),
            decoder_state=self.decoder.allocate_inference_cache(max_seqlen),
        )

    def forward(self, x: TT, inference_params: HNetState):
        # pad to current dim if needed
        d_orig = x.size(-1)
        x = x if self.pad_dimension is None else torch.cat(
            [x, self.pad_dimension.expand(*x.shape[:-1], -1)], dim=-1
        )

        # early exit if innermost
        if self.is_innermost: return self.main_network(x, inference_params.main_network_state)[...,:d_orig]

        # encoded residual
        r = self.encoder(x, inference_params.encoder_state)

        # calculate b/p
        bpred = self.routing_module(r, inference_params.routing_module_state)
        # select chunk-bytes from r. must be batch size 1 here
        assert r.shape[0] == 1
        h = r[:,bpred.b[0]]

        # compute main chunks && dechunk to outer seqlen
        h = self.main_network(h, inference_params.main_network_state)
        x = self.dechunk_layer(h, bpred, inference_params.dechunk_state)

        # residual add, decode to output
        x = self.residual_proj(r) + x
        return self.decoder(x, inference_params.decoder_state)[...,:d_orig]

    def step(self, x: TT, inference_params: HNetState):
        d_orig = x.size(-1)
        x = x if self.pad_dimension is None else torch.cat(
            [x, self.pad_dimension.expand(*x.shape[:-1], -1)], dim=-1
        )

        if self.is_innermost: return self.main_network.step(x, inference_params.main_network_state)[...,:d_orig]

        r = self.encoder.step(x, inference_params.encoder_state)
        bpred = self.routing_module.step(r, inference_params.routing_module_state)

        h = r[:,bpred.b[0]] # this is shaped [1,1,d] or [1,0,d]
        h = self.main_network.step(h, inference_params.main_network_state) if h.numel() else h
        x = self.dechunk_layer.step(h, bpred, inference_params.dechunk_state)

        x = self.residual_proj(r) + x
        return self.decoder.step(x, inference_params.decoder_state)[...,:d_orig]

class HNetLM(nn.Module):
    def __init__(self, c: HNetConfig):
        super().__init__()
        self.c, v, d = c, c.vocab_size, c.d_model[0]
        self.embeddings = nn.Embedding(v,d)
        self.backbone = HNet(c, stage_idx=0)
        self.lm_head = Lin(d,v)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype):
        assert batch_size == 1, dtype == torch.bfloat16
        with torch.device("cuda"):
            _default = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try: return self.backbone.allocate_inference_cache(max_seqlen)
            finally: torch.set_default_dtype(_default)

    def forward(self, iids: TT, inference_params: HNetState, *, mask=None):
        assert iids.ndim == 2 and iids.shape[0] == 1
        if mask is not None: assert mask.all()
        x = self.embeddings(iids)
        x = self.backbone(x, inference_params) # IP is modified inplace.
        return self.lm_head(x)

    def step(self, iids: TT, inference_params: HNetState):
        assert iids.ndim == 2 and iids.shape[0] == 1 == iids.shape[1]
        x = self.embeddings(iids)
        x = self.backbone.step(x, inference_params)
        return self.lm_head(x)
