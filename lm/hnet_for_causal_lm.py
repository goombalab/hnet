from collections import namedtuple
from dataclasses import dataclass

from flash_attn.utils.generation import GenerationMixin
from torch import Tensor, arange, int, tensor
from torch.nn import Linear, Module
from torch.nn.modules.sparse import Embedding

from hnet.modules.dc import RoutingModuleOutput
from lm.hnet_config import HnetConfig

from .hnet import HNet, HNetState


@dataclass
class CausalLmOutput:
    logits: Tensor
    bpred_output: list[RoutingModuleOutput]
    inference_params: HNetState


class HnetForCausalLm(Module, GenerationMixin):
    config: HnetConfig
    embeddings: Embedding
    backbone: HNet
    lm_head: Linear

    def __init__(
        self,
        config: HnetConfig,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.config = config

        vocab_size = self.config.vocab_size
        d_embed = self.config.d_model[0]

        # We consider the HNet as a map (B, L, D[0]) -> (B, L, D[0])
        # Thus, the embedding is defined outside of the HNet.
        self.embeddings = Embedding(
            vocab_size, d_embed, device=device, dtype=dtype
        )

        self.backbone = HNet(
            config=config,
            # We pass in the stage_idx as an HNet needs to know what
            # depth of the hierarchy it is in.
            stage_idx=0,
            device=device,
            dtype=dtype,
        )
        self.lm_head = Linear(
            d_embed, vocab_size, bias=False, device=device, dtype=dtype
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def allocate_inference_cache(
        self, batch_size, max_seqlen, dtype=None, **kwargs
    ):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.embeddings(input_ids)

        B, L, D = hidden_states.shape

        assert position_ids is None, (
            "Position ids are not supported for HNet due to the subsampling hierarchical structure"
        )

        if mask is None:
            # Absent a mask, we assume we are running in packed mode
            assert inference_params is None, (
                "Inference params are not supported in packed mode"
            )
            hidden_states = hidden_states.flatten(0, 1)
            cu_seqlens = arange(B + 1, device=hidden_states.device) * L
            max_seqlen = tensor(L, dtype=int, device=hidden_states.device)
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states, bpred_output = self.backbone(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params,
            **mixer_kwargs,
        )

        hidden_states = hidden_states.view(B, L, D)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple(
            "CausalLMOutput", ["logits", "bpred_output", "inference_params"]
        )
        return CausalLMOutput(
            logits=lm_logits,
            bpred_output=bpred_output,
            inference_params=inference_params,
        )

    def step(self, input_ids, inference_params):
        B = input_ids.shape[0]
        assert B == 1, (
            "HNetForCausalLM step currently only supports batch size 1 -- need to handle different-size lengths for each sample"
        )

        hidden_states = self.embeddings(input_ids)

        hidden_states, bpred_output = self.backbone.step(
            hidden_states, inference_params
        )
        logits = self.lm_head(hidden_states)

        return CausalLmOutput(
            logits=logits,
            bpred_output=bpred_output,
            inference_params=inference_params,
        )
