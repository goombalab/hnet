from collections import namedtuple
from dataclasses import dataclass

import torch
import torch.nn as nn

from flash_attn.utils.generation import GenerationMixin

from .hnet import HNet, HNetState
from .config_hnet import HNetConfig

from hnet.modules.dc import RoutingModuleOutput
from hnet.modules.utils import apply_optimization_params

@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    bpred_output: list[RoutingModuleOutput]
    inference_params: HNetState


class HNetForCausalLM(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: HNetConfig,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config

        vocab_size = self.config.vocab_size
        d_embed = self.config.d_model[0]
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        # We consider the HNet as a map (B, L, D[0]) -> (B, L, D[0])
        # Thus, the embedding is defined outside of the HNet.
        self.embeddings = nn.Embedding(vocab_size, d_embed, **factory_kwargs)

        self.backbone = HNet(
            config=config,
            # We pass in the stage_idx as an HNet needs to know what
            # depth of the hierarchy it is in.
            stage_idx=0,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False, **factory_kwargs)
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
    
    def init_weights(self, initializer_range: float = 0.02) -> None:
        """
        Initializes the weights of the model.
        """
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=initializer_range)
        # embeddings are initialized differently from linears
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)
        self.backbone._init_weights(initializer_range)

    def apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        """
        Applies the learning rate multipliers to the parameters of the model.
        NOTE: Must be ran before creating parameter groups, see hnet.utils.train.group_params for an example on how to run parameter groups.

        Inputs:
            lr_multiplier: A list of learning rate multipliers, one for each stage of the hierarchy, with the outer stages first (e.g. [3.0, 1.7, 0.9]).
        """
        for param in self.embeddings.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        for param in self.lm_head.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        self.backbone._apply_lr_multiplier(lr_multiplier)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
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

        assert (
            position_ids is None
        ), "Position ids are not supported for HNet due to the subsampling hierarchical structure"

        if mask is None:
            # Absent a mask, we assume we are running in packed mode
            assert (
                inference_params is None
            ), "Inference params are not supported in packed mode"
            hidden_states = hidden_states.flatten(0, 1)
            cu_seqlens = torch.arange(B + 1, device=hidden_states.device) * L
            max_seqlen = torch.tensor(L, dtype=torch.int, device=hidden_states.device)
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
        assert (
            B == 1
        ), "HNetForCausalLM step currently only supports batch size 1 -- need to handle different-size lengths for each sample"

        hidden_states = self.embeddings(input_ids)

        hidden_states, bpred_output = self.backbone.step(
            hidden_states, inference_params
        )
        logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits, bpred_output=bpred_output, inference_params=inference_params
        )
