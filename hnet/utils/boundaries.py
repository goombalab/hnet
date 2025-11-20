import torch

from hnet.models.mixer_seq import HNetForCausalLM

@torch.inference_mode()
def get_boundaries(model: HNetForCausalLM, input_ids: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    hidden_states = model.embeddings(input_ids)
    hnet_inner = model.backbone

    hierarchies = len(model.config.d_model) - 1
    boundaries = []
    for _ in range(hierarchies):
        if hnet_inner.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, hnet_inner.pad_dimension.expand(hidden_states.shape[:-1] + (-1,))),
                dim=-1,
            )
        hidden_states = hnet_inner.encoder(hidden_states, mask=mask)
        bpred_output = hnet_inner.routing_module(hidden_states, mask=mask)
        boundaries.append(bpred_output.boundary_mask.squeeze(0))
        hidden_states, _, _, mask = hnet_inner.chunk_layer(hidden_states, bpred_output.boundary_mask, mask=mask)

        hnet_inner = hnet_inner.main_network

    return boundaries
