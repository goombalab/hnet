from mamba_ssm.modules.mamba2 import Mamba2


class Mamba2Wrapper(Mamba2):
  """
  Mamba2 wrapper class that has the same inference interface as the CausalMHA class.
  """

  def next_step(self, hidden_states, inference_params):
    # Don't use _get_states_from_cache because we want to assert that they exist
    conv_state, ssm_state = inference_params.key_value_memory_dict[
      self.layer_idx
    ]  # init class of Mamba2 accepts layer_idx
    result, conv_state, ssm_state = super().step(
      hidden_states,
      conv_state,
      ssm_state,
    )

    # Update the state cache in-place
    inference_params.key_value_memory_dict[self.layer_idx][0].copy_(conv_state)
    inference_params.key_value_memory_dict[self.layer_idx][1].copy_(ssm_state)
    return result
