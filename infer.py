import torch
from omegaconf import ListConfig
from torch import bfloat16

from lm.byte_tokenizer import ByteTokenizer
from lm.hnet_config import HnetConfig
from lm.hnet_for_causal_lm import HnetForCausalLm


def generate(
  model,
  prompt: str,
  max_tokens: int = 1024,
  temperature: float = 1.0,
):
  device = next(model.parameters()).device
  tokenizer = ByteTokenizer()

  encoded = tokenizer.encode(prompt, add_bos=True)
  input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(
    0
  )

  inference_cache = model.allocate_inference_cache(
    1, input_ids.shape[1] + max_tokens, dtype=torch.bfloat16
  )

  mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
  output = model.forward(input_ids, mask=mask, inference_params=inference_cache)

  logits = output.logits[0, -1, :] / temperature

  for _ in range(max_tokens):
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1)

    if next_token.item() == tokenizer.eos_idx:
      break

    current_token = next_token.unsqueeze(0)
    yield current_token

    output = model.step(current_token, inference_cache)
    logits = output.logits[0, -1, :] / temperature


def main():
  model_path = "model/hnet_1stage_L.pt"
  config_path = "config/hnet_1stage_L.json"
  prompt = "sun is"
  max_tokens = 32
  temperature = 0.0001

  config = HnetConfig.load(config_path)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  dtype = bfloat16

  model = HnetForCausalLm(config, device=device, dtype=dtype)
  model.eval()

  with torch.serialization.safe_globals([ListConfig]):
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
  model.load_state_dict(state_dict)

  tokenizer = ByteTokenizer()

  buf = []
  for token in generate(
    model,
    prompt,
    max_tokens=max_tokens,
    temperature=temperature,
  ):
    buf.append(token)

    decoded = None
    res = None
    for j in range(1, min(len(buf), 4)):
      try:
        res = tokenizer.decode(buf[:j])
        decoded = j
      except:
        pass

    if res is not None:
      print(res, end="", flush=True)
      buf = buf[decoded:]


main()
