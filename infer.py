from torch import bfloat16, bool, cuda, long, multinomial, ones, softmax, tensor

from lm.byte_tokenizer import ByteTokenizer
from lm.hnet_config import HnetConfig
from lm.hnet_for_causal_lm import HnetForCausalLm


def generate(
  model: HnetForCausalLm,
  prompt: str,
  max_token: int = 1024,
  temperature: float = 1.0,
):
  device = next(model.parameters()).device
  tokenizer = ByteTokenizer()

  tokens_b = tokenizer.encode(prompt, add_bos=True)
  tokens = tensor(tokens_b, dtype=long, device=device).unsqueeze(0)

  inference_cache = model.allocate_inference_cache(
    batch_size=1,
    max_seqlen=tokens.shape[1] + max_token,
    dtype=bfloat16,
  )

  mask = ones(tokens.shape, device=device, dtype=bool)
  output = model.forward(tokens, inference_cache, mask)
  logits = output.logits[0, -1, :] / temperature

  for _ in range(max_token):
    probs = softmax(logits, dim=-1)
    next_token = multinomial(probs, 1)

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

  device = "cuda" if cuda.is_available() else "cpu"
  dtype = bfloat16

  config = HnetConfig.load(config_path)
  model = HnetForCausalLm(config, device, dtype).load(model_path)
  tokenizer = ByteTokenizer()

  tokens = []
  for token in generate(
    model,
    prompt,
    max_tokens,
    temperature,
  ):
    tokens.append(token)

    decoded = None
    size = None
    for i in range(1, min(len(tokens), 4)):
      decoded = tokenizer.decode(tokens[:i])
      size = i

    if decoded is not None:
      print(decoded, end="", flush=True)
      tokens = tokens[size:]


main()
