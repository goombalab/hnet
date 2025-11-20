"""
Script to inspect the boundaries of an H-Net model.
"""

import numpy as np
import json
import torch
import argparse
import sys
from omegaconf import ListConfig

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)
from hnet.utils.boundaries import get_boundaries
from hnet.utils.tokenizers import ByteTokenizer


def load_from_pretrained(model_path: str, model_config_path: str):
    """Load model from pretrained checkpoint.

    Args:
        model_path: Path to the model checkpoint (.pt file)
        model_config_path: Path to the model configuration (.json file)

    Returns:
        Loaded HNetForCausalLM model
    """
    # Load configuration
    with open(model_config_path, "r") as f:
        config = json.load(f)

    # Create config objects
    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=torch.bfloat16)
    model.eval()

    # Load checkpoint
    major, minor = map(int, torch.__version__.split('.')[:2])
    if (major, minor) >= (2, 6):
        with torch.serialization.safe_globals([ListConfig]):
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
    else:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    return model


def main():
    parser = argparse.ArgumentParser(description="Generate text from an H-Net model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the model configuration (.json file)",
    )
    args = parser.parse_args()

    print("Loading model...")
    try:
        model = load_from_pretrained(args.model_path, args.config_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    n_stages = len(model.config.d_model) - 1
    print(f"Model has {n_stages} stages.")

    tokenizer = ByteTokenizer()

    while True:
        text = input("\nText: ").strip()

        if not text:
            continue

        print("Getting boundaries...")
        encoded = tokenizer.encode([text], add_bos=True)[0]
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0)
        mask = torch.ones(input_ids.shape, device="cuda", dtype=torch.bool)

        boundaries = get_boundaries(model, input_ids, mask)

        averages = [x.float().mean().item() for x in boundaries]
        print("Downsampling factors: " + " ".join([f"{a:.3f}" for a in averages]))
        if input_ids.shape[1] != len(text) + 1:
            print("Warning: Alternate unicode bytes not yet supported, skipping text display")
            continue
        print(" " + text)
        mask = [True] * (len(text) + 1)
        for stage in range(n_stages):
            current_boundaries = boundaries[stage].tolist()
            idx = 0
            for i in range(len(mask)):
                if not mask[i]:
                    continue
                if not current_boundaries[idx]:
                    mask[i] = False
                idx += 1
        
            print("".join([str(int(i)) for i in mask]))

if __name__ == "__main__":
    main()
