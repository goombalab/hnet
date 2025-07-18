# H-Net (simple reference impl)

This repository contains simplified reimplementations of H-Net, for personal understanding.

## Installation
Fresh venv (with torch nightly):

```bash
uv venv --python 3.11
uv pip install setuptools ninja psutil
uv pip install "torch==2.9.0.dev20250715+cu126" --index-url "https://download.pytorch.org/whl/nightly/cu126" 
uv sync
uv sync --extra build --no-install-package triton # build no-isolation deps

# if you have caching issues:
uv pip uninstall mamba_ssm causal_conv1d flash_attn && uv cache clean mamba_ssm causal_conv1d flash_attn && rm uv.lock
uv sync --extra build --no-install-package triton # build no-isolation deps
# or give up:
# uv pip install --no-build-isolation --no-cache  "flash_attn==2.8.0.post2" \
#   "mamba_ssm @ git+https://github.com/state-spaces/mamba.git@a6a1dae6efbf804c9944a0c2282b437deb4886d8" \
#   "causal_conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@e940ead2fd962c56854455017541384909ca669f"

# if you accidentally pull triton:
uv pip uninstall triton pytorch-triton
uv pip install pytorch-triton "torch==2.9.0.dev20250715+cu126" --index-url "https://download.pytorch.org/whl/nightly/cu126" 
```

then download a model to cwd, e.g.
```python
uv run huggingface-cli download --local-dir . cartesia-ai/hnet_2stage_L hnet_2stage_L.pt 
```

## Inference

[`hnet_simple.py`](./hnet_simple.py) is a ~300LOC file that implements the non-isotropic blocks in a H-Net, while borrowing the rest from the original repo.

[`comparison.py`](./comparison.py) is a simple script to compare results with the original repo's impl, where it (mostly) matches.

I do not reimplement the transformer/mamba blocks, as civilization has done that 99999999 times and nobody needs to see another copy of them.

### Running
testing:
```python
$ uv run comparison.py --model hnet_1stage_L.pt --config configs/hnet_1stage_L.json
Loading model...
tensor[1, 17, 256] bf16 n=4352 (8.5Kb) x∈[-20.625, 14.688] μ=-7.469 σ=5.094 cuda:0
tensor[1, 17, 256] bf16 n=4352 (8.5Kb) x∈[-20.500, 14.625] μ=-7.469 σ=5.094 cuda:0
prefill diff: tensor[1, 17, 256] bf16 n=4352 (8.5Kb) x∈[-0.125, 0.125] μ=0.001 σ=0.030 cuda:0
tensor[1, 1, 256] bf16 x∈[-16.125, 5.688] μ=-7.000 σ=5.531 cuda:0
tensor[1, 1, 256] bf16 x∈[-16.125, 5.719] μ=-7.000 σ=5.531 cuda:0
 decode diff: tensor[1, 1, 256] bf16 x∈[-0.062, 0.062] μ=-0.004 σ=0.019 cuda:0
generation: ', programs hello world, programs hello world\nYou s'
```

prompting:

[![asciicast](https://asciinema.org/a/a9EOUrQemZUvAXHBzAqF8f4AX.svg)](https://asciinema.org/a/a9EOUrQemZUvAXHBzAqF8f4AX)

---

# H-Net

<table width="100%">
  <tr>
    <td><img src="assets/english.gif" alt="English" width="100%"></td>
    <td><img src="assets/code.gif" alt="Code" width="100%"></td>
  </tr>
  <tr>
    <td><img src="assets/chinese.gif" alt="Chinese" width="100%"></td>
    <td><img src="assets/korean.gif" alt="Korean" width="100%"></td>
  </tr>
</table>

> **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling**\
> Sukjun Hwang, Brandon Wang, Albert Gu\
> Paper: https://arxiv.org/abs/2507.07955

## About
![H-Net](assets/arch.png "H-Net Architecture")

This repository contains code of the H-Net architecture. Most of the code lies in `hnet/`, which has the following structure:

```
configs/
hnet/
├── models/            # Directory for H-Net
|   ├── config_hnet.py     (defines the config for the H-Net)
|   ├── hnet.py            (h-net as a (B, L, D) -> (B, L, D) sequence model)
│   └── mixer_seq.py       (wrapper to turn h-net into a language model)
└── modules/           # Directory of model components
    ├── dc.py              (modeling code for the dynamic chunking mechanism)
    └── isotropic.py       (code for isotropic, i.e. non-hierarchical components)
generate.py        # Script for inference/generation
```

## Installation

### Requirements:
- PyTorch >= 2.5.1

Clone the repository and install package.
``` sh
git clone https://github.com/goombalab/hnet
cd hnet
pip install -e .
```


We strongly recommend building **mamba_ssm** package from [**the latest source**](https://github.com/state-spaces/mamba) as follows:
``` sh
git clone https://github.com/state-spaces/mamba
cd mamba
pip install .
```

## Pretrained Models

Pretrained models are uploaded to
[Hugging Face](https://huggingface.co/cartesia-ai): `hnet_1stage_L`, `hnet_2stage_L`,
`hnet_1stage_XL`, `hnet_2stage_XL`.
We trained our models on the 100B-Token subset of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). <em>Large</em> and <em>XL</em> are compute-matched to GPT-3 <em>Large</em> and <em>XL</em>, respectively.

We also provide model weights for Chinese and Code, each trained using the 46B-Token subset of [FineWeb-Edu Chinese V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) and [Pile Github](https://huggingface.co/datasets/EleutherAI/pile): `hnet_2stage_XL_chinese`, `hnet_2stage_XL_code`.

You can find specifics of these models at [configs](configs), and more details from the paper.


## Text Generation

We provide [generate.py](generate.py) for text generation that you can use with the pretrained checkpoints.

### Examples
``` sh
python generate.py --model-path [MODEL_CKPT] --config-path [CONFIG]
python generate.py --model-path hnet_2stage_XL.pt --config-path configs/hnet_2stage_XL.json --max-tokens 1024 --temperature 1.0 --top-p 1.0
```


## Citation

If you use this codebase, or otherwise find our work valuable, please cite H-Net:

```
@article{hnet,
  title={Dynamic Chunking for End-to-End Hierarchical Sequence Modeling},
  author={Hwang, Sukjun and Wang, Brandon and Gu, Albert},
  journal={arXiv preprint arXiv:2507.07955},
  year={2025}
}
```
