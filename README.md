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


## Training

`train.py` trains a model from scratch on a flat byte corpus (no tokenizer; the corpus is raw UTF-8 bytes,
the vocabulary is 256). It is the script behind two Finnish runs of 195M and 501M parameters.

``` sh
python train.py --config configs/hnet_1stage_L.json \
                --train-bin data/train.bin --val-bin data/val.bin \
                --out runs/hnet.pt --steps 100000 --batch 16 --seq-len 512
```

The loop is standard: random byte windows, next-byte cross-entropy, plus `load_balancing_loss` on the router
outputs weighted at 0.01, AdamW with weight decay on matrices only, linear warmup then cosine decay, gradient
clipping, bf16 autocast, and held-out bits/byte every 250 steps. Rerunning the same command resumes from
`--out`.

Two practical notes:

- The script sets `use_mem_eff_path = False` on every Mamba2 module. The fused path calls
  `causal_conv1d_fwd_function` directly and produced `nan` losses on ROCm; the non-fused path is stable and
  also faster to compile.
- Throughput is dominated by sequence length and Triton autotuning on the first step of a new shape, not by
  step count. Judge nothing from step 0.

A probe tool that reports chunk-length distributions and boundary alignment (word edges, suffix onsets) lives
at [bqv/hnets](https://github.com/bqv/hnets), together with the ROCm shims needed to run this on non-CUDA hardware.

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