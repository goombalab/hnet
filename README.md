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

---

## H-Net (training)

This repository contains a simple training implementation of H-Net for langauge modeling.

![](https://main-horse.github.io/posts/hnet-trn/wandb-sample.png)

[`hnet_trainable.py`](./hnet_trainable.py) is my ~600LOC NJT-based impl of a trainable, block-compilable H-Net.

[`train.py`](./train.py) is a dumb handwritten script to train fineweb-10BT on H-Net.

Both of these scripts should be considered an **active WIP**. They are not efficient, but should be a good starting point for other people in the community to start work & fork from.

### test compilation on 1gpu
The following command should exit without errors on your machine:
```bash
uv run hnet_trainable.py
```

If it does not, please provide [`collect_env.py`](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py) information, as I have seen it working on multiple machines:

```bash
$ CUDA_VISIBLE_DEVICES=2 uv run hnet_trainable.py 
/hnet/.venv/lib/python3.11/site-packages/torch/backends/__init__.py:46: UserWarning: This API is going to be deprecated, please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:78.)
  self.setter(val)
/hnet/hnet_trainable.py:601: UserWarning: Anomaly Detection has been enabled. This mode will increase the runtime and should only be enabled for debugging.
  with torch.autograd.detect_anomaly(False):
Eager worked.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [01:03<00:00,  2.11s/it]
compile also worked (apparently)
```

### test fwd equivalence
This command verifies that the training H-Net impl produces the same logits as the inference impl, given a pretrained checkpoint + config.
```bash
LOVELY_TENSORS=1 uv run train.py -e generate -c configs/hnet_1stage_L.json -p hnet_1stage_L.pt
```

You should see no assertion errors if you run it.

### fineweb training
To try training a H-Net on Fineweb-10BT, start by downloading the dataset:
```bash
uv run --with huggingface-hub huggingface-cli download --repo-type dataset HuggingFaceFW/fineweb --include sample/10BT/*
uv run fineweb.py
```
This produces a local dump of `10BT/{seqlen}/{hash}.txt` files. It works on my SSD, but **do not try this on an NFS**, or on any storage device with low lifespans.

To use the dataset to test training with 1 GPU, run the following:
```bash
uv run torchrun --nproc-per-node 1 --standalone train.py
```

That should (after some waiting time) produce output logs like:

![](https://main-horse.github.io/posts/hnet-trn/training_log.jpeg)

Once you have confirmed that works, you can try training for slightly longer, on 8 GPUs:
```bash
LOVELY_TENSORS=1 OMP_NUM_THREADS=9 \
  uv run torchrun --nproc-per-node 8 --standalone train.py \
  --config configs/hnet_2stage_small.json \
  --n-compression 1-3-9 \
  --compile block \
  --steps 9999 \
  --save-dir /tmp/hnet \
  --logger wandb
```

If that also works for you, try training larger networks.

Although I hope this work encourages the broader community to partake in scaling laws analysis with H-Net, I would advise caution against using this codebase immediately, without verifying the correctness of its distributed impl.

### specifics of training

#### General correctness
Due to the inference check above, the forward pass for my training H-Net should be correct.

For the backward pass, there are three uncertainties I would consider:
1. Numerical problems. For example:
  - I intentionally eskew the original H-Net author's decision to use fp32 residuals + amp.autocast, in favor of defaulting to bfloat16. I do this because this is the default behavior of any FSDP2'd module with `nn.Embedding`, and is quite troublesome to fight against.
  - I removed some kernels from the original code for simplicity, but it is not obvious what potential impact this could have on numerical differences over a full training run.
2. Correctness problems.
  - I implement some bespoke ops to support NJT backwards. Although I think they are correct, I am not high confidence.
  - I use a slightly different solution from the authors to implement $p_0=0.0$ padding in computing the routing module's cosine sim. I added tests to verify my approach, but perhaps those are wrong too.
3. torch problems. I do not know if there are secret bugs in `torch.nested`, if dyanmo/aotautograd/inductor introduce silent bugs, or if the version of nightly I pulled is actually bugged.

Despite all of those potential problems, basic training does indicate the model is learning, and not fully broken.

- The celoss/bpb curves above indicate the model is learning.
- The behavior of $L_{ratio}$ matches what the paper says should occur (hovering around 1)
- The compression ratio graphs indicate the model is targetting the `N_compression` ratios assigned in config:

![](https://main-horse.github.io/posts/hnet-trn/comp-ratio.png)

So I think the architecture itself is implemented correctly.

#### NJTs
[Nested Tensors](https://docs.pytorch.org/docs/stable/nested.html) are the most sensible abstraction for HNets of arbitrary depth.

The sequence lengths involved throughout a HNet will always be unavoidably (const batch size) varlen, with those sequence lengths changing throughout the train step. Even if you invented your own bespoke datastructure to track cu_seqlens, it would end up quite similar to the structure of an NJT.

Unfortunately, NJTs are currently [abandonware](https://github.com/pytorch/pytorch/issues/145837#issuecomment-2830826574), with many bugs & key ops missing. I include various modifications in my implementation to ensure the model compiles & trains, but you should not be surprised if ostensibly benign modifications to the code cause pytorch errors to surface.

If you do not like NJTs, I recommend writing your own impl from scratch. Much like DTensors, NJTs are a [colored object](https://journal.stuffwithstuff.com/2015/02/01/what-color-is-your-function/); the code I've written is littered with assumptions of NJT presence.

#### Auxillary training work
From my understanding, the paper needs two extra compute steps during training:
1. **Ratio Loss**, which is implemented under `HNet.ratio_loss` and hardcoded to α=.03 in `train_step`
2. **Learning Rate Modulation**. I attempt to implement their [Equation (11)](https://arxiv.org/pdf/2507.07955#page=35) under `lr_modulation(...)`; empirically this returns something like `[12,6,2]`?

Please let me know if I have missed any significant steps.

#### Inference
I dislike co-locating inference & training code, so I do a pointer copy of weights to my original HNet Inference impl to test sampling where needed.

This necessarily bloats the amount of code you'd have to copy to make the train script work immediately. I recommend deleting the sampling code if you do not need it.

#### Efficiency
This codebase is not efficient. Actually, the MFU is quite bad.

This is primarily because I have not spent the time to optimize the Isotropic blocks, causing their execution profiles to be full of ugly white bubbles (tbd add image)

I would appreciate guidance from the original authors on this topic, as I am sure they have already spent great amounts of time optimizing their own internal training stack.

#### Future scaling
This codebase is NOT scalable beyond basic FSDP. There is no support for meta/distributed init (and much code assumes `__init__` can alloc tensors).

You will have to rewrite the codebase from scratch to implement tp/sp/cp/...

Additionally, there is 0 effort to adjust the init scheme of any modules away from torch defaults. I do this because such information is absent from the paper too, so I presume the defaults _are_ what goombalab used -- with the sole exception of the `.residual_proj`, which is zero-inited, following their code.

---

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
