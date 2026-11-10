import argparse
import json
import math
import os
import time

import numpy as np
import torch

from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.utils.train import load_balancing_loss


def build_config(path):
    cfg = json.load(open(path))
    attn_cfg = AttnConfig(**cfg.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**cfg.pop("ssm_cfg"))
    return HNetConfig(**cfg, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)


def sample_batch(arr, batch, seq_len, rng, device):
    out = np.empty((batch, seq_len + 1), dtype=np.uint8)
    hi = len(arr) - seq_len - 2
    for i in range(batch):
        start = int(rng.integers(0, hi))
        out[i] = arr[start:start + seq_len + 1]
    x = torch.from_numpy(out[:, :seq_len].astype(np.int64)).to(device)
    y = torch.from_numpy(out[:, 1:].astype(np.int64)).to(device)
    return x, y


def main():
    ap = argparse.ArgumentParser(description="Train an H-Net on a flat byte corpus")
    ap.add_argument("--config", required=True, help="HNetConfig JSON, as in configs/")
    ap.add_argument("--train-bin", required=True, help="flat uint8 corpus")
    ap.add_argument("--val-bin", required=True, help="flat uint8 held-out corpus")
    ap.add_argument("--out", required=True, help="checkpoint path; existing file resumes")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--router-weight", type=float, default=0.01)
    ap.add_argument("--target-ratio", type=float, default=4.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = HNetForCausalLM(build_config(args.config), device=args.device, dtype=torch.bfloat16)
    for module in model.modules():
        if hasattr(module, "use_mem_eff_path"):
            module.use_mem_eff_path = False
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params {nparams / 1e6:.1f}M", flush=True)

    data = np.fromfile(args.train_bin, dtype=np.uint8)
    val = np.fromfile(args.val_bin, dtype=np.uint8)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if (name.endswith(".bias") or ".norm." in name) else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    start_step = 0
    if os.path.exists(args.out):
        ck = torch.load(args.out, map_location=args.device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed at {start_step}", flush=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    def evaluate():
        model.eval()
        total, count = 0.0, 0
        rng_val = np.random.default_rng(1234)
        with torch.inference_mode():
            for _ in range(args.val_batches):
                x, y = sample_batch(val, args.batch, args.seq_len, rng_val, args.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                total += torch.nn.functional.cross_entropy(
                    out.logits.reshape(-1, 256).float(), y.reshape(-1), reduction="sum"
                ).item()
                count += y.numel()
        model.train()
        return total / count / math.log(2)

    rng = np.random.default_rng(0)
    started = time.time()
    for step in range(start_step, args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x, y = sample_batch(data, args.batch, args.seq_len, rng, args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            lm = torch.nn.functional.cross_entropy(out.logits.reshape(-1, 256).float(), y.reshape(-1))
            balance = sum(load_balancing_loss(r, args.target_ratio) for r in out.bpred_output) / len(out.bpred_output)
            loss = lm + args.router_weight * balance
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if step % args.log_every == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - started
            print(f"step {step} lm {lm.item():.4f} router {balance.item():.3f} lr {lr_at(step):.2e} "
                  f"bytes/s {args.batch * args.seq_len * (step - start_step + 1) / max(elapsed, 1e-9):.0f}",
                  flush=True)
        if step and step % args.val_every == 0:
            print(f"VAL step {step} bits/byte {evaluate():.4f}", flush=True)
        if step and step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, args.out)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": args.steps}, args.out)
    print(f"done {nparams / 1e6:.1f}M", flush=True)


if __name__ == "__main__":
    main()
