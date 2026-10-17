"""Training helpers shared by all training scripts."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch


def cosine_lr(step: int, base_lr: float, warmup: int, total: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def autocast_ctx(device: torch.device, amp: str):
    """``amp`` in {"auto", "bf16", "fp16", "none"}. "auto" = bf16 if supported, else fp32."""
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    if amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def needs_grad_scaler(device: torch.device, amp: str) -> bool:
    if device.type != "cuda":
        return False
    return amp == "fp16" or (amp == "auto" and not torch.cuda.is_bf16_supported())


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PSNR for images in [-1, 1], averaged over the batch."""
    mse = ((pred.float() - target.float()) / 2).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean()


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def log(self, record: dict) -> None:
        record = {"time": round(time.time() - self.t0, 1), **record}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        msg = "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items())
        print(msg, flush=True)


def infinite(loader):
    while True:
        yield from loader
