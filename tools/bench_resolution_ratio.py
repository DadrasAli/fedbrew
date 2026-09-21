"""Measure the real cost ratio between input resolutions for one model.

The feasibility projection scaled resolution by pixel count (96^2 = 2.25x of
64^2). That is an upper bound: small models at small resolutions are
kernel-launch bound rather than FLOP bound, so the realized ratio is lower.
This measures it instead of assuming it.

Usage: python tools/bench_resolution_ratio.py [--batch-size 32] [--iters 30]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from fedbrew.core.registry import models, register_builtin_components


def timed(fn, iters: int, warmup: int = 10) -> float:
    """Seconds per call, with CUDA synchronised around the timed region."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def _steps_for(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[Callable[[], None], Callable[[], None]]:
    """Build the two timed closures over one resolution's own objects.

    A factory rather than closures defined inside the loop: defined there they
    capture the loop variable rather than its value, so they would measure
    whatever the last iteration left behind if `timed` were ever made lazy or
    the calls were deferred. They are called immediately today, which is why
    this was only a warning -- but the fix costs a function and removes the
    class of bug rather than the symptom.
    """

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        optimizer.step()

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        model(x)
        model.train()

    return train_step, eval_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 96, 128])
    args = parser.parse_args()

    register_builtin_components()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    results = {}
    for res in args.resolutions:
        model = models.get("openimage_shufflenet")({"num_classes": 596}).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
        criterion = torch.nn.CrossEntropyLoss()
        x = torch.randint(
            0, 256, (args.batch_size, 3, res, res), dtype=torch.uint8, device=device
        ).float()
        y = torch.randint(0, 596, (args.batch_size,), device=device)

        train_step, eval_step = _steps_for(model, optimizer, criterion, x, y)

        train_s = timed(train_step, args.iters)
        eval_s = timed(eval_step, args.iters)
        results[res] = (train_s, eval_s)
        print(
            f"{res}^2  train {train_s * 1e3:7.2f} ms/batch "
            f"({args.batch_size / train_s:8.0f} img/s)   "
            f"eval {eval_s * 1e3:7.2f} ms/batch ({args.batch_size / eval_s:8.0f} img/s)"
        )

    base = args.resolutions[0]
    print(f"\nratios vs {base}^2 (pixel-count bound in brackets):")
    for res in args.resolutions:
        pixel = (res / base) ** 2
        t = results[res][0] / results[base][0]
        e = results[res][1] / results[base][1]
        print(f"  {res}^2  train {t:5.2f}x  eval {e:5.2f}x   [pixels {pixel:5.2f}x]")


if __name__ == "__main__":
    main()
