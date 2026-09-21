"""Run a CPU-only test without torch reaching for an accelerator.

A test that builds CPU tensors and steps a CPU optimizer should not depend on
what hardware the host has. Since torch 2.6, it does. `Optimizer.step` calls
`_accelerator_graph_capture_health_check`, which asks
`torch.accelerator.current_accelerator(check_available=True)` and, if the
answer is cuda or xpu, calls `current_stream()` on it -- for CPU parameters, on
every step. On a shared cluster login node whose GPU is in `Prohibited` compute
mode, `torch.cuda.is_available()` is True while any actual use raises, so three
CPU-only tests in this suite failed with

    torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or
    unavailable

on a machine where nothing they test involves a GPU. The same shape reaches any
host where a device is visible but unusable: an exclusive-mode GPU already held
by another job, a stale driver, a container without `--gpus`.

`CUDA_VISIBLE_DEVICES=""` also silences it, and is deliberately not what this
does. A suite that needs an environment variable to be CPU-only is not CPU-only;
the requirement would live in a README rather than in the code, and would be
discovered by whoever runs the suite on the wrong machine.

So the tests that step an optimizer on CPU declare it, and the declaration is
what makes them hardware-independent rather than incidentally passing wherever
CI happens to run.

Nothing here changes what the code under test does with a device: the
parameters were already CPU tensors and stay CPU tensors. Only torch's own
graph-capture probe, which is meaningless for them, is switched off.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from unittest import mock

import torch


@contextlib.contextmanager
def no_accelerator() -> Iterator[None]:
    """Report no accelerator for the duration, so a CPU step stays on the CPU.

    A no-op on torch builds without `torch.accelerator` (added in 2.6), where
    the probe this works around does not exist either -- so the same test runs
    unchanged on the older torch the frozen baseline names.
    """

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None or not hasattr(accelerator, "current_accelerator"):
        yield
        return
    with mock.patch.object(accelerator, "current_accelerator", return_value=None):
        yield
