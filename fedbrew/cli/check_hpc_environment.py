#!/usr/bin/env python
"""Print local runtime and path information for HPC jobs."""

import os
import socket
import sys
from pathlib import Path


def main(argv=None):
    """Print the runtime and path facts a job's environment actually has.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``. This command takes
            no arguments; ``-h`` / ``--help`` prints usage and returns, and
            anything else exits 2.

    Prints hostname and working directory, then the SLURM and fedbrew
    environment variables (each ``<unset>`` when absent), then torch and CUDA
    availability with device names, then an existence check for each of the
    path variables that is actually set. Read-only: it changes nothing and is
    safe to run inside a job.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in (["-h"], ["--help"]):
        print("usage: fedbrew check-hpc")
        return
    if argv:
        print("Usage: fedbrew check-hpc")
        raise SystemExit(2)

    print("HPC environment check")
    print("hostname:", socket.gethostname())
    print("cwd:", Path.cwd())

    for name in (
        "SLURM_JOB_ID",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "COMMON_DATASETS",
        "FL_DATA_ROOT",
        "FL_OUTPUT_ROOT",
        "FL_CACHE_ROOT",
        "FL_LOCAL_SCRATCH",
    ):
        print(f"{name}:", os.environ.get(name, "<unset>"))

    _print_torch_info()
    for name in (
        "COMMON_DATASETS",
        "FL_DATA_ROOT",
        "FL_OUTPUT_ROOT",
        "FL_CACHE_ROOT",
        "FL_LOCAL_SCRATCH",
    ):
        value = os.environ.get(name)
        if value:
            _print_path_status(value, label=name)


def _print_torch_info():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - manual diagnostic script.
        print("torch available: false")
        print("torch import error:", exc)
        return

    print("torch available: true")
    cuda_available = bool(torch.cuda.is_available())
    print("torch cuda available:", str(cuda_available).lower())
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            print(f"gpu {index}:", torch.cuda.get_device_name(index))


def _print_path_status(path_value, label=None):
    path = Path(os.path.expandvars(os.path.expanduser(path_value)))
    prefix = label or str(path)
    print(f"{prefix} exists:", str(path.exists()).lower())


if __name__ == "__main__":
    main()
