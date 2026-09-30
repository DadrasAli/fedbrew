"""A run's flush: one copy to the host, and the writes behind the loop.

A resident round (``fedbrew/core/resident.py``) leaves everything it
computes on the device until a flush, and every run, resident or per round
(``run_fl_loop``), writes its rounds behind the loop. This module holds what
the flush does without stopping the training:

- :class:`HostCopy` brings a window's values and models to the host in one
  wait: the copies are queued on a stream of their own behind the window's
  last round, into pinned memory, and the host waits for them alone -- not
  for the round the loop has already queued behind the window;
- :class:`WriterStaged` takes the checkpoints a round stages, as
  ``StagedCheckpoints`` does, and has the writer write each to its temporary
  file, in the order staged, while the loop goes on;
- :class:`FlushWriter` is that one thread. It runs what it is handed in
  order: each round's staged checkpoints, then at a flush the CSV rows,
  run.json, and only then the checkpoints' commit (POST-F24), so a checkpoint
  is never visible ahead of the history it continues. A flush writes the
  run's records as they were when it was handed over (:func:`frozen_state`),
  and the loop hands over a flush only once the last one is written, so a
  kill loses at most the rounds since the last flush it finished;
- :class:`RoundClock` times a round's phases on the device's own timeline,
  with events read back at the flush, so timing a round adds no wait.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class HostCopy:
    """Copies of device tensors on the host, made behind one event with one wait."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream(device) if self.cuda else None

    def copy(self, tensors: Sequence[Tensor], after: Any) -> list[Tensor]:
        """``tensors`` on the host, once everything queued before ``after`` has run.

        On the CPU a tensor is already where the host reads it, and is handed
        back as it is.
        """

        if not self.cuda:
            return [tensor.detach() for tensor in tensors]
        assert self.stream is not None
        self.stream.wait_event(after)
        with torch.cuda.stream(self.stream):
            copies = [
                torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True).copy_(
                    tensor.detach(), non_blocking=True
                )
                for tensor in tensors
            ]
            done = torch.cuda.Event()
            done.record(self.stream)
        done.synchronize()
        return copies

    def mark(self) -> Any:
        """An event at this point of the device's queue; None on the CPU."""

        if not self.cuda:
            return None
        event = torch.cuda.Event()
        event.record()
        return event


class RoundClock:
    """A round's phase boundaries: device events on CUDA, host clocks on the CPU."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.marks: dict[str, Any] = {}

    def mark(self, name: str) -> None:
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.marks[name] = event
        else:
            self.marks[name] = time.perf_counter()

    def seconds(self, first: str, last: str) -> float:
        """Seconds between two marks; on CUDA once both have run (after the flush's wait)."""

        start, stop = self.marks.get(first), self.marks.get(last)
        if start is None or stop is None:
            return 0.0
        if self.cuda:
            return float(start.elapsed_time(stop)) / 1000.0
        return float(stop - start)


class WriterStaged:
    """``StagedCheckpoints`` whose temporary files the writer writes, each as it is staged.

    The writer stages them into one ``StagedCheckpoints``, in the order
    staged, so a path staged again rewrites its one temporary file and keeps
    its place; the flush commits it (``written``). A payload is held only
    until the writer has written it.
    """

    def __init__(self, writer: FlushWriter) -> None:
        from fedbrew.core.checkpointing import StagedCheckpoints

        self.writer = writer
        self.staged = StagedCheckpoints()

    def stage(self, payload: Mapping[str, Any], path: Path) -> None:
        payload = dict(payload)
        self.writer.submit(lambda: self.staged.stage(payload, Path(path)))

    def written(self) -> Any:
        """The ``StagedCheckpoints`` to commit, once the writer has reached the flush."""

        return self.staged


def frozen_state(state: Any) -> Any:
    """``state`` as a flush writes it: its histories and their summaries as they are now.

    The loop goes on appending to them while the writer writes; the writer
    reads these copies, so what a flush writes is what the run held when it
    was handed over.
    """

    import copy

    frozen = copy.copy(state)
    for name in ("metrics_history", "client_metrics_history", "client_update_metrics_history"):
        history = getattr(state, name)
        if not hasattr(history, "summary"):
            setattr(frozen, name, list(history))
            continue
        copied = type(history).__new__(type(history))
        list.extend(copied, history)
        copied.summary = copy.deepcopy(history.summary)
        setattr(frozen, name, copied)
    return frozen


class FlushWriter:
    """One thread that runs the flushes it is handed, in order, one at a time.

    A flush that raises is raised again in the loop, at its next ``wait``.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="flush-writer", daemon=True)
        self._thread.start()

    def submit(self, job: Callable[[], None]) -> None:
        self._raise()
        self._jobs.put(job)

    def wait(self) -> None:
        """Until every flush handed over has been written."""

        self._jobs.join()
        self._raise()

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                if self._error is None:
                    job()
            except BaseException as error:  # noqa: BLE001 -- handed to the loop
                self._error = error
            finally:
                self._jobs.task_done()

    def _raise(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise error
