"""A resident round's training, captured once per shape and replayed: ``cuda_graphs: on``.

A resident round (``fedbrew/core/resident.py``) trains its buckets with a
few hundred small kernels, each launched from Python. With
``runtime.performance.cuda_graphs: on`` the round's training -- its buckets'
steps and post-fit passes, and the fold of their states into the mean -- is
recorded as a CUDA graph the second time a round of that shape comes, and
replayed from then on: the same kernels on the same inputs, in the same
order, launched as one.

What varies from round to round of one shape enters through inputs the graph
reads in place: the round's host-computed tensors (its batches' row indices
and lengths, the rows it gathers, its fold's weights and total) are copied
into the graph's input buffers from pinned memory before each replay, and the
model the round starts from into its model buffers. Everything else a round
of that shape does is the same -- its buckets, their clients' positions, their
batches' widths, which steps are sliced and which masked, its program and
its post-fit pass -- and is the shape's key (``RoundPlan.key``), so a replay
only ever runs the round it recorded. A round with a bucket of one client
(folded on the CPU) or a combined gradient (uploaded weights) has no key and
runs eagerly.

A capture or a replay that fails leaves every later round eager; stderr says
so once and run.json records it (``executor.cuda_graphs``), as compile does.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

#: Most round shapes a run holds a graph for; a new shape past it runs eagerly.
MAX_GRAPHS = 64


@dataclass(slots=True)
class _Captured:
    graph: Any
    inputs: list[Tensor]
    model: dict[str, Tensor]
    outputs: Any


class RoundGraphs:
    """Runs a round's training eagerly, or by the graph of its shape once one is recorded."""

    def __init__(self, device: torch.device, asked: bool, record: dict[str, Any]) -> None:
        self.device = device
        self.on = bool(asked) and device.type == "cuda"
        self.record = record
        if asked:
            record["cuda_graphs"] = (
                {"used": "on", "captured": 0, "replayed": 0}
                if self.on
                else {
                    "used": "off",
                    "fallback": f"CUDA graphs need a CUDA device, not {device.type}",
                }
            )
        self._seen: dict[Any, int] = {}
        self._graphs: dict[Any, _Captured] = {}
        self._pool: Any = None

    def run(
        self,
        key: Any,
        uploads: Sequence[Tensor],
        model: dict[str, Tensor],
        execute: Callable[[list[Tensor], dict[str, Tensor]], Any],
    ) -> tuple[Any, bool]:
        """The round's outputs, and whether they belong to a graph (replaced by its next replay)."""

        from fedbrew.core.batched_executor import uploaded

        if not self.on or key is None:
            return execute([uploaded(tensor, self.device) for tensor in uploads], model), False
        captured = self._graphs.get(key)
        if captured is None:
            seen = self._seen.get(key, 0)
            self._seen[key] = seen + 1
            if seen == 0 or len(self._graphs) >= MAX_GRAPHS:
                # The first round of a shape runs eagerly, which also warms up
                # what it touches (the cuBLAS handle, the cached values).
                return execute([uploaded(tensor, self.device) for tensor in uploads], model), False
            try:
                captured = self._capture(key, uploads, model, execute)
            except Exception as error:  # noqa: BLE001 -- the round runs eagerly, recorded
                self._fail("capturing", error)
                return execute([uploaded(tensor, self.device) for tensor in uploads], model), False
        try:
            return self._replay(captured, uploads, model), True
        except Exception as error:  # noqa: BLE001 -- the round runs eagerly, recorded
            self._fail("replaying", error)
            return execute([uploaded(tensor, self.device) for tensor in uploads], model), False

    def _capture(
        self,
        key: Any,
        uploads: Sequence[Tensor],
        model: dict[str, Tensor],
        execute: Callable[[list[Tensor], dict[str, Tensor]], Any],
    ) -> _Captured:
        inputs = [
            torch.empty(tensor.shape, dtype=tensor.dtype, device=self.device) for tensor in uploads
        ]
        buffers = {name: torch.empty_like(value) for name, value in model.items()}
        _fill(inputs, uploads, buffers, model)
        graph = torch.cuda.CUDAGraph()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        # Captured on a stream of its own behind the work already queued, as
        # ``torch.cuda.graph`` captures, without its synchronize and its
        # empty_cache: the round's buffers are already in the queue.
        stream = torch.cuda.Stream(self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            graph.capture_begin(self._pool)
            try:
                outputs = execute(inputs, buffers)
            finally:
                graph.capture_end()
        current.wait_stream(stream)
        captured = _Captured(graph, inputs, buffers, outputs)
        self._graphs[key] = captured
        self.record["cuda_graphs"]["captured"] = len(self._graphs)
        return captured

    def _replay(
        self, captured: _Captured, uploads: Sequence[Tensor], model: dict[str, Tensor]
    ) -> Any:
        _fill(captured.inputs, uploads, captured.model, model)
        captured.graph.replay()
        self.record["cuda_graphs"]["replayed"] += 1
        return captured.outputs

    def _fail(self, doing: str, error: BaseException) -> None:
        lines = str(error).strip().splitlines()
        reason = f"{doing} a round failed: {type(error).__name__}: {lines[0] if lines else ''}"[
            :300
        ]
        self.on = False
        self._graphs.clear()
        self.record["cuda_graphs"].update(used="off", fallback=reason)
        print(
            f"runtime.performance.cuda_graphs: {reason}; every later round runs eagerly",
            file=sys.stderr,
            flush=True,
        )
        # Where it failed, for whoever reads the log; run.json keeps the line.
        traceback.print_exception(type(error), error, error.__traceback__, limit=-12)


def _fill(
    inputs: list[Tensor],
    uploads: Sequence[Tensor],
    buffers: dict[str, Tensor],
    model: dict[str, Tensor],
) -> None:
    """The round's host tensors and model into a graph's buffers, queued without waiting."""

    for buffer, tensor in zip(inputs, uploads, strict=True):
        buffer.copy_(tensor.pin_memory(), non_blocking=True)
    for name, buffer in buffers.items():
        buffer.copy_(model[name])
