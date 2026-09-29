"""PyTorch classification task adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from fedbrew.core.torch_utils import OptimizerLike, SeedWorker, resolve_torch_device, uploaded
from fedbrew.tasks.base import (
    LoaderOrder,
    TaskAdapter,
    batch_row_numbers,
    model_config_key,
    row_count,
    row_mean,
    row_numbers,
)


class TorchClassificationTask(TaskAdapter):
    """Task-specific PyTorch helpers for classification."""

    #: Cross-entropy and the argmax's hit rate (``compute_metrics``).
    METRICS = {"loss": "min", "accuracy": "max"}
    METRIC_GLOSSES = {"loss": "cross-entropy", "accuracy": "top-1 accuracy"}

    #: The batched executor's form of a stacked step's gradients: vmap(grad).
    #: One backward through the per-client losses' sum was slower on the CPU
    #: -- 141 against 101 ms of local steps a round for the MNIST MLP at 1000
    #: clients, one thread -- and no faster on an A100, 2.6 against 2.7 ms
    #: (measured on 2026-09-27).
    batched_gradient = "vmap_grad"

    #: What its grad_norm_sq measures (TaskAdapter.GRAD_NORM_GLOSS).
    GRAD_NORM_GLOSS = (
        "squared norm of the gradient of F, the example-weighted mean cross-entropy "
        "over every client's train split, at the global model in its trainable "
        "parameters"
    )

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        batch_size: int = 32,
        device: str = "cpu",
        dataloader_config: Mapping[str, Any] | None = None,
        reuse_model: bool = True,
        use_amp: bool = False,
        fast_batching: bool = True,
        eval_batch_size: int | None = None,
    ) -> None:
        """Configure batching, device and model caching for classification.

        Args:
            model_config: The ``model`` config block, copied on entry and
                merged with any per-call override in :meth:`build_model`.
            batch_size: Training mini-batch size, in examples.
            device: Torch device string; ``"auto"`` resolves to CUDA when
                available.
            dataloader_config: DataLoader options -- ``num_workers``,
                ``pin_memory``, and the worker-only settings that follow from
                them. ``num_workers > 0`` disables ``fast_batching``.
            reuse_model: Keep one model instance per architecture and load each
                incoming state into it, instead of rebuilding per client per
                phase. Safe because every caller overwrites the weights before
                use; throughput-only, results unchanged.
            use_amp: Train under float16 autocast. Silently ignored off CUDA.
                **Changes the numbers where it applies.**
            fast_batching: Slice resident device tensors instead of going
                through a DataLoader. Reproduces the DataLoader RNG protocol
                exactly, so batch order is unchanged epoch for epoch.
            eval_batch_size: Batch size for gradient-free passes, in examples.
                Defaults to ``batch_size``. Larger only changes floating-point
                summation order; metrics stay example-weighted.

        Batches are (features, targets): features float of shape
        (N, *feature_dims) exactly as the shards store them -- **not
        normalised** here: femnist_resnet18 and openimage_shufflenet scale raw
        pixels themselves, and mlp, cnn and small_cnn take the features as
        stored -- and targets
        int64 of shape (N,) holding class indices in [0, num_classes).
        """

        self.model_config = dict(model_config or {})
        self.batch_size = batch_size
        # Centralized evaluation is gradient-free; batching it at the training
        # size would run the global test set in thousands of tiny forward passes.
        self.eval_batch_size = int(eval_batch_size or batch_size)
        self.device = torch.device(resolve_torch_device(device))
        self.dataloader_config = dict(dataloader_config or {})
        self.reuse_model = bool(reuse_model)
        self.fast_batching = bool(fast_batching)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self._criterion = nn.CrossEntropyLoss()
        self._model_cache: dict[str, nn.Module] = {}
        self._scaler: Any = torch.amp.GradScaler("cuda") if self.use_amp else None

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        """Return a classification model for the resolved model config.

        Cross-device rounds call this once per client per phase, and rebuilding
        a model means re-running initialization on CPU and re-uploading every
        parameter to the GPU each time. Because every caller overwrites the
        weights with a received state before using the model, one cached
        instance per architecture is reused instead. Set ``reuse_model=False``
        for callers that need independent instances.
        """

        model_config = dict(self.model_config)
        if config is not None:
            model_config.update(config)

        if not self.reuse_model:
            return self._construct_model(model_config)

        cache_key = model_config_key(model_config)
        model = self._model_cache.get(cache_key)
        if model is None:
            model = self._construct_model(model_config)
            self._model_cache[cache_key] = model
        return model

    def _construct_model(self, model_config: Mapping[str, Any]) -> nn.Module:
        model_name = str(model_config.get("name", "mlp"))

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        model_factory = models.get(model_name)
        return model_factory(dict(model_config)).to(self.device)

    def federated_model_state_metadata(self, model: nn.Module) -> dict[str, Any]:
        """Describe the communication contract, cached per model instance.

        The base implementation clones the whole state dict just to measure it.
        These numbers depend only on the architecture, so they are computed once
        per model object and cached on it.
        """

        cached = getattr(model, "_fl_state_metadata", None)
        if cached is None:
            cached = super().federated_model_state_metadata(model)
            model._fl_state_metadata = cached  # type: ignore[assignment]
        return dict(cached)

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
        shuffle: bool = False,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        """Wrap tensor data in a DataLoader."""

        if isinstance(config, bool):
            shuffle = config
            config = None
        loader_config = dict(self.dataloader_config)
        if config is not None:
            loader_config.update(config)

        shuffle = bool(loader_config.get("shuffle", shuffle))
        batch_size = int(loader_config.get("batch_size", self.batch_size))
        drop_last = bool(loader_config.get("drop_last", False))
        seed = loader_config.get("seed")
        generator = loader_config.get("generator")
        worker_init_fn = loader_config.get("worker_init_fn")
        if seed is not None:
            seed_value = int(seed)
            if generator is None:
                generator = torch.Generator()
                generator.manual_seed(seed_value)
            if worker_init_fn is None:
                # Seeded from the per-epoch value torch hands the worker, not
                # from seed_value: the loader's generator is already seeded
                # from it, and re-deriving here froze every epoch to one
                # in-worker stream. P09-F08(c).
                worker_init_fn = SeedWorker()

        dataloader_kwargs = _dataloader_kwargs(loader_config, self.device)

        if self.fast_batching and int(dataloader_kwargs["num_workers"]) == 0:
            raw_features, raw_targets = _raw_tensors(data)
            return cast(
                Any,
                _DeviceTensorBatches(
                    features=raw_features,
                    targets=raw_targets,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    device=self.device,
                    generator=generator,
                ),
            )

        features, targets = _extract_tensors(data)
        dataset = cast(
            Dataset[tuple[Tensor, Tensor]],
            TensorDataset(features, targets),
        )
        if generator is not None:
            dataloader_kwargs["generator"] = generator
        if worker_init_fn is not None:
            dataloader_kwargs["worker_init_fn"] = worker_init_fn
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **dataloader_kwargs,
        )

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: OptimizerLike | None = None,
    ) -> dict[str, float]:
        """Run one supervised training step.

        Raises:
            ValueError: If no optimizer is passed. The optimizer carries the
                configured rule, so this task cannot supply one per call.
        """

        if optimizer is None:
            # This used to build a fresh SGD(lr=0.01) here. Plain SGD has no
            # moments to reset, so this is the milder half of what
            # TorchCausalLMTask.train_step had -- but it is the same defect:
            # the rate is hard-coded, and so are momentum, weight_decay and
            # nesterov, all of which the client config carries. Measured, one
            # step on a 4x3 linear model, max |parameter delta|:
            #
            #   no optimizer passed      0.00184
            #   SGD(lr=0.01)             0.00184
            #   SGD(lr=0.5)              0.09219
            #
            # A caller that omits the optimizer trains at 0.01 whatever the
            # config says, so it is refused instead. Kept in step with the
            # causal-LM task deliberately: one contract for both.
            raise ValueError(
                "train_step requires an optimizer: the update rule's learning rate, "
                "momentum, weight decay and nesterov flag all live in the optimizer, "
                "and one built per call would silently replace the configured ones"
            )
        model.train()
        features, targets = self._move_batch(batch)

        optimizer.zero_grad(set_to_none=True)
        scaler = self._scaler
        if scaler is not None:
            # What GradScaler actually needs is param_groups: it reads them in
            # unscale_ and tracks per-device inf counts against the object
            # itself. It does not need the nominal torch type.
            #
            # This used to test the nominal torch type instead, which is
            # stricter, and that extra strictness was load-bearing -- it is
            # what made scaffold, fedprox and max_grad_norm refuse use_amp;
            # all three compose, and all three now run.
            # tests/test_amp_composes_with_wrapped_optimizers.py records the
            # wrappers measured with the guard lifted: every one that wraps an
            # optimizer took an AMP step, and _GradientOnlyOptimizer raised
            # AttributeError on param_groups.
            #
            # So the line below is the requirement rather than a proxy for it,
            # and the one wrapper that fails fails on the requirement itself.
            # _GradientOnlyOptimizer computes gradients and never steps, so it
            # has no optimizer to expose groups from; the rules that use it --
            # fedlalr, delta_sgd, update_mode: frozen_batch_gradients -- are
            # refused at config load, and this is the backstop for a client
            # constructed directly. P03-F05.
            if not hasattr(optimizer, "param_groups"):
                raise TypeError(
                    "numerics.use_amp needs an optimizer exposing param_groups; "
                    f"{type(optimizer).__name__} has none, so GradScaler cannot unscale it"
                )
            with torch.autocast("cuda", dtype=torch.float16):
                loss = self._criterion(model(features), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # functional_loss's loss, without the outputs it also builds.
            loss = self._cross_entropy(self._logits(model, None, None, features), targets)
            loss.backward()
            optimizer.step()
        return {"loss": float(loss.detach())}

    def eval_step(self, model: nn.Module, batch: Any) -> dict[str, float]:
        """Run one supervised evaluation step."""

        model.eval()
        features, targets = self._move_batch(batch)
        with torch.no_grad():
            if self.use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(features)
                outputs = outputs.float()
            else:
                outputs = self._logits(model, None, None, features)
            loss, correct = self._scored(outputs, targets)
        return {
            "loss": float(loss),
            "correct": float(correct),
            "total": float(int(targets.numel())),
        }

    # -- the batched executor (fedbrew.tasks.base.BatchableTask) --------------

    def split_rows(self, data: Any) -> tuple[Tensor, Tensor]:
        """A split's features and targets as the loader yields them: widened, on the device."""

        return _resident_rows(*_raw_tensors(data), self.device)

    def row_batches(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
    ) -> _RowNumbers:
        """``build_dataloader(data, config)``'s batches as row indices, by the same loader.

        The loader itself says which rows it yields (``index_batches``), or,
        on the ``DataLoader`` path, is built on the row numbers in place of the
        rows, so its generator, shuffle, batching and ``drop_last`` are the
        ones it applies; each iteration of the result iterates it once,
        drawing what an epoch of it draws. In the main process: the order a
        DataLoader yields is its sampler's, whichever process fetches the
        rows.
        """

        if isinstance(config, bool):
            config = {"shuffle": config}
        loader = self.build_dataloader(data, {**(config or {}), "num_workers": 0})
        if not isinstance(loader, _DeviceTensorBatches):
            rows = len(_raw_tensors(data)[1])
            numbered = {"x": torch.zeros((rows, 1)), "y": row_numbers(rows).long()}
            loader = self.build_dataloader(numbered, {**(config or {}), "num_workers": 0})
        return _RowNumbers(loader)

    def loader_order(
        self, data: Any, config: Mapping[str, Any] | bool | None = None
    ) -> LoaderOrder | None:
        """What ``build_dataloader(data, config)`` yields, declared (``LoaderOrder``).

        Both of its loaders draw a ``DataLoader``'s order per epoch --
        ``_DeviceTensorBatches`` reproduces it -- from the generator seeded
        with ``seed``. None for a loader handed a generator of its own.
        """

        if isinstance(config, bool):
            config = {"shuffle": config}
        values = {**self.dataloader_config, **(config or {})}
        shuffle = bool(values.get("shuffle", False))
        seed = values.get("seed")
        if values.get("generator") is not None:
            return None
        return LoaderOrder(
            rows=len(_raw_tensors(data)[1]),
            batch_size=max(1, int(values.get("batch_size", self.batch_size))),
            shuffle=shuffle,
            drop_last=bool(values.get("drop_last", False)),
            seed=None if seed is None else int(seed),
            per_epoch=True,
        )

    def functional_loss(
        self,
        model: nn.Module,
        params: Mapping[str, Tensor] | None,
        buffers: Mapping[str, Tensor] | None,
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """The cross-entropy ``train_step`` backpropagates, at ``params`` or the model's own."""

        features, targets = batch
        loss = self._cross_entropy(self._logits(model, params, buffers, features), targets, mask)
        return loss, {"loss": loss.detach()}

    def functional_eval(
        self,
        model: nn.Module,
        params: Mapping[str, Tensor] | None,
        buffers: Mapping[str, Tensor] | None,
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``eval_step``'s loss, correct and total, as tensors."""

        features, targets = batch
        return self._measured(self._logits(model, params, buffers, features), targets, mask)

    def stacked_metrics(
        self, outputs: Sequence[Mapping[str, Tensor]], counts: Sequence[int]
    ) -> tuple[dict[str, Tensor], Tensor]:
        """``compute_metrics`` of many splits' eval outputs at once, and their example counts.

        ``outputs[p][key]`` holds position ``p``'s value for every split, and
        split ``k`` has ``counts[k]`` positions: what it returns, per split, is
        ``compute_metrics`` of that split's own outputs -- the same sums, in
        float64, over the same batches -- and the sum of their ``total``.
        """

        device = outputs[0]["loss"].device
        values = {
            key: torch.stack([output[key] for output in outputs]).to(torch.float64)
            for key in ("loss", "correct", "total")
        }
        positions = torch.arange(len(outputs), device=device).unsqueeze(1)
        counts_tensor = uploaded(torch.tensor(list(counts)), device)
        # Selected, not multiplied: a split's padding positions are empty
        # batches whose mean loss is NaN, and NaN * 0 is NaN.
        real = positions < counts_tensor.unsqueeze(0)
        loss, correct, total = (
            torch.where(real, values[key], 0.0) for key in ("loss", "correct", "total")
        )
        examples = total.sum(dim=0)
        batches = counts_tensor.to(torch.float64)
        counted = examples != 0.0
        safe = torch.where(counted, examples, 1.0)
        weighted = torch.where(
            counted, (loss * total).sum(dim=0) / safe, loss.sum(dim=0) / batches.clamp(min=1.0)
        )
        accuracy = torch.where(counted, correct.sum(dim=0) / safe, 0.0)
        empty = counts_tensor == 0
        return (
            {
                "loss": torch.where(empty, 0.0, weighted),
                "accuracy": torch.where(empty, 0.0, accuracy),
            },
            examples,
        )

    def _logits(
        self,
        model: nn.Module,
        params: Mapping[str, Tensor] | None,
        buffers: Mapping[str, Tensor] | None,
        features: Tensor,
    ) -> Tensor:
        if params is None:
            return model(features)
        return torch.func.functional_call(model, (dict(params), dict(buffers or {})), (features,))

    def _cross_entropy(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        """The criterion's mean, or its mean over the real rows under ``mask``."""

        if mask is None:
            return self._criterion(logits, targets)
        return row_mean(nn.functional.cross_entropy(logits, targets, reduction="none"), mask)

    def _measured(
        self, logits: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        """What eval_step reports for a batch's logits: loss, correct and total."""

        loss, correct = self._scored(logits, targets, mask)
        return {"loss": loss, "correct": correct, "total": row_count(targets, mask)}

    def _scored(
        self, logits: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """A batch's mean loss and its count of correct predictions, over its real rows."""

        loss = self._cross_entropy(logits, targets, mask).detach()
        hits = logits.argmax(dim=1) == targets
        return loss, hits.sum() if mask is None else (hits * mask).sum()

    def objective_loss(self, model: nn.Module, batch: Any) -> tuple[Tensor, float]:
        """The batch's mean cross-entropy, as ``train_step`` takes it, and its examples."""

        features, targets = self._move_batch(batch)
        loss, _ = self.functional_loss(model, None, None, (features, targets))
        return loss, float(int(targets.numel()))

    def evaluation_total(self, batch: Any) -> float | None:
        """eval_step's "total": how many targets the batch holds."""

        _, targets = _extract_tensors(batch)
        return float(int(targets.numel()))

    def compute_metrics(
        self,
        outputs: Sequence[Any] | Tensor,
        targets: Tensor | None = None,
    ) -> dict[str, float]:
        """Compute loss and accuracy from logits or eval-step outputs."""

        if targets is not None:
            if not isinstance(outputs, Tensor):
                raise TypeError("outputs must be a tensor when targets are provided")
            logits = outputs.to(self.device)
            labels = targets.to(self.device)
            loss = self._criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            accuracy = float((predictions == labels).float().mean().item())
            return {"loss": float(loss.detach().cpu().item()), "accuracy": accuracy}

        records = [record for record in outputs if isinstance(record, dict)]
        if not records:
            return {"loss": 0.0, "accuracy": 0.0}

        total_examples = sum(float(record.get("total", 0.0)) for record in records)
        if total_examples == 0.0:
            loss = sum(float(record.get("loss", 0.0)) for record in records) / len(records)
            return {"loss": loss, "accuracy": 0.0}

        weighted_loss = (
            sum(
                float(record.get("loss", 0.0)) * float(record.get("total", 0.0))
                for record in records
            )
            / total_examples
        )
        total_correct = sum(float(record.get("correct", 0.0)) for record in records)
        return {"loss": weighted_loss, "accuracy": total_correct / total_examples}

    def evaluate_model(self, model: nn.Module, data: Any) -> dict[str, float]:
        """Evaluate a model on tensor data and return loss/accuracy."""

        dataloader = self.build_dataloader(
            data,
            {"batch_size": self.eval_batch_size, "shuffle": False},
        )
        outputs = [self.eval_step(model, batch) for batch in dataloader]
        return self.compute_metrics(outputs)

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        features, targets = _extract_tensors(batch)
        return features.to(self.device), targets.to(self.device)


class _DeviceTensorBatches:
    """Iterate mini-batches of resident tensors without per-sample collation.

    ``DataLoader`` over a ``TensorDataset`` slices one example at a time and
    re-stacks them in Python, then copies each batch to the GPU separately.
    Federated rounds rebuild that machinery for every client, so for the common
    case of in-memory tensors and no worker processes this uploads the client's
    whole split once and yields slices of it instead.

    Shuffling reproduces the DataLoader RNG protocol exactly -- the per-epoch
    base-seed draw that ``_BaseDataLoaderIter`` makes, then ``RandomSampler``'s
    ``torch.randperm`` -- so a seeded run yields the same batch order as the
    DataLoader path, epoch for epoch. ``tests/test_fast_batching.py`` pins this.
    """

    def __init__(
        self,
        features: Tensor,
        targets: Tensor,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have matching lengths")
        self._raw_features = features
        self._raw_targets = targets
        self._batch_size = max(1, int(batch_size))
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._device = device
        self._generator = generator
        self._features: Tensor | None = None
        self._targets: Tensor | None = None

    def _resident(self) -> tuple[Tensor, Tensor]:
        if self._features is None or self._targets is None:
            self._features, self._targets = _resident_rows(
                self._raw_features, self._raw_targets, self._device
            )
        return self._features, self._targets

    def __len__(self) -> int:
        total = self._raw_features.shape[0]
        if self._drop_last:
            return total // self._batch_size
        return (total + self._batch_size - 1) // self._batch_size

    def _epoch_generator(self) -> torch.Generator | None:
        """Consume the base-seed draw a DataLoader iterator makes per epoch."""

        torch.empty((), dtype=torch.int64).random_(generator=self._generator)
        if self._generator is not None:
            return self._generator

        # An unseeded RandomSampler reseeds a private generator from global RNG.
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        return sampler_generator

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        features, targets = self._resident()
        for index in self.index_batches(self._device):
            if isinstance(index, slice):
                yield features[index], targets[index]
            else:
                yield features.index_select(0, index), targets.index_select(0, index)

    def index_batches(self, device: torch.device | str = "cpu") -> Iterator[Tensor | slice]:
        """One epoch's batches as the rows they hold: an index on ``device``, or a slice.

        The epoch's draws are made here, so iterating this is iterating the
        loader as far as its generator can tell; it is what ``__iter__``
        yields the rows of, and what the batched executor reads the order
        from (``TorchClassificationTask.row_batches``).
        """

        total = self._raw_features.shape[0]
        batch_size = self._batch_size

        if self._shuffle:
            generator = self._epoch_generator()
            order = torch.randperm(total, generator=generator).to(device)
            for start in range(0, total, batch_size):
                index = order[start : start + batch_size]
                if self._drop_last and int(index.numel()) < batch_size:
                    break
                yield index
            # RandomSampler always evaluates one more permutation before it
            # stops, and discards it. Draining it here keeps the generator state
            # aligned with the DataLoader path across epochs. Consumers that
            # break out early skip this, exactly as they would there.
            torch.randperm(total, generator=generator)
            return

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            if self._drop_last and end - start < batch_size:
                break
            yield slice(start, end)


class _RowNumbers:
    """A loader's batches as CPU index tensors, each iteration one epoch of it.

    ``_DeviceTensorBatches`` says which rows it yields (``index_batches``); a
    ``DataLoader`` is built on the row numbers, and its batches are them.
    """

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __iter__(self) -> Iterator[Tensor]:
        if isinstance(self._loader, _DeviceTensorBatches):
            for index in self._loader.index_batches():
                if isinstance(index, slice):
                    yield torch.arange(index.start, index.stop)
                else:
                    yield index
            return
        for _, numbers in self._loader:
            yield batch_row_numbers(numbers)

    def __len__(self) -> int:
        return len(self._loader)


def _resident_rows(
    features: Tensor, targets: Tensor, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Rows as ``_DeviceTensorBatches`` and ``split_rows`` hold them: widened on the device.

    Transferred in the stored dtype (uint8 images stay 1 byte per pixel) and
    widened there, so the copy is as small as possible.
    """

    return features.to(device).float(), targets.to(device).long()


def _raw_tensors(data: Any) -> tuple[Tensor, Tensor]:
    """Return the stored feature/target tensors without converting their dtype."""

    if isinstance(data, Mapping):
        features = data.get("X", data.get("x"))
        targets = data.get("y")
    elif isinstance(data, tuple | list) and len(data) == 2:
        features, targets = data
    else:
        raise TypeError("data must be a mapping or a pair of tensors")

    if not isinstance(features, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("features and targets must be tensors")
    return features, targets


def _extract_tensors(data: Any) -> tuple[Tensor, Tensor]:
    features, targets = _raw_tensors(data)
    return features.float(), targets.long()


def _dataloader_kwargs(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    num_workers = int(config.get("num_workers", 0))
    kwargs["num_workers"] = num_workers

    pin_memory = bool(config.get("pin_memory", False))
    if pin_memory and device.type == "cuda":
        kwargs["pin_memory"] = True

    if num_workers > 0:
        if "persistent_workers" in config:
            kwargs["persistent_workers"] = bool(config["persistent_workers"])
        if config.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(config["prefetch_factor"])

    return kwargs
