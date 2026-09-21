"""PyTorch task adapter for next-token causal language modeling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from fedbrew.core.federated_state import model_state_size
from fedbrew.core.torch_utils import (
    OptimizerLike,
    SeedWorker,
    clone_model_state,
    forget_resident_state,
    get_untied_model_state,
    load_untied_model_state,
    resolve_torch_device,
)
from fedbrew.tasks.base import TaskAdapter, model_config_key


class TorchCausalLMTask(TaskAdapter):
    """Train and evaluate a causal LM using explicit next-token targets.

    Accuracy is token-level next-token accuracy over non-padding targets.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        batch_size: int = 8,
        device: str = "cpu",
        dataloader_config: Mapping[str, Any] | None = None,
        reuse_model: bool = True,
    ) -> None:
        """Configure batching, padding and target masking for causal LM.

        Args:
            model_config: The ``model`` config block, copied on entry. Also
                carries the generator's ``dataset_*`` metadata, which is where
                ``ignore_index`` and the SFT flag come from. A ``pad_token_id``
                absent from it means the dataset declares no padding token, not
                token 0.
            batch_size: Mini-batch size, in sequences. Must be positive. There
                is no separate evaluation batch size for this task: a causal-LM
                forward produces (N, sequence_length, vocabulary) logits, which
                reaches tens of gigabytes at the classification default, so
                evaluation runs at the training size.
            device: Torch device string; ``"auto"`` resolves to CUDA when
                available.
            dataloader_config: DataLoader options.
            reuse_model: Keep one model instance per architecture. Matters more
                here than for classification: an uncached build is a
                ``from_pretrained`` of a 0.5B model per client per phase, and
                under LoRA it also advances the process-wide RNG, which moves
                the training trajectory.

        Raises:
            ValueError: If ``batch_size`` is not positive, ``ignore_index`` is
                not an integer, ``active_target_weighting`` is not a bool, or
                ``pad_token_id`` equals the dataset's ``eos_token_id``.

        Batches are (inputs, targets), both int64 of shape
        (N, sequence_length). **The targets arrive already shifted** -- the
        generator stores ``tokens[1:]`` against ``tokens[:-1]`` -- and this
        adapter does no shifting of its own: it aligns ``logits[..., t]`` with
        ``targets[..., t]`` position for position and raises on a shape
        mismatch. A target position is excluded from both loss and accuracy
        when it equals ``ignore_index`` (-100 by default, covering padding and,
        for SFT data, the prompt tokens) or ``pad_token_id`` when that is set.
        That second filter is by token *value*, so a ``pad_token_id`` that is
        also the dataset's EOS id is refused rather than applied: it would
        remove every real end-of-document target as well as the padding.
        Reported accuracy is token-level next-token accuracy over the surviving
        positions, so it is not comparable to a classification accuracy.
        """

        self.model_config = dict(model_config or {})
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.device = torch.device(resolve_torch_device(device))
        self.dataloader_config = dict(dataloader_config or {})
        # No padding token unless the dataset names one. This used to default
        # to 0, which is the tiny byte-level generator's padding id and a real
        # word on every tokenizer that is not it -- "!" on Qwen2.5. A manifest
        # with no padding_token_id would then have every id-0 target dropped
        # from the loss, the accuracy and the aggregation weight, and every
        # id-0 input position zeroed in the attention mask, on the strength of
        # a default it never asked for. factory._add_causal_manifest_metadata
        # writes model.pad_token_id whenever the manifest declares one, so the
        # datasets that do pad are unaffected.
        raw_pad_token_id = self.model_config.get("pad_token_id")
        self.pad_token_id = None if raw_pad_token_id is None else int(raw_pad_token_id)
        # pad_token_id filters targets by *value*, not by position, so a
        # padding id that is also a real vocabulary token takes every genuine
        # occurrence of that token with it. EOS is the one that matters:
        # hf_causal_lm_text falls back to the EOS id when the tokenizer has no
        # distinct padding token (_padding_token_id), and the model would then
        # never be trained to emit end-of-document -- silently, because loss,
        # accuracy and the aggregation weight are all measured over the same
        # surviving positions and stay consistent with each other.
        eos_token_id = self.model_config.get("dataset_eos_token_id")
        if (
            self.pad_token_id is not None
            and isinstance(eos_token_id, int)
            and not isinstance(eos_token_id, bool)
            and self.pad_token_id == int(eos_token_id)
        ):
            raise ValueError(
                "pad_token_id equals the dataset's eos_token_id "
                f"({self.pad_token_id}): this task masks targets by value, so every "
                "end-of-document target would be dropped from the loss, the accuracy "
                "and the aggregation weight. Use a dataset whose padding token is "
                "distinct from its EOS token, or one that records no padding token."
            )
        raw_ignore_index = self.model_config.get(
            "dataset_ignore_index",
            self.model_config.get("ignore_index", -100),
        )
        if isinstance(raw_ignore_index, bool) or not isinstance(raw_ignore_index, int):
            raise ValueError("ignore_index must be an integer")
        self.ignore_index = int(raw_ignore_index)
        dataset_task = self.model_config.get("dataset_task")
        raw_active_weighting = self.model_config.get(
            "active_target_weighting",
            dataset_task == "causal_lm_sft",
        )
        if not isinstance(raw_active_weighting, bool):
            raise ValueError("active_target_weighting must be a bool")
        self.active_target_weighting = raw_active_weighting
        self.reuse_model = bool(reuse_model)
        self._model_cache: dict[str, nn.Module] = {}

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        """Return a causal LM for the resolved model config, cached by default.

        Cross-device rounds call this once per client per phase, and every
        caller overwrites the trainable parameters with a received state before
        using the model, so one instance per architecture is reused.

        Rebuilding is not only a from_pretrained of a 0.5B-parameter model per
        client per phase. PEFT initialises lora_A with kaiming_uniform_ from the
        process-wide RNG, so an uncached build also advances the global stream a
        number of times proportional to how many clients are fitted *and
        evaluated*. The random init itself is harmless -- the broadcast state
        overwrites it -- but the draws are not: with the LoRA state pinned, one
        extra build before a client's step changed that client's LoRA gradient
        digest from 0bf8cc20169ff0ec to a960bca5f439c687. Evaluation settings
        then move the training trajectory -- an evaluation-only setting changing
        what the arm learns, arriving here by another route.

        Set reuse_model=False for callers that need independent instances.
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
        model_name = str(model_config.get("name", "tiny_gpt2"))

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        model_factory = models.get(model_name)
        return model_factory(dict(model_config)).to(self.device)

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
        shuffle: bool = False,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        """Wrap integer input and next-token target tensors in a DataLoader."""

        if isinstance(config, bool):
            shuffle = config
            config = None
        loader_config = dict(self.dataloader_config)
        if config is not None:
            loader_config.update(config)

        shuffle = bool(loader_config.get("shuffle", shuffle))
        batch_size = int(loader_config.get("batch_size", self.batch_size))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
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

        inputs, targets = _extract_token_tensors(data)
        dataset = cast(
            Dataset[tuple[Tensor, Tensor]],
            TensorDataset(inputs, targets),
        )
        dataloader_kwargs = _dataloader_kwargs(loader_config, self.device)
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
        """Run one next-token optimization step and return token statistics.

        Raises:
            ValueError: If no optimizer is passed. The optimizer carries the
                state the update rule is made of, so this task cannot supply
                one per call.
        """

        if optimizer is None:
            # This used to build a fresh AdamW(lr=1e-3) here. A per-call
            # optimizer resets the Adam moments every batch, and Adam's first
            # step is lr * m_hat / (sqrt(v_hat) + eps) = lr * sign(g): the rule
            # degenerates to sign-SGD at a rate no config can reach. Measured
            # on tiny_gpt2 over six steps of two alternating batches, max
            # |parameter delta| per step:
            #
            #   fresh AdamW per call  .001001 .001001 .001001 .001001 .001001 .001001
            #   one AdamW(lr=1e-3)    .001001 .001001 .001002 .001002 .001007 .001013
            #   one AdamW(lr=0.05)    .050034 .050042 .043664 .043127 .044233 .041134
            #
            # The first row never leaves lr*sign(g); the others move as the
            # moments accumulate. Batches have to differ for that gap to open
            # at all -- on one batch repeated, a reused Adam converges to
            # sign(g) too and the rows agree, which is why the guard for this
            # alternates batches. And the 1e-3 was unreachable from a config
            # either way. A caller that omits the optimizer gets a plausible
            # loss curve from the wrong algorithm, so it is refused instead.
            raise ValueError(
                "train_step requires an optimizer: the causal-LM update rule keeps its "
                "state (Adam moments, the configured learning rate) in the optimizer, "
                "and one built per call would reset that state every batch"
            )
        model.to(self.device)
        model.train()
        inputs, targets = self._move_batch(batch)

        optimizer.zero_grad()
        logits = _extract_logits(model(**self._model_inputs(inputs)))
        loss, correct, total = self._loss_and_counts(logits, targets)
        loss.backward()
        optimizer.step()
        return {
            "loss": float(loss.detach().cpu().item()),
            "correct": float(correct),
            "total": float(total),
        }

    def eval_step(self, model: nn.Module, batch: Any) -> dict[str, float]:
        """Evaluate one batch and return non-padding token statistics."""

        model.to(self.device)
        model.eval()
        inputs, targets = self._move_batch(batch)
        with torch.no_grad():
            logits = _extract_logits(model(**self._model_inputs(inputs)))
            loss, correct, total = self._loss_and_counts(logits, targets)
        return {
            "loss": float(loss.detach().cpu().item()),
            "correct": float(correct),
            "total": float(total),
        }

    def compute_metrics(
        self,
        outputs: Sequence[Any] | Tensor,
        targets: Tensor | None = None,
    ) -> dict[str, float]:
        """Return loss and token-level next-token accuracy."""

        if targets is not None:
            if not isinstance(outputs, Tensor):
                raise TypeError("outputs must be a tensor when targets are provided")
            logits = outputs.to(self.device)
            labels = targets.to(self.device).long()
            batch_loss, batch_correct, batch_total = self._loss_and_counts(logits, labels)
            accuracy = float(batch_correct / batch_total) if batch_total else 0.0
            loss_value = float(batch_loss.detach().cpu().item())
            return {"loss": loss_value, "accuracy": accuracy}

        records = [record for record in outputs if isinstance(record, dict)]
        if not records:
            return {"loss": 0.0, "accuracy": 0.0}

        total_tokens = sum(float(record.get("total", 0.0)) for record in records)
        if total_tokens == 0.0:
            mean_loss = sum(float(record.get("loss", 0.0)) for record in records)
            return {"loss": mean_loss / len(records), "accuracy": 0.0}

        aggregated_loss = (
            sum(
                float(record.get("loss", 0.0)) * float(record.get("total", 0.0))
                for record in records
            )
            / total_tokens
        )
        total_correct = sum(float(record.get("correct", 0.0)) for record in records)
        return {"loss": aggregated_loss, "accuracy": total_correct / total_tokens}

    def evaluate_model(self, model: nn.Module, data: Any) -> dict[str, float]:
        """Evaluate a causal LM over all supplied token sequences."""

        dataloader = self.build_dataloader(
            data,
            {"batch_size": self.batch_size, "shuffle": False},
        )
        outputs = [self.eval_step(model, batch) for batch in dataloader]
        return self.compute_metrics(outputs)

    def get_federated_model_state(self, model: nn.Module) -> dict[str, Any]:
        """Extract adapter-only tensors from PEFT models, or full state otherwise."""

        if getattr(model, "_fl_model_state_scope", "full") != "adapter":
            # Drop tied duplicates so a full-model round does not transmit and
            # average the same tensor twice.
            return get_untied_model_state(model)
        try:
            from peft import get_peft_model_state_dict
        except ModuleNotFoundError as error:  # pragma: no cover - install dependent.
            raise ModuleNotFoundError(
                'adapter state extraction requires peft; install it with pip install -e ".[llm]"'
            ) from error

        adapter_name = _adapter_model_attribute(model, "_fl_adapter_name")
        state = get_peft_model_state_dict(model, adapter_name=adapter_name)
        cloned = clone_model_state(state)
        if not cloned:
            raise ValueError("PEFT adapter state extraction returned no tensors")
        return cloned

    def load_federated_model_state(
        self,
        model: nn.Module,
        state: Mapping[str, Any],
    ) -> None:
        """Load adapter-only tensors through PEFT, or full state otherwise."""

        if getattr(model, "_fl_model_state_scope", "full") != "adapter":
            load_untied_model_state(model, state)
            return
        forget_resident_state(model)
        try:
            from peft import set_peft_model_state_dict
        except ModuleNotFoundError as error:  # pragma: no cover - install dependent.
            raise ModuleNotFoundError(
                'adapter state loading requires peft; install it with pip install -e ".[llm]"'
            ) from error

        adapter_name = _adapter_model_attribute(model, "_fl_adapter_name")
        expected_state = self.get_federated_model_state(model)
        received_state = clone_model_state(state)
        missing = sorted(set(expected_state).difference(received_state))
        extra = sorted(set(received_state).difference(expected_state))
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("unexpected " + ", ".join(extra))
            raise ValueError(
                "adapter state tensor keys do not match the configured adapter: "
                + "; ".join(details)
            )
        status = set_peft_model_state_dict(
            model,
            received_state,
            adapter_name=adapter_name,
        )
        unexpected = getattr(status, "unexpected_keys", ())
        if unexpected:
            raise ValueError(
                "adapter state contains unexpected tensors: "
                + ", ".join(str(key) for key in unexpected)
            )

    def federated_model_state_metadata(self, model: nn.Module) -> dict[str, Any]:
        """Return stable adapter identity plus communication diagnostics."""

        if getattr(model, "_fl_model_state_scope", "full") != "adapter":
            return super().federated_model_state_metadata(model)
        state = self.get_federated_model_state(model)
        communicated_parameters, communicated_bytes = model_state_size(state)
        lora_config = getattr(model, "_fl_lora_config", None)
        if not isinstance(lora_config, Mapping):
            raise ValueError("LoRA model is missing normalized adapter configuration")
        return {
            "model_state_scope": "adapter",
            "base_model_identifier": _adapter_model_attribute(model, "_fl_base_model_identifier"),
            "base_model_resolved_revision": _adapter_model_attribute(
                model, "_fl_base_model_resolved_revision"
            ),
            "adapter_name": _adapter_model_attribute(model, "_fl_adapter_name"),
            "lora_config": dict(lora_config),
            "total_parameters": sum(int(parameter.numel()) for parameter in model.parameters()),
            "trainable_parameters": sum(
                int(parameter.numel())
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "communicated_parameters": communicated_parameters,
            "communicated_bytes": communicated_bytes,
        }

    def federated_aggregation_weight(
        self,
        training_outputs: Sequence[Mapping[str, float]],
        evaluated_num_examples: int,
    ) -> int:
        """Use processed active target tokens for SFT and legacy counts otherwise."""

        if not self.active_target_weighting:
            return super().federated_aggregation_weight(training_outputs, evaluated_num_examples)
        return int(sum(float(output.get("total", 0.0)) for output in training_outputs))

    def train_loss_denominator(self, batch: Any, output: Mapping[str, float]) -> float:
        """The active target tokens `train_step` averaged its loss over.

        `_loss_and_counts` takes the cross-entropy mean over exactly the tokens
        it counts as `total`, and a batch with none contributes a zero loss, so
        weighting it by zero is what the whole split's mean would do too.
        Not gated on `active_target_weighting`, which decides the client's
        aggregation weight: the loss is a token mean either way.
        """

        del batch
        return float(output["total"])

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        inputs, targets = _extract_token_tensors(batch)
        return inputs.to(self.device), targets.to(self.device)

    def _model_inputs(self, inputs: Tensor) -> dict[str, Tensor]:
        model_inputs = {"input_ids": inputs}
        if self.pad_token_id is not None:
            model_inputs["attention_mask"] = inputs.ne(self.pad_token_id).long()
        return model_inputs

    def _loss_and_counts(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, int, int]:
        if logits.shape[:-1] != targets.shape:
            raise ValueError(
                "causal LM logits and targets have incompatible shapes: "
                f"{tuple(logits.shape)} versus {tuple(targets.shape)}"
            )
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        active = flat_targets.ne(self.ignore_index)
        if self.pad_token_id is not None:
            active = active & flat_targets.ne(self.pad_token_id)
        total = int(active.sum().item())
        if total:
            loss = F.cross_entropy(flat_logits[active], flat_targets[active])
            predictions = flat_logits.detach().argmax(dim=-1)
            correct = int(((predictions == flat_targets) & active).sum().item())
        else:
            # A batch with no supervised token must contribute nothing, and the
            # loss must still carry a gradient path so backward() and step()
            # behave as on any other batch. Slicing to zero rows gives both:
            # 0.0, grad-connected, and backward scatters zeros over the whole
            # logit tensor.
            #
            # It used to be flat_logits.sum() * 0.0, which reads every logit to
            # produce a constant. One non-finite logit anywhere in the block
            # makes that sum non-finite, and inf * 0.0 is nan -- so the batch
            # that should have contributed nothing injects a nan loss and nan
            # gradients instead, on an already-diverging model, which is when
            # the run most needs an honest number.
            loss = flat_logits[:0].sum()
            correct = 0
        return loss, correct, total


def _adapter_model_attribute(model: nn.Module, name: str) -> str:
    value = getattr(model, name, None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"LoRA model is missing required attribute {name}")
    return value


def _extract_logits(outputs: Any) -> Tensor:
    if isinstance(outputs, Tensor):
        return outputs
    logits = getattr(outputs, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    if isinstance(outputs, Mapping) and isinstance(outputs.get("logits"), Tensor):
        return cast(Tensor, outputs["logits"])
    if isinstance(outputs, tuple | list) and outputs and isinstance(outputs[0], Tensor):
        return outputs[0]
    raise TypeError("causal LM model output must contain a logits tensor")


def _extract_token_tensors(data: Any) -> tuple[Tensor, Tensor]:
    if isinstance(data, Mapping):
        inputs = data.get("X", data.get("x"))
        targets = data.get("y")
    elif isinstance(data, tuple | list) and len(data) == 2:
        inputs, targets = data
    else:
        raise TypeError("data must be a mapping or a pair of tensors")

    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("inputs and targets must be tensors")
    if inputs.shape != targets.shape:
        raise ValueError("inputs and next-token targets must have matching shapes")
    return inputs.long(), targets.long()


def _dataloader_kwargs(
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"num_workers": int(config.get("num_workers", 0))}
    if bool(config.get("pin_memory", False)) and device.type == "cuda":
        kwargs["pin_memory"] = True
    if kwargs["num_workers"] > 0:
        if "persistent_workers" in config:
            kwargs["persistent_workers"] = bool(config["persistent_workers"])
        if config.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    return kwargs
