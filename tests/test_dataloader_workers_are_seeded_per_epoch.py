"""A DataLoader worker gets a fresh seed each epoch, and the run stays reproducible.

`_SeedWorker` took a `base_seed` fixed at construction and set every worker RNG
to `base_seed + worker_id`. A DataLoader spawns its workers afresh for each
iterator, so that value was the same on every epoch: worker *w* replayed one
in-worker stream for every epoch of a round. torch's own recipe reads
`torch.initial_seed()` inside the worker, which the loader derives from its
`generator` per epoch, so it varies with the epoch and still reproduces under a
fixed seed. That variation is torch's own: its worker loop already seeds
`random` and NumPy per worker. What `SeedWorker` adds is one stated definition,
shared by both task adapters, instead of the rule being inherited.

It was unreachable while `runtime.num_workers` never reached the loader --
P09-F08(a) and (b), since closed -- so `--num-workers 4` on any shipped config
now reaches it. Nothing in a shipped dataset draws randomness in `__getitem__`
today, so this was dormant rather than wrong; enabling workers for throughput
is the change that wakes it. P09-F08(c).

The two claims are measured through a real DataLoader with real worker
processes, because the defect is entirely about what torch does between epochs.
"""

from __future__ import annotations

import random
import unittest

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fedbrew.core.torch_utils import SeedWorker

EPOCHS = 3
WORKERS = 2


class _DrawsInGetItem(Dataset):
    """What an augmentation would be: randomness inside `__getitem__`."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([random.random()])


def _body(function: object) -> str:
    """A function's source with its docstring removed."""

    import inspect

    source = inspect.getsource(function)  # type: ignore[arg-type]
    parts = source.split('"""')
    return parts[-1] if len(parts) > 1 else source


def _epochs(seed: int, worker_init_fn: object) -> list[list[float]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        _DrawsInGetItem(),
        batch_size=8,
        num_workers=WORKERS,
        worker_init_fn=worker_init_fn,  # type: ignore[arg-type]
        generator=generator,
    )
    return [[round(float(v), 9) for v in next(iter(loader))[:, 0].tolist()] for _ in range(EPOCHS)]


class WorkerSeedingTest(unittest.TestCase):
    def test_every_epoch_draws_a_different_in_worker_stream(self) -> None:
        drawn = _epochs(1234, SeedWorker())
        for later in drawn[1:]:
            self.assertNotEqual(drawn[0], later, "an epoch replayed the previous one's stream")

    def test_the_same_generator_seed_reproduces_every_epoch(self) -> None:
        """Varying per epoch must not cost the reproducibility the seed buys."""

        self.assertEqual(_epochs(1234, SeedWorker()), _epochs(1234, SeedWorker()))

    def test_a_different_generator_seed_draws_a_different_stream(self) -> None:
        self.assertNotEqual(_epochs(1234, SeedWorker()), _epochs(4321, SeedWorker()))

    def test_the_previous_construction_is_what_replayed(self) -> None:
        """The defect itself, so the guard cannot pass for the wrong reason."""

        class _FixedSeedWorker:
            def __init__(self, base_seed: int) -> None:
                self.base_seed = int(base_seed)

            def __call__(self, worker_id: int) -> None:
                worker_seed = (self.base_seed + int(worker_id)) % (2**32)
                random.seed(worker_seed)

        drawn = _epochs(1234, _FixedSeedWorker(1234))
        self.assertTrue(
            all(epoch == drawn[0] for epoch in drawn),
            "the pre-fix construction no longer replays, so this guard proves nothing",
        )

    @pytest.mark.fast
    def test_the_seed_comes_from_the_worker_and_nowhere_else(self) -> None:
        """Read the body, not the prose above it.

        torch has already folded `worker_id` into `initial_seed()`, so adding
        it again would make worker *w* of one epoch collide with worker *w+1*
        of an epoch whose base seed happened to be one lower. And torch has
        already seeded the worker with the full 64-bit value, so re-seeding it
        from a 32-bit truncation of that would discard entropy.
        """

        body = _body(SeedWorker.__call__)
        self.assertIn("torch.initial_seed()", body)
        self.assertNotIn("worker_id", body)
        self.assertNotIn("manual_seed", body)
        self.assertNotIn("base_seed", body)


@pytest.mark.fast
class BothTasksUseTheOneDefinitionTest(unittest.TestCase):
    def test_neither_task_carries_its_own_copy(self) -> None:
        from fedbrew.tasks.causal_lm import torch_causal_lm
        from fedbrew.tasks.classification import torch_classification

        for module in (torch_classification, torch_causal_lm):
            with self.subTest(module=module.__name__):
                self.assertFalse(
                    hasattr(module, "_SeedWorker"),
                    "a second copy of the worker seeding rule is back",
                )
                self.assertIs(module.SeedWorker, SeedWorker)

    def test_a_seeded_loader_config_installs_it(self) -> None:
        """The reachable path: build_dataloader with a seed and no explicit fn."""

        from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

        task = TorchClassificationTask(device="cpu")
        data = {
            "train": {
                "x": torch.zeros(8, 4),
                "y": torch.zeros(8, dtype=torch.long),
            }
        }
        # num_workers 1, so this is a real DataLoader: at 0 the classification
        # task returns its own fast-batching iterable, which has no workers to
        # seed. That branch is exactly why this was unreachable for so long.
        loader = task.build_dataloader(
            data["train"],
            {"seed": 7, "num_workers": 1, "batch_size": 4, "shuffle": True},
        )
        self.assertIsInstance(loader.worker_init_fn, SeedWorker)
        self.assertIsNotNone(loader.generator)


if __name__ == "__main__":
    unittest.main()
