"""A cost the client measures must not be dropped on the way to the CSV.

Delta-SGD and FedLALR compute communicated_parameters and communicated_bytes
correctly -- dtype-aware, from the exact tensor dict placed in the payload --
and then return filter_metrics(all_metrics, self.metrics).
self.metrics is the config's client.metrics list, and filter_metrics keeps
only names that appear in it:

    return {name: metrics[name] for name in requested if name in metrics}

A name the config omits is therefore dropped by a dict comprehension: no
warning, no error, no column in client_update_metrics.csv for the whole run.
configs/femnist/fedlalr.yaml listed both names; delta_sgd.yaml did not, so a
cost-per-round table across arms had the column for the fedavg family (whose
client does not filter) and for fedlalr, and a hole where delta_sgd should be.

The configs were fixed, and then the cause went: a run gives its clients no
metrics list now (``reporting.fit_metrics`` is the server's, applied once to
the round), so every client reports the cost it measures and
client_update_metrics.csv carries it for every rule. This pins both: no
client is built with a list, and the audited arms keep the cost columns in
round_metrics.csv, where the one list still decides.
"""

from __future__ import annotations

import glob
import inspect
import re
import unittest
from importlib import import_module

import pytest

from fedbrew.core.config import is_family_base, load_config_mapping
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.registry import client_updates, register_builtin_components

pytestmark = pytest.mark.fast

#: The names a client reports for the run's communication volume.
COST_METRICS = ("communicated_parameters", "communicated_bytes")


def _client_class(update_rule: str) -> type | None:
    """The class the registry's builder for this update rule constructs.

    The builders are one-line lazy imports, so the class name and its module
    are right there in the source; importing the module is what the factory
    does at runtime anyway.
    """

    register_builtin_components()
    if not client_updates.exists(update_rule):
        return None
    source = inspect.getsource(client_updates.get(update_rule))
    match = re.search(r"from (fedbrew\.clients\.[\w.]+) import (\w+)", source)
    if match is None:
        return None
    return getattr(import_module(match.group(1)), match.group(2))


def _measures_and_filters(client: type) -> bool:
    """Whether this client computes the cost metrics and then filters them out.

    Both halves have to be checked against the fit path specifically.
    TorchSGDClient computes them in fit and calls filter_metrics elsewhere, in
    its evaluation path, so a module-level search would flag local_sgd -- which
    is not at risk, because its fit metrics reach the FitResult unfiltered.
    """

    fit_sources = []
    for ancestor in client.__mro__:
        own_fit = ancestor.__dict__.get("fit")
        if own_fit is None:
            continue
        try:
            fit_sources.append(inspect.getsource(own_fit))
        except (OSError, TypeError):  # pragma: no cover - source is available
            continue
    if not fit_sources:
        return False

    effective = fit_sources[0]
    # A fit that builds its result in a method of its own -- the tail it shares
    # with the batched executor -- is read together with that method.
    for name in re.findall(r"self\.(_\w+_result)\(", effective):
        effective += inspect.getsource(getattr(client, name))
    measures = all(name in effective for name in COST_METRICS)
    if not measures and "super().fit(" in effective:
        measures = any(all(name in source for name in COST_METRICS) for source in fit_sources[1:])
    # An ancestor computing the cost is not enough: SCAFFOLD and FedProx
    # override fit outright and build their own metrics dict, so until 13 F2
    # they computed nothing at all despite inheriting from a client that does.
    return measures and "filter_metrics(" in effective


def _configs() -> list[tuple[str, dict]]:
    loaded = []
    for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
        if is_family_base(path):
            continue
        document = load_config_mapping(path)
        if isinstance(document, dict) and isinstance(document.get("client"), dict):
            loaded.append((path, document))
    return loaded


class FilterMechanismTest(unittest.TestCase):
    """The premise, pinned so it is not taken on trust."""

    def test_a_name_absent_from_the_list_is_dropped_without_a_sound(self) -> None:
        computed = {"fit_loss": 1.0, "communicated_bytes": 4096.0}
        self.assertEqual(filter_metrics(computed, ["fit_loss"]), {"fit_loss": 1.0})

    def test_an_empty_list_keeps_everything(self) -> None:
        """Which is why the unfiltered fedavg arms never lost the column."""

        computed = {"fit_loss": 1.0, "communicated_bytes": 4096.0}
        self.assertEqual(filter_metrics(computed, []), computed)


class ShippedConfigTest(unittest.TestCase):
    """Every config whose client both measures and filters must ask for it."""

    def _rules_that_measure_and_filter(self) -> set[str]:
        register_builtin_components()
        rules = set()
        for rule in client_updates.builtin():
            client = _client_class(rule)
            if client is not None and _measures_and_filters(client):
                rules.add(rule)
        return rules

    def test_the_filtering_clients_are_the_ones_at_risk(self) -> None:
        """If this set changes, the config check below has to be revisited."""

        self.assertEqual(
            self._rules_that_measure_and_filter(),
            {"delta_sgd", "fedlada", "fedlalr", "fedprox", "scaffold"},
        )

    def test_no_run_gives_a_client_a_list(self) -> None:
        """The factory hands a client every key but ``metrics``, so none filters."""

        from fedbrew.core import factory

        source = inspect.getsource(factory)
        self.assertNotIn("config.client.metrics", source)
        self.assertNotIn("metrics", factory.EXTENSION_CLIENT_KEYS)

    def test_every_config_the_audit_named_carries_it(self) -> None:
        expected = {
            "configs/femnist/delta_sgd.yaml",
            "configs/femnist/fedlalr.yaml",
            "configs/femnist/scaffold.yaml",
            "configs/femnist/fedprox.yaml",
        }
        carrying = {
            path
            for path, document in _configs()
            if all(
                name in ((document.get("reporting") or {}).get("fit_metrics") or [])
                for name in COST_METRICS
            )
        }
        self.assertEqual(expected - carrying, set())


if __name__ == "__main__":
    unittest.main()
