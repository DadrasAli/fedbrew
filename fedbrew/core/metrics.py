"""Metric filtering and JSON-serialisation helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with null so the record stays valid JSON.

    Every writer that serialises measured metrics has to run its record
    through this. json.dumps defaults to allow_nan=True and emits the bare
    tokens NaN and Infinity, which RFC 8259 does not define: Python's json and
    pandas.read_json (since pandas 1.0) read them back, jq 1.6 turns NaN into
    null and Infinity into the largest double, and Go, serde_json and
    JSON.parse refuse the file outright. A diverged run's loss is exactly such
    a value.

    It lives here, in a leaf module, so a CLI can import it without pulling in
    the artifact layer. Pair it with allow_nan=False at the json.dumps call,
    which turns a field that slipped past it into a loud failure rather than
    an invalid file.
    """

    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


#: Metrics a server strategy adds to the round record itself, keyed by the
#: strategy name in the server registry. Every one is added *after*
#: filter_metrics, so ``server.metrics`` never governs them -- which is what
#: makes this list worth having in a leaf module: preflight needs to know
#: which names a metrics list is unable to remove, and the alternative is
#: guessing.
#:
#: tests/test_metric_filter_scope.py aggregates a round on each of these two
#: servers and diffs what they actually emitted against this mapping, so a
#: diagnostic added to a server and not to this dict fails there rather than
#: producing a wrong preflight verdict.
SERVER_DIAGNOSTIC_METRICS: Mapping[str, frozenset[str]] = {
    "scaffold": frozenset(
        {
            "server_control_norm",
            "mean_client_control_delta_norm",
        }
    ),
    "fedlalr": frozenset(
        {
            "momentum_norm",
            "second_moment_norm",
            "effective_learning_rate_across_clients_mean",
            "effective_learning_rate_across_clients_std",
            "effective_learning_rate_across_clients_min",
            "effective_learning_rate_across_clients_max",
        }
    ),
}

#: The server diagnostics that are built from a client metric, and so exist
#: only in a round whose clients reported it. FedLALR's spread of effective
#: rates across clients is computed from each client's
#: ``effective_learning_rate_coordinate_mean``; a client whose
#: ``client.metrics`` does not name it filters it out, the server has nothing
#: to spread, and the four columns are absent from the round. Config load
#: refuses a config that asks for one of the four -- in ``divergence.metric``
#: or either metrics list -- while ``client.metrics`` filters out its source
#: (FINDINGS.csv POST-F12).
SERVER_DIAGNOSTIC_SOURCES: Mapping[str, Mapping[str, str]] = {
    "fedlalr": {
        f"effective_learning_rate_across_clients_{statistic}": (
            "effective_learning_rate_coordinate_mean"
        )
        for statistic in ("mean", "std", "min", "max")
    },
}

#: Metric names no longer written, mapped to what replaced them. FedLALR's
#: client and server both wrote ``client_effective_learning_rate_min``,
#: ``_max`` and ``_mean`` as different quantities -- one client's statistic
#: over its coordinates, and a statistic across clients of those clients'
#: means -- and in one config shape the round record carried the first under
#: the second's name (FINDINGS.csv POST-F14). Both families were renamed, so
#: no old name keeps a meaning a config written for the other could read.
#: A config naming one is refused at load, and a resume onto CSVs whose
#: header carries one is refused, so no file mixes the two spellings.
RETIRED_METRIC_NAMES: Mapping[str, str] = {
    "client_effective_learning_rate_mean": (
        "effective_learning_rate_coordinate_mean for one client's mean over its "
        "coordinates, or effective_learning_rate_across_clients_mean for the "
        "mean across clients of those means"
    ),
    "client_effective_learning_rate_min": (
        "effective_learning_rate_coordinate_min for one client's minimum over its "
        "coordinates, or effective_learning_rate_across_clients_min for the "
        "minimum across clients of their coordinate means"
    ),
    "client_effective_learning_rate_max": (
        "effective_learning_rate_coordinate_max for one client's maximum over its "
        "coordinates, or effective_learning_rate_across_clients_max for the "
        "maximum across clients of their coordinate means"
    ),
    "client_effective_learning_rate_std": (
        "effective_learning_rate_across_clients_std, the population standard "
        "deviation across clients of their coordinate means"
    ),
}


def retired_metric_message(name: str) -> str:
    """What to write instead of the retired metric ``name``."""

    return (
        f"{name!r} is a retired FedLALR diagnostic name, split into two by what "
        f"it computes: use {RETIRED_METRIC_NAMES[name]}. Chapter 08 section 7.3"
    )


def server_diagnostic_metrics(strategy: str, client_metrics: Sequence[str]) -> frozenset[str]:
    """The diagnostics ``strategy`` adds to a round, given the clients' metrics list.

    :data:`SERVER_DIAGNOSTIC_METRICS` less any whose source metric the clients
    filter out. An empty ``client_metrics`` keeps everything, as
    :func:`filter_metrics` does, so every source reaches the server.
    """

    names = SERVER_DIAGNOSTIC_METRICS.get(strategy, frozenset())
    sources = SERVER_DIAGNOSTIC_SOURCES.get(strategy, {})
    requested = set(client_metrics)
    return frozenset(
        name for name in names if name not in sources or not requested or sources[name] in requested
    )


#: What a client adds to its fit result *after* ``filter_metrics``, keyed by
#: the update rule in the client registry. The client-side twin of
#: SERVER_DIAGNOSTIC_METRICS, and needed for the same reason: preflight has to
#: know which names ``client.metrics`` is unable to remove.
#:
#: The two halves of the convention below are not applied uniformly. The rules
#: built on TorchSGDClient.fit and FedAvgClient.fit filter the task metrics and
#: then add their extras, so those extras always survive; fedprox, scaffold,
#: delta_sgd and fedlalr assemble everything first and filter the lot, so for
#: those four nothing survives a list that does not name it. Those four have no
#: row.
#:
#: tests/test_divergence_metric_reachable.py runs every rule's fit under a
#: metrics list naming nothing real and diffs what survived against this
#: mapping, so a client that starts or stops filtering its extras fails there
#: rather than producing a wrong preflight verdict.
_BASE_CLIENT_FIT_METRICS = frozenset(
    {
        "optimizer_steps",
        "active_target_tokens",
        "trainable_parameters",
        "communicated_parameters",
        "communicated_bytes",
    }
)

CLIENT_UNFILTERED_FIT_METRICS: Mapping[str, frozenset[str]] = {
    "local_sgd": _BASE_CLIENT_FIT_METRICS,
    "local_adamw": _BASE_CLIENT_FIT_METRICS,
    "fedavg": _BASE_CLIENT_FIT_METRICS | {"client_learning_rate"},
    "centralized": _BASE_CLIENT_FIT_METRICS | {"client_learning_rate"},
    "fedavg_ft": _BASE_CLIENT_FIT_METRICS | {"client_learning_rate"},
}


def client_fit_extras(update_rule: str) -> frozenset[str]:
    """What this rule's fit result carries past the client's own filter."""

    return CLIENT_UNFILTERED_FIT_METRICS.get(update_rule, frozenset())


def surviving_client_fit_extras(update_rule: str, server_metrics: Sequence[str]) -> list[str]:
    """The rule's extras that reach round_metrics.csv, sorted.

    The client exempts its extras from ``client.metrics`` and then the server
    runs the whole aggregated dict through :func:`filter_metrics` again against
    ``server.metrics``, which un-exempts them. So the convention holds on one
    side of the round and not the other, and the names most likely to be
    listed in a config -- ``optimizer_steps``, ``client_learning_rate`` -- are
    exactly the ones that vanish.

    An empty ``server_metrics`` keeps everything, matching
    :func:`filter_metrics`, so everything survives.
    """

    extras = client_fit_extras(update_rule)
    requested = list(server_metrics)
    if not requested:
        return sorted(extras)
    return sorted(extras & set(requested))


def dropped_client_fit_extras(update_rule: str, server_metrics: Sequence[str]) -> list[str]:
    """The rule's extras that ``server.metrics`` removes, sorted.

    The complement of :func:`surviving_client_fit_extras`, which is what
    preflight reports: a config naming one of these under ``client.metrics``
    reads as though it asked for a column it will not get.
    """

    surviving = set(surviving_client_fit_extras(update_rule, server_metrics))
    return sorted(client_fit_extras(update_rule) - surviving)


def filter_metrics(
    metrics: dict[str, float],
    requested: list[str],
) -> dict[str, float]:
    """Return all metrics or only requested metric keys that exist.

    An empty ``requested`` keeps everything. A non-empty one keeps the named
    metrics that exist and silently ignores names that do not, so a list may
    name a metric only some arms emit.

    The convention on both sides is that this filter governs *measured* metrics
    and nothing else: clients apply it to the task metrics and then add their
    algorithm extras (torch_sgd_client.py), and the two servers that produce
    diagnostics apply it to the aggregated client metrics and then add theirs
    (scaffold.py, fedlalr.py). Filtering a framework
    diagnostic would let a metrics list silently drop a column that
    checkpointing.best_metric or the divergence guard names.
    """

    if not requested:
        return dict(metrics)
    return {name: metrics[name] for name in requested if name in metrics}


# ---------------------------------------------------------------------------
# Plain-language glosses
# ---------------------------------------------------------------------------
#
# One sentence per column, for the plan header printed before a run starts and
# for `--validate-only`. A column name is not self-explanatory to anyone who
# did not write it, and two of them -- `_avg` and `_sample_weighted_avg` --
# differ by numbers that look interchangeable and by nothing else in the name.
# The glosses have to separate those two in words, because that is the pair a
# reader most often quotes the wrong member of.
#
# Composed from base plus suffix rather than written out per column. There are
# twelve columns per split, three splits, optionally doubled by
# `evaluation.model_scope: both` -- 72 hand-written sentences, of which 71
# would say the same six things. Composition also means a column cannot exist
# without a gloss: `client_metric_names` builds a name from a base and a
# suffix, and `metric_gloss` reads the same two pieces back.


#: What each base metric measures, as the opening noun phrase of a sentence.
#:
#: The keys must be exactly `fedbrew.core.config.CLIENT_METRIC_BASES`. That
#: tuple is not imported here: this module is a leaf by design -- config.py
#: imports *it*, lazily, from inside two functions -- so importing back would
#: be a cycle. tests/test_metric_glosses.py diffs the two instead, which is
#: what keeps a third base metric from arriving with no gloss.
METRIC_BASE_GLOSSES: Mapping[str, str] = {
    "loss": "Cross-entropy",
    "accuracy": "Top-1 accuracy",
}

#: The data each evaluated split is measured on.
SPLIT_GLOSSES: Mapping[str, str] = {
    "train": "client train data",
    "val": "client validation data",
    "test": "client test data",
}

#: Inserted for a `personal_`-prefixed split. The prefix names the *model*, not
#: the data: the same examples, measured with a different set of weights.
PERSONAL_GLOSS = " under each client's own model"

#: How the per-client numbers were combined. This is the half of the name a
#: reader skims, and the half that decides what the number means.
METRIC_SUFFIX_GLOSSES: Mapping[str, str] = {
    "sample_weighted_avg": "pooled over examples — the largest clients move it most",
    "avg": ("averaged over clients — a 9-example client counts as much as a 900-example one"),
    "std": "spread across clients (population standard deviation)",
    "variance": "spread across clients, before the square root (population variance)",
    "min": "the single lowest client value",
    "max": "the single highest client value",
    "worst{P}": "the mean over the worst {P}% of clients — the tail, not the average",
}

#: Columns that are not built from a split, a base and a suffix, and so cannot
#: be composed: the fit phase's own numbers, the central pass, and each
#: algorithm's diagnostics.
FIXED_METRIC_GLOSSES: Mapping[str, str] = {
    "fit_loss": (
        "Example-weighted mean cross-entropy of selected clients' post-fit "
        "local models on their train sets."
    ),
    "fit_accuracy": (
        "Correct predictions / examples for selected clients' post-fit local "
        "models on their train sets."
    ),
    # One per split, and the personal_ twins classify through the same three:
    # classify_metric strips that prefix before looking a fixed name up.
    "train_num_clients": (
        "Clients whose train aggregates this round is over, after dropping "
        "any reporting zero train examples."
    ),
    "val_num_clients": (
        "Clients whose val aggregates this round is over, after dropping any "
        "reporting zero val examples. Two runs at different evaluation.val."
        "clients settings produce the same column names over different "
        "populations; this is the number that tells them apart."
    ),
    "test_num_clients": (
        "Clients whose test aggregates this round is over, after dropping any "
        "reporting zero test examples."
    ),
    "central_test_loss": "Average loss of the global model on the complete global test set.",
    "central_test_accuracy": (
        "Accuracy of the global model on the complete global test set. This is "
        "computed as total correct predictions divided by total test samples."
    ),
    "fit_proximal_loss": (
        "Example-weighted mean client (mu/2) * ||w_local - w_round_start||^2 after local training."
    ),
    "fit_total_loss": (
        "Example-weighted mean client (cross-entropy + proximal loss) after local training."
    ),
    "control_delta_norm": (
        "Example-weighted mean ||c_i,new - c_i,old||_2 across selected clients."
    ),
    "client_control_norm": "Example-weighted mean ||c_i,new||_2 across selected clients.",
    "local_steps": "Example-weighted mean minibatch optimizer steps per selected client.",
    "mean_client_control_delta_norm": (
        "Unweighted mean ||c_i,new - c_i,old||_2 across selected clients."
    ),
    "server_control_norm": (
        "||c_server,new||_2 after adding sum(selected client deltas) / all clients."
    ),
    # The remaining diagnostics had no gloss at all before the plan header
    # needed one, and fell through to a sentence generated from the column
    # name -- which for `communicated_bytes` produced "Example-weighted mean
    # client communicated bytes after local training on client train sets",
    # a description of a cost accumulator as if it were a loss.
    "optimizer_steps": "Minibatch optimizer steps a client actually took this round.",
    "active_target_tokens": (
        "Non-padding, non-prompt target tokens a client trained on this round."
    ),
    "trainable_parameters": "Parameters a client updated locally, after any freezing.",
    "communicated_parameters": "Parameters a client sent back to the server this round.",
    "communicated_bytes": (
        "Bytes a client sent back this round, at the dtype the tensors are stored in."
    ),
    "client_learning_rate": "The step size a client actually used, after any schedule.",
    "momentum_norm": "||m||_2 of the server's first-moment buffer after this round's update.",
    "second_moment_norm": "||v||_2 of the server's second-moment buffer after this round's update.",
    "effective_learning_rate_coordinate_mean": (
        "A client's per-coordinate step size alpha/sqrt(v_hat), averaged over its "
        "coordinates; across clients, example-weighted."
    ),
    "effective_learning_rate_coordinate_min": (
        "A client's smallest per-coordinate step size; across clients, the "
        "example-weighted mean of those minima."
    ),
    "effective_learning_rate_coordinate_max": (
        "A client's largest per-coordinate step size; across clients, the "
        "example-weighted mean of those maxima."
    ),
    "effective_learning_rate_across_clients_mean": (
        "Unweighted mean across clients of each client's mean step size."
    ),
    "effective_learning_rate_across_clients_std": (
        "Spread of the clients' mean step sizes -- large values mean the clients "
        "disagreed about how far to step."
    ),
    "effective_learning_rate_across_clients_min": (
        "The smallest mean step size any client derived this round."
    ),
    "effective_learning_rate_across_clients_max": (
        "The largest mean step size any client derived this round."
    ),
}

_WORST_SUFFIX = re.compile(r"^worst([0-9p]+)$")


def metric_gloss(name: str, *, split_glosses: Mapping[str, str] | None = None) -> str:
    """One plain-language sentence for one column name.

    `split_glosses` overrides which data a split is measured on. The train
    split is the only caller of it today: under
    `evaluation.train.clients: participating` it is this round's trainers
    rather than every client, and a gloss that said "every client" would be
    describing a different measurement than the one the run performs.
    """

    fixed = FIXED_METRIC_GLOSSES.get(name)
    if fixed is not None:
        return fixed

    composed = _composed_gloss(name, split_glosses or SPLIT_GLOSSES)
    if composed is not None:
        return composed

    # Unknown column. Say what can be said from the name rather than nothing:
    # a new algorithm diagnostic reaches the header before anyone writes it a
    # gloss, and a blank cell reads as a bug in the run.
    if name.startswith("central_test_"):
        # A task's own central-pass diagnostic (loop._evaluate_central_test_set
        # passes through anything evaluate_global reports besides loss and
        # accuracy) rather than one of the two names above with a gloss
        # already on file.
        return (
            f"{_titled(name.removeprefix('central_test_'))} of the global model "
            "on the complete global test set."
        )
    if name.startswith("global_"):
        return f"{_titled(name.removeprefix('global_'))} of the new global model over all clients."
    return f"Example-weighted mean client {_titled(name).lower()} after local training."


def _composed_gloss(name: str, split_glosses: Mapping[str, str]) -> str | None:
    """`{split}_{base}_{suffix}` read back as a sentence, or None."""

    personal = name.startswith("personal_")
    remainder = name.removeprefix("personal_") if personal else name

    for split, data in split_glosses.items():
        for base, measured in METRIC_BASE_GLOSSES.items():
            prefix = f"{split}_{base}_"
            if not remainder.startswith(prefix):
                continue
            combination = _suffix_gloss(remainder[len(prefix) :])
            if combination is None:
                return None
            data = f"{data}{PERSONAL_GLOSS}" if personal else data
            return f"{measured} on {data}, {combination}."
    return None


def _suffix_gloss(suffix: str) -> str | None:
    known = METRIC_SUFFIX_GLOSSES.get(suffix)
    if known is not None:
        return known
    # The percentage lives in the name, so the text is built from it: worst2p5
    # is as legal a column as worst10, and a lookup table cannot hold both.
    match = _WORST_SUFFIX.match(suffix)
    if match is None:
        return None
    return METRIC_SUFFIX_GLOSSES["worst{P}"].replace("{P}", match.group(1).replace("p", "."))


def _titled(name: str) -> str:
    return name.replace("_", " ").title()
