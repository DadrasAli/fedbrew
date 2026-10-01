"""Metric filtering and JSON-serialisation helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
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
#: strategy name in the server registry. Each is added *before* the server's
#: one filter_metrics pass, so ``reporting.fit_metrics`` governs them like
#: every other fit-side column; the plan header needs the names to list them,
#: and the alternative is guessing.
#:
#: tests/test_metric_filter_scope.py aggregates a round on each of these two
#: servers and diffs what they actually emitted against this mapping, so a
#: diagnostic added to a server and not to this dict fails there rather than
#: producing a wrong plan header.
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


#: What a client adds to its fit result *after* its own ``filter_metrics``,
#: keyed by the update rule in the client registry. A run never gives a
#: client a list -- ``reporting.fit_metrics`` is the server's, applied once to
#: the whole round -- so a built client reports everything it computes and
#: this matters only to a client constructed with a ``metrics`` list, as tests
#: do. It is also the FedAvg family's share of RULE_FIT_METRICS below.
#:
#: The convention is not applied uniformly. The rules built on
#: TorchSGDClient.fit and FedAvgClient.fit filter the task metrics and then
#: add their extras, so those extras always survive; fedprox, scaffold,
#: delta_sgd and fedlalr assemble everything first and filter the lot, so for
#: those four nothing survives a list that does not name it. Those four have
#: no row.
#:
#: tests/test_divergence_metric_reachable.py runs every rule's fit under a
#: metrics list naming nothing real and diffs what survived against this
#: mapping, so a client that starts or stops filtering its extras fails there.
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


_VOLUME = frozenset({"communicated_parameters", "communicated_bytes"})

#: Everything an update rule's fit result carries beside the task's own
#: ``fit_<metric>`` names, as it reaches the server: the rule's algorithm
#: columns (docs/08 §4.2). The server diagnostics are
#: SERVER_DIAGNOSTIC_METRICS; ``reporting.fit_metrics`` filters both, and the
#: task's, once at the server. The plan header's fit columns
#: are built from this, and tests/test_planned_columns_are_written.py runs
#: every rule on every task to hold it to what is written. An update rule
#: not listed here (an extension) is planned with the task's metrics alone.
RULE_FIT_METRICS: Mapping[str, frozenset[str]] = {
    **CLIENT_UNFILTERED_FIT_METRICS,
    "fedprox": _VOLUME | {"fit_proximal_loss", "fit_total_loss"},
    "scaffold": _VOLUME | {"control_delta_norm", "client_control_norm", "local_steps"},
    "delta_sgd": _BASE_CLIENT_FIT_METRICS
    | {
        "client_eta_0",
        "client_step_size_mean",
        "client_step_size_min",
        "client_step_size_max",
        "client_step_size_final",
        "step_size_clamp_fraction",
        "undefined_curvature_fraction",
    },
    "fedlalr": _VOLUME
    | {
        "client_alpha",
        "local_steps",
        "optimizer_steps",
        "effective_learning_rate_coordinate_mean",
        "effective_learning_rate_coordinate_min",
        "effective_learning_rate_coordinate_max",
    },
}

#: The rule columns that exist only on a round with a post-fit pass, because
#: they are built from the task's fit_loss: fedprox's fit_total_loss.
POST_FIT_RULE_METRICS = frozenset({"fit_total_loss"})


#: The split a round-record name starts with. ``fit_`` is the post-fit pass,
#: the other four the evaluation passes; ``personal_`` goes in front of a
#: personalized pass's split and is taken off first.
#: evaluation.grad_norm's column (fedbrew/core/grad_norm.py): a squared
#: gradient norm, better at zero whatever the task.
GRAD_NORM_COLUMN = "grad_norm_sq"
GRAD_NORM_DIRECTION = "min"

#: The suffix of a column holding the exact mean of another column over the
#: run's iterates (``convergence.metrics``, fedbrew/core/convergence.py): the
#: column keeps the direction of the one it is the mean of.
RUNNING_MEAN_SUFFIX = "_running_mean"

_SPLIT_PREFIXES = ("central_test_", "fit_", "train_", "val_", "test_")
#: Aggregates of one metric across clients that keep its direction: an
#: average, an extreme or the worst-percent mean of accuracies is still better
#: higher. ``_std``, ``_variance`` and ``_num_clients`` are not.
_SAME_DIRECTION_AGGREGATE = re.compile(r"_(?:sample_weighted_avg|avg|min|max|worst[0-9p]+)$")


def declared_direction(name: str, directions: Mapping[str, str]) -> str | None:
    """ "min" or "max" for a round-record column, from the task's declared metrics.

    ``directions`` is the task's ``METRICS``. The column is a split prefix,
    one declared metric and, for an evaluation aggregate, a suffix that keeps
    the metric's direction: ``fit_accuracy``, ``val_loss_sample_weighted_avg``,
    ``test_accuracy_worst10`` and ``central_test_optimality_gap`` all have
    one. None for anything else -- a rule's own column such as
    ``fit_proximal_loss``, a server diagnostic, a spread.
    """

    if name.endswith(RUNNING_MEAN_SUFFIX):
        return declared_direction(name.removesuffix(RUNNING_MEAN_SUFFIX), directions)
    if name == GRAD_NORM_COLUMN:
        return GRAD_NORM_DIRECTION
    rest = name.removeprefix("personal_")
    for prefix in _SPLIT_PREFIXES:
        if rest.startswith(prefix):
            rest = rest[len(prefix) :]
            break
    else:
        return None
    if rest in directions:
        return directions[rest]
    base = _SAME_DIRECTION_AGGREGATE.sub("", rest)
    if base != rest and base in directions:
        return directions[base]
    return None


def filter_metrics(
    metrics: dict[str, float],
    requested: list[str],
) -> dict[str, float]:
    """Return all metrics or only requested metric keys that exist.

    An empty ``requested`` keeps everything. A non-empty one keeps the named
    metrics that exist and silently ignores names that do not, so a list may
    name a metric only some arms emit.

    A run applies it once, at the server, with ``reporting.fit_metrics``: to
    the aggregated round after the strategy has added its own diagnostics
    (scaffold.py, fedlalr.py), so every fit-side column goes through the same
    list. Config load refuses a divergence.metric the list would drop, which
    is the one column whose absence would silence something.
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


#: What each base metric measures, as a noun phrase that reads mid-sentence,
#: for a task that declares nothing (``TaskAdapter.METRIC_GLOSSES``): such a
#: task is taken to be classification-shaped, as every consumer does. A task
#: that declares its metrics says what its own ``loss`` is -- a quadratic's is
#: not a cross-entropy, which is what every loss was glossed as before.
#:
#: The keys must be exactly `fedbrew.core.config.CLIENT_METRIC_BASES`. That
#: tuple is not imported here: this module is a leaf by design -- config.py
#: imports *it*, lazily, from inside two functions -- so importing back would
#: be a cycle. tests/test_metric_glosses.py diffs the two instead, which is
#: what keeps a third base metric from arriving with no gloss.
METRIC_BASE_GLOSSES: Mapping[str, str] = {
    "loss": "cross-entropy",
    "accuracy": "top-1 accuracy",
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
    "sample_weighted_avg": (
        "pooled over examples (active target tokens for causal LM) — the largest clients "
        "move it most"
    ),
    "avg": ("averaged over clients — a 9-example client counts as much as a 900-example one"),
    "std": "spread across clients (population standard deviation)",
    "variance": "spread across clients, before the square root (population variance)",
    "min": "the single lowest client value",
    "max": "the single highest client value",
    "worst{P}": "the mean over the worst {P}% of clients — the tail, not the average",
}


def _task_metric_gloss(name: str, glosses: Mapping[str, str]) -> str | None:
    """A task metric's own column -- ``fit_<m>``, ``central_test_<m>`` -- from what ``m`` is.

    ``glosses`` says what each of the task's metrics measures; None for a
    column that is not one of them.
    """

    if name == GRAD_NORM_COLUMN and name in glosses:
        return f"{_sentence(glosses[name])}."
    if name == "fit_total_loss":
        loss = glosses.get("loss", "loss")
        return f"Example-weighted mean client ({loss} + proximal loss) after local training."
    metric = name.removeprefix("fit_")
    if metric != name and metric in glosses:
        return (
            f"Example-weighted mean {glosses[metric]} of selected clients' post-fit "
            "local models on their train sets."
        )
    metric = name.removeprefix("central_test_")
    if metric != name and metric in glosses:
        return f"{_sentence(glosses[metric])} of the global model on the complete global test set."
    return None


def _sentence(phrase: str) -> str:
    """A mid-sentence noun phrase at the start of one: its first letter raised."""

    return phrase[:1].upper() + phrase[1:]


#: Columns that are not built from a split, a base and a suffix: the fit
#: phase's own numbers, the central pass, the client counts and each
#: algorithm's diagnostics. The first five are a task metric's own columns,
#: here as a task that declares nothing (classification) reads them;
#: ``metric_gloss`` composes them from the run's task instead when it is
#: given one (``_task_metric_gloss``), so the two cannot disagree.
FIXED_METRIC_GLOSSES: Mapping[str, str] = {
    **{
        name: str(_task_metric_gloss(name, METRIC_BASE_GLOSSES))
        for name in (
            "fit_loss",
            "fit_accuracy",
            "fit_total_loss",
            "central_test_loss",
            "central_test_accuracy",
        )
    },
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
    "fit_proximal_loss": (
        "Example-weighted mean client (mu/2) * ||w_local - w_round_start||^2 after local training."
    ),
    "control_delta_norm": (
        "Example-weighted mean ||c_i,new - c_i,old||_2 across selected clients (float64)."
    ),
    "client_control_norm": (
        "Example-weighted mean ||c_i,new||_2 across selected clients (float64)."
    ),
    "local_steps": (
        "Example-weighted mean optimizer steps per selected client: one per "
        "minibatch, or one per iteration under update_mode: full_gradient."
    ),
    "mean_client_control_delta_norm": (
        "Unweighted mean ||c_i,new - c_i,old||_2 across selected clients (float64)."
    ),
    # evaluation.grad_norm's column, as a task that says nothing more reads
    # it; a task's own gloss (TaskAdapter.GRAD_NORM_GLOSS) says what its F is.
    GRAD_NORM_COLUMN: (
        "Squared norm ||grad F(x)||^2 of the gradient of the global objective F at the "
        "global model: the task's training loss over every client's train split, "
        "weighted as the loss averages, in the trainable parameters (float64)."
    ),
    "server_control_norm": (
        "||c_server,new||_2 after adding sum(selected client deltas) / all clients (float64)."
    ),
    # The remaining diagnostics had no gloss at all before the plan header
    # needed one, and fell through to a sentence generated from the column
    # name -- which for `communicated_bytes` produced "Example-weighted mean
    # client communicated bytes after local training on client train sets",
    # a description of a cost accumulator as if it were a loss.
    "optimizer_steps": (
        "Optimizer steps a client actually took this round: one per minibatch, "
        "or one per iteration under update_mode: frozen_batch_gradients or "
        "full_gradient."
    ),
    "active_target_tokens": (
        "Non-padding, non-prompt target tokens a client trained on this round; 0.0 "
        "for a task whose train step reports no count (classification, the examples)."
    ),
    "trainable_parameters": "Parameters a client updated locally, after any freezing.",
    "communicated_parameters": (
        "Parameters a client sent back to the server this round; the round's value is "
        "the example-weighted mean over clients, not their total."
    ),
    "communicated_bytes": (
        "Bytes a client sent back this round, at the dtype the tensors are stored in; "
        "the round's value is the example-weighted mean over clients, not their total."
    ),
    "client_learning_rate": "The step size a client actually used, after any schedule.",
    "client_alpha": "FedLALR's base learning rate alpha, as the client was configured with it.",
    "client_eta_0": "Delta-SGD's starting step size eta_0, as the client was configured with it.",
    "client_step_size_mean": (
        "Mean of the step sizes Delta-SGD chose over a client's local steps this round."
    ),
    "client_step_size_min": "The smallest step size Delta-SGD chose over a client's local steps.",
    "client_step_size_max": "The largest step size Delta-SGD chose over a client's local steps.",
    "client_step_size_final": "The step size of a client's last local step this round.",
    "step_size_clamp_fraction": (
        "Fraction of a client's local steps where client.eta_max bound the step size "
        "rather than the local smoothness estimate."
    ),
    "undefined_curvature_fraction": (
        "Fraction of a client's local steps where the smoothness estimate was undefined "
        "(no parameter or gradient change) and only the growth term applied."
    ),
    "momentum_norm": (
        "||m||_2 of the server's first-moment buffer after this round's update (float64)."
    ),
    "second_moment_norm": (
        "||v||_2 of the server's second-moment buffer after this round's update (float64)."
    ),
    "effective_learning_rate_coordinate_mean": (
        "A client's per-coordinate step size alpha/sqrt(v_hat) in float32, averaged over "
        "its coordinates; across clients, example-weighted."
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


def metric_gloss(
    name: str,
    *,
    split_glosses: Mapping[str, str] | None = None,
    metric_glosses: Mapping[str, str] | None = None,
) -> str:
    """One plain-language sentence for one column name.

    `split_glosses` overrides which data a split is measured on. The train
    split is the only caller of it today: under
    `evaluation.train.clients: participating` it is this round's trainers
    rather than every client, and a gloss that said "every client" would be
    describing a different measurement than the one the run performs.

    `metric_glosses` says what each of the task's metrics measures
    (``config.task_metric_glosses``, from ``TaskAdapter.METRIC_GLOSSES``);
    without it, a classification task's.
    """

    if name.endswith(RUNNING_MEAN_SUFFIX):
        base = metric_gloss(
            name.removesuffix(RUNNING_MEAN_SUFFIX),
            split_glosses=split_glosses,
            metric_glosses=metric_glosses,
        )
        return (
            "Exact mean over the run's iterates x_1 .. x_t (the global model after each of "
            f"rounds 1 to t) of this metric, the expected value at a uniformly random output "
            f"iterate: {base[:1].lower()}{base[1:]}"
        )
    glosses = METRIC_BASE_GLOSSES if metric_glosses is None else metric_glosses
    task = _task_metric_gloss(name, glosses)
    if task is not None:
        return task
    fixed = FIXED_METRIC_GLOSSES.get(name)
    if fixed is not None:
        return fixed

    composed = _composed_gloss(name, split_glosses or SPLIT_GLOSSES, glosses)
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


def _composed_gloss(
    name: str, split_glosses: Mapping[str, str], metric_glosses: Mapping[str, str]
) -> str | None:
    """`{split}_{base}_{suffix}` read back as a sentence, or None."""

    personal = name.startswith("personal_")
    remainder = name.removeprefix("personal_") if personal else name

    for split, data in split_glosses.items():
        for base, default in METRIC_BASE_GLOSSES.items():
            measured = _sentence(metric_glosses.get(base, default))
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
