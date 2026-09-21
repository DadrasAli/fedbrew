"""Multiple-choice accuracy on the held-out MedMCQA split.

The logged ``central_test_accuracy`` is token-level teacher-forced next-token
accuracy over every supervised token, so a model that reproduces the answer
explanation's phrasing scores well without answering anything. This reports the
metric MedMCQA is actually benchmarked on: which of the four options the model
picks, against a 25% random baseline.

The packed ``global_test`` shard cannot be used. Packing concatenates
EOS-separated examples and slices at a fixed stride, which destroys the option
boundaries. This re-reads the parquet snapshot and recovers the same held-out
questions unpacked, by replaying the generator's own row filters and split hash.

Prompt construction goes through ``tokenize_assistant_target`` -- the function
that built the training windows -- so the scored prompt is byte-identical to
what the model was trained on. Any drift there would silently measure a
different distribution.

Two metrics are reported:

``letter``
    One forward pass per question. The four candidate answers diverge at exactly
    one token position (the option letter), so the model's preference is read
    from the logits at that position. This is the headline number.
``option_sum`` / ``option_lengthnorm``
    Four forward passes per question, scoring each complete candidate response
    the way the training loss would. Reported as a robustness check: agreement
    with ``letter`` separates knowing the answer from learning the format.

The two are over **different question counts**, and the record says both.
``evaluated`` is the letter pass's: every question whose prompt plus one letter
token fits inside ``--max-length``. ``option_evaluated`` is the option pass's,
smaller by ``skipped_option_too_long`` -- the questions whose longest complete
candidate response does not fit, which is a strictly longer string than the
letter pass measures.

Both accuracies used to divide by ``evaluated``, so every question the option
pass skipped for length was a guaranteed miss in a denominator it contributed
no numerator to, biasing ``option_sum`` and ``option_lengthnorm`` low with
nothing in the record saying by how much. Measured on the real held-out sets at
the shipped ``--max-length 1024``, replaying the row filters and both length
tests against the Qwen2.5-0.5B tokenizer::

    split         held-out  letter skips  option skips  longest candidate
    client_eval     16,367             0             0          416
    global_test     16,242             0             0          417

Zero at that cap, against 2.5x headroom. At ``--max-length 384`` the second
test does fire. FINDINGS.csv P10-F27.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from fedbrew.core.config import load_config
from fedbrew.core.factory import _add_causal_manifest_metadata, _model_config
from fedbrew.core.metrics import json_safe
from fedbrew.core.registry import register_builtin_components, tasks
from fedbrew.data.generic_sft import (
    _load_dataset_assets,
    _load_dataset_rows,
    _text_or_none,
    render_template,
    resolve_choice_fields,
)
from fedbrew.data.oasst1_sft import (
    IGNORE_INDEX,
    ChatMessage,
    TokenizedSFTExample,
    assign_tree_split,
    tokenize_assistant_target,
)

# The base model has no adapter, so its config keys must be dropped rather than
# passed to the plain causal-LM builder.
LORA_ONLY_KEYS = (
    "adapter_name",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
)

# Recorded in the generated manifest, so a mismatch against the generation
# config means the two describe different data and the eval would be invalid.
CROSS_CHECKED_FIELDS = (
    ("prompt_template", "prompt_template"),
    ("response_template", "response_template"),
    ("system_prompt", "system_prompt"),
    ("group_field", "group_field"),
    ("client_field", "partition_field"),
)


# --------------------------------------------------------------------------
# Held-out question recovery
# --------------------------------------------------------------------------


def _cross_check(sft_config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Fail loudly when the generation config no longer describes the run's data."""

    for config_key, manifest_key in CROSS_CHECKED_FIELDS:
        expected = manifest.get(manifest_key)
        actual = sft_config.get(config_key)
        if expected is None and actual is None:
            continue
        if str(expected or "") != str(actual or ""):
            raise ValueError(
                f"generation config {config_key!r} does not match the run's data "
                f"manifest {manifest_key!r}:\n  manifest: {expected!r}\n  config:   {actual!r}"
            )
    expected_required = [str(name) for name in manifest.get("required_fields", ())]
    actual_required = [str(name) for name in sft_config.get("required_fields", ())]
    if expected_required != actual_required:
        raise ValueError(
            f"generation config required_fields {actual_required} do not match the "
            f"run's data manifest {expected_required}"
        )


def _split_ratios(manifest: Mapping[str, Any]) -> tuple[float, float, float]:
    ratios = manifest.get("split_ratios") or {}
    return (
        float(ratios["train"]),
        float(ratios["client_eval"]),
        float(ratios["global_test"]),
    )


#: The three splits generic_sft assigns. client_eval is the validation half --
#: disjoint from global_test, already carried in the shards, and already
#: evaluated every 5 rounds by the shipped configs.
SPLITS = ("client_eval", "global_test")
DEFAULT_SPLIT = "client_eval"


def _held_out_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sft_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    split: str = DEFAULT_SPLIT,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replay the generator's filters, keeping only rows of one split.

    The filter chain has to match ``generic_sft._prepare_rows`` exactly. Applying
    a looser or stricter one would evaluate on a different question set than the
    one training held out, which no downstream check would catch.

    ``split`` defaults to client_eval, the validation half. This used to be
    hardwired to global_test, which meant the only multiple-choice number the
    tooling could produce was one on the test set, so any arm chosen with it
    was chosen on test-set accuracy.
    """

    counts = Counter[str]()
    counts["source_rows"] = len(rows)
    client_field = str(manifest["partition_field"])
    group_field = str(manifest["group_field"])
    required = tuple(str(name) for name in manifest.get("required_fields", ()))
    choice = sft_config.get("choice")
    prompt_template = str(manifest["prompt_template"])
    response_template = str(manifest["response_template"])
    seed = int(manifest["seed"])
    ratios = _split_ratios(manifest)

    kept: list[dict[str, Any]] = []
    for row in rows:
        client_key = _text_or_none(row.get(client_field))
        group_key = _text_or_none(row.get(group_field))
        if client_key is None or group_key is None:
            counts["rows_missing_client_or_group_key"] += 1
            continue
        if any(_text_or_none(row.get(field)) is None for field in required):
            counts["rows_missing_required_fields"] += 1
            continue
        try:
            values = dict(row)
            values.update(resolve_choice_fields(row, choice))
            prompt = render_template(prompt_template, values)
            response = render_template(response_template, values)
        except ValueError:
            counts["rows_failing_template_rendering"] += 1
            continue
        if not prompt or not response:
            counts["rows_with_empty_prompt_or_response"] += 1
            continue

        counts["qualifying_rows"] += 1
        row_split = assign_tree_split(
            group_key,
            seed=seed,
            train_ratio=ratios[0],
            client_eval_ratio=ratios[1],
            global_test_ratio=ratios[2],
        )
        counts[f"split_{row_split}"] += 1
        if row_split != split:
            continue
        kept.append(
            {
                "id": group_key,
                "subject": client_key,
                "prompt": prompt,
                "answer_index": int(row[str(choice["index_field"])]),
                "options": [str(row.get(str(name), "")) for name in choice["option_fields"]],
            }
        )
    return kept, dict(counts)


def _subsample(questions: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Deterministic uniform subsample, independent of the split hash.

    A different salt from the split hash keeps the subsample from correlating
    with which questions landed in global_test.
    """

    if limit is None or limit >= len(questions):
        return questions
    ordered = sorted(
        questions,
        key=lambda item: hashlib.sha256(f"mc-eval:{item['id']}".encode()).digest(),
    )
    return ordered[:limit]


# --------------------------------------------------------------------------
# Candidate construction
# --------------------------------------------------------------------------


def _candidate_examples(
    question: Mapping[str, Any],
    *,
    labels: Sequence[str],
    system_prompt: str | None,
    answer_prefix: str,
    tokenizer: Any,
) -> list[TokenizedSFTExample]:
    """Tokenize the four complete candidate responses through the training path."""

    examples: list[TokenizedSFTExample] = []
    for index, label in enumerate(labels):
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=question["prompt"]))
        messages.append(
            ChatMessage(
                role="assistant",
                content=f"{answer_prefix}{label}. {question['options'][index]}".strip(),
            )
        )
        examples.append(tokenize_assistant_target(messages, tokenizer))
    return examples


def _letter_positions(
    question: Mapping[str, Any],
    *,
    labels: Sequence[str],
    system_prompt: str | None,
    answer_prefix: str,
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Return the shared context ids and the four diverging letter token ids.

    Rather than assuming ``" A"`` tokenizes to a single id, this locates the
    position where the four candidate token sequences first differ. That
    position is the option letter by construction, whatever the tokenizer does
    with spacing.
    """

    sequences = []
    for label in labels:
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=question["prompt"]))
        messages.append(ChatMessage(role="assistant", content=f"{answer_prefix}{label}".strip()))
        sequences.append(list(tokenize_assistant_target(messages, tokenizer).token_ids))

    shortest = min(len(sequence) for sequence in sequences)
    divergence = shortest
    for position in range(shortest):
        column = {sequence[position] for sequence in sequences}
        if len(column) > 1:
            divergence = position
            break
    else:
        raise ValueError("the candidate answers never diverge; check answer_prefix")

    letter_ids = [sequence[divergence] for sequence in sequences]
    if len(set(letter_ids)) != len(letter_ids):
        raise ValueError(
            f"option letters share a token id at the divergence position: {letter_ids}"
        )
    return sequences[0][:divergence], letter_ids


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _pad_batch(
    sequences: Sequence[Sequence[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths)
    input_ids = torch.zeros((len(sequences), width), dtype=torch.long)
    attention = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention[row, : len(sequence)] = 1
    return input_ids.to(device), attention.to(device), lengths


@dataclass
class ChoiceTotals:
    """What one evaluation counted, with the two denominators kept apart.

    `scored` is the letter pass's: every question whose shared context plus
    one letter token fits inside `max_length`. `option_scored` is the option
    pass's, and is smaller by the questions whose longest *complete* candidate
    response does not fit -- a strictly longer string, so clearing the first
    test says nothing about the second.

    They were one number. `accuracy_option_sum` and
    `accuracy_option_lengthnorm` divided by `scored`, so every question the
    option pass skipped for length was a guaranteed miss in their numerator's
    denominator, biasing both low, and nothing in the record said how many
    such questions there were. `accuracy_letter` was and is over `scored`,
    which is correct for it. P10-F27.
    """

    scored: int = 0
    option_scored: int = 0
    skipped_too_long: int = 0
    skipped_option_too_long: int = 0
    letter_correct: int = 0
    option_sum_correct: int = 0
    option_norm_correct: int = 0
    predictions: Counter[str] = field(default_factory=Counter)


def accuracy_record(
    totals: ChoiceTotals,
    *,
    labels: Sequence[str],
    letter_only: bool,
) -> dict[str, Any]:
    """The measured half of the output record, each rate over its own count.

    Args:
        totals: What `score_questions` counted.
        labels: The choice letters, in order, for the prediction histogram.
        letter_only: Whether the option pass ran at all. When it did not, its
            three keys are absent rather than zero -- a zero would read as a
            measurement, and there was none.

    Returns:
        The counts and rates, with `evaluated` and `option_evaluated` beside
        the accuracies they divide by so a reader can check either.
    """

    record: dict[str, Any] = {
        "evaluated": totals.scored,
        "skipped_too_long": totals.skipped_too_long,
        "accuracy_letter": totals.letter_correct / totals.scored if totals.scored else 0.0,
        "prediction_distribution": {label: totals.predictions[label] for label in labels},
    }
    if letter_only:
        return record
    option_scored = totals.option_scored
    record["option_evaluated"] = option_scored
    record["skipped_option_too_long"] = totals.skipped_option_too_long
    record["accuracy_option_sum"] = (
        totals.option_sum_correct / option_scored if option_scored else 0.0
    )
    record["accuracy_option_lengthnorm"] = (
        totals.option_norm_correct / option_scored if option_scored else 0.0
    )
    return record


def score_questions(
    questions: Sequence[Mapping[str, Any]],
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    labels: Sequence[str],
    system_prompt: str | None,
    answer_prefix: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    letter_only: bool = False,
    on_progress: Callable[[int], None] | None = None,
) -> ChoiceTotals:
    """Score every question, counting each pass against its own denominator.

    Lifted out of `main` so the arithmetic has a seam: the defect this
    replaces was one counter serving two passes, and there was nowhere to
    check that from without a checkpoint and a GPU. P10-F27.

    Args:
        questions: The held-out rows, as `_held_out_rows` returns them.
        model: The evaluated model, already on `device` and in eval mode.
        tokenizer: The generator's tokenizer.
        labels: The choice letters, in order.
        system_prompt: The generator's system prompt, or None.
        answer_prefix: The response template up to the answer label.
        device: Where the batches go.
        batch_size: Questions per letter-pass forward.
        max_length: The token cap both passes are held to.
        letter_only: Skip the option pass entirely.
        on_progress: Called with the running `scored` count, occasionally.

    Returns:
        The counts, as `ChoiceTotals`.
    """

    totals = ChoiceTotals()
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        contexts: list[list[int]] = []
        letter_ids: list[list[int]] = []
        usable: list[Mapping[str, Any]] = []
        for question in batch:
            context, ids = _letter_positions(
                question,
                labels=labels,
                system_prompt=system_prompt,
                answer_prefix=answer_prefix,
                tokenizer=tokenizer,
            )
            if len(context) + 1 > max_length:
                totals.skipped_too_long += 1
                continue
            contexts.append(context)
            letter_ids.append(ids)
            usable.append(question)
        if not usable:
            continue

        for index, choice_index in enumerate(_letter_choice(model, contexts, letter_ids, device)):
            totals.predictions[labels[choice_index]] += 1
            if choice_index == usable[index]["answer_index"]:
                totals.letter_correct += 1

        if not letter_only:
            for question in usable:
                examples = _candidate_examples(
                    question,
                    labels=labels,
                    system_prompt=system_prompt,
                    answer_prefix=answer_prefix,
                    tokenizer=tokenizer,
                )
                if max(example.input_ids.shape[0] for example in examples) > max_length:
                    # Counted, and counted out of the option denominator. It
                    # used to be neither: the question stayed in `scored` and
                    # contributed to no option numerator.
                    totals.skipped_option_too_long += 1
                    continue
                totals.option_scored += 1
                scores = _sequence_logprobs(model, examples, device)
                summed = [total for total, _ in scores]
                normalized = [total / max(count, 1) for total, count in scores]
                if max(range(len(summed)), key=summed.__getitem__) == question["answer_index"]:
                    totals.option_sum_correct += 1
                if (
                    max(range(len(normalized)), key=normalized.__getitem__)
                    == question["answer_index"]
                ):
                    totals.option_norm_correct += 1

        totals.scored += len(usable)
        if on_progress is not None and start % (batch_size * 20) == 0:
            on_progress(totals.scored)
    return totals


@torch.no_grad()
def _letter_choice(
    model: torch.nn.Module,
    contexts: Sequence[Sequence[int]],
    letter_ids: Sequence[Sequence[int]],
    device: torch.device,
) -> list[int]:
    """Argmax over the four letter token ids at the position that predicts them."""

    input_ids, attention, lengths = _pad_batch(contexts, device)
    logits = model(input_ids=input_ids, attention_mask=attention).logits
    choices: list[int] = []
    for row, length in enumerate(lengths):
        candidates = torch.tensor(letter_ids[row], device=device)
        scores = logits[row, length - 1].index_select(0, candidates)
        choices.append(int(torch.argmax(scores).item()))
    return choices


@torch.no_grad()
def _sequence_logprobs(
    model: torch.nn.Module,
    examples: Sequence[TokenizedSFTExample],
    device: torch.device,
) -> list[tuple[float, int]]:
    """Summed log-probability of each example's supervised tokens.

    This is the negative of the training loss restricted to that candidate, so
    the score is computed exactly as the model was optimised.
    """

    input_ids, attention, _ = _pad_batch(
        [example.input_ids.tolist() for example in examples], device
    )
    width = input_ids.shape[1]
    labels = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    for row, example in enumerate(examples):
        labels[row, : example.labels.shape[0]] = example.labels
    labels = labels.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    active = labels != IGNORE_INDEX
    gathered = logprobs.gather(2, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    gathered = gathered.masked_fill(~active, 0.0)
    totals = gathered.sum(dim=1)
    counts = active.sum(dim=1)
    return [(float(totals[row]), int(counts[row])) for row in range(len(examples))]


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------


def _build_model(config: Any, manifest: Mapping[str, Any], checkpoint: Path | None) -> Any:
    """Build the run's model, optionally loading a federated checkpoint into it.

    Model config comes from the manifest directly rather than from a built
    dataset, so nothing loads the packed shards this evaluator does not use.
    """

    register_builtin_components()
    model_config = _model_config(config)
    _add_causal_manifest_metadata(model_config, manifest)

    if checkpoint is None:
        # LoRA initialises B to zero, so round 0 is exactly the base model.
        model_config["name"] = "hf_causal_lm"
        for key in LORA_ONLY_KEYS:
            model_config.pop(key, None)

    task = _build_task_for(config, model_config)
    model = task.build_model(model_config)

    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        scope = payload.get("model_state_scope", "full")
        expected = getattr(model, "_fl_model_state_scope", "full")
        if scope != expected:
            raise ValueError(
                f"checkpoint scope {scope!r} does not match the model built from "
                f"{config.model.name!r} (scope {expected!r})"
            )
        task.load_federated_model_state(model, payload["model_state"])
    return task, model, model_config


def _build_task_for(config: Any, model_config: Mapping[str, Any]) -> Any:
    from fedbrew.core.factory import _build_task

    return _build_task(config, tasks.get(config.task.name), dict(model_config))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    """Score multiple-choice accuracy on a held-out MedMCQA split, post hoc.

    Reads a training run config (``--config``) and the generation config
    (``--generation-config``) that defines the choice field mapping, then
    scores either a federated checkpoint (``--checkpoint``) or, when that is
    omitted, the pretrained base model. ``--limit`` caps the questions scored
    (0 means all), and ``--batch-size`` / ``--max-length`` control the forward
    passes.

    ``--split`` defaults to ``client_eval`` on purpose. Scoring ``global_test``
    while comparing arms is selection on the test set: score ``global_test``
    once, for the arm already chosen. Nothing here writes into the run
    directory unless ``--output`` names a path.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training run config")
    parser.add_argument(
        "--generation-config",
        default="data/configs/medmcqa_qwen05b_20clients.yaml",
        help="data generation config, read for the choice field mapping",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="federated checkpoint (.pt); omitted evaluates the pretrained base model",
    )
    parser.add_argument("--limit", type=int, default=2000, help="0 evaluates every question")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--letter-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default=DEFAULT_SPLIT,
        help=(
            "which held-out split to score. Default client_eval: comparing arms "
            "on global_test and then reporting the winner's global_test number "
            "is selection on the test set. Score global_test once, for the arm "
            "already chosen."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.generation_config, encoding="utf-8") as handle:
        generation_config = yaml.safe_load(handle)
    sft_config = generation_config["generic_sft"]
    with open(config.data.path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    _cross_check(sft_config, manifest)

    choice = sft_config.get("choice")
    if not choice:
        raise ValueError("generation config has no generic_sft.choice block")
    labels = [str(label) for label in choice["labels"]]
    system_prompt = manifest.get("system_prompt")
    answer_prefix = str(manifest["response_template"]).split("{answer_label}")[0]

    from fedbrew.data.oasst1_sft import _load_tokenizer, _resolve_tokenizer_assets

    tokenizer = _load_tokenizer(_resolve_tokenizer_assets(sft_config))
    _, dataset_manifest = _load_dataset_assets(sft_config)
    rows = _load_dataset_rows(dataset_manifest)

    questions, counts = _held_out_rows(
        rows, sft_config=sft_config, manifest=manifest, split=args.split
    )
    total_held_out = len(questions)
    questions = _subsample(questions, args.limit or None)
    print(
        f"held-out questions: {total_held_out} "
        f"(of {counts.get('qualifying_rows', 0)} qualifying); evaluating {len(questions)}",
        flush=True,
    )

    device = torch.device(args.device)
    task, model, model_config = _build_model(
        config, manifest, Path(args.checkpoint) if args.checkpoint else None
    )
    model.to(device)
    model.eval()

    totals = score_questions(
        questions,
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        system_prompt=system_prompt,
        answer_prefix=answer_prefix,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        letter_only=args.letter_only,
        on_progress=lambda done: print(f"  {done}/{len(questions)} scored", flush=True),
    )

    record: dict[str, Any] = {
        "metric": "medmcqa_multiple_choice_accuracy",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "model": model_config.get("name"),
        "data": config.data.path,
        "split": args.split,
        "split_seed": int(manifest["seed"]),
        "held_out_questions": total_held_out,
        "random_baseline": 1.0 / len(labels),
        "filtering_summary": counts,
        **accuracy_record(totals, labels=labels, letter_only=args.letter_only),
    }

    # A measured metric can be inf or nan; json.dumps would write the bare
    # token, which strict readers refuse. Same contract as run.json.
    record = json_safe(record)
    print(json.dumps(record, indent=2, allow_nan=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
