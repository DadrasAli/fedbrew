"""The average drift at a model: rho-hat, over one full-participation round, with no update.

Wang, Das, Joshi, Kale, Xu and Zhang (arXiv:2206.04723, eq. 15) define each
client's pseudo-gradient after H local steps at step eta from a model w as
``G_c(w) = (w - w_c^(H)) / (eta H)``, and the average drift at w as the norm of
their mean. Measured here: every client of a run's config takes its local
update from w -- its own rule, its own loader, as a round would -- and nothing
is aggregated or written.

    rho-hat = || (1/N) sum_c (w - w_c^(H)) / (eta H_c) ||

H_c is the steps client c took (its ``optimizer_steps``, or the config's
``local_iterations`` when the rule reports none). To first order in eta,
``rho = (eta (H - 1) / 2) ||delta*||`` at the optimum, delta* the
curvature-weighted gradient dissimilarity, so rho-hat places a dataset on that
axis.

w is the config's initial model, a checkpoint's (``--checkpoint``), or a state
dict saved with ``torch.save`` (``--model-state``), e.g. one exported from
another code base. Parameters it does not name keep the model's own.

Usage:
    python tools/drift_diagnostic.py --config configs/.../arm.yaml
        [--checkpoint <run>/checkpoints/latest.pt | --model-state state.pt]
        [--local-iterations H] [--learning-rate ETA] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


def measure(
    config_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    model_state: str | Path | dict[str, Any] | None = None,
    local_iterations: int | None = None,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    """rho-hat and what it was measured at; see the module docstring."""

    from fedbrew.core.config import load_config
    from fedbrew.core.factory import build_components
    from fedbrew.core.protocol import FitRequest

    config = load_config(config_path)
    client = config.client
    if local_iterations is not None:
        client = replace(client, local_iterations=int(local_iterations))
    if learning_rate is not None:
        client = replace(client, learning_rate=float(learning_rate))
    config = replace(config, client=client)
    if config.client.learning_rate is None:
        raise SystemExit("the drift needs a fixed step: client.learning_rate is unset")
    eta = float(config.client.learning_rate)

    components = build_components(config)
    server = components.server
    server.initialize()
    state = dict(server._model_state)
    source = "the config's initial model"
    if checkpoint is not None:
        from fedbrew.core.checkpointing import load_checkpoint

        loaded = load_checkpoint(checkpoint)["model_state"]
        source = str(checkpoint)
    elif model_state is not None:
        loaded = model_state if isinstance(model_state, dict) else torch.load(model_state)
        source = str(model_state) if not isinstance(model_state, dict) else "a state dict"
    else:
        loaded = {}
    for key, value in loaded.items():
        if key not in state:
            raise SystemExit(f"the model has no {key!r}; it has {sorted(state)}")
        state[key] = value.detach().to(state[key].device, state[key].dtype)
    server._model_state = state
    payload = server._federated_payload()

    keys = [key for key, value in state.items() if torch.is_floating_point(value)]
    total = {key: torch.zeros_like(state[key], dtype=torch.float64) for key in keys}
    steps_seen: set[int] = set()
    norms = []
    clients = components.dataset.list_clients()
    for client_id in clients:
        request = FitRequest(
            round_id=1,
            client_id=client_id,
            payload=payload,
            total_rounds=1,
            post_fit_evaluation=False,
        )
        result = components.clients[client_id].fit(request)
        trained = result.payload["model_state"]
        steps = int(result.metrics.get("optimizer_steps", config.client.local_iterations))
        steps_seen.add(steps)
        squared = 0.0
        for key in keys:
            pseudo = (state[key].double() - trained[key].double().to(state[key].device)) / (
                eta * steps
            )
            total[key] += pseudo
            squared += float(torch.sum(pseudo * pseudo))
        norms.append(math.sqrt(squared))
    rho = math.sqrt(sum(float(torch.sum((value / len(clients)) ** 2)) for value in total.values()))
    return {
        "rho_hat": rho,
        "clients": len(clients),
        "learning_rate": eta,
        "local_steps": sorted(steps_seen),
        "mean_client_pseudo_gradient_norm": sum(norms) / len(norms),
        "update_rule": config.client.update_rule,
        "model": source,
        "config": str(config_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", help="a checkpoint whose model_state is w")
    group.add_argument("--model-state", help="a torch.save'd state dict that is w")
    parser.add_argument("--local-iterations", type=int, help="H, in place of the config's")
    parser.add_argument("--learning-rate", type=float, help="eta, in place of the config's")
    parser.add_argument("--json", help="also write the result here")
    args = parser.parse_args(argv)
    result = measure(
        args.config,
        checkpoint=args.checkpoint,
        model_state=args.model_state,
        local_iterations=args.local_iterations,
        learning_rate=args.learning_rate,
    )
    from fedbrew.core.metrics import json_safe

    text = json.dumps(json_safe(result), indent=1, allow_nan=False)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
