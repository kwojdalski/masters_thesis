"""Quick TD3 debug runner to verify that parameters update during training.

Usage (from repo root):
  PYTHONPATH=src python scripts/debug_td3_learning.py \
    --config src/configs/scenarios/sine_wave_td3_no_trend_tradingenv.yaml \
    --max-steps 40000 --actor-lr 0.0003 --init-rand-steps 5000

This script overrides config values directly and prints parameter stats
before/after training so you can confirm the actor/critics are actually learning.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch

from trading_rl.config import ExperimentConfig
from trading_rl.data_utils import prepare_data
from trading_rl.train_trading_agent import env_builder, set_seed
from trading_rl.trainers.td3 import TD3Trainer


def _flatten_params(params: Iterable[torch.Tensor]) -> torch.Tensor:
    flat = [p.detach().flatten() for p in params]
    return torch.cat(flat) if flat else torch.tensor([])


def _summarize_params(name: str, params: Iterable[torch.Tensor]) -> dict[str, float]:
    flat = _flatten_params(params)
    return {
        f"{name}_numel": float(flat.numel()),
        f"{name}_mean": float(flat.mean().item()) if flat.numel() else 0.0,
        f"{name}_std": float(flat.std().item()) if flat.numel() else 0.0,
        f"{name}_abs_max": float(flat.abs().max().item()) if flat.numel() else 0.0,
    }


def _print_summary(label: str, stats: dict[str, float]) -> None:
    print(f"\n{label}:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TD3 learning debug script")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max-steps", type=int, help="Override max_steps")
    parser.add_argument("--init-rand-steps", type=int, help="Override init_rand_steps")
    parser.add_argument("--actor-lr", type=float, help="Override actor LR")
    parser.add_argument("--value-lr", type=float, help="Override critic LR")
    parser.add_argument("--buffer-size", type=int, help="Override replay buffer size")
    parser.add_argument("--seed", type=int, help="Override seed for reproducibility")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.init_rand_steps is not None:
        config.training.init_rand_steps = args.init_rand_steps
    if args.actor_lr is not None:
        config.training.actor_lr = args.actor_lr
    if args.value_lr is not None:
        config.training.value_lr = args.value_lr
    if args.buffer_size is not None:
        config.training.buffer_size = args.buffer_size
    if args.seed is not None:
        config.seed = args.seed

    set_seed(config.seed)

    df = prepare_data(
        data_path=config.data.data_path,
        download_if_missing=config.data.download_data,
        exchange_names=config.data.exchange_names,
        symbols=config.data.symbols,
        timeframe=config.data.timeframe,
        data_dir=config.data.data_dir,
        since=config.data.download_since,
        no_features=getattr(config.data, "no_features", False),
    )

    env = env_builder.create(df, config)
    n_obs = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]
    actor, qnets = TD3Trainer.build_models(n_obs, n_act, config, env)
    trainer = TD3Trainer(actor=actor, qvalue_nets=qnets, env=env, config=config.training)

    actor_before = _summarize_params("actor_before", trainer.actor.parameters())
    critics_before = _summarize_params(
        "critics_before", list(trainer.value_net[0].parameters()) + list(trainer.value_net[1].parameters())
    )
    _print_summary("Parameter stats before training", {**actor_before, **critics_before})

    logs = trainer.train(callback=None)

    actor_after = _summarize_params("actor_after", trainer.actor.parameters())
    critics_after = _summarize_params(
        "critics_after", list(trainer.value_net[0].parameters()) + list(trainer.value_net[1].parameters())
    )
    _print_summary("Parameter stats after training", {**actor_after, **critics_after})

    print("\nDeltas:")
    for key in actor_after:
        before_key = key.replace("after", "before")
        delta = actor_after[key] - actor_before.get(before_key, 0.0)
        print(f"  {key} delta: {delta:.6f}")
    for key in critics_after:
        before_key = key.replace("after", "before")
        delta = critics_after[key] - critics_before.get(before_key, 0.0)
        print(f"  {key} delta: {delta:.6f}")

    if logs.get("loss_value"):
        print(f"\nFinal critic loss: {logs['loss_value'][-1]:.6f}")
    if logs.get("loss_actor"):
        print(f"Final actor loss: {logs['loss_actor'][-1]:.6f}")


if __name__ == "__main__":
    main()
