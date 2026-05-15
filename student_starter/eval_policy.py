"""Evaluate a saved PPO policy on the 3D MetaDrive sky-road track."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from env import RacingEnv
from model import CNNActorCritic


def evaluate(
    checkpoint: str,
    episodes: int = 20,
    render: bool = False,
    map_name: str = "sky_chicane",
    seed: int = 10_000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CNNActorCritic.load(checkpoint, device=device)
    policy.eval()
    env = RacingEnv(map_name=map_name, render=render, seed=seed)

    returns: list[float] = []
    lengths: list[int] = []
    completions: list[float] = []

    try:
        for ep in range(episodes):
            obs, _info = env.reset(seed=seed + ep)
            total_return = 0.0
            length = 0
            final_info = {}
            while True:
                with torch.no_grad():
                    dist, _value = policy(torch.as_tensor(obs[None], device=device))
                    action = dist.mean.squeeze(0).cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, final_info = env.step(action)
                if render:
                    env.render()
                total_return += float(reward)
                length += 1
                if terminated or truncated:
                    break
            completion = float(final_info.get("route_completion", 0.0))
            returns.append(total_return)
            lengths.append(length)
            completions.append(completion)
            print(
                f"episode={ep + 1:02d}/{episodes} return={total_return:8.2f} "
                f"length={length:4d} route_completion={completion:.3f}"
            )
    finally:
        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "route_completion": float(np.mean(completions) * 100.0),
        "returns": returns,
        "lengths": lengths,
        "completions": completions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained pixel PPO policy.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/policy.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--map", type=str, default="sky_chicane")
    args = parser.parse_args()

    results = evaluate(args.checkpoint, episodes=args.episodes, render=args.render, map_name=args.map)
    print("\nEvaluation summary")
    print(f"mean return:          {results['mean_return']:.2f}")
    print(f"mean episode length:  {results['mean_episode_length']:.1f}")
    print(f"route_completion %:   {results['route_completion']:.1f}")


if __name__ == "__main__":
    main()
