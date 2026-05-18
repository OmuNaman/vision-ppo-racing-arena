"""CNN actor-critic policy for stacked RGB observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

LOG_STD_MIN = -2.5
LOG_STD_MAX = 0.0
ACTION_EPS = 1e-6


class CNNActorCritic(nn.Module):
    def __init__(self, action_dim: int = 2, init_log_std: float = -1.0) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        if action_dim >= 2:
            with torch.no_grad():
                self.actor.bias[1] = 0.8

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.encode(obs)
        mean = torch.tanh(self.actor(features))
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        value = self.critic(features).squeeze(-1)
        return Normal(mean, std), value

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 5:
            batch, stack, channels, height, width = obs.shape
            obs = obs.reshape(batch, stack * channels, height, width)
        elif obs.ndim != 4:
            raise ValueError(f"expected obs with 4 or 5 dims, got shape {tuple(obs.shape)}")
        obs = obs.float() / 255.0
        return self.encoder(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self.forward(obs)
        if deterministic:
            raw_action = dist.mean
            action = raw_action.clamp(-1.0, 1.0)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
        else:
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            log_prob = self._squashed_log_prob(dist, raw_action, action)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, values = self.forward(obs)
        clipped = actions.clamp(-1.0 + ACTION_EPS, 1.0 - ACTION_EPS)
        raw_actions = torch.atanh(clipped)
        log_probs = self._squashed_log_prob(dist, raw_actions, clipped)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, values

    def _squashed_log_prob(self, dist: Normal, raw_actions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        correction = torch.log(1.0 - actions.pow(2) + ACTION_EPS)
        return (dist.log_prob(raw_actions) - correction).sum(dim=-1)

    def save(self, path: str | Path, **metadata: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "action_dim": self.action_dim,
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "CNNActorCritic":
        checkpoint = torch.load(path, map_location=device)
        model = cls(action_dim=int(checkpoint.get("action_dim", 2)))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model


ActorCritic = CNNActorCritic
