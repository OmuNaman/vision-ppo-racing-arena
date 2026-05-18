"""Train PPO from scratch on the 3D pixel-only MetaDrive sky-road arena."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from env import RacingEnv
from model import CNNActorCritic


CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
HORIZON = 512
EPOCHS = 4
MINIBATCH = 64
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
SAVE_EVERY = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pixel-only PPO driver in MetaDrive.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total environment steps.")
    parser.add_argument("--render", action="store_true", help="Open the Panda3D 3D window while training.")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Training device.")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="PPO rollout length.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="PPO optimization epochs per rollout.")
    parser.add_argument("--minibatch", type=int, default=MINIBATCH, help="PPO minibatch size.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Checkpoint interval in env steps.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/policy.pt", help="Checkpoint path.")
    parser.add_argument("--logdir", type=str, default="runs/ppo_pixels", help="TensorBoard log directory.")
    return parser.parse_args()


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for step in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - dones[step]
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    device_name = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    if args.device in {"cuda", "cpu"}:
        device_name = args.device
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device} cuda_available={torch.cuda.is_available()}")
    env = RacingEnv(render=args.render, seed=args.seed)
    model = CNNActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    writer = SummaryWriter(args.logdir)

    checkpoint_path = Path(args.checkpoint)
    global_step = 0
    next_save_step = args.save_every
    update_idx = 0
    obs, _info = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_length = 0
    start_time = time.time()

    try:
        while global_step < args.steps:
            obs_buf = np.zeros((args.horizon, 4, 3, 64, 64), dtype=np.uint8)
            action_buf = np.zeros((args.horizon, 2), dtype=np.float32)
            logprob_buf = np.zeros(args.horizon, dtype=np.float32)
            reward_buf = np.zeros(args.horizon, dtype=np.float32)
            done_buf = np.zeros(args.horizon, dtype=np.float32)
            value_buf = np.zeros(args.horizon, dtype=np.float32)

            for step in range(args.horizon):
                obs_buf[step] = obs
                obs_t = torch.as_tensor(obs[None], device=device)
                with torch.no_grad():
                    bounded_action, log_prob, _entropy, value = model.act(obs_t, deterministic=False)

                action = bounded_action.squeeze(0).cpu().numpy().astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                if args.render:
                    env.render()

                done = bool(terminated or truncated)
                action_buf[step] = action
                logprob_buf[step] = float(log_prob.item())
                reward_buf[step] = float(reward)
                done_buf[step] = float(done)
                value_buf[step] = float(value.item())

                ep_return += float(reward)
                ep_length += 1
                global_step += 1
                obs = next_obs

                if done:
                    writer.add_scalar("episode_reward", ep_return, global_step)
                    writer.add_scalar("episode_length", ep_length, global_step)
                    print(
                        f"step={global_step:8d} episode_return={ep_return:8.2f} "
                        f"episode_length={ep_length:4d} completion={info.get('route_completion', 0.0):.3f}"
                    )
                    obs, _info = env.reset()
                    ep_return = 0.0
                    ep_length = 0

                if global_step >= args.steps:
                    break

            rollout_len = step + 1
            obs_buf = obs_buf[:rollout_len]
            action_buf = action_buf[:rollout_len]
            logprob_buf = logprob_buf[:rollout_len]
            reward_buf = reward_buf[:rollout_len]
            done_buf = done_buf[:rollout_len]
            value_buf = value_buf[:rollout_len]

            with torch.no_grad():
                if done_buf[-1] > 0.5:
                    last_value = 0.0
                else:
                    _dist, bootstrap_value = model(torch.as_tensor(obs[None], device=device))
                    last_value = float(bootstrap_value.item())

            advantages, returns = compute_gae(reward_buf, done_buf, value_buf, last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            b_obs = torch.as_tensor(obs_buf, device=device)
            b_actions = torch.as_tensor(action_buf, device=device)
            b_old_logprobs = torch.as_tensor(logprob_buf, device=device)
            b_advantages = torch.as_tensor(advantages, device=device)
            b_returns = torch.as_tensor(returns, device=device)

            last_policy_loss = 0.0
            last_value_loss = 0.0
            last_entropy = 0.0
            indices = np.arange(rollout_len)
            for _epoch in range(args.epochs):
                np.random.shuffle(indices)
                for start in range(0, rollout_len, args.minibatch):
                    mb = indices[start : start + args.minibatch]
                    new_logprobs, entropy, values = model.evaluate_actions(b_obs[mb], b_actions[mb])
                    ratio = (new_logprobs - b_old_logprobs[mb]).exp()
                    unclipped = ratio * b_advantages[mb]
                    clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages[mb]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * (b_returns[mb] - values).pow(2).mean()
                    entropy_loss = entropy.mean()

                    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                    last_policy_loss = float(policy_loss.item())
                    last_value_loss = float(value_loss.item())
                    last_entropy = float(entropy_loss.item())

            update_idx += 1
            writer.add_scalar("policy_loss", last_policy_loss, global_step)
            writer.add_scalar("value_loss", last_value_loss, global_step)
            writer.add_scalar("entropy", last_entropy, global_step)
            writer.add_scalar("actions/mean_steer", float(action_buf[:, 0].mean()), global_step)
            writer.add_scalar("actions/mean_abs_steer", float(np.abs(action_buf[:, 0]).mean()), global_step)
            writer.add_scalar("actions/mean_throttle", float(action_buf[:, 1].mean()), global_step)
            writer.add_scalar("actions/mean_brake", float(np.maximum(-action_buf[:, 1], 0.0).mean()), global_step)

            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"update={update_idx:4d} step={global_step:8d} "
                f"policy_loss={last_policy_loss:8.4f} value_loss={last_value_loss:8.4f} "
                f"entropy={last_entropy:6.3f} "
                f"steer={action_buf[:, 0].mean():6.3f} abs_steer={np.abs(action_buf[:, 0]).mean():6.3f} "
                f"throttle={action_buf[:, 1].mean():6.3f} "
                f"brake={np.maximum(-action_buf[:, 1], 0.0).mean():6.3f} "
                f"fps={global_step / elapsed:6.1f}"
            )

            while global_step >= next_save_step:
                model.save(checkpoint_path, global_step=global_step, map_name="sky_chicane")
                print(f"saved checkpoint -> {checkpoint_path}")
                next_save_step += args.save_every

        model.save(checkpoint_path, global_step=global_step, map_name="sky_chicane")
        print(f"final checkpoint -> {checkpoint_path}")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
