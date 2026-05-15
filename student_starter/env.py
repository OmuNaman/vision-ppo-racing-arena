"""Vision-only MetaDrive racing environment.

The policy observation is strictly a deque stack of four RGB camera frames with
shape (4, 3, 64, 64). Simulator state is used only for reward/termination
bookkeeping, never returned to the agent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


FRAME_STACK = 4
CAMERA_SIZE = 64
ROAD_WIDTH = 3.2


MAP_VARIANTS: dict[str, dict[str, Any]] = {
    "winding": {"sequence": "SCSCS", "seed": 0},
    "winding_0": {"sequence": "SCSCS", "seed": 0},
    "winding_1": {"sequence": "CCSCC", "seed": 17},
    "winding_2": {"sequence": "SCCSC", "seed": 31},
    "winding_3": {"sequence": "CSCSC", "seed": 43},
    "winding_4": {"sequence": "SCSCCS", "seed": 59},
}


@dataclass(frozen=True)
class RewardParts:
    progress: float
    center_reward: float
    road_reward: float
    speed_reward: float
    offroad_penalty: float
    crash_penalty: float
    finish_bonus: float

    @property
    def total(self) -> float:
        return (
            self.progress
            + self.center_reward
            + self.road_reward
            + self.speed_reward
            + self.offroad_penalty
            + self.crash_penalty
            + self.finish_bonus
        )


def _make_map_config(sequence: str) -> dict[str, Any]:
    return {
        "type": "block_sequence",
        "config": sequence,
        "lane_width": ROAD_WIDTH,
        "lane_num": 1,
    }


def _policy_action(name: str) -> np.ndarray:
    if name != "still":
        raise ValueError("This starter is single-ego; only opponent_policy='still' is supported.")
    return np.array([0.0, 0.0], dtype=np.float32)


class RacingEnv(gym.Env):
    """Single-ego MetaDrive wrapper with pixel-only observations."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_name: str = "winding",
        opponent_policy: str = "still",
        render: bool = False,
        seed: int | None = None,
        horizon: int = 1000,
    ) -> None:
        super().__init__()
        _policy_action(opponent_policy)

        from metadrive.component.sensors.rgb_camera import RGBCamera
        from metadrive.envs.metadrive_env import MetaDriveEnv

        variant = MAP_VARIANTS.get(map_name, {"sequence": map_name, "seed": 0})
        self.map_name = map_name
        self.base_seed = int(variant.get("seed", 0) if seed is None else seed)
        self._frames: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self._last_completion = 0.0

        config = {
            "use_render": render,
            "image_observation": True,
            "norm_pixel": False,
            "stack_size": 1,
            "sensors": {"rgb_camera": (RGBCamera, CAMERA_SIZE, CAMERA_SIZE)},
            "vehicle_config": {"image_source": "rgb_camera"},
            "map_config": _make_map_config(str(variant["sequence"])),
            "num_scenarios": 1,
            "start_seed": self.base_seed,
            "traffic_density": 0.0,
            "random_traffic": False,
            "out_of_route_done": True,
            "out_of_road_done": True,
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "crash_human_done": True,
            "horizon": horizon,
            "log_level": 50,
        }
        if "idle_done" in MetaDriveEnv.default_config():
            config["idle_done"] = False
        self._env = MetaDriveEnv(config=config)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(FRAME_STACK, 3, CAMERA_SIZE, CAMERA_SIZE),
            dtype=np.uint8,
        )
        self.action_space = self._env.action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        raw_obs, info = self._env.reset(seed=self.base_seed if seed is None else seed)
        frame = self._extract_frame(raw_obs)
        self._frames.clear()
        for _ in range(FRAME_STACK):
            self._frames.append(frame.copy())
        self._last_completion = self._route_completion(info)
        return self._stacked_obs(), self._clean_info(info, RewardParts(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        raw_obs, _default_reward, terminated, truncated, info = self._env.step(action)
        self._frames.append(self._extract_frame(raw_obs))

        reward_parts = self._compute_reward(info)
        terminated = bool(terminated or self._crashed(info) or not self._on_road(info))
        clean_info = self._clean_info(info, reward_parts)
        return self._stacked_obs(), float(reward_parts.total), bool(terminated), bool(truncated), clean_info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped_metadrive(self):
        return self._env

    def _extract_frame(self, raw_obs: Any) -> np.ndarray:
        image = raw_obs["image"] if isinstance(raw_obs, dict) else raw_obs
        image = np.asarray(image)
        if image.ndim == 4:
            image = image[..., -1]
        if image.shape[:2] != (CAMERA_SIZE, CAMERA_SIZE):
            raise ValueError(f"expected {CAMERA_SIZE}x{CAMERA_SIZE} RGB image, got {image.shape}")
        if image.shape[-1] != 3:
            raise ValueError(f"expected RGB image with 3 channels, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.transpose(image, (2, 0, 1))

    def _stacked_obs(self) -> np.ndarray:
        return np.stack(tuple(self._frames), axis=0).astype(np.uint8, copy=False)

    def _compute_reward(self, info: dict[str, Any]) -> RewardParts:
        completion = self._route_completion(info)
        progress = max(0.0, completion - self._last_completion) * 100.0
        self._last_completion = max(self._last_completion, completion)

        distance_from_center = self._distance_from_center()
        on_road = self._on_road(info)
        # Bounded shaping: +1.0 on the centerline, smoothly falling to -0.5
        # near/off the lane edge. This avoids huge negative rewards from raw
        # lateral distances while still making center driving obviously best.
        center_score = 1.0 - min(distance_from_center / (ROAD_WIDTH * 0.5), 1.0)
        moving_forward = progress > 1e-4
        center_reward = (-0.5 + 1.5 * center_score) if moving_forward else -0.05
        road_reward = 0.2 if on_road and moving_forward else 0.0
        speed_km_h = self._speed_km_h()
        speed_score = min(max(speed_km_h, 0.0) / 35.0, 1.0)
        speed_reward = 0.6 * speed_score if on_road and moving_forward else 0.0
        offroad_penalty = -1.0 if not on_road else 0.0
        crash_penalty = -10.0 if self._crashed(info) else 0.0
        finish_bonus = 20.0 if self._arrived(info) else 0.0
        return RewardParts(progress, center_reward, road_reward, speed_reward, offroad_penalty, crash_penalty, finish_bonus)

    def _clean_info(self, info: dict[str, Any], reward_parts: RewardParts) -> dict[str, Any]:
        clean = dict(info)
        clean["route_completion"] = self._route_completion(info)
        clean["distance_from_center"] = self._distance_from_center()
        clean["on_road"] = self._on_road(info)
        clean["reward_progress"] = reward_parts.progress
        clean["reward_center"] = reward_parts.center_reward
        clean["reward_road"] = reward_parts.road_reward
        clean["reward_speed"] = reward_parts.speed_reward
        clean["reward_offroad"] = reward_parts.offroad_penalty
        clean["reward_crash"] = reward_parts.crash_penalty
        clean["reward_finish"] = reward_parts.finish_bonus
        clean["reward_total"] = reward_parts.total
        return clean

    @staticmethod
    def _route_completion(info: dict[str, Any]) -> float:
        for key in ("route_completion", "progress"):
            if key in info:
                return float(np.clip(info[key], 0.0, 1.0))
        return 0.0

    @staticmethod
    def _crashed(info: dict[str, Any]) -> bool:
        return any(
            bool(info.get(key, False))
            for key in ("crash", "crash_vehicle", "crash_object", "crash_human", "crash_sidewalk")
        )

    @staticmethod
    def _arrived(info: dict[str, Any]) -> bool:
        return bool(info.get("arrive_dest", False) or info.get("success", False))

    def _on_road(self, info: dict[str, Any]) -> bool:
        if bool(info.get("out_of_road", False)):
            return False
        vehicle = getattr(self._env, "agent", None)
        if vehicle is None and hasattr(self._env, "agents"):
            vehicle = self._env.agents.get("agent0")
        if vehicle is None:
            return True
        if bool(getattr(vehicle, "crash_sidewalk", False)):
            return False
        if hasattr(vehicle, "on_lane"):
            return bool(vehicle.on_lane)
        return True

    def _speed_km_h(self) -> float:
        vehicle = getattr(self._env, "agent", None)
        if vehicle is None and hasattr(self._env, "agents"):
            vehicle = self._env.agents.get("agent0")
        return float(max(getattr(vehicle, "speed_km_h", 0.0), 0.0)) if vehicle is not None else 0.0

    def _distance_from_center(self) -> float:
        vehicle = getattr(self._env, "agent", None)
        if vehicle is None and hasattr(self._env, "agents"):
            vehicle = self._env.agents.get("agent0")
        lane = getattr(vehicle, "lane", None)
        position = getattr(vehicle, "position", None)
        if lane is not None and position is not None and hasattr(lane, "local_coordinates"):
            try:
                _longitudinal, lateral = lane.local_coordinates(position)
                return float(abs(lateral))
            except Exception:
                pass
        nav = getattr(vehicle, "navigation", None)
        current_lane = getattr(nav, "current_lane", None)
        if current_lane is not None and position is not None and hasattr(current_lane, "local_coordinates"):
            try:
                _longitudinal, lateral = current_lane.local_coordinates(position)
                return float(abs(lateral))
            except Exception:
                pass
        return 0.0


def make_env(map_name: str = "winding", render: bool = False, seed: int | None = None) -> RacingEnv:
    return RacingEnv(map_name=map_name, opponent_policy="still", render=render, seed=seed)
