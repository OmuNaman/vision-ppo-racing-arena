"""3D MetaDrive sky-road arena with stacked RGB camera observations.

The student policy sees only pixels: four consecutive 64x64 RGB frames from
MetaDrive's vehicle-mounted camera. The world is configured like a narrow
"sky road": terrain and sidewalks are hidden, the lane is tight, and leaving
the drivable surface ends the episode with a large fall-style penalty.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


FRAME_STACK = 4
CAMERA_SIZE = 64


EVAL_MAPS: dict[str, dict[str, Any]] = {
    "sky_chicane": {"map_config": {"type": "block_sequence", "config": "SCSCS"}},
    "sky_curves": {"map_config": {"type": "block_sequence", "config": "CCSCC"}},
    "sky_slalom": {"map_config": {"type": "block_sequence", "config": "CrCSC"}},
    "sky_sprint": {"map_config": {"type": "block_sequence", "config": "SSCSS"}},
    "sky_gauntlet": {"map_config": {"type": "block_sequence", "config": "CSCSC"}},
}
EPISODES_PER_MAP = 4

# Backward-compatible aliases used by older local scripts/checkpoints.
MAP_VARIANTS = EVAL_MAPS
MAP_ALIASES = {
    "winding": "sky_chicane",
    "winding_0": "sky_chicane",
    "winding_1": "sky_curves",
    "winding_2": "sky_slalom",
    "winding_3": "sky_sprint",
    "winding_4": "sky_gauntlet",
    "chicane": "sky_chicane",
    "tight_s": "sky_curves",
    "curve_a": "sky_slalom",
    "long_straight": "sky_sprint",
    "oval": "sky_gauntlet",
    "circuit": "sky_chicane",
}


class RacingEnv(gym.Env):
    """Single-agent Gymnasium wrapper around MetaDriveEnv."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_name: str = "sky_chicane",
        opponent_policy: str = "still",
        render: bool = False,
        seed: int | None = None,
        horizon: int = 3000,
    ) -> None:
        super().__init__()
        from metadrive.component.sensors.rgb_camera import RGBCamera
        from metadrive.envs.metadrive_env import MetaDriveEnv

        del opponent_policy  # Kept only so older training scripts still import.

        self._map_name = MAP_ALIASES.get(map_name, map_name)
        self._base_seed = 0 if seed is None else int(seed)
        self._frames: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self._last_info: dict[str, Any] = {}
        self._episode_step = 0

        map_cfg = EVAL_MAPS.get(self._map_name, EVAL_MAPS["sky_chicane"])
        self._env = MetaDriveEnv(
            config={
                "num_scenarios": 100_000,
                "start_seed": 0,
                "use_render": bool(render),
                "image_observation": True,
                "image_on_cuda": False,
                "norm_pixel": False,
                "stack_size": 1,
                "sensors": {"rgb_camera": (RGBCamera, CAMERA_SIZE, CAMERA_SIZE)},
                "vehicle_config": {
                    "image_source": "rgb_camera",
                    "show_navi_mark": False,
                    "show_dest_mark": False,
                    "show_navigation_arrow": False,
                    "show_lidar": False,
                    "show_lane_line_detector": False,
                    "show_side_detector": False,
                },
                "interface_panel": ["dashboard", "rgb_camera"] if render else [],
                "show_interface": bool(render),
                "show_logo": False,
                "show_fps": bool(render),
                # Keep rendered terrain on; otherwise MetaDrive clears the
                # hidden world to white and the camera sees a blank slab.
                "show_terrain": True,
                "show_sidewalk": True,
                "show_crosswalk": False,
                "horizon": horizon,
                "out_of_road_done": True,
                "crash_vehicle_done": True,
                "crash_object_done": True,
                "on_continuous_line_done": False,
                "on_broken_line_done": False,
                "traffic_density": 0.06,
                "random_traffic": True,
                "accident_prob": 0.35,
                "static_traffic_object": True,
                "random_agent_model": False,
                "random_spawn_lane_index": False,
                "map": 3,
                "map_config": {
                    "lane_num": 1,
                    "lane_width": 3.0,
                    "exit_length": 35,
                    **map_cfg.get("map_config", {}),
                },
                # Base MetaDrive shaping: forward progress, speed, lateral
                # centering, then large sparse penalties/bonus below.
                "driving_reward": 1.2,
                "speed_reward": 0.35,
                "use_lateral_reward": True,
                "success_reward": 80.0,
                "out_of_road_penalty": 50.0,
                "crash_vehicle_penalty": 30.0,
                "crash_object_penalty": 30.0,
                "crash_sidewalk_penalty": 50.0,
                "log_level": 50,
            }
        )

        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(FRAME_STACK, 3, CAMERA_SIZE, CAMERA_SIZE),
            dtype=np.uint8,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        raw_obs, info = self._env.reset(seed=self._base_seed if seed is None else seed)
        self._episode_step = 0
        frame = self._rgb_frame(raw_obs)
        self._frames.clear()
        for _ in range(FRAME_STACK):
            self._frames.append(frame.copy())
        self._last_info = self._augment_info(info)
        return self._stacked_obs(), self._last_info

    def step(self, action):
        clipped_action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        raw_obs, default_reward, terminated, truncated, info = self._env.step(clipped_action)
        self._episode_step += 1
        self._frames.append(self._rgb_frame(raw_obs))
        info = self._augment_info(info)
        reward = self._sky_reward(float(default_reward), info, clipped_action)
        self._last_info = info
        return self._stacked_obs(), reward, bool(terminated), bool(truncated), info

    def render(self):
        text = {
            "map": self._map_name,
            "reward": f"{self._last_info.get('sky_reward', 0.0):.2f}",
            "completion": f"{self._last_info.get('route_completion', 0.0):.1%}",
            "speed": f"{self._last_info.get('speed_km_h', 0.0):.1f} km/h",
        }
        return self._env.render(text=text)

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    @property
    def unwrapped_metadrive(self):
        return self._env

    def _rgb_frame(self, raw_obs: Any) -> np.ndarray:
        image = raw_obs["image"] if isinstance(raw_obs, dict) else raw_obs
        image = np.asarray(image)
        if image.ndim == 4:
            image = image[..., -1]
        if image.dtype != np.uint8:
            if float(np.nanmax(image)) <= 1.5:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[:2] != (CAMERA_SIZE, CAMERA_SIZE):
            raise ValueError(f"expected RGB camera {(CAMERA_SIZE, CAMERA_SIZE)}, got {image.shape}")
        return np.transpose(image[..., :3], (2, 0, 1))

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        clean = dict(info)
        clean["map_name"] = self._map_name
        agent = getattr(self._env, "agent", None)
        if agent is not None:
            clean["speed_km_h"] = float(max(getattr(agent, "speed_km_h", 0.0), 0.0))
            clean["route_completion"] = float(getattr(agent.navigation, "route_completion", 0.0))
            clean["center_score"] = self._center_score(agent)
            clean["on_lane"] = bool(getattr(agent, "on_lane", False))
        else:
            clean.setdefault("speed_km_h", 0.0)
            clean.setdefault("route_completion", 0.0)
            clean.setdefault("center_score", 0.0)
            clean.setdefault("on_lane", False)
        return clean

    def _center_score(self, agent) -> float:
        if not getattr(agent, "on_lane", False) or getattr(agent, "lane", None) is None:
            return 0.0
        _longitudinal, lateral = agent.lane.local_coordinates(agent.position)
        width = max(float(agent.navigation.get_current_lane_width()), 1e-6)
        return float(np.clip(1.0 - abs(lateral) / (0.5 * width), 0.0, 1.0))

    def _sky_reward(self, default_reward: float, info: dict[str, Any], action: np.ndarray) -> float:
        reward = default_reward
        speed = float(info.get("speed_km_h", 0.0))
        center = float(info.get("center_score", 0.0))
        throttle = max(float(action[1]), 0.0)
        brake = max(float(-action[1]), 0.0)

        reward += 0.08 * center
        reward -= 0.10 * (1.0 - center)
        reward += 0.04 * min(speed / 35.0, 1.0)
        reward += 0.04 * throttle * (1.0 if speed < 35.0 else 0.25)

        brake_penalty = 0.18 * brake
        if speed < 12.0:
            brake_penalty += 0.35 * brake
        if self._episode_step > 20 and speed < 3.0:
            brake_penalty += 0.35 * brake
        reward -= brake_penalty
        info["brake_amount"] = brake
        info["brake_penalty"] = float(brake_penalty)

        if self._episode_step > 20 and speed < 2.0:
            reward -= 0.35
            info["idle_penalty_active"] = True
        else:
            info["idle_penalty_active"] = False

        if info.get("arrive_dest", False):
            reward = 100.0
            info["terminal_reason"] = "finish"
        elif info.get("out_of_road", False):
            reward = -60.0
            info["terminal_reason"] = "fell_off_sky_road"
        elif info.get("crash_object", False):
            reward = -35.0
            info["terminal_reason"] = "hit_obstacle"
        elif info.get("crash_vehicle", False):
            reward = -35.0
            info["terminal_reason"] = "hit_traffic"
        elif info.get("crash_sidewalk", False):
            reward = -60.0
            info["terminal_reason"] = "hit_edge"

        info["sky_reward"] = float(reward)
        return float(reward)

    def _stacked_obs(self) -> np.ndarray:
        return np.stack(tuple(self._frames), axis=0).astype(np.uint8, copy=False)


def make_env(map_name: str = "sky_chicane", render: bool = False, seed: int | None = None) -> RacingEnv:
    return RacingEnv(map_name=map_name, opponent_policy="still", render=render, seed=seed)
