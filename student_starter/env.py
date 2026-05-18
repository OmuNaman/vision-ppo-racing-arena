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

import reward_config as cfg


FRAME_STACK = 4
CAMERA_SIZE = 64
DEFAULT_MAP_NAME = "sky_chicane"
DEFAULT_MAP_CONFIG: dict[str, Any] = {"type": "block_sequence", "config": "SSCSS"}


EVAL_MAPS: dict[str, dict[str, Any]] = {
    DEFAULT_MAP_NAME: {"map_config": DEFAULT_MAP_CONFIG},
}
EPISODES_PER_MAP = 4

# Backward-compatible aliases used by older local scripts/checkpoints. They all
# resolve to the single default map so there is no hidden curriculum behavior.
MAP_VARIANTS = EVAL_MAPS
MAP_ALIASES = {
    "winding": DEFAULT_MAP_NAME,
    "winding_0": DEFAULT_MAP_NAME,
    "winding_1": DEFAULT_MAP_NAME,
    "winding_2": DEFAULT_MAP_NAME,
    "winding_3": DEFAULT_MAP_NAME,
    "winding_4": DEFAULT_MAP_NAME,
    "chicane": DEFAULT_MAP_NAME,
    "tight_s": DEFAULT_MAP_NAME,
    "curve_a": DEFAULT_MAP_NAME,
    "long_straight": DEFAULT_MAP_NAME,
    "oval": DEFAULT_MAP_NAME,
    "circuit": DEFAULT_MAP_NAME,
}


class RacingEnv(gym.Env):
    """Single-agent Gymnasium wrapper around MetaDriveEnv."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_name: str = DEFAULT_MAP_NAME,
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
        self._stall_steps = 0
        self._last_completion = 0.0

        map_cfg = EVAL_MAPS[DEFAULT_MAP_NAME]
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
                "traffic_density": cfg.TRAFFIC_DENSITY,
                "random_traffic": cfg.RANDOM_TRAFFIC,
                "accident_prob": cfg.ACCIDENT_PROB,
                "static_traffic_object": cfg.STATIC_TRAFFIC_OBJECTS,
                "random_agent_model": False,
                "random_spawn_lane_index": False,
                "map": 3,
                "map_config": {
                    "lane_num": 1,
                    "lane_width": cfg.LANE_WIDTH,
                    "exit_length": 35,
                    **map_cfg.get("map_config", {}),
                },
                # Base MetaDrive shaping: forward progress, speed, lateral
                # centering, then large sparse penalties/bonus below.
                "driving_reward": cfg.BASE_DRIVING_REWARD,
                "speed_reward": cfg.BASE_SPEED_REWARD,
                "use_lateral_reward": cfg.USE_LATERAL_REWARD,
                "success_reward": cfg.BASE_SUCCESS_REWARD,
                "out_of_road_penalty": cfg.BASE_OUT_OF_ROAD_PENALTY,
                "crash_vehicle_penalty": cfg.BASE_CRASH_VEHICLE_PENALTY,
                "crash_object_penalty": cfg.BASE_CRASH_OBJECT_PENALTY,
                "crash_sidewalk_penalty": cfg.BASE_CRASH_SIDEWALK_PENALTY,
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
        self._stall_steps = 0
        frame = self._rgb_frame(raw_obs)
        self._frames.clear()
        for _ in range(FRAME_STACK):
            self._frames.append(frame.copy())
        self._last_info = self._augment_info(info)
        self._last_completion = float(self._last_info.get("route_completion", 0.0))
        return self._stacked_obs(), self._last_info

    def step(self, action):
        clipped_action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        env_action = clipped_action.copy()
        env_action[0] *= cfg.STEERING_SCALE
        if env_action[1] > 0.0:
            env_action[1] *= cfg.THROTTLE_SCALE
        raw_obs, _default_reward, terminated, truncated, info = self._env.step(env_action)
        self._episode_step += 1
        self._frames.append(self._rgb_frame(raw_obs))
        info = self._augment_info(info)
        info["policy_steer"] = float(clipped_action[0])
        info["applied_steer"] = float(env_action[0])
        info["applied_throttle"] = float(env_action[1])
        reward = self._sky_reward(info, clipped_action)
        self._last_completion = float(info.get("route_completion", self._last_completion))
        if self._is_stalling(info, clipped_action):
            self._stall_steps += 1
        else:
            self._stall_steps = 0
        info["stall_steps"] = self._stall_steps
        if self._stall_steps >= cfg.STALL_AFTER_STEPS:
            reward = cfg.STALL_TERMINAL_REWARD
            info["sky_reward"] = reward
            info["terminal_reason"] = "stalled_on_sky_road"
            terminated = True
        if info.get("terminal_reason") in {"fell_off_sky_road", "hit_edge"}:
            terminated = True
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

    def _sky_reward(self, info: dict[str, Any], action: np.ndarray) -> float:
        speed = float(info.get("speed_km_h", 0.0))
        center = float(info.get("center_score", 0.0))
        steer = abs(float(action[0]))
        throttle = max(float(action[1]), 0.0)
        brake = max(float(-action[1]), 0.0)
        completion = float(info.get("route_completion", self._last_completion))
        progress_delta = completion - self._last_completion

        if progress_delta >= 0.0:
            reward = cfg.PROGRESS_REWARD_SCALE * progress_delta
        else:
            reward = cfg.BACKWARD_PROGRESS_PENALTY_SCALE * progress_delta
        info["progress_delta"] = float(progress_delta)

        off_center = 1.0 - center
        reward += cfg.CENTER_BONUS * center
        reward -= cfg.OFF_CENTER_PENALTY * (off_center**2)
        reward += cfg.SPEED_BONUS * min(speed / cfg.MAX_REWARDED_SPEED_KMH, 1.0)
        if speed > cfg.MAX_REWARDED_SPEED_KMH:
            reward -= cfg.OVERSPEED_PENALTY * (
                (speed - cfg.MAX_REWARDED_SPEED_KMH) / cfg.OVERSPEED_PENALTY_STEP_KMH
            )
        reward += cfg.THROTTLE_BONUS * throttle * (
            1.0 if speed < cfg.MAX_REWARDED_SPEED_KMH else cfg.THROTTLE_BONUS_WHEN_OVERSPEED
        )

        steering_penalty = (
            cfg.STEERING_BASE_PENALTY * steer
            + cfg.STEERING_SPEED_PENALTY * steer * min(speed / cfg.STEERING_SPEED_REFERENCE_KMH, 1.0)
            + cfg.STEERING_EDGE_PENALTY * steer * off_center
        )
        reward -= steering_penalty
        info["steering_penalty"] = float(steering_penalty)

        brake_penalty = cfg.BRAKE_BASE_PENALTY * brake
        if speed < cfg.LOW_SPEED_BRAKE_THRESHOLD_KMH:
            brake_penalty += cfg.LOW_SPEED_BRAKE_EXTRA_PENALTY * brake
        if self._episode_step > cfg.IDLE_AFTER_STEPS and speed < cfg.IDLE_BRAKE_THRESHOLD_KMH:
            brake_penalty += cfg.IDLE_BRAKE_EXTRA_PENALTY * brake
        reward -= brake_penalty
        info["brake_amount"] = brake
        info["brake_penalty"] = float(brake_penalty)

        if self._episode_step > cfg.IDLE_AFTER_STEPS and speed < cfg.IDLE_SPEED_THRESHOLD_KMH:
            reward -= cfg.IDLE_STEP_PENALTY
            info["idle_penalty_active"] = True
        else:
            info["idle_penalty_active"] = False

        if info.get("arrive_dest", False):
            reward = cfg.FINISH_REWARD
            info["terminal_reason"] = "finish"
        elif info.get("out_of_road", False):
            reward = cfg.FALL_OFF_ROAD_REWARD
            info["terminal_reason"] = "fell_off_sky_road"
        elif info.get("crash_object", False):
            reward = cfg.HIT_OBSTACLE_REWARD
            info["terminal_reason"] = "hit_obstacle"
        elif info.get("crash_vehicle", False):
            reward = cfg.HIT_TRAFFIC_REWARD
            info["terminal_reason"] = "hit_traffic"
        elif info.get("crash_sidewalk", False):
            reward = cfg.HIT_EDGE_REWARD
            info["terminal_reason"] = "hit_edge"

        info["sky_reward"] = float(reward)
        return float(reward)

    def _is_stalling(self, info: dict[str, Any], action: np.ndarray) -> bool:
        speed = float(info.get("speed_km_h", 0.0))
        throttle = float(action[1])
        return (
            self._episode_step > cfg.IDLE_AFTER_STEPS
            and speed < cfg.STALL_SPEED_THRESHOLD_KMH
            and throttle < cfg.STALL_THROTTLE_THRESHOLD
        )

    def _stacked_obs(self) -> np.ndarray:
        return np.stack(tuple(self._frames), axis=0).astype(np.uint8, copy=False)


def make_env(map_name: str = DEFAULT_MAP_NAME, render: bool = False, seed: int | None = None) -> RacingEnv:
    return RacingEnv(map_name=map_name, opponent_policy="still", render=render, seed=seed)
