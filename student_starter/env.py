"""MetaDrive racing environment wrapper with top-down image observations.

This follows the World Model Arena starter pattern: a thin single-agent
interface over MetaDrive's two-agent ``MultiAgentRacingEnv``. ``agent0`` is the
student policy, while ``agent1`` is driven by a simple opponent policy.

Difference from the reference starter: this wrapper returns only stacked 2D
top-down RGB frames, not MetaDrive's vector/LiDAR observation.

Usage:
    env = RacingEnv(map_name="chicane", opponent_policy="still")
    obs, _ = env.reset()
    assert obs.shape == (4, 3, 64, 64)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image


FRAME_STACK = 4
CAMERA_SIZE = 64
TOPDOWN_SCREEN_SIZE = 256
ROUTE_LENGTH_METERS = 1000.0


EVAL_MAPS: dict[str, dict[str, Any]] = {
    "curve_a": {"map_config": {"type": "block_sequence", "config": "CrCSC"}},
    "chicane": {"map_config": {"type": "block_sequence", "config": "SCSCS"}},
    "long_straight": {"map_config": {"type": "block_sequence", "config": "SSSSS"}},
    "tight_s": {"map_config": {"type": "block_sequence", "config": "CCSCC"}},
    "oval": {"map_config": {"type": "block_sequence", "config": "SCCS"}},
}
EPISODES_PER_MAP = 4

# Backward-compatible aliases used by older local scripts.
MAP_VARIANTS = EVAL_MAPS
MAP_ALIASES = {
    "winding": "chicane",
    "winding_0": "chicane",
    "winding_1": "tight_s",
    "winding_2": "curve_a",
    "winding_3": "long_straight",
    "winding_4": "oval",
    "circuit": "chicane",
}


def _make_opponent(name: str):
    """Returns a callable ``obs -> action`` for agent1."""
    rng = np.random.default_rng(0)

    def random_policy(_obs):
        return rng.uniform(-1, 1, size=(2,)).astype(np.float32)

    def aggressive_policy(_obs):
        return np.array([0.0, 1.0], dtype=np.float32)

    def still_policy(_obs):
        return np.array([0.0, 0.0], dtype=np.float32)

    table = {
        "random": random_policy,
        "aggressive": aggressive_policy,
        "still": still_policy,
    }
    if name not in table:
        raise ValueError(f"unknown opponent policy: {name}")
    return table[name]


class _FakeMainCamera:
    """Small camera shim so MetaDrive's top-down renderer tracks agent0."""

    CHASE_TASK_NAME = "_fake_topdown_chase_task"

    def __init__(self, agent):
        self.current_track_agent = agent

    def destroy(self) -> None:
        pass


class RacingEnv(gym.Env):
    """Single-agent Gymnasium wrapper around MetaDrive's 2-agent racing env."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_name: str = "chicane",
        opponent_policy: str = "still",
        render: bool = False,
        seed: int | None = None,
        horizon: int = 3000,
    ) -> None:
        super().__init__()
        from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

        self._opponent = _make_opponent(opponent_policy)
        self._map_name = MAP_ALIASES.get(map_name, map_name)
        self._base_seed = 0 if seed is None else int(seed)
        self._show_topdown = bool(render)
        self._last_opp_obs = None
        self._frames: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self._route_completion = 0.0

        map_cfg = EVAL_MAPS.get(self._map_name, EVAL_MAPS["chicane"])
        config: dict[str, Any] = {
            "num_agents": 2,
            "num_scenarios": 100_000,
            "start_seed": 0,
            # We keep Panda3D's 3D window off. render=True means show top-down.
            "use_render": False,
            "horizon": horizon,
            "out_of_road_done": True,
            "idle_done": False,
            "traffic_density": 0.0,
            "random_agent_model": False,
            "map_config": {
                "lane_num": 2,
                "lane_width": 3.5,
                **map_cfg.get("map_config", {}),
            },
            # A little extra push against slow/stalled policies.
            "driving_reward": 1.0,
            "speed_reward": 0.3,
            "idle_penalty": 5.0,
            "success_reward": 20.0,
            "out_of_road_penalty": 5.0,
            "crash_sidewalk_penalty": 5.0,
            "log_level": 50,
        }
        self._env = MultiAgentRacingEnv(config=config)

        self.action_space = self._env.action_space["agent0"]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(FRAME_STACK, 3, CAMERA_SIZE, CAMERA_SIZE),
            dtype=np.uint8,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs_dict, info_dict = self._env.reset(seed=self._base_seed if seed is None else seed)
        self._last_opp_obs = obs_dict.get("agent1")
        self._route_completion = 0.0
        self._set_topdown_track_agent()

        frame = self._observation_frame()
        self._frames.clear()
        for _ in range(FRAME_STACK):
            self._frames.append(frame.copy())

        info = self._augment_info(info_dict.get("agent0", {}))
        return self._stacked_obs(), info

    def step(self, action):
        opp_action = self._opponent(self._last_opp_obs)
        obs_dict, r_dict, d_dict, t_dict, info_dict = self._env.step(
            {
                "agent0": np.asarray(action, dtype=np.float32).clip(-1.0, 1.0),
                "agent1": opp_action,
            }
        )
        self._last_opp_obs = obs_dict.get("agent1", self._last_opp_obs)
        self._set_topdown_track_agent()
        self._frames.append(self._observation_frame())

        info = self._augment_info(info_dict.get("agent0", {}))
        return (
            self._stacked_obs(),
            float(r_dict.get("agent0", 0.0)),
            bool(d_dict.get("agent0", False)),
            bool(t_dict.get("agent0", False)),
            info,
        )

    def render(self):
        self._set_topdown_track_agent()
        return self._topdown_frame(window=self._show_topdown, size=TOPDOWN_SCREEN_SIZE)

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    @property
    def unwrapped_metadrive(self):
        return self._env

    def _observation_frame(self) -> np.ndarray:
        size = TOPDOWN_SCREEN_SIZE if self._show_topdown else CAMERA_SIZE
        return self._topdown_frame(window=self._show_topdown, size=size)

    def _set_topdown_track_agent(self) -> None:
        agent = self._env.agents.get("agent0")
        if agent is None:
            return
        main_camera = getattr(self._env.engine, "main_camera", None)
        if main_camera is not None and hasattr(main_camera, "current_track_agent"):
            main_camera.current_track_agent = agent
        else:
            self._env.engine.main_camera = _FakeMainCamera(agent)

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        clean = dict(info)
        clean["map_name"] = self._map_name
        progress = float(clean.get("progress", 0.0))
        self._route_completion = float(np.clip(self._route_completion + max(progress, 0.0) / ROUTE_LENGTH_METERS, 0.0, 1.0))
        agent = self._env.agents.get("agent0")
        if agent is not None:
            nav_completion = float(getattr(agent.navigation, "route_completion", 0.0))
            clean["route_completion"] = max(self._route_completion, nav_completion)
            clean["speed_km_h"] = float(max(getattr(agent, "speed_km_h", 0.0), 0.0))
        else:
            clean["route_completion"] = self._route_completion
        return clean

    def _topdown_frame(self, window: bool, size: int = CAMERA_SIZE) -> np.ndarray:
        frame = self._env.render(
            mode="topdown",
            window=window,
            screen_size=(size, size),
            film_size=(3000, 3000),
            semantic_map=False,
            show_agent_name=False,
            target_agent_heading_up=True,
            draw_target_vehicle_trajectory=False,
            draw_contour=True,
            num_stack=1,
        )
        if frame is None:
            raise RuntimeError("MetaDrive top-down renderer returned no frame")
        image = np.asarray(frame)
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[0] != CAMERA_SIZE or image.shape[1] != CAMERA_SIZE:
            image = np.asarray(Image.fromarray(image).resize((CAMERA_SIZE, CAMERA_SIZE), Image.Resampling.BILINEAR))
        return np.transpose(image, (2, 0, 1))

    def _stacked_obs(self) -> np.ndarray:
        return np.stack(tuple(self._frames), axis=0).astype(np.uint8, copy=False)


def make_env(map_name: str = "chicane", render: bool = False, seed: int | None = None) -> RacingEnv:
    return RacingEnv(map_name=map_name, opponent_policy="still", render=render, seed=seed)
