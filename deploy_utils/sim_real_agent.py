"""
A simulation-backed stand-in for a real robot, for testing the deploy pipeline
without hardware (sim-to-sim).

`SimBackedRealAgent` implements the same `BaseRealAgent` interface as
`deploy_utils.manipulator.LeRobotRealAgent`, but instead of talking to LeRobot
servos and USB cameras it drives a second ManiSkill simulation instance. When
plugged into `mani_skill.envs.sim2real_env.Sim2RealEnv` it makes the deploy loop
believe it is talking to a real robot: the policy's target-qpos signal is
"intercepted" by this agent and replayed inside the inner simulation, and camera
frames / joint positions are read back out of that same inner simulation.

Because the inner env is a full simulation we also get ground-truth success and
reward for free (unlike a real robot), exposed via `last_info` / `last_reward`.
"""

from typing import Optional

import numpy as np
import torch
import gymnasium as gym

from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.utils import common

# Registers the SO101* task ids (SO101ReachCube-v1 etc.)
import envs  # noqa: F401
import mani_skill.envs  # noqa: F401


class SimBackedRealAgent(BaseRealAgent):
    """A `BaseRealAgent` whose "real robot" is a second simulation instance.

    Args:
        env_id: task id, must match the outer sim env used by `Sim2RealEnv`.
        env_kwargs: kwargs forwarded to `gym.make` for the inner env. Should
            match the outer env (obs_mode, control_mode, sensor_configs, ...)
            except `reward_mode`, which is forced to something computable so we
            can report success/return. Any `render_mode` key is ignored (it is
            derived from `viewer`).
        device: torch device the policy runs on (only used for tensor moves).
        seed: base seed for the inner env's first reset.
        viewer: if True, open a live SAPIEN 3D viewer on the inner env so the
            robot can be watched in real time.
    """

    def __init__(
        self,
        env_id: str,
        env_kwargs: dict,
        device: Optional[torch.device] = None,
        seed: int = 0,
        viewer: bool = False,
    ):
        super().__init__()
        self.device = device
        self.viewer = viewer
        self._seed = seed

        env_kwargs = dict(env_kwargs)
        env_kwargs.pop("render_mode", None)
        env_kwargs.pop("num_envs", None)
        self.inner_env = gym.make(
            env_id,
            num_envs=1,
            sim_backend="physx_cpu",
            render_mode="human" if viewer else "rgb_array",
            **env_kwargs,
        )
        self.inner_env.reset(seed=seed)

        # Filled in by `set_target_qpos` / `reset`.
        self.pending_action: Optional[np.ndarray] = None
        self.last_reward: float = 0.0
        self.last_info: dict = {}
        self._captured_sensor_data: Optional[dict] = None
        self._last_sensor_data: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def _unwrapped(self):
        return self.inner_env.unwrapped

    def _render_viewer(self):
        if self.viewer:
            self.inner_env.render()

    def _zero_action(self) -> np.ndarray:
        return np.zeros(self._unwrapped.single_action_space.shape, dtype=np.float32)

    def _refresh_sensor_cache_from_env(self):
        self._unwrapped.scene.update_render()
        self._last_sensor_data = self._unwrapped.get_obs()["sensor_data"]

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        pass

    def stop(self):
        try:
            self.inner_env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # control
    # ------------------------------------------------------------------ #
    def set_target_qpos(self, qpos):
        """Advance the inner sim by one control step.

        `Sim2RealEnv` hands us the drive targets it computed with the outer
        controller, but since the inner env uses the identical controller we
        just replay the original (pre-scaled) policy action stashed in
        `pending_action` -- this keeps PD gains / substepping / rendering all
        handled by the inner env itself.
        """
        action = self.pending_action
        if action is None:
            action = self._zero_action()
        obs, reward, terminated, truncated, info = self.inner_env.step(action)
        self._last_sensor_data = obs["sensor_data"]
        self.last_reward = float(common.to_tensor(reward).flatten()[0].item())
        self.last_info = info
        self._render_viewer()

    # set_target_qvel intentionally not implemented: the SO101
    # pd_joint_target_delta_pos controller has sets_target_qvel == False.

    def reset(self, qpos, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the inner env and move its robot to `qpos`.

        `qpos` is whatever the outer sim env sampled on its own reset. Passing
        the matching `seed` makes the inner scene layout identical to the outer
        one (zero sim-to-sim gap); a different seed / domain randomization
        injects a gap on purpose.
        """
        reset_seed = self._seed if seed is None else seed
        self.inner_env.reset(seed=reset_seed)

        qpos_t = common.to_tensor(qpos).reshape(1, -1).float()
        robot = self._unwrapped.agent.robot
        robot.set_qpos(qpos_t)
        try:
            robot.set_joint_drive_targets(qpos_t, robot.active_joints)
        except Exception:
            pass
        if self._unwrapped.gpu_sim_enabled:
            self._unwrapped.scene._gpu_apply_all()

        self.pending_action = None
        self.last_reward = 0.0
        self.last_info = {}
        self._refresh_sensor_cache_from_env()
        self._render_viewer()

    # ------------------------------------------------------------------ #
    # data access
    # ------------------------------------------------------------------ #
    def capture_sensor_data(self, sensor_names: Optional[list[str]] = None):
        data = self._last_sensor_data
        if data is None:
            self._refresh_sensor_cache_from_env()
            data = self._last_sensor_data

        if sensor_names is None:
            sensor_names = list(data.keys())

        captured = {}
        for name in sensor_names:
            cam = data.get(name, {})
            if "rgb" not in cam:
                continue
            rgb = cam["rgb"]
            if not isinstance(rgb, torch.Tensor):
                rgb = common.to_tensor(rgb)
            captured[name] = dict(rgb=rgb.cpu())
        self._captured_sensor_data = captured

    def get_sensor_data(self, sensor_names: Optional[list[str]] = None):
        if self._captured_sensor_data is None:
            raise RuntimeError(
                "No sensor data captured yet. Call capture_sensor_data() first."
            )
        if sensor_names is None:
            return self._captured_sensor_data
        return {
            k: v for k, v in self._captured_sensor_data.items() if k in sensor_names
        }

    def get_qpos(self):
        return self._unwrapped.agent.robot.get_qpos().cpu()

    def get_qvel(self):
        return self._unwrapped.agent.robot.get_qvel().cpu()
