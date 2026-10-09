"""
Sim-to-sim stand-in that ALSO exercises the LeRobot calibration / hardware
conversion path.

`SimBackedRealAgent` (see `sim_real_agent.py`) bypasses LeRobot entirely. This
module instead keeps the *real* `LeRobotRealAgent` + real `FeetechMotorsBus`
(with calibration loaded from `deploy_utils/so101_follower.json`) and only
replaces the serial transport with an emulated set of servos coupled to a second
ManiSkill simulation. The full chain that `deploy.py` relies on is executed:

    Sim2RealEnv
      -> LeRobotRealAgent      (rad<->deg, SO101 gripper sim<->servo remap, joint order)
      -> FeetechMotorsBus       (_normalize / _unnormalize using JSON range/drive_mode/norm_mode,
                                 sign-magnitude encode/decode)
      -> emulated servos        (Goal/Present position registers, homing-offset register)
      -> inner ManiSkill sim

What this validates: that the calibration file is parsed and wired correctly,
that per-motor `norm_mode` (gripper vs body) is honoured, that the SO100/SO101
detection fires, sign handling, joint ordering, and the rad<->deg + gripper
remap math round-trip.

What it CANNOT validate (needs real hardware): the absolute correctness of the
recorded `homing_offset` values, and camera extrinsic calibration (real camera
pose/FOV vs the sim wrist-camera constants). The emulated encoder is defined as
perfectly zeroed, so `homing_offset` is loaded but not put under test.
"""

import math
import types
from typing import Optional

import numpy as np
import torch
import gymnasium as gym

from mani_skill.utils import common

# Registers SO101* task ids.
import envs  # noqa: F401
import mani_skill.envs  # noqa: F401

from lerobot.motors.encoding_utils import encode_sign_magnitude, decode_sign_magnitude

from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.robot_config import create_real_robot

_SIGN_BIT = 15            # sts3215 Present/Goal_Position sign bit
_MAX_RES = 4095           # sts3215 resolution - 1
_CENTER = _MAX_RES / 2.0
_TICKS_PER_RAD = _MAX_RES / (2.0 * math.pi)


class _FakeCamera:
    """Minimal LeRobot-camera stand-in that returns the inner sim's wrist frame."""

    def __init__(self, inner_env, sensor_name: str = "base_camera"):
        self.inner_env = inner_env
        self.sensor_name = sensor_name
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def async_read(self, *args, **kwargs) -> np.ndarray:
        uenv = self.inner_env.unwrapped
        uenv.scene.update_render()
        data = uenv.get_obs()["sensor_data"][self.sensor_name]["rgb"]
        img = data[0]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        return img.astype(np.uint8)


class _FakeFeetechServos:
    """Emulates the 6 SO101 servos + wire protocol, coupled to an inner sim env.

    Registered as `bus._sync_read` / `bus._sync_write` so the real `sync_read` /
    `sync_write` bodies (normalisation + sign handling) run unchanged; only the
    raw register I/O is faked.
    """

    def __init__(self, inner_env, bus, motor_names, sim_steps: int):
        self.inner_env = inner_env
        self.bus = bus
        self.motor_names = list(motor_names)
        self.sim_steps = max(1, int(sim_steps))
        self._id_to_name = {bus.motors[m].id: m for m in self.motor_names}
        self._name_to_jidx = {m: i for i, m in enumerate(self.motor_names)}
        # Perfectly-zeroed encoder: the JSON homing_offset is loaded into the bus
        # calibration but the emulated servo applies none, so it is not tested.
        self._homing = {m: 0 for m in self.motor_names}
        # Optional gripper linkage (servo_min, servo_max, sim_min, sim_max) in
        # degrees -- set from the agent after LeRobotRealAgent.__init__ so the
        # emulated mechanical linkage matches the remap the agent applies.
        self._grip = None
        self._goal_rad = list(map(float, self._sim_qpos()))
        self.last_reward = 0.0
        self.last_info: dict = {}
        self._render_cb = None

    # ---- sim helpers ----
    def _sim_qpos(self) -> np.ndarray:
        return self.inner_env.unwrapped.agent.robot.get_qpos().cpu().flatten().numpy()

    def _mid(self, name: str) -> float:
        c = self.bus.calibration[name]
        return (c.range_min + c.range_max) / 2.0

    def _servo_deg_to_ticks(self, name: str, servo_deg: float) -> float:
        return servo_deg * _MAX_RES / 360.0 + self._mid(name)

    def _ticks_to_servo_deg(self, name: str, ticks: float) -> float:
        return (ticks - self._mid(name)) * 360.0 / _MAX_RES

    def _servo_deg_to_sim_rad(self, name: str, servo_deg: float) -> float:
        if name == "gripper" and self._grip is not None:
            s_min, s_max, q_min, q_max = self._grip
            sim_deg = (servo_deg - s_min) / (s_max - s_min) * (q_max - q_min) + q_min
            return math.radians(sim_deg)
        return math.radians(servo_deg)

    def _sim_rad_to_servo_deg(self, name: str, rad: float) -> float:
        sim_deg = math.degrees(rad)
        if name == "gripper" and self._grip is not None:
            s_min, s_max, q_min, q_max = self._grip
            return (sim_deg - q_min) / (q_max - q_min) * (s_max - s_min) + s_min
        return sim_deg

    # ---- raw register I/O (installed onto the bus) ----
    def _sync_read(self, addr, length, motor_ids, *, num_retry=0, raise_on_error=True, err_msg=""):
        qpos = self._sim_qpos()
        out = {}
        for mid in motor_ids:
            name = self._id_to_name[mid]
            servo_deg = self._sim_rad_to_servo_deg(name, float(qpos[self._name_to_jidx[name]]))
            actual = self._servo_deg_to_ticks(name, servo_deg)
            present_signed = int(round(actual - self._homing[name]))
            out[mid] = encode_sign_magnitude(present_signed, _SIGN_BIT)
        return out, 0

    def _sync_write(self, addr, length, ids_values, num_retry=0, raise_on_error=True, err_msg=""):
        for mid, raw in ids_values.items():
            name = self._id_to_name[mid]
            goal_actual = decode_sign_magnitude(int(raw), _SIGN_BIT) + self._homing[name]
            servo_deg = self._ticks_to_servo_deg(name, goal_actual)
            self._goal_rad[self._name_to_jidx[name]] = self._servo_deg_to_sim_rad(name, servo_deg)
        self._drive_and_step()
        return 0

    # ---- inner sim stepping ----
    def _drive_and_step(self):
        uenv = self.inner_env.unwrapped
        robot = uenv.agent.robot
        target = torch.tensor([self._goal_rad], dtype=torch.float32)
        try:
            robot.set_joint_drive_targets(target, robot.active_joints)
        except Exception:
            robot.set_joint_drive_targets(target)
        for _ in range(self.sim_steps):
            uenv.scene.step()
        uenv.scene.update_render()
        self.refresh_metrics()
        if self._render_cb is not None:
            self._render_cb()

    def refresh_metrics(self):
        uenv = self.inner_env.unwrapped
        try:
            info = uenv.evaluate()
        except Exception:
            info = {}
        self.last_info = info
        try:
            obs = uenv.get_obs(info)
            reward = uenv.get_reward(obs=obs, action=None, info=info)
            self.last_reward = float(common.to_tensor(reward).flatten()[0].item())
        except Exception:
            self.last_reward = 0.0


class CalibratedSimRealAgent(LeRobotRealAgent):
    """`LeRobotRealAgent` whose real robot is emulated servos + an inner sim."""

    def __init__(self, env_id: str, env_kwargs: dict, device=None, seed: int = 0, viewer: bool = False):
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

        robot = create_real_robot()
        motor_names = list(robot.bus.motors.keys())
        uenv = self.inner_env.unwrapped
        sim_steps = int(round(uenv.sim_freq / uenv.control_freq))

        # Replace the serial layer with the emulator.
        robot.bus.port_handler = types.SimpleNamespace(is_open=True)
        self._servos = _FakeFeetechServos(self.inner_env, robot.bus, motor_names, sim_steps)
        robot.bus._sync_read = self._servos._sync_read
        robot.bus._sync_write = self._servos._sync_write
        robot.bus._is_comm_success = lambda comm: True

        cam_names = list(robot.cameras.keys()) or ["base_camera"]
        robot.cameras = {name: _FakeCamera(self.inner_env, name) for name in cam_names}

        super().__init__(robot)  # sets gripper norm_mode, infers SO100/SO101, etc.

        self._motor_keys = list(robot.bus.motors.keys())
        if all(hasattr(self, a) for a in ("_gripper_servo_min", "_gripper_servo_max",
                                          "_gripper_sim_min", "_gripper_sim_max")):
            self._servos._grip = (self._gripper_servo_min, self._gripper_servo_max,
                                  self._gripper_sim_min, self._gripper_sim_max)
        self._servos._render_cb = self._render_viewer

    # ---- metrics passthrough (used by deploy_sim.py) ----
    @property
    def last_reward(self) -> float:
        return self._servos.last_reward

    @property
    def last_info(self) -> dict:
        return self._servos.last_info

    # ---- overrides ----
    def _render_viewer(self):
        if self.viewer:
            self.inner_env.render()

    def reset(self, qpos, seed: Optional[int] = None, options: Optional[dict] = None):
        # Skip LeRobotRealAgent's slow gradual-approach loop (a real-robot safety
        # measure); teleport the inner sim robot instead.
        reset_seed = self._seed if seed is None else seed
        self.inner_env.reset(seed=reset_seed)
        uenv = self.inner_env.unwrapped
        qpos_t = common.to_tensor(qpos).reshape(1, -1).float()
        uenv.agent.robot.set_qpos(qpos_t)
        try:
            uenv.agent.robot.set_joint_drive_targets(qpos_t, uenv.agent.robot.active_joints)
        except Exception:
            pass
        if uenv.gpu_sim_enabled:
            uenv.scene._gpu_apply_all()
        uenv.scene.update_render()
        self._servos._goal_rad = list(map(float, qpos_t.flatten().tolist()))
        self._servos.refresh_metrics()
        self._cached_qpos = None
        self._motor_keys = list(self.real_robot.bus.motors.keys())
        self._render_viewer()

    def get_qvel(self):
        return self.inner_env.unwrapped.agent.robot.get_qvel().cpu()

    def stop(self):
        try:
            self.inner_env.close()
        except Exception:
            pass


def calibration_roundtrip_check(agent: CalibratedSimRealAgent, n_targets: int = 5, settle: int = 200, frac: float = 0.4):
    """Drive the emulated arm to random targets and read them back through the
    full LeRobot calibration path. Returns (joint_names, max_abs_error_rad).

    A correctly wired stack round-trips to ~0 on every joint. A large error on
    `gripper` points at the SO100/SO101 detection or gripper norm_mode/remap; a
    large error across all joints points at a sign / unit / joint-order bug.
    """
    uenv = agent.inner_env.unwrapped
    names = [j.name for j in uenv.agent.robot.active_joints]
    qlim = uenv.agent.robot.get_qlimits().cpu().numpy()[0]  # (n, 2)
    lo, hi = qlim[:, 0], qlim[:, 1]
    mid = 0.5 * (lo + hi)
    max_err = np.zeros(len(names))
    rng = np.random.default_rng(0)
    for _ in range(n_targets):
        q = (mid + frac * (rng.uniform(lo, hi) - mid)).astype(np.float32)
        for _ in range(settle):
            agent.set_target_qpos(torch.tensor(q))
        agent._cached_qpos = None
        back = agent.get_qpos().cpu().flatten().numpy()
        max_err = np.maximum(max_err, np.abs(q - back))
    return names, max_err
