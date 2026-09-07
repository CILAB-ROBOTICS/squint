"""
Evaluate a trained RL agent in simulation using the *real* deployment code path.

This is the sim-to-sim counterpart of `deploy.py`. Instead of driving a physical
SO101/SO100 arm, it plugs a simulation-backed fake robot (`SimBackedRealAgent`)
into the exact same `Sim2RealEnv` pipeline that `deploy.py` uses. The policy's
target-qpos signal is "intercepted" and replayed inside a second simulation
instance, so you can shake out the whole deploy pipeline (sensor preprocessing,
controller alignment, timing, safe-exit, recording, wandb checkpoint loading)
without any hardware. Because the stand-in robot is itself a simulation we also
get ground-truth success / return, which is reported per episode.

Usage:
    python deploy_sim.py --checkpoint path/to/checkpoint.pt --env_id SO101ReachCube-v1
    python deploy_sim.py --checkpoint wandb --env_id SO101ReachCube-v1
    python deploy_sim.py --env_id SO101ReachCube-v1                       # random agent
    python deploy_sim.py --env_id SO101ReachCube-v1 --interactive False   # auto N episodes
    python deploy_sim.py --env_id SO101ReachCube-v1 --viewer              # live SAPIEN 3D view

Keyboard Controls (interactive mode):
    's' - Skip current episode
    'q' - Quit evaluation
"""

from dataclasses import dataclass
import contextlib
import random
from typing import Optional
from pathlib import Path
import sys
import signal
import select
import termios
import tty
import atexit
import time
import threading
import queue

import gymnasium as gym
import numpy as np
import torch
import tyro
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.visualization import tile_images

# Registers the SO101* task ids.
import envs  # noqa: F401
import mani_skill.envs  # noqa: F401

from deploy_utils.sim_real_agent import SimBackedRealAgent
from deploy_utils.sim_calibrated_agent import CalibratedSimRealAgent, calibration_roundtrip_check

from train_squint import DeployAgent


class SimToSimEnv(Sim2RealEnv):
    """`Sim2RealEnv` with `base_sim_env` pinned to the real task env.

    Stock `Sim2RealEnv.base_sim_env` resolves via `self.sim_env.unwrapped`, which
    during the wrapper-swap performed inside `reset()`/`step()` walks into the
    internal `RealEnvStepReset` whose `.unwrapped` points back at the
    `Sim2RealEnv` -- making `base_sim_env.__class__.get_obs(...)` recurse forever.
    Pinning the underlying task env at construction time avoids that.
    """

    def __init__(self, sim_env, *args, throttle: bool = True, **kwargs):
        self._pinned_base_sim_env = sim_env.unwrapped
        self._throttle = throttle
        super().__init__(sim_env, *args, **kwargs)

    @property
    def base_sim_env(self):
        return self._pinned_base_sim_env

    def _step_action(self, action):
        # When not emulating real-time control, skip the wall-clock throttle in
        # the parent (which otherwise logs a "Control dt not reached" warning
        # every single step).
        if not self._throttle:
            self.last_control_time = None
        return super()._step_action(action)

# ============================================================
# ARGUMENTS
# ============================================================

@dataclass
class Args:
    checkpoint: Optional[str] = None
    """Path to checkpoint file, or 'wandb' to load from wandb. If None, uses random agent."""
    env_id: str = "SO101ReachCube-v1"
    """Environment ID (must match training environment)."""
    obs_mode: str = "rgb+segmentation"
    """Observation mode for the environment."""
    control_mode: str = "pd_joint_target_delta_pos"
    """Control mode for the robot."""
    max_episode_steps: int = 200
    """Maximum steps per episode."""
    continuous_eval: bool = True
    """If True, runs without pausing. If False, waits for Enter at each step."""
    control_freq: Optional[int] = 30
    """Control frequency in Hz. Only enforced as a real-time throttle when --realtime is set."""
    realtime: bool = False
    """If True, throttle each step to control_freq (like the real robot). If False, run as fast as possible."""
    action_scale: float = 0.15
    """Action scaling factor. Values < 1.0 make movements smaller/slower."""
    record_dir: Optional[str] = None
    """Directory to save episode recordings. If None, no recording."""
    record_resolution: int = 256
    """Resolution (HxW) for each frame in the recorded video. Use 128, 256, or 480."""
    debug: bool = False
    """If True, shows sim/real overlay visualization."""
    viewer: bool = False
    """If True, open a live SAPIEN 3D viewer on the inner (stand-in robot) simulation."""
    seed: int = 1
    """Random seed for reproducibility."""
    image_size: int = 128
    """HxW of input image to agent"""

    # Sim-to-sim evaluation settings
    interactive: bool = True
    """If True, drive episodes with the keyboard. If False, auto-run num_eval_episodes and print a summary."""
    num_eval_episodes: int = 10
    """Number of episodes to run in non-interactive mode."""
    inner_domain_randomization: bool = False
    """Enable domain randomization on the inner (stand-in robot) simulation to inject a sim-to-sim gap."""
    inner_seed_offset: int = 0
    """Seed offset for the inner simulation relative to the outer one. 0 => identical scene layout (zero gap)."""
    use_calibration: bool = False
    """If True, route control through the real LeRobot + FeetechMotorsBus calibration path (emulated servos) instead of the plain sim stand-in."""

    # Wandb checkpoint download settings (only used when checkpoint='wandb')
    wandb_entity: Optional[str] = None  # CHANGE THIS: your wandb username/entity
    wandb_project: str = "maniskill-so101"  # Your wandb project name
    wandb_agent_name: str = "squint"  # Agent name used in wandb
    wandb_version: str = "latest"  # Version of the checkpoint to download
    wandb_seeds: tuple[int, ...] = (1, 2, 3, 4, 5)  # Seeds to check for best checkpoint


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def create_wrist_camera_preprocessor(sim_env):
    """Create a preprocessing function for the wrist camera images.

    Handles:
    - Cropping to square aspect ratio
    - Resizing to match simulation camera resolution

    Args:
        sim_env: The base simulation environment (unwrapped)

    Returns:
        Preprocessing function for sensor data
    """

    def preprocess(sensor_data, sensor_names=None):
        """Preprocess sensor data to match simulation format."""
        if sensor_names is None:
            sensor_names = list(sensor_data.keys())

        for sensor_name in sensor_names:
            sim_sensor_cfg = sim_env._sensor_configs[sensor_name]
            assert isinstance(sim_sensor_cfg, CameraConfig)

            target_h, target_w = sim_sensor_cfg.height, sim_sensor_cfg.width
            real_data = sensor_data[sensor_name]

            if "rgb" not in real_data:
                continue

            img = real_data["rgb"][0].numpy()

            # Crop to square aspect ratio
            h, w = img.shape[:2]
            crop_size = min(h, w)

            if h > w:
                offset = (h - crop_size) // 2
                img = img[offset:offset + crop_size, :, :]
            elif w > h:
                offset = (w - crop_size) // 2
                img = img[:, offset:offset + crop_size, :]

            img = cv2.resize(img, (target_w, target_h))
            real_data["rgb"] = common.to_tensor(img).unsqueeze(0)
            sensor_data[sensor_name] = real_data

        return sensor_data

    return preprocess


def make_sim_real_reset(inner_seed_offset: int):
    """Build a `real_reset_function` for `Sim2RealEnv` (no 'press enter' prompt).

    Draws an explicit seed so the inner (stand-in robot) simulation can be reset
    to the *same* scene layout as the outer sim env (offset 0 => zero gap).
    """

    def sim_real_reset(env, seed=None, options=None):
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        env.sim_env.reset(seed=seed, options=options)
        qpos = env.base_sim_env.agent.robot.qpos.cpu().flatten()
        env.agent.reset(qpos=qpos, seed=seed + inner_seed_offset, options=options)

    return sim_real_reset


def setup_safe_exit(sim_env, real_env, real_agent, recorder=None):
    """Register handlers for graceful shutdown on Ctrl+C or script exit."""
    def cleanup():
        print("\nCleaning up...")
        try:
            if recorder is not None:
                recorder.close()
        except Exception:
            pass
        try:
            if real_agent is not None:
                real_agent.reset(sim_env.unwrapped.agent.keyframes["rest"].qpos)
        except Exception:
            pass
        for env in [real_env, sim_env]:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

    def signal_handler(sig=None, frame=None):
        print("\nCtrl+C detected. Exiting gracefully...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup)


def overlay_envs(sim_env, real_env):
    """Overlay sim and real observations for visual debugging."""
    real_obs = real_env.unwrapped.get_obs()["sensor_data"]
    sim_obs = sim_env.unwrapped.get_obs()["sensor_data"]

    assert sorted(real_obs.keys()) == sorted(sim_obs.keys()), \
        f"Camera mismatch: real={real_obs.keys()}, sim={sim_obs.keys()}"

    overlaid_imgs = []
    for name in sim_obs:
        real_img = real_obs[name]["rgb"][0] / 255
        sim_img = sim_obs[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_img + 0.5 * sim_img)

    return tile_images(overlaid_imgs), real_img, sim_img


def print_timing_stats(timing_stats: dict, episode_num: int, target_freq: int):
    """Print timing statistics for an episode."""
    if not timing_stats["total"]:
        return

    print(f"\n--- Timing Stats (Episode {episode_num}) ---")
    print(f"Inference: {np.mean(timing_stats['inference'])*1000:.1f}ms avg, "
          f"{np.max(timing_stats['inference'])*1000:.1f}ms max")
    print(f"Step:      {np.mean(timing_stats['step'])*1000:.1f}ms avg, "
          f"{np.max(timing_stats['step'])*1000:.1f}ms max")
    print(f"Total:     {np.mean(timing_stats['total'])*1000:.1f}ms avg, "
          f"{np.max(timing_stats['total'])*1000:.1f}ms max")
    print(f"Achieved freq: {1/np.mean(timing_stats['total']):.1f} Hz (target: {target_freq} Hz)")


def _print_calibration_report(real_agent):
    """Print the loaded servo calibration and a round-trip check through the
    full LeRobot + FeetechMotorsBus calibration path."""
    bus = real_agent.real_robot.bus
    print("\n" + "=" * 60)
    print("CALIBRATION CHECK (LeRobot path, emulated servos)")
    print(f"detected robot kind: {real_agent._robot_kind}")
    print(f"gripper norm_mode:   {bus.motors['gripper'].norm_mode}")
    print(f"\n{'motor':<14} {'id':<4} {'drive':<6} {'homing':<8} {'range_min':<10} {'range_max':<10}")
    print("-" * 60)
    for m, c in bus.calibration.items():
        print(f"{m:<14} {c.id:<4} {c.drive_mode:<6} {c.homing_offset:<8} {c.range_min:<10} {c.range_max:<10}")
    print("(homing_offset is loaded but not verifiable without hardware)")

    names, max_err = calibration_roundtrip_check(real_agent)
    print(f"\nround-trip |target - readback| (rad), max over samples:")
    print(f"{'motor':<14} {'max_err':<12}")
    print("-" * 28)
    worst = 0.0
    for n, e in zip(names, max_err):
        flag = "  <-- CHECK" if e > 0.1 else ""
        print(f"{n:<14} {e:<12.4f}{flag}")
        worst = max(worst, float(e))
    verdict = "PASS" if worst <= 0.1 else "WARN - calibration/conversion wiring likely wrong"
    print(f"\nverdict: {verdict} (worst joint error {worst:.4f} rad)")
    print("=" * 60 + "\n")


def coerce_flag(x) -> bool:
    """Turn a possibly-batched tensor/array/scalar success flag into a python bool."""
    if isinstance(x, torch.Tensor):
        return bool(x.flatten()[0].item())
    if isinstance(x, np.ndarray):
        return bool(x.flatten()[0])
    return bool(x)


def select_best_wandb_seed(entity: str, project: str, agent_name: str, env_id: str, seeds: list[int], version: str = "latest") -> Optional[int]:
    """Check wandb for checkpoints across seeds and return the one with highest eval/success_at_end."""
    import wandb
    api = wandb.Api()

    metrics_keys = ["eval/return", "eval/reward", "eval/success_at_end", "eval/success_once"]
    results = []

    for seed in seeds:
        artifact_name = f"{entity}/{project}/model_{agent_name}_{env_id}_{seed}:{version}"
        try:
            artifact = api.artifact(artifact_name)
            run = artifact.logged_by()
            metrics = {k: run.summary.get(k) for k in metrics_keys}
            results.append({"seed": seed, **metrics})
        except wandb.errors.CommError:
            print(f"  Seed {seed}: checkpoint not found ({artifact_name})")

    if not results:
        print("No checkpoints found for any seed!")
        return None

    print(f"\n{'Seed':<6} {'return':<12} {'reward':<12} {'success_end':<12} {'success_once':<12}")
    print("-" * 54)
    for r in results:
        print(f"{r['seed']:<6} {r.get('eval/return', 'N/A'):<12.4f} {r.get('eval/reward', 'N/A'):<12.4f} "
              f"{r.get('eval/success_at_end', 'N/A'):<12.4f} {r.get('eval/success_once', 'N/A'):<12.4f}")

    best = max(results, key=lambda x: x.get("eval/return", -float("inf")))
    print(f"\nSelected seed {best['seed']} (return: {best.get('eval/return', 'N/A'):.4f})")
    return best["seed"]


def extract_recording_frame(real_obs: dict) -> Optional[np.ndarray]:
    """Extract a BGR frame from the real observation for recording.

    Looks for the first RGB image in the observation dict.
    Returns a uint8 BGR numpy array, or None if no image found.
    """
    for key in real_obs:
        if "rgb" in key or "image" in key:
            img = real_obs[key]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            # Handle batched observations (B, H, W, C)
            if img.ndim == 4:
                img = img[0]
            # Convert to uint8 if normalized
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            # RGB -> BGR for cv2
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
    return None


# ============================================================
# HELPER CLASSES
# ============================================================

class KeyboardController:
    """Non-blocking keyboard input handler for episode control."""

    def __init__(self):
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *args):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def check_key(self) -> Optional[str]:
        """Check for keyboard input without blocking. Returns key or None."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class AsyncRecorder:
    """Non-blocking video recorder that writes frames in a background thread."""

    def __init__(self, output_dir: str, fps: int = 30, resolution: int = 256):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.resolution = resolution
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._write_loop, daemon=True)
        self._worker.start()
        self._episode_count = 0
        self._writer = None

    def push(self, frame: np.ndarray):
        """Push a BGR frame to be written. Non-blocking."""
        self._queue.put(("frame", frame))

    def end_episode(self):
        """Signal end of current episode. Starts a new video file on next push."""
        self._queue.put(("end_episode", None))

    def _write_loop(self):
        while True:
            msg_type, data = self._queue.get()

            if msg_type == "close":
                if self._writer is not None:
                    self._writer.release()
                break

            elif msg_type == "end_episode":
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
                self._episode_count += 1

            elif msg_type == "frame":
                # Crop to square and resize to target resolution
                h, w = data.shape[:2]
                crop = min(h, w)
                y0 = (h - crop) // 2
                x0 = (w - crop) // 2
                data = data[y0:y0 + crop, x0:x0 + crop]
                if data.shape[0] != self.resolution or data.shape[1] != self.resolution:
                    data = cv2.resize(data, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

                if self._writer is None:
                    path = str(self.output_dir / f"episode_{self._episode_count}.mp4")
                    self._writer = cv2.VideoWriter(
                        path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (self.resolution, self.resolution),
                    )
                self._writer.write(data)

    @property
    def queue_size(self):
        return self._queue.qsize()

    def close(self):
        """Flush remaining frames and shut down the writer thread."""
        self._queue.put(("close", None))
        self._worker.join()


# ============================================================
# MAIN
# ============================================================

def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --------------------------------------------------
    # Phase 1: Robot & Environment Setup
    # --------------------------------------------------
    print("Setting up stand-in robot and environment...")

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode="sensors",
        max_episode_steps=args.max_episode_steps,
        domain_randomization=False,
        reward_mode="none",
        control_mode=args.control_mode,
        sensor_configs=dict(width=args.image_size, height=args.image_size)
    )

    # The inner simulation stands in for the real robot. It needs a computable
    # reward mode so we can report success/return, and it can optionally diverge
    # from the outer env (domain randomization / seed) to inject a sim-to-sim gap.
    inner_env_kwargs = dict(
        obs_mode=args.obs_mode,
        max_episode_steps=args.max_episode_steps,
        domain_randomization=args.inner_domain_randomization,
        reward_mode="normalized_dense",
        control_mode=args.control_mode,
        sensor_configs=dict(width=args.image_size, height=args.image_size),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_calibration:
        real_agent = CalibratedSimRealAgent(
            env_id=args.env_id,
            env_kwargs=inner_env_kwargs,
            device=device,
            seed=args.seed + args.inner_seed_offset,
            viewer=args.viewer,
        )
        _print_calibration_report(real_agent)
    else:
        real_agent = SimBackedRealAgent(
            env_id=args.env_id,
            env_kwargs=inner_env_kwargs,
            device=device,
            seed=args.seed + args.inner_seed_offset,
            viewer=args.viewer,
        )

    sim_env = gym.make(args.env_id, **env_kwargs)
    sim_env = FlattenRGBDObservationWrapper(sim_env, rgb=True, depth=False, state=True)

    # Async recorder for recording videos
    recorder = None
    if args.record_dir:
        recorder = AsyncRecorder(
            output_dir=args.record_dir,
            fps=args.control_freq or 30,
            resolution=args.record_resolution,
        )

    preprocessor = create_wrist_camera_preprocessor(sim_env.unwrapped)
    real_env = SimToSimEnv(
        sim_env=sim_env,
        agent=real_agent,
        control_freq=args.control_freq,
        sensor_data_preprocessing_function=preprocessor,
        real_reset_function=make_sim_real_reset(args.inner_seed_offset),
        throttle=args.realtime,
    )

    sim_obs, _ = sim_env.reset()
    real_obs, _ = real_env.reset()

    print("\nObservation shapes:")
    for k in sim_obs.keys():
        print(f"  {k}: sim={sim_obs[k].shape}, real={real_obs[k].shape}")

    setup_safe_exit(sim_env, real_env, real_agent, recorder=recorder)

    # --------------------------------------------------
    # Phase 2: Agent Loading
    # --------------------------------------------------
    print("\nLoading agent...")

    agent = DeployAgent(sim_env, sample_obs=real_obs)

    if args.checkpoint:
        seed_to_use = args.seed
        if args.checkpoint == "wandb" and args.wandb_seeds:
            seed_to_use = select_best_wandb_seed(
                args.wandb_entity, args.wandb_project, args.wandb_agent_name,
                args.env_id, list(args.wandb_seeds), args.wandb_version
            )
            if seed_to_use is None:
                print("No valid checkpoints found. Using random agent.")
                args.checkpoint = None

        if args.checkpoint:
            checkpoint_config = {
                "wandb_entity": args.wandb_entity,
                "wandb_project_name": args.wandb_project,
                "agent_name": args.wandb_agent_name,
                "env_id": args.env_id,
                "seed": seed_to_use,
                "version": args.wandb_version
            }
            agent.load_checkpoint(args.checkpoint, checkpoint_config)
            print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided - using random agent")

    agent.to(device)

    # --------------------------------------------------
    # Phase 3: Debug Visualization Setup
    # --------------------------------------------------
    if args.debug:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

        overlaid, real_img, sim_img = overlay_envs(sim_env, real_env)
        im1 = ax1.imshow(overlaid)
        ax1.set_title('Overlaid (Sim + Real)')
        im2 = ax2.imshow(sim_img)
        ax2.set_title('Simulation')
        im3 = ax3.imshow(real_img)
        ax3.set_title('Real (inner sim)')
        plt.tight_layout()

    # --------------------------------------------------
    # Phase 4: Evaluation Loop
    # --------------------------------------------------
    print("\n" + "=" * 40)
    if args.interactive:
        print("KEYBOARD CONTROLS")
        print("  's' - Skip to next episode")
        print("  'q' - Quit evaluation")
    else:
        print(f"AUTO EVAL: {args.num_eval_episodes} episodes")
    print("=" * 40 + "\n")

    episode_count = 0
    episode_stats = []  # list of (return, success_once, success_at_end)
    timing_stats = {"inference": [], "step": [], "total": []}

    kb_ctx = KeyboardController() if args.interactive else contextlib.nullcontext()
    with kb_ctx as kb:
        while True:
            print("=============================")
            if args.interactive:
                print(f"Episode {episode_count} - Press Enter to start, 'q' to quit")
                while (key := kb.check_key()) not in ('\n', '\r', 'q'):
                    time.sleep(0.01)
                if key == 'q':
                    print("[Quitting...]")
                    break
            else:
                if episode_count >= args.num_eval_episodes:
                    break
                key = None
                print(f"Episode {episode_count} / {args.num_eval_episodes}")

            skip_episode = False
            ep_return = 0.0
            ep_success_once = False
            ep_success_at_end = False

            for _ in tqdm(range(args.max_episode_steps), desc="Steps"):
                if args.interactive:
                    key = kb.check_key()
                    if key == 's':
                        print("\n[Skipping episode...]")
                        real_obs, _ = real_env.reset()
                        skip_episode = True
                        break
                    elif key == 'q':
                        print("\n[Quitting...]")
                        skip_episode = True
                        break

                loop_start = time.perf_counter()

                obs = {k: v.to(device) for k, v in real_obs.items()}

                t0 = time.perf_counter()
                action = agent.get_action(obs)
                timing_stats["inference"].append(time.perf_counter() - t0)

                if not args.continuous_eval:
                    input("Press Enter to continue...")

                action = action.cpu().numpy()
                scaled_action = np.clip(action * args.action_scale, -1, 1)
                real_agent.pending_action = scaled_action

                t0 = time.perf_counter()
                real_obs, _, terminated, truncated, info = real_env.step(scaled_action)
                timing_stats["step"].append(time.perf_counter() - t0)

                timing_stats["total"].append(time.perf_counter() - loop_start)

                # Ground-truth metrics come from the inner (stand-in) simulation.
                step_success = coerce_flag(real_agent.last_info.get("success", False))
                ep_return += real_agent.last_reward
                ep_success_once = ep_success_once or step_success
                ep_success_at_end = step_success

                # Async recording: push frame without blocking
                if recorder:
                    frame = extract_recording_frame(real_obs)
                    if frame is not None:
                        recorder.push(frame)

                if args.debug:
                    # Step sim env with zero action to update wrist camera (qpos already synced)
                    sim_env.step(np.zeros_like(scaled_action))

                    overlaid, real_img, sim_img = overlay_envs(sim_env, real_env)
                    im1.set_data(overlaid)
                    im2.set_data(sim_img)
                    im3.set_data(real_img)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.001)

            # If 'q' was pressed mid-episode, break out of the while loop too
            if key == 'q':
                break

            print_timing_stats(timing_stats, episode_count, args.control_freq)
            if not skip_episode:
                print(f"--- Eval Stats (Episode {episode_count}) ---")
                print(f"return:         {ep_return:.3f}")
                print(f"success_once:   {ep_success_once}")
                print(f"success_at_end: {ep_success_at_end}")
                episode_stats.append((ep_return, ep_success_once, ep_success_at_end))

            if recorder and recorder.queue_size > 0:
                print(f"  Recorder queue: {recorder.queue_size} frames pending write")
            timing_stats = {"inference": [], "step": [], "total": []}

            # Signal end of episode to recorder
            if recorder:
                recorder.end_episode()

            episode_count += 1

            if not skip_episode:
                real_obs, _ = real_env.reset()

    # Summary over all completed episodes
    if episode_stats:
        rets = np.array([s[0] for s in episode_stats], dtype=np.float64)
        so = np.array([s[1] for s in episode_stats], dtype=np.float64)
        se = np.array([s[2] for s in episode_stats], dtype=np.float64)
        print("\n" + "=" * 40)
        print(f"=== Summary over {len(episode_stats)} episodes ===")
        print(f"success_once:   {so.mean():.3f}")
        print(f"success_at_end: {se.mean():.3f}")
        print(f"return:         {rets.mean():.3f} +/- {rets.std():.3f}")
        print("=" * 40)

    # Cleanup
    print("\nReturning robot to rest position...")
    try:
        real_agent.reset(sim_env.unwrapped.agent.keyframes["rest"].qpos)
    except Exception as e:
        print(f"Warning: failed to reset robot to rest: {e}")

    if recorder:
        recorder.close()

    print("Evaluation complete.")
    for env in [sim_env, real_env]:
        try:
            env.close()
        except Exception as e:
            print(f"Warning: error closing environment: {e}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
