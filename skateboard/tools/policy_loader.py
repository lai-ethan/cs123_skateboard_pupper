"""Build play-mode skate envs and load a saved policy into them.

Shared by `render_policy.py`, `stance_audit.py` and `gait_report.py`. Takes a
training checkpoint (`model_*.pt`) or an exported `policy.json`.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.pupper.export import json_forward, to_robot_order
from mjlab.tasks.pupper_skate.mdp import (
  PupperVelocityCommandCfg,
  SteerVelocityCommandCfg,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def fix_speed(env_cfg: Any, speed: float) -> None:
  """Command a fixed board speed in m/s and turn off standing commands.

  Without the second edit, each command resample has a `rel_standing_envs`
  chance of commanding 0 m/s, so a check at a fixed speed would sometimes
  measure a robot told to stand still.
  """
  twist = env_cfg.commands["twist"]
  assert isinstance(twist, PupperVelocityCommandCfg)
  twist.ranges.lin_vel_x = (speed, speed)
  twist.rel_standing_envs = 0.0


def fix_yaw_rate(env_cfg: Any, yaw_rate: float) -> None:
  """Command a fixed yaw rate in rad/s in every env.

  Also turns off the forward-only and turn-only command shares, which would
  zero or replace the yaw rate (and lift or zero the speed) in some envs. On
  the tasks with no yaw command (all but the steer task) a yaw rate of 0 is
  the trained range already, and both shares are already 0, so nothing
  changes.

  On the steer task the yaw command is a carve signal. A yaw rate of 0 makes
  every env ride straight, so the heading hold is on, as in the ride. Any
  other value is the raw carve command in every env, with its sign. The task
  trains on carves of 0.1 to 0.6 rad/s in size; under the hold's threshold
  (0.1) the hold replaces the command. Speed commands under the task's
  `yaw_min_speed` still get no carve.

  Raises ValueError on the steer task for a yaw rate outside the task's
  command clip (its `ang_vel_z` range, the export's `wz` clip): the robot
  would clip it, so the sim must not run it.
  """
  twist = env_cfg.commands["twist"]
  assert isinstance(twist, PupperVelocityCommandCfg)
  if isinstance(twist, SteerVelocityCommandCfg):
    low, high = twist.ranges.ang_vel_z
    if not low <= yaw_rate <= high:
      raise ValueError(
        f"yaw rate {yaw_rate} rad/s is outside the command clip {low} to {high}"
      )
  twist.ranges.ang_vel_z = (yaw_rate, yaw_rate)
  twist.rel_forward_envs = 0.0
  twist.rel_turn_envs = 0.0
  if isinstance(twist, SteerVelocityCommandCfg):
    # The steer command draws its carve from carve_range, not ang_vel_z.
    if yaw_rate == 0.0:
      twist.rel_straight_envs = 1.0
    else:
      twist.rel_straight_envs = 0.0
      twist.carve_range = (abs(yaw_rate), abs(yaw_rate))
      twist.carve_left_share = 1.0 if yaw_rate > 0.0 else 0.0


def load_policy(
  task: str,
  policy_path: Path,
  speed: float | None,
  *,
  render_mode: str | None = None,
  viewer: dict[str, float] | None = None,
  configure: Callable[[Any], None] | None = None,
  num_envs: int = 1,
  seed: int | None = None,
  yaw_rate: float | None = None,
) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper, Callable]:
  """Play-mode envs, their rsl_rl wrapper, and `policy(obs) -> actions`.

  Builds `num_envs` envs (1 by default). `seed` fixes the envs' random draws
  (board, gains, commands) so two runs match; None draws a fresh seed.
  `speed` fixes the board speed command in m/s (ride) and turns off standing
  commands (see `fix_speed`); None keeps the task's range and standing share.
  `yaw_rate` fixes the board yaw-rate command in rad/s (see `fix_yaw_rate`);
  None keeps the task's yaw range.
  `render_mode` and `viewer` (attributes set on the env's viewer config) are
  only needed for rendering. `configure(env_cfg)`, if given, edits the env
  config after the edits above and before the env is built.
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = num_envs
  if seed is not None:
    env_cfg.seed = seed
  for key, value in (viewer or {}).items():
    setattr(env_cfg.viewer, key, value)
  if speed is not None:
    fix_speed(env_cfg, speed)
  if yaw_rate is not None:
    fix_yaw_rate(env_cfg, yaw_rate)
  if configure is not None:
    configure(env_cfg)
  agent_cfg = load_rl_cfg(task)
  raw = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)

  if str(policy_path).endswith(".json"):
    exported = json.loads(Path(policy_path).read_text())
    history = exported["observation_history"]
    frame = exported["single_observation_size"]

    def policy(obs):
      x = to_robot_order(obs["actor"], history, frame)
      return torch.as_tensor(
        json_forward(exported, x), dtype=torch.float32, device=device
      )

  else:
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(policy_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
  return raw, env, policy
