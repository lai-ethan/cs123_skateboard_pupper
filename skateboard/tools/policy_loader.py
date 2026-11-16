"""Build one play-mode skate env and load a saved policy into it.

Shared by `render_policy.py` and `gait_report.py`. Takes a training checkpoint
(`model_*.pt`) or an exported `policy.json`.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.pupper.export import json_forward, to_robot_order
from mjlab.tasks.pupper_skate.mdp import PupperVelocityCommandCfg
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


def load_policy(
  task: str,
  policy_path: Path,
  speed: float | None,
  *,
  render_mode: str | None = None,
  viewer: dict[str, float] | None = None,
  configure: Callable[[Any], None] | None = None,
) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper, Callable]:
  """One env in play mode, its rsl_rl wrapper, and `policy(obs) -> actions`.

  `speed` fixes the board speed command in m/s (ride) and turns off standing
  commands (see `fix_speed`); None keeps the task's range and standing share.
  `render_mode` and `viewer` (attributes set on the env's viewer config) are
  only needed for rendering. `configure(env_cfg)`, if given, edits the env
  config after the edits above and before the env is built.
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = 1
  for key, value in (viewer or {}).items():
    setattr(env_cfg.viewer, key, value)
  if speed is not None:
    fix_speed(env_cfg, speed)
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
