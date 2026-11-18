"""Tests for skateboard/tools/policy_loader.py (config only, no env is built)."""

import importlib.util
from pathlib import Path

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.tasks.pupper_skate.mdp import PupperVelocityCommandCfg
from mjlab.tasks.registry import load_env_cfg

REPO = Path(__file__).parent.parent


def _load_policy_loader():
  path = REPO / "skateboard" / "tools" / "policy_loader.py"
  spec = importlib.util.spec_from_file_location("policy_loader", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


pl = _load_policy_loader()


def test_fix_speed_pins_the_command_and_turns_off_standing():
  cfg = load_env_cfg("Mjlab-Skate-Ride-Pupper-v3", play=True)
  twist = cfg.commands["twist"]
  assert isinstance(twist, PupperVelocityCommandCfg)
  assert twist.rel_standing_envs > 0.0
  pl.fix_speed(cfg, 0.4)
  assert twist.ranges.lin_vel_x == (0.4, 0.4)
  assert twist.rel_standing_envs == 0.0


def test_fix_yaw_rate_pins_the_yaw_command_on_the_steer_task():
  cfg = load_env_cfg("Mjlab-SkateSplit-Steer-Pupper-v3", play=True)
  twist = cfg.commands["twist"]
  assert isinstance(twist, PupperVelocityCommandCfg)
  assert isinstance(twist, pl.SteerVelocityCommandCfg)
  assert twist.rel_straight_envs > 0.0
  pl.fix_yaw_rate(cfg, 0.1)
  assert twist.ranges.ang_vel_z == (0.1, 0.1)
  assert twist.rel_forward_envs == 0.0 and twist.rel_turn_envs == 0.0
  assert twist.rel_straight_envs == 0.0


def test_zero_yaw_rate_leaves_the_other_skate_commands_alone():
  for task in (
    "Mjlab-Skate-Balance-Pupper-v3",
    "Mjlab-Skate-Ride-Pupper-v3",
    "Mjlab-SkateSplit-Balance-Pupper-v3",
    "Mjlab-SkateSplit-Ride-Pupper-v3",
  ):
    cfg = load_env_cfg(task, play=True)
    before = repr(cfg.commands["twist"])
    pl.fix_yaw_rate(cfg, 0.0)
    assert repr(cfg.commands["twist"]) == before, task
