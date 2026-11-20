"""Tests for skateboard/tools/policy_loader.py (config only, no env is built)."""

import importlib.util
from pathlib import Path

import pytest

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


def test_fix_yaw_rate_pins_a_raw_carve_on_the_steer_task():
  for yaw_rate, share in ((0.5, 1.0), (-0.4, 0.0), (0.2, 1.0), (-0.15, 0.0)):
    cfg = load_env_cfg("Mjlab-SkateSplit-Steer-Pupper-v3", play=True)
    twist = cfg.commands["twist"]
    assert isinstance(twist, pl.SteerVelocityCommandCfg)
    assert twist.rel_straight_envs > 0.0
    pl.fix_yaw_rate(cfg, yaw_rate)
    assert twist.carve_range == (abs(yaw_rate), abs(yaw_rate))
    assert twist.carve_left_share == share
    assert twist.rel_forward_envs == 0.0 and twist.rel_turn_envs == 0.0
    assert twist.rel_straight_envs == 0.0
    # The heading hold stays on; a carve is above its threshold.
    assert twist.heading_hold_kp is not None
    assert abs(yaw_rate) > twist.heading_hold_yaw_threshold


def test_zero_yaw_rate_rides_the_steer_task_straight_with_the_hold_on():
  cfg = load_env_cfg("Mjlab-SkateSplit-Steer-Pupper-v3", play=True)
  twist = cfg.commands["twist"]
  assert isinstance(twist, pl.SteerVelocityCommandCfg)
  pl.fix_yaw_rate(cfg, 0.0)
  assert twist.rel_straight_envs == 1.0
  assert twist.heading_hold_kp is not None


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


def test_fix_yaw_rate_rejects_only_values_outside_the_steer_command_clip():
  # Partial turns are trained now: every value inside the 0.6 clip is accepted.
  for yaw_rate in (0.0, 0.05, 0.1, 0.2, -0.3, 0.35, 0.4, -0.6, 0.6):
    cfg = load_env_cfg("Mjlab-SkateSplit-Steer-Pupper-v3", play=True)
    pl.fix_yaw_rate(cfg, yaw_rate)
  for yaw_rate in (0.61, -0.7, 1.0):
    cfg = load_env_cfg("Mjlab-SkateSplit-Steer-Pupper-v3", play=True)
    with pytest.raises(ValueError, match="outside the command clip"):
      pl.fix_yaw_rate(cfg, yaw_rate)
