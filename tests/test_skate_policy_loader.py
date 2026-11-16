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
