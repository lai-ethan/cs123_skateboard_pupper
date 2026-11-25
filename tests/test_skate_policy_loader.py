"""Tests for skateboard/tools/policy_loader.py (config only, no env is built)."""

import importlib.util
from pathlib import Path

import pytest

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.tasks.pupper_skate.mdp import PupperVelocityCommandCfg
from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import DECK_HEIGHT_RANGE
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
  for yaw_rate in (0.05, 0.1, 0.2, -0.3, 0.35, 0.4, -0.6, 0.6):
    cfg = load_env_cfg("Mjlab-SkateSplit-Steer-Pupper-v3", play=True)
    pl.fix_yaw_rate(cfg, yaw_rate)
    twist = cfg.commands["twist"]
    assert isinstance(twist, pl.SteerVelocityCommandCfg)
    assert twist.carve_range == (abs(yaw_rate), abs(yaw_rate)), yaw_rate
  for yaw_rate in (0.61, -0.7, 1.0):
    cfg = load_env_cfg("Mjlab-SkateSplit-Steer-Pupper-v3", play=True)
    with pytest.raises(ValueError, match="outside the command clip"):
      pl.fix_yaw_rate(cfg, yaw_rate)


STEER = "Mjlab-SkateSplit-Steer-Pupper-v3"
POLICIES = REPO / "skateboard" / "policies"
STEER_V5 = POLICIES / "split_steer_v5_max0.3_policy.json"


def test_policy_hold_clip_reads_the_json_or_the_flag():
  # Steer v5 trained and ships with the 0.3 clip, not the task's 0.1 now.
  assert pl.policy_hold_clip(STEER_V5, None) == 0.3
  assert pl.policy_hold_clip(STEER_V5, 0.3) == 0.3
  with pytest.raises(ValueError, match="disagrees"):
    pl.policy_hold_clip(STEER_V5, 0.1)
  # A block with kp 0 has the hold off: nothing to apply.
  assert pl.policy_hold_clip(POLICIES / "split_balance_v3_policy.json", None) is None
  # A checkpoint carries no block: the flag, or the task's own.
  ckpt = POLICIES / "split_steer_v5_model_23999.pt"
  assert pl.policy_hold_clip(ckpt, 0.3) == 0.3
  assert pl.policy_hold_clip(ckpt, None) is None


def test_fix_hold_clip_sets_only_tasks_with_the_hold():
  cfg = load_env_cfg(STEER, play=True)
  pl.fix_hold_clip(cfg, 0.3)
  assert cfg.commands["twist"].heading_hold_clip == 0.3
  cfg = load_env_cfg("Mjlab-SkateSplit-Balance-Pupper-v3", play=True)
  before = repr(cfg.commands["twist"])
  pl.fix_hold_clip(cfg, 0.05)
  assert repr(cfg.commands["twist"]) == before


def test_fix_yaw_rate_rejects_the_hold_band_of_a_0_3_clip_policy():
  # With v5's 0.3 clip a stick of 0.1 to 0.3 looks like a hold correction.
  for yaw_rate in (0.1, 0.2, -0.3):
    cfg = load_env_cfg(STEER, play=True)
    pl.fix_hold_clip(cfg, 0.3)
    with pytest.raises(ValueError, match="hold's clip"):
      pl.fix_yaw_rate(cfg, yaw_rate)
  for yaw_rate in (0.0, 0.05, 0.4, -0.6):
    cfg = load_env_cfg(STEER, play=True)
    pl.fix_hold_clip(cfg, 0.3)
    pl.fix_yaw_rate(cfg, yaw_rate)


class _Built(Exception):
  """Raised by the fake env so load_policy stops once the config is final."""


def _capture_env_cfg(monkeypatch, policy_path, **kwargs):
  def fake_env(cfg, **_):
    raise _Built(cfg)

  monkeypatch.setattr(pl, "ManagerBasedRlEnv", fake_env)
  with pytest.raises(_Built) as built:
    pl.load_policy(STEER, policy_path, 0.3, **kwargs)
  return built.value.args[0]


def test_load_policy_replays_a_json_with_its_own_hold_clip(monkeypatch):
  cfg = _capture_env_cfg(monkeypatch, STEER_V5, yaw_rate=0.5)
  assert cfg.commands["twist"].heading_hold_clip == 0.3
  # And v5's untrained band is refused once its clip is in place.
  with pytest.raises(ValueError, match="hold's clip"):
    _capture_env_cfg(monkeypatch, STEER_V5, yaw_rate=0.2)


def test_load_policy_takes_the_hold_clip_flag_for_a_checkpoint(monkeypatch):
  ckpt = POLICIES / "split_steer_v5_model_23999.pt"
  cfg = _capture_env_cfg(monkeypatch, ckpt, hold_clip=0.3)
  assert cfg.commands["twist"].heading_hold_clip == 0.3
  # Without the flag a checkpoint runs on the task's current clip.
  cfg = _capture_env_cfg(monkeypatch, ckpt)
  assert cfg.commands["twist"].heading_hold_clip == 0.1


SPLIT_RIDE = "Mjlab-SkateSplit-Ride-Pupper-v3"
SPLIT_POLICY = POLICIES / "split_ride_v4_policy.json"


@pytest.mark.parametrize("task", [SPLIT_RIDE, STEER])
def test_load_policy_holds_the_carpet_events_at_the_hard_floor(monkeypatch, task):
  # The tools judge every policy on the conditions v6 was judged on; only the
  # ride audit's carpet rows go past them.
  def fake_env(cfg, **_):
    raise _Built(cfg)

  monkeypatch.setattr(pl, "ManagerBasedRlEnv", fake_env)
  with pytest.raises(_Built) as built:
    pl.load_policy(task, SPLIT_POLICY, 0.3)
  events = built.value.args[0].events
  assert events["wheel_drag"].params["scale_range"] == (0.5, 3.0)
  assert events["wheel_drag"].params["log_uniform"] is False
  assert events["floor_foot_friction"].params["ranges"] == (0.5, 1.2)
  assert events["deck_foot_friction"].params["ranges"] == (0.5, 1.4)
  assert events["deck_height"].params["ranges"] == DECK_HEIGHT_RANGE
  # The deck geom keeps its compiled friction (1.0), as before.
  assert "deck_friction" not in events


def test_hard_floor_leaves_the_other_skate_tasks_alone():
  for task in (
    "Mjlab-Skate-Balance-Pupper-v3",
    "Mjlab-Skate-Ride-Pupper-v3",
    "Mjlab-SkateSplit-Balance-Pupper-v3",
  ):
    cfg = load_env_cfg(task, play=True)
    before = repr(cfg.events)
    pl.hold_hard_floor(cfg)
    assert repr(cfg.events) == before, task
