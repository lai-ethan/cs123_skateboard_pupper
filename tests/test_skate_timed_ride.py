"""Tests for skateboard/tools/timed_ride.py (the pure functions)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.tasks.registry import load_env_cfg

TOOLS = Path(__file__).parent.parent / "skateboard" / "tools"


def _load_timed_ride():
  sys.path.insert(0, str(TOOLS))
  spec = importlib.util.spec_from_file_location("timed_ride", TOOLS / "timed_ride.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


tr = _load_timed_ride()
from policy_loader import (  # noqa: E402  (on the path once timed_ride loads)
  HARD_FLOOR_WHEEL_DRAG_SCALE_RANGE,
  hold_hard_floor,
)


def _straight(metres_per_step, steps, envs=2):
  """Boards moving along x; env 1 stands still."""
  pos = np.zeros((steps + 1, envs, 2))
  pos[:, 0, 0] = np.cumsum([0.0, *metres_per_step])
  return pos


def test_path_length_adds_up_each_step():
  pos = _straight([1.0, 1.0, 0.5], 3)
  dist = tr.path_length(pos)
  assert dist.shape == (4, 2)
  assert dist[:, 0].tolist() == [0.0, 1.0, 2.0, 2.5]
  assert dist[:, 1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_turn_so_far_unwraps_across_pi():
  yaw = np.array([[3.0], [3.1], [-3.1], [-3.0]])
  turn = tr.turn_so_far(yaw)
  assert turn[-1, 0] == pytest.approx(2 * np.pi - 6.0, abs=1e-9)


def test_summarise_splits_ride_and_coast():
  # 2 ride steps of 0.5 m, then 1 coast step of 0.1 m, at 0.5 s a step.
  pos = _straight([0.5, 0.5, 0.1], 3, envs=1)
  yaw = np.radians([[0.0], [1.0], [2.0], [2.5]])
  row = tr.summarise(pos, yaw, ride_steps=2, step_dt=0.5, keep=np.array([True]))
  assert row["envs"] == 1
  assert row["ride_m_mean"] == pytest.approx(1.0)
  assert row["ride_m_max"] == pytest.approx(1.0)
  assert row["m_per_s"] == pytest.approx(1.0)
  assert row["turn_deg_mean"] == pytest.approx(2.0)
  assert row["turn_deg_per_s"] == pytest.approx(2.0)
  assert row["coast_m_mean"] == pytest.approx(0.1)
  assert row["coast_m_max"] == pytest.approx(0.1)
  assert row["coast_turn_deg_mean"] == pytest.approx(0.5)


def test_summarise_drops_envs_that_reset():
  pos = _straight([0.5, 0.5, 0.1], 3, envs=2)
  yaw = np.zeros((4, 2))
  row = tr.summarise(pos, yaw, 2, 0.5, keep=np.array([True, False]))
  assert row["envs"] == 1
  assert row["ride_m_mean"] == pytest.approx(1.0)


def test_summarise_refuses_when_every_env_reset():
  pos = _straight([0.5, 0.5, 0.1], 3, envs=1)
  with pytest.raises(ValueError, match="every env reset"):
    tr.summarise(pos, np.zeros((4, 1)), 2, 0.5, keep=np.array([False]))


DT = 0.02


def _ride(fwd, left, yaw):
  """One env riding from the origin: forward and left along heading +x."""
  pos = np.stack([fwd, left], axis=-1)[:, None, :]
  return pos, np.asarray(yaw, dtype=float)[:, None]


def _times(ride_s):
  return np.arange(round(ride_s / DT) + 1) * DT


def test_straight_ahead_has_no_side_and_no_swing():
  t = _times(4.0)
  pos, yaw = _ride(0.3 * t, 0.0 * t, 0.0 * t)
  row = tr.summarise(pos, yaw, len(t) - 1, DT, keep=np.array([True]))
  assert row["side_m_mean"] == pytest.approx(0.0, abs=1e-12)
  assert row["side_m_abs_max"] == pytest.approx(0.0, abs=1e-12)
  assert row["side_per_m_mean"] == pytest.approx(0.0, abs=1e-12)
  assert row["yaw_swing_deg_mean"] == pytest.approx(0.0, abs=1e-9)
  assert row["side_swing_cm_mean"] == pytest.approx(0.0, abs=1e-9)


def test_side_is_measured_from_the_first_heading():
  # Heading 30 degrees at the start; the board runs 10 degrees left of it.
  t = _times(4.0)
  head, off = np.radians(30.0), np.radians(10.0)
  d = 0.3 * t
  x, y = d * np.cos(head + off), d * np.sin(head + off)
  pos, yaw = _ride(x + 1.0, y - 2.0, np.full_like(t, head))
  row = tr.summarise(pos, yaw, len(t) - 1, DT, keep=np.array([True]))
  assert row["side_per_m_mean"] == pytest.approx(np.tan(off), abs=5e-4)
  assert row["side_m_mean"] == pytest.approx(1.2 * np.sin(off))
  assert row["side_m_abs_max"] == pytest.approx(1.2 * np.sin(off))
  assert row["yaw_swing_deg_mean"] == pytest.approx(0.0, abs=1e-9)
  assert row["side_swing_cm_mean"] == pytest.approx(0.0, abs=1e-9)


def test_side_per_m_skips_boards_that_barely_move():
  t = _times(2.0)
  pos = np.zeros((len(t), 2, 2))
  pos[:, 0, 0] = 0.3 * t
  pos[:, 0, 1] = 0.03 * t
  pos[:, 1, 0] = 0.02 * t  # 0.04 m forward: under 0.05 m
  pos[:, 1, 1] = -0.02 * t
  row = tr.summarise(pos, np.zeros((len(t), 2)), len(t) - 1, DT, np.array([True] * 2))
  assert row["side_per_m_mean"] == pytest.approx(0.1)
  assert row["side_m_mean"] == pytest.approx((0.06 - 0.04) / 2)
  assert row["side_m_abs_max"] == pytest.approx(0.06)
  pos[:, 0] = pos[:, 1]
  row = tr.summarise(pos, np.zeros((len(t), 2)), len(t) - 1, DT, np.array([True] * 2))
  assert row["side_per_m_mean"] is None


def test_steady_turn_has_no_yaw_swing():
  t = _times(5.0)
  pos, yaw = _ride(0.3 * t, 0.0 * t, 0.4 * t)
  row = tr.summarise(pos, yaw, len(t) - 1, DT, keep=np.array([True]))
  assert row["yaw_swing_deg_mean"] < 0.01


def test_nose_swing_on_a_steady_turn_is_its_peak_to_peak():
  # 3 s after the skip, 4.5 cycles: the fitted line takes none of the wobble.
  t = _times(4.0)
  yaw = 0.4 * t + np.radians(2.0 * np.sin(2 * np.pi * 1.5 * t))
  pos, yaw = _ride(0.3 * t, 0.0 * t, yaw)
  row = tr.summarise(pos, yaw, len(t) - 1, DT, keep=np.array([True]))
  assert row["yaw_swing_deg_mean"] == pytest.approx(4.0, abs=0.1)


def test_side_swing_is_in_cm_after_the_first_second():
  t = _times(4.0)
  # A 1 m step sideways in the first second is not swing.
  left = 0.01 * np.sin(2 * np.pi * 1.5 * t) + np.where(t < 0.5, 0.0, 1.0)
  pos, yaw = _ride(0.3 * t, left, 0.0 * t)
  row = tr.summarise(pos, yaw, len(t) - 1, DT, keep=np.array([True]))
  assert row["side_swing_cm_mean"] == pytest.approx(2.0, abs=0.05)


def test_swings_are_null_on_a_short_ride():
  t = _times(1.4)
  pos, yaw = _ride(0.3 * t, 0.0 * t, 0.4 * t)
  row = tr.summarise(pos, yaw, len(t) - 1, DT, keep=np.array([True]))
  assert row["yaw_swing_deg_mean"] is None
  assert row["side_swing_cm_mean"] is None
  assert row["side_m_mean"] == pytest.approx(0.0)


def test_swings_ignore_the_coast():
  t = _times(6.0)
  ride_steps = round(4.0 / DT)
  yaw = np.where(t > 4.0, np.radians(30.0) * (t - 4.0), 0.0)
  left = np.where(t > 4.0, 0.5 * (t - 4.0), 0.0)
  pos, yaw = _ride(0.3 * t, left, yaw)
  row = tr.summarise(pos, yaw, ride_steps, DT, keep=np.array([True]))
  assert row["side_m_mean"] == pytest.approx(0.0)
  assert row["yaw_swing_deg_mean"] == pytest.approx(0.0)
  assert row["side_swing_cm_mean"] == pytest.approx(0.0)


def test_wheel_drag_pins_both_wheel_terms_in_every_env():
  cfg = load_env_cfg(tr.STEER, play=True)
  hold_hard_floor(cfg)
  tr.configure_drag(cfg, 6.0)
  events = list(cfg.events)
  for audit in ("audit_wheel_damping", "audit_wheel_drag"):
    term = cfg.events[audit]
    assert term.params["operation"] == "scale"
    assert term.params["ranges"] == (6.0, 6.0)
    assert term.params["asset_cfg"].joint_names == (r"wheel_.*_spin",)
    # Scales the compiled value after the task's own draw, so it wins.
    assert events.index(audit) > events.index("wheel_drag")


def test_no_wheel_drag_leaves_the_hard_floor_range():
  cfg = load_env_cfg(tr.STEER, play=True)
  hold_hard_floor(cfg)
  before = {name: dict(term.params) for name, term in cfg.events.items()}
  tr.configure_drag(cfg, None)
  assert {name: dict(term.params) for name, term in cfg.events.items()} == before
  assert cfg.events["wheel_drag"].params["scale_range"] == (
    HARD_FLOOR_WHEEL_DRAG_SCALE_RANGE
  )
