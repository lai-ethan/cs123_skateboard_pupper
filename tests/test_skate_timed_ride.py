"""Tests for skateboard/tools/timed_ride.py (the pure functions)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).parent.parent / "skateboard" / "tools"


def _load_timed_ride():
  sys.path.insert(0, str(TOOLS))
  spec = importlib.util.spec_from_file_location("timed_ride", TOOLS / "timed_ride.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


tr = _load_timed_ride()


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
