"""Tests for skateboard/tools/gait_report.py (no simulator needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
T, DT, EFFORT = 500, 0.02, 3.0


def _load_gait_report():
  path = REPO / "skateboard" / "tools" / "gait_report.py"
  spec = importlib.util.spec_from_file_location("gait_report", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


gr = _load_gait_report()


def test_gait_limits():
  assert gr.GAIT_LIMITS == {
    "peak_joint_speed": ("max", 8.0),
    "saturated_fraction": ("max", 0.05),
    "min_foot_duty": ("min", 0.4),
    "max_stride_rate": ("max", 2.0),
    "board_speed_ratio": ("min", 0.5),
  }


def _calm_gait():
  joint_vel = np.full((T, 12), 2.0)
  torque = np.full((T, 12), 1.0)
  on = (np.arange(T) % 50 < 40).astype(float)
  contact = np.stack([on, on], axis=1)
  return joint_vel, torque, contact


def test_calm_gait_passes():
  joint_vel, torque, contact = _calm_gait()
  metrics = gr.gait_metrics(
    joint_vel,
    torque,
    contact,
    DT,
    EFFORT,
    board_speed=np.full(T, 0.38),
    command_speed=0.4,
  )
  assert set(metrics) == set(gr.GAIT_LIMITS)
  assert metrics["peak_joint_speed"] == 2.0
  assert metrics["saturated_fraction"] == 0.0
  assert metrics["min_foot_duty"] == pytest.approx(0.8)
  assert metrics["max_stride_rate"] == pytest.approx(1.0, abs=0.11)
  assert metrics["board_speed_ratio"] == pytest.approx(0.95)
  rows = gr.gait_verdict(metrics)
  assert [row[0] for row in rows] == list(gr.GAIT_LIMITS)
  assert all(row[4] for row in rows)


def test_windmill_gait_fails():
  steps = np.arange(T)
  joint_vel = np.zeros((T, 12))
  # 39 rad/s on 5% of steps of one joint: above its own 99th percentile.
  joint_vel[steps % 20 == 0, 0] = 39.0
  torque = np.zeros((T, 12))
  torque[steps % 2 == 1, 1] = 3.0
  contact = np.ones((T, 2))
  contact[:, 0] = (steps % 16 == 0).astype(float)
  metrics = gr.gait_metrics(
    joint_vel,
    torque,
    contact,
    DT,
    EFFORT,
    board_speed=np.full(T, 0.1),
    command_speed=0.4,
  )
  rows = {row[0]: row for row in gr.gait_verdict(metrics)}
  for name in gr.GAIT_LIMITS:
    assert not rows[name][4], name
  assert rows["peak_joint_speed"][1:4] == (39.0, "max", 8.0)
  assert rows["min_foot_duty"][2] == "min"


def test_still_board_fails_only_on_board_speed():
  joint_vel, torque, contact = _calm_gait()
  metrics = gr.gait_metrics(
    joint_vel,
    torque,
    contact,
    DT,
    EFFORT,
    board_speed=np.zeros(T),
    command_speed=0.4,
  )
  failed = [row[0] for row in gr.gait_verdict(metrics) if not row[4]]
  assert failed == ["board_speed_ratio"]
  assert metrics["board_speed_ratio"] == 0.0


@pytest.mark.parametrize("command_speed", [None, 0.0])
def test_no_speed_command_passes_board_speed(command_speed):
  joint_vel, torque, contact = _calm_gait()
  metrics = gr.gait_metrics(
    joint_vel,
    torque,
    contact,
    DT,
    EFFORT,
    board_speed=np.zeros(T),
    command_speed=command_speed,
  )
  assert metrics["board_speed_ratio"] == 1.0
