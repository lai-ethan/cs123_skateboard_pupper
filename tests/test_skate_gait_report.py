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
  # One foot pushes once a second: 0.6 s on the deck, 0.2 s in the air, 0.2 s on
  # the floor. The other three stay on the deck.
  phase = np.arange(T) % 50
  deck = np.ones((T, 4))
  deck[:, 0] = (phase < 30).astype(float)
  ground = np.zeros((T, 4))
  ground[:, 0] = (phase >= 40).astype(float)
  return joint_vel, torque, deck, ground


def test_calm_gait_passes():
  joint_vel, torque, deck, ground = _calm_gait()
  metrics = gr.gait_metrics(
    joint_vel,
    torque,
    deck,
    ground,
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
  deck = np.ones((T, 4))
  deck[:, 2] = 0.0
  # Foot 2 hardly touches anything, and lands on the floor about 3 times a second.
  ground = np.zeros((T, 4))
  ground[:, 2] = (steps % 16 == 0).astype(float)
  metrics = gr.gait_metrics(
    joint_vel,
    torque,
    deck,
    ground,
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
  joint_vel, torque, deck, ground = _calm_gait()
  metrics = gr.gait_metrics(
    joint_vel,
    torque,
    deck,
    ground,
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
  joint_vel, torque, deck, ground = _calm_gait()
  metrics = gr.gait_metrics(
    joint_vel,
    torque,
    deck,
    ground,
    DT,
    EFFORT,
    board_speed=np.zeros(T),
    command_speed=command_speed,
  )
  assert metrics["board_speed_ratio"] == 1.0


def test_foot_order_matches_the_sensors():
  assert gr.FOOT_ORDER == ("FR", "FL", "BR", "BL")


def test_floor_share_names_the_pushing_foot():
  ground = np.zeros((T, 4))
  ground[np.arange(T) % 50 >= 40, 0] = 1.0
  share = gr.floor_share(ground)
  assert list(share) == ["FR", "FL", "BR", "BL"]
  assert share["FR"] == pytest.approx(0.2)
  assert share["FL"] == share["BR"] == share["BL"] == 0.0


def _contact_metrics(deck: np.ndarray, ground: np.ndarray) -> dict[str, float]:
  return gr.gait_metrics(
    np.zeros((T, 12)),
    np.zeros((T, 12)),
    deck,
    ground,
    DT,
    EFFORT,
    board_speed=np.full(T, 0.4),
    command_speed=0.4,
  )


def test_tapping_the_floor_fails_stride_rate():
  # Foot 0 taps the floor 4 times a second and is back on the deck in between.
  phase = np.arange(T) % 12
  deck = np.ones((T, 4))
  deck[:, 0] = (phase < 9).astype(float)
  ground = np.zeros((T, 4))
  ground[:, 0] = (phase >= 10).astype(float)
  metrics = _contact_metrics(deck, ground)
  assert metrics["max_stride_rate"] == pytest.approx(4.0, abs=0.2)
  rows = {row[0]: row for row in gr.gait_verdict(metrics)}
  assert not rows["max_stride_rate"][4]
  assert rows["min_foot_duty"][4]


def test_a_foot_on_deck_then_floor_counts_as_in_contact():
  # Feet 0 and 1 move straight between deck and floor each second, never in the
  # air. Foot 1 is on the deck only 0.3 of the time, which would fail on its
  # own; counting the floor, both feet are in contact all the time.
  phase = np.arange(T) % 50
  deck = np.ones((T, 4))
  deck[:, 0] = (phase < 25).astype(float)
  deck[:, 1] = (phase < 15).astype(float)
  ground = np.zeros((T, 4))
  ground[:, 0] = (phase >= 25).astype(float)
  ground[:, 1] = (phase >= 15).astype(float)
  metrics = _contact_metrics(deck, ground)
  assert metrics["min_foot_duty"] == pytest.approx(1.0)
  assert metrics["max_stride_rate"] == pytest.approx(1.0, abs=0.11)
  rows = {row[0]: row for row in gr.gait_verdict(metrics)}
  assert rows["min_foot_duty"][4] and rows["max_stride_rate"][4]


def test_stance_comes_from_the_task_id():
  assert not gr.is_split("Mjlab-Skate-Ride-Pupper-v3")
  assert gr.is_split("Mjlab-SkateSplit-Ride-Pupper-v3")
  assert gr.floor_feet("Mjlab-Skate-Ride-Pupper-v3") == ("FR", "FL", "BR", "BL")
  # Split: the ground sensor holds the two right (pushing) feet.
  assert gr.floor_feet("Mjlab-SkateSplit-Ride-Pupper-v3") == ("FR", "BR")


def _split_metrics(ground: np.ndarray) -> dict[str, float]:
  # Split stance: no deck contacts, only the two right feet on the floor.
  return gr.gait_metrics(
    np.full((T, 12), 2.0),
    np.full((T, 12), 1.0),
    None,
    ground,
    DT,
    EFFORT,
    board_speed=np.full(T, 0.38),
    command_speed=0.4,
  )


def test_split_calm_push_passes():
  on = (np.arange(T) % 50 < 40).astype(float)
  metrics = _split_metrics(np.stack([on, on], axis=1))
  assert metrics["min_foot_duty"] == pytest.approx(0.8)
  assert metrics["max_stride_rate"] == pytest.approx(1.0, abs=0.11)
  assert all(row[4] for row in gr.gait_verdict(metrics))


def test_split_pushing_foot_in_the_air_fails_duty_and_rate():
  # The front right foot taps the floor about 3 times a second and is in the air
  # in between: with no deck to count, its duty is tiny.
  ground = np.ones((T, 2))
  ground[:, 0] = (np.arange(T) % 16 == 0).astype(float)
  metrics = _split_metrics(ground)
  rows = {row[0]: row for row in gr.gait_verdict(metrics)}
  assert not rows["min_foot_duty"][4]
  assert not rows["max_stride_rate"][4]


def test_floor_share_for_split_names_the_right_feet():
  ground = np.zeros((T, 2))
  ground[:, 1] = 1.0
  share = gr.floor_share(ground, ("FR", "BR"))
  assert share == {"FR": 0.0, "BR": 1.0}
