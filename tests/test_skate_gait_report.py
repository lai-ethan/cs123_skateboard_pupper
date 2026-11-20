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
  assert gr.GAIT_LIMITS == (
    ("peak_joint_speed", "max", 12.0),
    ("saturated_fraction", "max", 0.05),
    ("min_foot_duty", "min", 0.4),
    ("max_stride_rate", "max", 2.0),
    ("board_speed_ratio", "min", 0.5),
    ("board_speed_ratio", "max", 1.5),
  )


_LIMITED = {row[0] for row in gr.GAIT_LIMITS}


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
  assert set(metrics) == _LIMITED | {"max_joint_speed"}
  assert metrics["peak_joint_speed"] == 2.0
  assert metrics["max_joint_speed"] == 2.0
  assert metrics["saturated_fraction"] == 0.0
  assert metrics["min_foot_duty"] == pytest.approx(0.8)
  assert metrics["max_stride_rate"] == pytest.approx(1.0, abs=0.11)
  assert metrics["board_speed_ratio"] == pytest.approx(0.95)
  rows = gr.gait_verdict([metrics])
  assert [row[:4] for row in rows] == [
    (name, metrics[name], kind, limit) for name, kind, limit in gr.GAIT_LIMITS
  ]
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
  rows = {(row[0], row[2]): row for row in gr.gait_verdict([metrics])}
  for name, kind, _ in gr.GAIT_LIMITS:
    # A board at a quarter of the command fails the minimum, not the maximum.
    expect_pass = (name, kind) == ("board_speed_ratio", "max")
    assert rows[name, kind][4] is expect_pass, name
  assert rows["peak_joint_speed", "max"][1:4] == (39.0, "max", 12.0)
  assert metrics["max_joint_speed"] == 39.0
  assert rows["min_foot_duty", "min"][2] == "min"


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
  failed = [row[:3] for row in gr.gait_verdict([metrics]) if not row[4]]
  assert failed == [("board_speed_ratio", 0.0, "min")]


def test_board_faster_than_the_command_fails():
  joint_vel, torque, deck, ground = _calm_gait()
  metrics = gr.gait_metrics(
    joint_vel,
    torque,
    deck,
    ground,
    DT,
    EFFORT,
    board_speed=np.full(T, 0.8),
    command_speed=0.4,
  )
  failed = [row[:3] for row in gr.gait_verdict([metrics]) if not row[4]]
  assert failed == [("board_speed_ratio", pytest.approx(2.0), "max")]


def test_max_joint_speed_reports_a_rare_spike():
  joint_vel, torque, deck, ground = _calm_gait()
  # 20 rad/s on 2 of 500 steps: under the 99th percentile, so the verdict
  # passes, but the true maximum shows it.
  joint_vel[[100, 300], 3] = 20.0
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
  assert metrics["peak_joint_speed"] == pytest.approx(2.0)
  assert metrics["max_joint_speed"] == 20.0
  assert all(row[4] for row in gr.gait_verdict([metrics]))


def test_verdict_judges_the_worst_env():
  good = {
    "peak_joint_speed": 2.0,
    "saturated_fraction": 0.0,
    "min_foot_duty": 0.8,
    "max_stride_rate": 1.0,
    "board_speed_ratio": 1.0,
  }
  # Env 1 is too slow, env 2 too fast, env 3 has fast legs.
  per_env = [
    good,
    {**good, "board_speed_ratio": 0.3},
    {**good, "board_speed_ratio": 1.7},
    {**good, "peak_joint_speed": 13.0},
  ]
  rows = {(row[0], row[2]): row for row in gr.gait_verdict(per_env)}
  assert rows["board_speed_ratio", "min"][1:] == (0.3, "min", 0.5, False, 1)
  assert rows["board_speed_ratio", "max"][1:] == (1.7, "max", 1.5, False, 2)
  assert rows["peak_joint_speed", "max"][1:] == (13.0, "max", 12.0, False, 3)
  assert rows["min_foot_duty", "min"][1:] == (0.8, "min", 0.4, True, 0)
  # One env alone passes.
  assert all(row[4] for row in gr.gait_verdict([good]))


@pytest.mark.parametrize("speed", [[], ["--speed", "0"], ["--speed", "-0.2"]])
def test_ride_task_needs_a_positive_speed(speed, capsys):
  argv = ["Mjlab-SkateSplit-Ride-Pupper-v3", "model_1.pt", *speed]
  with pytest.raises(SystemExit) as exit_info:
    gr.main(argv)
  assert exit_info.value.code == 2
  assert "--speed" in capsys.readouterr().err


def test_deck_height_and_envs_are_checked(capsys):
  for extra in (["--deck-height", "0"], ["--envs", "0"]):
    with pytest.raises(SystemExit):
      gr.main(["Mjlab-Skate-Balance-Pupper-v3", "model_1.pt", *extra])
    capsys.readouterr()


def test_deck_offset_is_from_the_measured_deck():
  assert gr.deck_offset(0.103, nominal=0.103) == 0.0
  assert gr.deck_offset(0.073, nominal=0.103) == pytest.approx(-0.03)


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
  rows = {row[0]: row for row in gr.gait_verdict([metrics])}
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
  rows = {row[0]: row for row in gr.gait_verdict([metrics])}
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
  assert all(row[4] for row in gr.gait_verdict([metrics]))


def test_split_pushing_foot_in_the_air_fails_duty_and_rate():
  # The front right foot taps the floor about 3 times a second and is in the air
  # in between: with no deck to count, its duty is tiny.
  ground = np.ones((T, 2))
  ground[:, 0] = (np.arange(T) % 16 == 0).astype(float)
  metrics = _split_metrics(ground)
  rows = {row[0]: row for row in gr.gait_verdict([metrics])}
  assert not rows["min_foot_duty"][4]
  assert not rows["max_stride_rate"][4]


def test_floor_share_for_split_names_the_right_feet():
  ground = np.zeros((T, 2))
  ground[:, 1] = 1.0
  share = gr.floor_share(ground, ("FR", "BR"))
  assert share == {"FR": 0.0, "BR": 1.0}


def test_steer_task_needs_a_positive_speed(capsys):
  with pytest.raises(SystemExit) as exit_info:
    gr.main(["Mjlab-SkateSplit-Steer-Pupper-v3", "model_1.pt"])
  assert exit_info.value.code == 2
  assert "--speed" in capsys.readouterr().err
  assert gr.needs_speed("Mjlab-SkateSplit-Ride-Pupper-v3")
  assert not gr.needs_speed("Mjlab-SkateSplit-Balance-Pupper-v3")


def test_yaw_rate_ratio_compares_the_board_to_the_command():
  mean, ratio = gr.yaw_rate_ratio(np.full((50, 4), 0.1), 0.2)
  assert mean == pytest.approx(0.1) and ratio == pytest.approx(0.5)
  mean, ratio = gr.yaw_rate_ratio(np.array([-0.3, -0.1]), -0.2)
  assert mean == pytest.approx(-0.2) and ratio == pytest.approx(1.0)


def test_board_yaw_target_covers_every_steer_carve():
  steer = "Mjlab-SkateSplit-Steer-Pupper-v3"

  def target(task, yaw_rate):
    return gr.board_yaw_target(task, yaw_rate, 0.1, 0.1)

  # Partial and full carves, either sign, ask for a tenth of themselves.
  for yaw_rate in (0.15, 0.2, -0.3, 0.5, -0.6):
    assert target(steer, yaw_rate) == pytest.approx(0.1 * yaw_rate), yaw_rate
  # 0 and anything up to the threshold are the hold's: no target.
  for yaw_rate in (0.0, 0.05, -0.1, 0.1):
    assert target(steer, yaw_rate) is None, yaw_rate
  # The other tasks ask for the command itself.
  assert target("Mjlab-SkateSplit-Ride-Pupper-v3", 0.2) == 0.2
  assert target("Mjlab-SkateSplit-Ride-Pupper-v3", 0.0) is None


def test_fastest_creep_names_the_env_and_its_way():
  # Env 0 holds still, env 1 creeps towards the nose at 0.5 cm/s, env 2 towards
  # the tail at 0.3 cm/s.
  t = np.arange(100)[:, None] * DT
  slide = t * np.array([0.0, -0.005, 0.003])
  done = np.zeros_like(slide, dtype=bool)
  rate, env = gr.fastest_creep(slide, done, DT)
  assert env == 1 and rate == pytest.approx(-0.005)


def test_fastest_creep_leaves_out_the_reset_jump():
  # Env 0 creeps at 0.2 cm/s and resets 5 cm back at step 50; env 1 is still.
  t = np.arange(100) * DT
  slide = np.stack([0.002 * t, np.zeros(100)], axis=1)
  slide[50:, 0] -= 0.05
  done = np.zeros_like(slide, dtype=bool)
  done[50, 0] = True
  rate, env = gr.fastest_creep(slide, done, DT)
  assert env == 0 and rate == pytest.approx(0.002)
  # Counted in, the jump would swamp the creep.
  rate, _ = gr.fastest_creep(slide, np.zeros_like(done), DT)
  assert rate < 0


def test_fastest_creep_of_an_env_that_always_resets_is_0():
  slide = np.array([[0.0], [0.05], [-0.05]])
  done = np.ones_like(slide, dtype=bool)
  assert gr.fastest_creep(slide, done, DT) == (0.0, 0)
