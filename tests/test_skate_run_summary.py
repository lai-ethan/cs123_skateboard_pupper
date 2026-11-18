"""Tests for skateboard/tools/run_summary.py and the skate notebook."""

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

REPO = Path(__file__).parent.parent
BALANCE = "Mjlab-Skate-Balance-Pupper-v3"
RIDE = "Mjlab-Skate-Ride-Pupper-v3"
SPLIT_BALANCE = "Mjlab-SkateSplit-Balance-Pupper-v3"
SPLIT_RIDE = "Mjlab-SkateSplit-Ride-Pupper-v3"


def _load_run_summary():
  path = REPO / "skateboard" / "tools" / "run_summary.py"
  spec = importlib.util.spec_from_file_location("run_summary", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


rs = _load_run_summary()


def _write_run(run_dir: Path, scalars: dict[str, list[float]]) -> None:
  writer = SummaryWriter(log_dir=str(run_dir))
  for tag, values in scalars.items():
    for step, value in enumerate(values):
      writer.add_scalar(tag, value, step)
  writer.close()


def test_read_scalars_averages_last_points(tmp_path):
  run_dir = tmp_path / "exp" / "run1"
  _write_run(run_dir, {"Train/mean_reward": [float(i) for i in range(30)]})
  scalars = rs.read_scalars(run_dir, last_n=10)
  assert scalars["Train/mean_reward"] == pytest.approx(24.5)


def test_latest_run_dir_picks_newest(tmp_path):
  old = tmp_path / "exp" / "old"
  new = tmp_path / "exp" / "new"
  old.mkdir(parents=True)
  new.mkdir(parents=True)
  os.utime(old, (1_000, 1_000))
  os.utime(new, (2_000, 2_000))
  assert rs.latest_run_dir(tmp_path) == new


def test_latest_run_dir_raises_when_empty(tmp_path):
  with pytest.raises(FileNotFoundError, match="No runs"):
    rs.latest_run_dir(tmp_path)


TORQUE = "Episode_Reward/torque_saturation"


def test_verdict_passes_and_fails():
  length, deck = "Train/mean_episode_length", "Episode_Reward/feet_on_deck"
  passing = rs.verdict(BALANCE, {length: 480.0, deck: 3.8, TORQUE: -0.01})
  assert [row[0] for row in passing] == [length, deck, TORQUE]
  assert all(row[3] for row in passing)

  failing = rs.verdict(BALANCE, {length: 480.0, deck: 1.9, TORQUE: -0.01})
  assert failing[0][3] and not failing[1][3]
  assert failing[1][1:3] == (1.9, 3.6)

  missing = rs.verdict(BALANCE, {length: 480.0, TORQUE: -0.01})
  assert missing[1][0] == deck and not missing[1][3]
  assert math.isnan(missing[1][1])


def test_split_balance_needs_two_feet_on_deck():
  length, deck = "Train/mean_episode_length", "Episode_Reward/feet_on_deck"
  passing = rs.verdict(SPLIT_BALANCE, {length: 480.0, deck: 1.9, TORQUE: -0.01})
  assert [row[0] for row in passing] == [length, deck, TORQUE]
  assert all(row[3] for row in passing)

  failing = rs.verdict(SPLIT_BALANCE, {length: 480.0, deck: 1.0, TORQUE: -0.01})
  assert failing[1][1:3] == (1.0, 1.8) and not failing[1][3]


@pytest.mark.parametrize("task, deck", [(BALANCE, 3.8), (SPLIT_BALANCE, 1.9)])
def test_balance_fails_when_motors_saturate(tmp_path, capsys, task, deck):
  good = {
    "Train/mean_episode_length": [480.0] * 5,
    "Episode_Reward/feet_on_deck": [deck] * 5,
  }
  for name, torque, result in (
    ("honest", -0.01, "PASS"),
    ("jammed", -0.2, "FAIL"),
  ):
    run_dir = tmp_path / "exp" / name
    _write_run(run_dir, {**good, TORQUE: [torque] * 5})
    assert rs.main([task, "--run-dir", str(run_dir)]) == 0
    assert capsys.readouterr().out.rstrip().endswith(f"RESULT: {result}")

  # Runs from before the balance torque penalty do not log it.
  old = tmp_path / "exp" / "old"
  _write_run(old, good)
  assert rs.main([task, "--run-dir", str(old)]) == 0
  out = capsys.readouterr().out
  assert out.rstrip().endswith("RESULT: FAIL")
  assert f"FAIL  {TORQUE} is missing from the log" in out


def test_split_ride_criteria_match_centred_ride():
  assert rs.PASS_CRITERIA[SPLIT_RIDE] == rs.PASS_CRITERIA[RIDE]


def test_steer_reuses_the_split_ride_criteria():
  steer = "Mjlab-SkateSplit-Steer-Pupper-v3"
  assert rs.PASS_CRITERIA[steer] == rs.PASS_CRITERIA[SPLIT_RIDE]


def test_cli_accepts_all_four_tasks(tmp_path, capsys):
  run_dir = tmp_path / "exp" / "run1"
  run_dir.mkdir(parents=True)
  for task in (BALANCE, RIDE, SPLIT_BALANCE, SPLIT_RIDE):
    assert rs.main([task, "--run-dir", str(run_dir)]) == 0
    assert capsys.readouterr().out.rstrip().endswith("RESULT: UNKNOWN")


def test_cli_prints_result_line(tmp_path, capsys):
  run_dir = tmp_path / "exp" / "run1"
  _write_run(
    run_dir,
    {
      "Train/mean_episode_length": [480.0] * 5,
      "Episode_Reward/feet_on_deck": [3.8] * 5,
      TORQUE: [-0.01] * 5,
      "Episode_Termination/time_out": [1.0] * 5,
      "Loss/value": [0.1] * 5,
    },
  )
  assert rs.main([BALANCE, "--run-dir", str(run_dir)]) == 0
  out = capsys.readouterr().out
  assert out.rstrip().endswith("RESULT: PASS")
  assert "Episode_Termination/time_out" in out
  assert "Loss/value" not in out
  assert "iterations: 5" in out


def test_ride_fails_on_joint_speed_penalty(tmp_path, capsys):
  good = {
    "Train/mean_episode_length": [480.0] * 5,
    "Episode_Reward/track_board_velocity": [2.5] * 5,
  }
  windmill = tmp_path / "exp" / "windmill"
  _write_run(
    windmill,
    {
      **good,
      "Episode_Reward/joint_speed_limit": [-0.5] * 5,
      "Episode_Reward/torque_saturation": [-0.05] * 5,
    },
  )
  assert rs.main([RIDE, "--run-dir", str(windmill)]) == 0
  assert capsys.readouterr().out.rstrip().endswith("RESULT: FAIL")

  calm = tmp_path / "exp" / "calm"
  _write_run(
    calm,
    {
      **good,
      "Episode_Reward/joint_speed_limit": [-0.01] * 5,
      "Episode_Reward/torque_saturation": [-0.05] * 5,
    },
  )
  assert rs.main([RIDE, "--run-dir", str(calm)]) == 0
  assert capsys.readouterr().out.rstrip().endswith("RESULT: PASS")


def test_cli_without_event_files_is_unknown(tmp_path, capsys):
  run_dir = tmp_path / "exp" / "run1"
  run_dir.mkdir(parents=True)
  assert rs.main([BALANCE, "--run-dir", str(run_dir)]) == 0
  out = capsys.readouterr().out
  assert "TensorBoard" in out
  assert out.rstrip().endswith("RESULT: UNKNOWN")


def test_notebook_has_no_interactive_login():
  nb = json.loads((REPO / "notebooks" / "CS123_Pupper_Skate.ipynb").read_text())
  sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
  assert not any("wandb.login()" in s for s in sources)
  assert any("run_summary.py" in s for s in sources)
  train = [s for s in sources if "launch_training(" in s]
  assert len(train) == 1 and "agent_cfg.logger = LOGGER" in train[0]
