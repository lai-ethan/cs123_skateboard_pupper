"""Tests for skateboard/tools/ride_audit.py."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.tasks.registry import load_env_cfg

REPO = Path(__file__).parent.parent
TOOLS = REPO / "skateboard" / "tools"
SPLIT_RIDE = "Mjlab-SkateSplit-Ride-Pupper-v3"
TRIPOD_RIDE = "Mjlab-Skate-Ride-Pupper-v3"
SPLIT_POLICY = REPO / "skateboard" / "policies" / "split_ride_iznjwbka_policy.json"
ROW_NAMES = [
  "nominal_10s",
  "nominal_30s",
  "deck_grip_0.4",
  "floor_grip_0.4",
  "board_y_+2cm",
  "board_y_-2cm",
  "board_yaw_0.15",
  "shove_side_0.5",
  "shove_fwd_0.5",
  "wheel_drag_x2",
  "wheel_drag_x3",
  "truck_x0.5",
  "truck_x4",
  "deck_93mm",
  "deck_113mm",
  "kp_x0.7",
]


def _load_ride_audit():
  # run_row imports policy_loader by name, as the script does from its folder.
  if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
  spec = importlib.util.spec_from_file_location("ride_audit", TOOLS / "ride_audit.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  # Registered first: its dataclasses look their module up by name.
  sys.modules["ride_audit"] = module
  spec.loader.exec_module(module)
  return module


ra = _load_ride_audit()
GOOD = {"envs_with_reset": 0, "speed_ratio_mean": 1.0, "deck_contact_worst_env": 1.0}


def test_rows_are_the_sixteen_in_order():
  assert list(ra.ROWS) == ROW_NAMES
  assert ra.ROWS["nominal_10s"].steps == 500
  assert ra.ROWS["nominal_30s"].steps == 1500
  assert all(
    row.steps == 500 for name, row in ra.ROWS.items() if name[:9] != "nominal_3"
  )


def test_ride_limits():
  assert ra.RIDE_LIMITS == {
    "max_reset_share": 0.0625,
    "min_speed_ratio": 0.7,
    "max_speed_ratio": 1.3,
    "min_deck_contact": 0.9,
  }


def test_reset_allowance_scales_with_envs_rounded_up():
  assert ra.max_resets(16) == 1
  assert ra.max_resets(2) == 1
  assert ra.max_resets(17) == 2
  assert ra.max_resets(32) == 2
  assert ra.max_resets(64) == 4


def test_verdict_at_each_limit():
  assert ra.row_verdict(GOOD, 16)
  # On the limit passes; just past it fails.
  for key, edge, past in (
    ("envs_with_reset", 1, 2),
    ("speed_ratio_mean", 0.7, 0.69),
    ("speed_ratio_mean", 1.3, 1.31),
    ("deck_contact_worst_env", 0.9, 0.89),
  ):
    assert ra.row_verdict({**GOOD, key: edge}, 16), (key, edge)
    assert not ra.row_verdict({**GOOD, key: past}, 16), (key, past)
  assert ra.row_verdict({**GOOD, "envs_with_reset": 2}, 32)


def test_split_ride_feet_come_from_its_sensors():
  sides = ra.foot_sides(load_env_cfg(SPLIT_RIDE, play=True))
  assert sides.deck == ("leg_front_l_3_collision", "leg_back_l_3_collision")
  assert sides.floor == ("leg_front_r_3_collision", "leg_back_r_3_collision")
  assert sides.deck_slots == (0, 1)


def test_tripod_ride_has_three_deck_feet_and_one_floor_foot():
  sides = ra.foot_sides(load_env_cfg(TRIPOD_RIDE, play=True))
  assert sides.deck == (
    "leg_front_r_3_collision",
    "leg_front_l_3_collision",
    "leg_back_l_3_collision",
  )
  assert sides.floor == ("leg_back_r_3_collision",)
  assert sides.deck_slots == (0, 1, 3)


def test_every_row_configures_the_split_and_tripod_rides():
  for task in (SPLIT_RIDE, TRIPOD_RIDE):
    for name, row in ra.ROWS.items():
      cfg = load_env_cfg(task, play=True)
      ra.configure_row(cfg, row, ra.foot_sides(cfg))
      if row.kind == "deck_grip":
        grip = cfg.events["audit_deck_grip"].params
        assert grip["ranges"] == (0.4, 0.4), (task, name)
        # Runs after the task's own foot friction draw.
        assert list(cfg.events).index("audit_deck_grip") > list(cfg.events).index(
          "foot_friction"
        )


def test_board_yaw_adds_to_the_task_noise():
  cfg = load_env_cfg(SPLIT_RIDE, play=True)
  lo, hi = cfg.events["reset_board"].params["pose_range"]["yaw"]
  ra.configure_row(cfg, ra.ROWS["board_yaw_0.15"], ra.foot_sides(cfg))
  new = cfg.events["reset_board"].params["pose_range"]["yaw"]
  assert new == pytest.approx((lo + 0.15, hi + 0.15))


def test_deck_rows_pin_the_deck_top():
  cfg = load_env_cfg(SPLIT_RIDE, play=True)
  ra.configure_row(cfg, ra.ROWS["deck_93mm"], ra.foot_sides(cfg))
  offset = cfg.events["deck_height"].params["ranges"]
  assert offset == pytest.approx((-0.01, -0.01))


def test_missing_event_is_not_applicable():
  cfg = load_env_cfg(SPLIT_RIDE, play=True)
  del cfg.events["truck_stiffness"]
  with pytest.raises(ra.NotApplicable):
    ra.configure_row(cfg, ra.ROWS["truck_x4"], ra.foot_sides(cfg))


def test_unknown_row_is_an_error():
  with pytest.raises(SystemExit):
    ra.main([SPLIT_RIDE, str(SPLIT_POLICY), "--speed", "0.3", "--rows", "nope"])


@pytest.mark.slow
def test_nominal_row_smoke(tmp_path):
  # 160 steps: 60 measured after the first 100.
  out = tmp_path / "audit.json"
  argv = [SPLIT_RIDE, str(SPLIT_POLICY), "--speed", "0.3", "--num-envs", "2"]
  argv += ["--rows", "nominal_10s", "--steps", "160", "--json", str(out)]
  assert ra.main(argv) in (0, 1)
  metrics = json.loads(out.read_text())["rows"]["nominal_10s"]
  assert metrics["verdict"] in ("PASS", "FAIL")
  assert metrics["steps"] == 160
  assert len(metrics["deck_contact_mean"]) == 2
  numbers = [v for v in metrics.values() if isinstance(v, float)]
  numbers += metrics["deck_contact_mean"] + metrics["deck_contact_min_env"]
  assert numbers and all(math.isfinite(v) for v in numbers)
  assert 0.0 <= metrics["deck_contact_worst_env"] <= 1.0
