"""Tests for skateboard/tools/stance_audit.py (no simulator needed)."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
OVERRIDE_KEYS = {
  "deck",
  "truck",
  "mu",
  "kp",
  "effort",
  "board_x",
  "board_y",
  "board_yaw",
  "robot_yaw",
  "joint_offset",
  "push_y",
  "board_kick",
  "zero_action",
}


def _load_stance_audit():
  path = REPO / "skateboard" / "tools" / "stance_audit.py"
  spec = importlib.util.spec_from_file_location("stance_audit", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


sa = _load_stance_audit()


def test_support_margin_unit_square():
  assert sa.support_margin(SQUARE, (0.5, 0.5)) == pytest.approx(0.5)
  assert sa.support_margin(SQUARE, (0.5, 0.1)) == pytest.approx(0.1)
  assert sa.support_margin(SQUARE, (0.5, -0.2)) == pytest.approx(-0.2)


def test_support_margin_ignores_foot_order():
  shuffled = [SQUARE[2], SQUARE[0], SQUARE[3], SQUARE[1]]
  for point in ((0.5, 0.5), (0.5, 0.1), (0.5, -0.2), (1.3, 0.7)):
    assert sa.support_margin(shuffled, point) == pytest.approx(
      sa.support_margin(SQUARE, point)
    )


def test_lifted_foot_margins():
  margins = sa.lifted_foot_margins(SQUARE, (0.4, 0.4), ["a", "b", "c", "d"])
  assert set(margins) == {"a", "b", "c", "d"}
  assert margins["a"] == pytest.approx(-0.2 / 2**0.5, abs=1e-3)
  assert margins["c"] == pytest.approx(0.2 / 2**0.5, abs=1e-3)


def test_stance_limits():
  assert sa.STANCE_LIMITS == {
    "min_deck_contact": 0.9,
    "max_torque_use": 0.6,
    "margin": 0.03,
  }


def test_condition_verdict():
  good = {"resets": 0, "min_deck_contact": 1.0, "max_torque_use": 0.2, "margin": 0.06}
  assert sa.condition_verdict(good)
  for key, value in (
    ("resets", 1),
    ("min_deck_contact", 0.5),
    ("max_torque_use", 0.8),
    ("margin", 0.01),
  ):
    assert not sa.condition_verdict({**good, key: value}), key


def test_conditions_use_known_overrides():
  assert set(sa.OVERRIDE_DEFAULTS) == OVERRIDE_KEYS
  assert list(sa.CONDITIONS)[:2] == ["baseline", "zero_action"]
  assert sa.CONDITIONS["baseline"] == {}
  assert len(sa.CONDITIONS) == 21
  for name, overrides in sa.CONDITIONS.items():
    assert set(overrides) <= OVERRIDE_KEYS, name


def test_deck_and_sideways_conditions_fit_the_centred_stance():
  # Single-change conditions only (worst_combo also moves the deck).
  single = [c for c in sa.CONDITIONS.values() if len(c) == 1]
  deck = sorted(c["deck"] for c in single if "deck" in c)
  assert deck == [-0.035, -0.02, 0.01, 0.02, 0.03]
  board_y = sorted(c["board_y"] for c in single if "board_y" in c)
  assert board_y == [-0.02, -0.01, 0.01, 0.02]


def test_lift_feet_cover_all_four_feet_in_sensor_order():
  assert list(sa.LIFT_FEET) == ["lift_fr", "lift_fl", "lift_br", "lift_bl"]
  assert list(sa.LIFT_FEET.values()) == [
    "leg_front_r_3_foot_site",
    "leg_front_l_3_foot_site",
    "leg_back_r_3_foot_site",
    "leg_back_l_3_foot_site",
  ]


def test_unknown_condition_is_an_error():
  with pytest.raises(SystemExit):
    sa.main(["--conditions", "baseline,not_a_condition"])
