"""Physics checks for the mini cruiser skateboard asset (plain MuJoCo, CPU)."""

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
  DECK_ROLL_JOINT_NAME,
  SkateboardParams,
  get_skateboard_cfg,
  get_skateboard_spec,
)
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import PUPPER_MUJOCO_OPTIONS
from mjlab.entity import Entity
from mjlab.sim.sim import MujocoCfg

WHEEL_SPIN_JOINTS = (
  "wheel_fl_spin",
  "wheel_fr_spin",
  "wheel_rl_spin",
  "wheel_rr_spin",
)


@pytest.fixture
def board() -> tuple[mujoco.MjModel, mujoco.MjData]:
  """The board alone on a floor, under Pupper's solver settings, settled."""
  spec = get_skateboard_spec()
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05)
  )
  model = spec.compile()
  MujocoCfg(**PUPPER_MUJOCO_OPTIONS).apply(model)
  data = mujoco.MjData(model)
  _step(model, data, 1.0)
  return model, data


def _step(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
  for _ in range(round(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)


def _push_forward(model: mujoco.MjModel, data: mujoco.MjData, speed: float) -> None:
  data.qvel[0] = speed
  for name in WHEEL_SPIN_JOINTS:
    data.qvel[model.joint(name).dofadr[0]] = speed / SkateboardParams().wheel_radius


def test_skateboard_entity_compiles() -> None:
  model = Entity(get_skateboard_cfg()).compile()
  assert model.nq == 7 + 7  # Freejoint, deck roll, two truck yaws, four wheels.
  assert model.nu == 0
  assert model.neq == 2
  assert model.body_subtreemass[1] == pytest.approx(SkateboardParams().mass, rel=0.02)


def test_deck_rests_at_listed_height(board) -> None:
  model, data = board
  p = SkateboardParams()
  deck_top = data.site("deck_site").xpos[2]
  assert deck_top == pytest.approx(p.deck_top_height, abs=0.003)
  assert np.abs(data.qvel).max() < 1e-2


def test_board_rolls_straight_and_coasts_to_a_stop(board) -> None:
  model, data = board
  _push_forward(model, data, 1.0)
  _step(model, data, 10.0)
  assert 0.5 < data.qpos[0] < 5.0
  assert abs(data.qpos[1]) < 0.01
  assert abs(data.qvel[0]) < 0.05


def test_board_resists_a_sideways_shove(board) -> None:
  model, data = board
  data.qvel[1] = 1.0
  _step(model, data, 1.0)
  assert abs(data.qpos[1]) < 0.05
  deck_up = data.body("deck").xmat.reshape(3, 3)[2, 2]
  assert deck_up > 0.99


def test_leaning_right_steers_right(board) -> None:
  model, data = board
  _push_forward(model, data, 1.0)
  roll_dof = model.joint(DECK_ROLL_JOINT_NAME).dofadr[0]
  for _ in range(750):  # 3 s: the board has coasted to a stop.
    data.qfrc_applied[roll_dof] = 3.0  # +x torque pushes the right edge down.
    mujoco.mj_step(model, data)
  roll = data.joint(DECK_ROLL_JOINT_NAME).qpos[0]
  full_yaw = SkateboardParams().steer_gain * roll
  front = data.joint("truck_f_yaw").qpos[0]
  rear = data.joint("truck_r_yaw").qpos[0]
  assert roll > 0.01
  # Front yaws against the lean and rear with it, by the same amount: the full
  # steer_gain * roll. (The 45% shortfall seen before was the truck yaw limit,
  # compiled as 0.0029 rad, fighting the joint equality.)
  assert rear == pytest.approx(-front, abs=1e-4)
  assert rear == pytest.approx(full_yaw, rel=0.01)
  assert data.qpos[1] < -0.02  # Measured -0.22.


def test_board_joint_limits_are_in_radians() -> None:
  p = SkateboardParams()
  model = get_skateboard_spec().compile()
  roll = model.joint(DECK_ROLL_JOINT_NAME).range
  np.testing.assert_allclose(roll, (-p.roll_limit, p.roll_limit), atol=1e-6)
  yaw = p.steer_gain * p.roll_limit
  for name in ("truck_f_yaw", "truck_r_yaw"):
    np.testing.assert_allclose(model.joint(name).range, (-yaw, yaw), atol=1e-6)


def test_load_on_the_deck_edge_leans_the_deck(board) -> None:
  model, data = board
  deck = model.body("deck").id
  edge = np.array([0.0, -SkateboardParams().deck_width / 2, 0.0])
  force, torque = np.array([0.0, 0.0, -15.0]), np.zeros(3)
  for _ in range(500):  # 2 s: 15 N straight down on the right edge.
    data.qfrc_applied[:] = 0.0
    point = data.body("deck").xpos + data.body("deck").xmat.reshape(3, 3) @ edge
    mujoco.mj_applyFT(model, data, force, torque, point, deck, data.qfrc_applied)
    mujoco.mj_step(model, data)
  # With the limit compiled as 0.0035 rad the soft limit let this reach 0.015;
  # now it is 0.042.
  assert data.joint(DECK_ROLL_JOINT_NAME).qpos[0] > 0.02
