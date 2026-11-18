"""Physics checks for the mini cruiser skateboard asset (plain MuJoCo, CPU)."""

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
  DECK_GEOM_NAME,
  DECK_ROLL_JOINT_NAME,
  SkateboardParams,
  deck_half_width_at,
  get_skateboard_cfg,
  get_skateboard_spec,
)
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import PUPPER_MUJOCO_OPTIONS
from mjlab.entity import Entity
from mjlab.sim.sim import MujocoCfg
from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts

WHEEL_SPIN_JOINTS = (
  "wheel_fl_spin",
  "wheel_fr_spin",
  "wheel_rl_spin",
  "wheel_rr_spin",
)


def _board_on_floor(
  spec: mujoco.MjSpec | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
  """The board alone on a floor, under Pupper's solver settings, not yet stepped."""
  spec = spec or get_skateboard_spec()
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05)
  )
  model = spec.compile()
  MujocoCfg(**PUPPER_MUJOCO_OPTIONS).apply(model)
  return model, mujoco.MjData(model)


@pytest.fixture
def board() -> tuple[mujoco.MjModel, mujoco.MjData]:
  """The board alone on a floor, under Pupper's solver settings, settled."""
  model, data = _board_on_floor()
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
  yaw = p.steer_gain * p.roll_limit + 0.05  # Margin above the steer coupling.
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


##
# The deck outline traced from the team's board (assets/board_geometry.json).
##

_BALL_RADIUS = 0.005


def _deck_vertices(model: mujoco.MjModel) -> np.ndarray:
  """The deck's collision hull vertices in the deck body's frame (x = nose)."""
  geom = model.geom(DECK_GEOM_NAME)
  mesh = geom.dataid[0]
  # mesh_graph: vertex count, face count, then each hull vertex's edge address
  # and its index into the mesh's vertices.
  adr = model.mesh_graphadr[mesh]
  count = model.mesh_graph[adr]
  hull = model.mesh_graph[adr + 2 + count : adr + 2 + 2 * count]
  vert = model.mesh_vert[model.mesh_vertadr[mesh] + hull]
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, geom.quat)
  return vert @ rot.reshape(3, 3).T + geom.pos


def _hull_half_width(vert: np.ndarray, x: float) -> float:
  """Largest |y| of the hull's top-view outline on the line at ``x``."""
  outline = vert[ConvexHull(vert[:, :2]).vertices, :2]
  a, b = outline, np.roll(outline, -1, axis=0)
  cross = (a[:, 0] - x) * (b[:, 0] - x) <= 0
  a, b = a[cross], b[cross]
  t = (x - a[:, 0]) / np.where(b[:, 0] == a[:, 0], 1.0, b[:, 0] - a[:, 0])
  return float(np.abs(a[:, 1] + t * (b[:, 1] - a[:, 1])).max())


def _with_ball(spec: mujoco.MjSpec, x: float, y: float, z: float) -> mujoco.MjSpec:
  """Add a free 10 g ball centred at (x, y, z) to the board spec."""
  ball = spec.worldbody.add_body(name="ball", pos=(x, y, z))
  ball.add_freejoint(name="ball_free")
  ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(_BALL_RADIUS, 0, 0), mass=0.01)
  return spec


def _ball_rest_height(
  y: float, params: SkateboardParams | None = None, deck_offset: float = 0.0
) -> float:
  """Drop the ball just above the deck top at x = -0.114 m; its height after 1 s.

  ``deck_offset`` moves the deck geom in z as the ``deck_height`` event does.
  Like mujoco_warp, the CPU model then skips the same-frame shortcuts, which
  would ignore the moved geom_pos.
  """
  p = params or SkateboardParams()
  top = p.deck_top_height + deck_offset
  spec = _with_ball(get_skateboard_spec(p), -0.114, y, top + _BALL_RADIUS + 0.003)
  model, data = _board_on_floor(spec)
  disable_model_sameframe_shortcuts(model)
  model.geom_pos[model.geom(DECK_GEOM_NAME).id, 2] += deck_offset
  _step(model, data, 1.0)
  return float(data.body("ball").xpos[2])


def test_deck_half_width_follows_the_traced_stations() -> None:
  # Deck centre is 292.1 mm from the nose, between the 279.4 mm (156.9) and
  # 304.8 mm (155.7) stations.
  assert deck_half_width_at(0.0) == pytest.approx(0.1563 / 2, abs=0.001)
  for x in (0.2922, -0.2922, 0.4, -0.4):
    assert deck_half_width_at(x) == 0.0
  # 1 inch either side of the 7 and 16 inch stations from the nose: the nose
  # half is the wider one.
  assert 2 * deck_half_width_at(0.114) == pytest.approx(0.1486, abs=0.0015)
  assert 2 * deck_half_width_at(-0.114) == pytest.approx(0.1370, abs=0.0015)


def test_deck_half_width_of_the_box_is_flat() -> None:
  p = SkateboardParams(deck_shape="box")
  for x in (-0.25, 0.0, 0.25):
    assert deck_half_width_at(x, p) == p.deck_width / 2
  assert deck_half_width_at(p.deck_length / 2 + 0.001, p) == 0.0


def test_deck_mesh_matches_the_outline_with_the_nose_forward() -> None:
  model = get_skateboard_spec().compile()
  assert model.geom(DECK_GEOM_NAME).type[0] == mujoco.mjtGeom.mjGEOM_MESH
  vert = _deck_vertices(model)
  # Centred on the deck body, tip to tip, after MuJoCo's recentring.
  np.testing.assert_allclose(vert[:, 0].min(), -0.2921, atol=0.001)
  np.testing.assert_allclose(vert[:, 0].max(), 0.2921, atol=0.001)
  np.testing.assert_allclose(vert[:, 1].min(), -vert[:, 1].max(), atol=0.001)
  for x in (-0.24, -0.18, -0.114, 0.0, 0.114, 0.18, 0.24):
    half = _hull_half_width(vert, x)
    assert half == pytest.approx(deck_half_width_at(x), abs=0.002), x
  # Nose forward: the deck is wider 0.114 m ahead of centre than behind it.
  assert _hull_half_width(vert, 0.114) - _hull_half_width(vert, -0.114) > 0.004


def test_deck_mesh_keeps_the_box_frame(board) -> None:
  # What base_near_deck, reset_tripod_layout and the tools assume: the deck
  # geom's world z is the deck body's, its top face is that plus half the
  # thickness, and the top is at deck_top_height with the board on its wheels.
  model, data = board
  p = SkateboardParams()
  geom = model.geom(DECK_GEOM_NAME).id
  assert model.geom_pos[geom, 2] == pytest.approx(0.0, abs=1e-9)
  assert data.geom_xpos[geom, 2] == pytest.approx(data.body("deck").xpos[2], abs=1e-6)
  vert = _deck_vertices(model)
  np.testing.assert_allclose(vert[:, 2].max(), p.deck_thickness / 2, atol=1e-6)
  np.testing.assert_allclose(vert[:, 2].min(), -p.deck_thickness / 2, atol=1e-6)
  top = data.geom_xpos[geom, 2] + p.deck_thickness / 2
  assert top == pytest.approx(p.deck_top_height, abs=0.003)
  assert data.site("deck_site").xpos[2] == pytest.approx(top, abs=1e-6)
  assert model.body("deck").mass[0] == pytest.approx(0.6 * p.mass)


def test_ball_rests_on_the_deck_inside_the_outline() -> None:
  top = SkateboardParams().deck_top_height
  assert _ball_rest_height(0.060) == pytest.approx(top + _BALL_RADIUS, abs=0.005)


def test_ball_misses_the_deck_past_the_outline() -> None:
  # 0.076 m out at x = -0.114 m is past the traced edge (0.0685 m) but on the
  # old 0.161 m box.
  top = SkateboardParams().deck_top_height
  assert _ball_rest_height(0.076) < top - 0.02
  old_box = SkateboardParams(deck_shape="box", deck_width=0.161)
  assert _ball_rest_height(0.076, old_box) == pytest.approx(
    top + _BALL_RADIUS, abs=0.005
  )


def test_moving_the_deck_geom_moves_the_collision_surface() -> None:
  # The deck_height event adds to the deck geom's z; the ball lands on the
  # moved surface.
  top = SkateboardParams().deck_top_height
  for offset in (-0.018, 0.010):
    rest = _ball_rest_height(0.0, deck_offset=offset)
    assert rest == pytest.approx(top + offset + _BALL_RADIUS, abs=0.005), offset


def test_box_deck_is_the_old_box() -> None:
  p = SkateboardParams(deck_shape="box", deck_width=0.161)
  model = get_skateboard_spec(p).compile()
  geom = model.geom(DECK_GEOM_NAME)
  assert geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX
  np.testing.assert_allclose(geom.size, (0.292, 0.0805, 0.015), atol=1e-9)
  np.testing.assert_allclose(geom.pos, 0.0, atol=1e-12)
  assert model.nmesh == 0
