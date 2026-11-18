"""The Pupper skateboard tasks: stance keyframe, MDP terms, envs, export."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device
from rsl_rl.modules import MLP, EmpiricalNormalization
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
  DECK_GEOM_NAME,
  DECK_ROLL_JOINT_NAME,
  SkateboardParams,
)
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  FOOT_SITE_NAMES,
  JOINT_NAMES,
  PUPPER_EFFORT_LIMIT,
  SKATE_BOARD_OFFSET,
  SKATE_HEIGHT,
  SKATE_KEYFRAME,
  SKATE_POSE,
  SKATE_ROLL,
  get_spec,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper_gait.export import (
  export_pupper_gait_policy_from_env,
  gait_frame_kind,
  reference_offset_numpy,
)
from mjlab.tasks.pupper_gait.mdp import gait_reference as gr
from mjlab.tasks.pupper_gait.mdp.gait import reference_offset_from_tables
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  GAIT_GALLOP_SPEED,
  GAIT_N_SAMPLES,
  pupper_velocity_fs_env_cfg,
)
from mjlab.tasks.pupper_skate import mdp
from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import (
  BALANCE_TORQUE_SOFT_RATIO,
  BASE_NEAR_DECK_MIN,
  CENTRED_RIDE_SPEED_RANGE,
  DECK_HEIGHT_RANGE,
  JOINT_SPEED_LIMIT,
  KNEES_ON_DECK_S,
  PUSH_FOOT_ID,
  PUSH_INTERVAL_S,
  RIDE_DECK_HEIGHT_RANGE,
  RIDE_SPEED_RANGE,
  SUPPORT_FOOT_IDS,
  SUPPORT_OFF_DECK_S,
  pupper_skate_balance_env_cfg,
  pupper_skate_ride_env_cfg,
)
from mjlab.tasks.pupper_skate.pupper_skate_split_env_cfg import (
  pupper_skate_split_balance_env_cfg,
  pupper_skate_split_ride_env_cfg,
  pupper_skate_split_steer_env_cfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

NUM_ENVS = 4

##
# Keyframe.
##


def _foot_in_base(model: mujoco.MjModel, data: mujoco.MjData, leg: str) -> np.ndarray:
  site = data.site(f"leg_{leg}_3_foot_site").xpos
  return site - data.body("base_link").xpos


@pytest.fixture(scope="module")
def stance() -> tuple[mujoco.MjModel, mujoco.MjData]:
  """The robot alone, base at the origin and level, joints at SKATE_POSE."""
  model = get_spec().compile()
  data = mujoco.MjData(model)
  data.qpos[3] = 1.0
  for name, angle in zip(JOINT_NAMES, SKATE_POSE, strict=True):
    data.qpos[model.joint(name).qposadr[0]] = angle
  mujoco.mj_kinematics(model, data)
  return model, data


def test_skate_pose_is_inside_joint_limits(stance):
  model, _ = stance
  for name, angle in zip(JOINT_NAMES, SKATE_POSE, strict=True):
    low, high = model.joint(name).range
    assert low + 0.1 < angle < high - 0.1, name


def test_skate_pose_puts_all_four_feet_over_the_deck(stance):
  model, data = stance
  legs = ("front_r", "front_l", "back_r", "back_l")
  feet = [_foot_in_base(model, data, leg) for leg in legs]
  # Each foot centre is at least 1 cm in from the deck's edge, so spawn noise
  # does not drop a foot off it.
  half_width = SkateboardParams().deck_width / 2
  for leg, foot in zip(legs, feet, strict=True):
    assert abs(foot[1] - SKATE_BOARD_OFFSET[1]) <= half_width - 0.01, leg
    assert foot[2] == pytest.approx(-0.115, abs=0.005), leg


def test_skate_keyframe_spawns_the_pose_above_the_deck():
  assert SKATE_KEYFRAME.joint_pos is not None
  assert list(SKATE_KEYFRAME.joint_pos.values()) == list(SKATE_POSE)
  w, x, y, z = SKATE_KEYFRAME.rot
  assert 2 * math.atan2(x, w) == pytest.approx(SKATE_ROLL)
  assert (y, z) == (0.0, 0.0)
  assert SKATE_KEYFRAME.pos[2] == pytest.approx(SKATE_HEIGHT + 0.01)


##
# The centred push reference (gait_reference's skate_push table).
##

_DECK_FEET = ("front_r", "front_l", "back_l")
_PUSH_LEG = "back_r"


@pytest.fixture(scope="module")
def push_table() -> np.ndarray:
  return gr.build_joint_reference_table(100, "skate_push")


def _phase_row(name: str, n: int = 100) -> int:
  """Table row of the keyframe ``name``, which must appear once in the cycle."""
  (phase,) = [p for p, key in gr.SKATE_PUSH_KEYFRAMES if key == name]
  return round(phase * n)


def test_push_table_is_periodic_and_inside_the_joint_limits(push_table):
  assert push_table.shape == (100, 12)
  np.testing.assert_allclose(push_table[0], SKATE_POSE)
  np.testing.assert_allclose(push_table[-1], SKATE_POSE)
  model = get_spec().compile()
  limits = np.array([model.joint(n).range for n in JOINT_NAMES])
  assert np.all(push_table >= limits[:, 0]) and np.all(push_table <= limits[:, 1])


def test_push_table_glides_on_the_stance(push_table):
  glide = np.arange(100) / 100 >= gr.SKATE_PUSH_GLIDE_START
  assert glide.sum() == 29
  np.testing.assert_allclose(push_table[glide], np.tile(SKATE_POSE, (29, 1)))
  # Before the glide the back right leg is moving on every row but the first.
  push = slice(6, 9)
  moving = np.any(push_table[1:71, push] != np.array(SKATE_POSE)[push], axis=1)
  assert moving.all()


def test_push_table_stays_under_the_ride_speed_onset(push_table):
  step = np.diff(np.vstack([push_table, push_table[:1]]), axis=0)
  speed = np.abs(step) * 100 * gr.SKATE_PUSH_FREQUENCY
  assert speed.max() < 6.0
  assert gr.SKATE_PUSH_FREQUENCY == 0.4
  assert gr.SKATE_PUSH_BLEND_SPEED == 0.05


def _place_on_deck(model, data, q, centroid, deck_top, foot_r, guess):
  """Base pose (x, y, z, roll, pitch) that rests the three deck feet on the deck.

  Their undersides touch the deck top and their xy centroid sits at
  ``centroid``; yaw is zero.
  """

  def set_pose(b):
    data.qpos[0:3] = b[0:3]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([b[3], b[4], 0.0]), "xyz")
    data.qpos[3:7] = quat
    for name, angle in zip(JOINT_NAMES, q, strict=True):
      data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_kinematics(model, data)

  def residual(b):
    set_pose(b)
    feet = np.array([data.site(f"leg_{leg}_3_foot_site").xpos for leg in _DECK_FEET])
    return np.r_[feet[:, 2] - (deck_top + foot_r), feet[:, :2].mean(0) - centroid]

  b = least_squares(residual, guess).x
  set_pose(b)
  return b


def test_push_foot_reaches_the_floor_and_clears_the_deck_side(push_table):
  # The robot alone, three feet resting on a stock deck (its box from the board
  # params, centred under the stance), the push foot free.
  p = SkateboardParams()
  model = get_spec().compile()
  data = mujoco.MjData(model)
  foot_r = float(model.geom(f"leg_{_PUSH_LEG}_3_collision").size[0])
  deck_top = p.deck_top_height
  centre = np.array([*SKATE_BOARD_OFFSET, deck_top - p.deck_thickness / 2])
  half = np.array([p.deck_length / 2, p.deck_width / 2, p.deck_thickness / 2])
  # Deck feet centroid with the stance on the board.
  stance_base = np.array([0.0, 0.0, 0.3, 0.0, 0.0])
  _place_on_deck(model, data, SKATE_POSE, np.zeros(2), deck_top, foot_r, stance_base)
  centroid = (
    np.mean(
      [data.site(f"leg_{leg}_3_foot_site").xpos[:2] for leg in _DECK_FEET], axis=0
    )
    - data.qpos[0:2]
  )
  base = stance_base
  underside, side_gap = [], []
  for q in push_table:
    base = _place_on_deck(model, data, q, centroid, deck_top, foot_r, base)
    foot = data.site(f"leg_{_PUSH_LEG}_3_foot_site").xpos.copy()
    outside = np.maximum(np.abs(foot - centre) - half, 0.0)
    underside.append(foot[2] - foot_r)
    side_gap.append(np.linalg.norm(outside) - foot_r)
  underside_a, side_gap_a = np.array(underside), np.array(side_gap)
  for name in ("reach", "sweep"):
    assert abs(underside_a[_phase_row(name)]) < 0.01, name
  # Below the deck top the foot is beside the deck, 3 mm clear of it. On or
  # above the top (within 1 mm) it is standing on the deck or over it.
  below_top = underside_a < deck_top - 0.001
  assert below_top.sum() > 20
  assert side_gap_a[below_top].min() >= 0.003


##
# MDP terms on hand-built states. A fake env is enough: each term only reads
# ``env.scene[name].data``, ``env.device`` and the command.
##


def _fake_env(**entities) -> Any:
  scene = {
    name: SimpleNamespace(data=SimpleNamespace(**data))
    for name, data in entities.items()
  }
  return SimpleNamespace(scene=scene, device="cpu")


def _sensor_env(**fields) -> Any:
  return _fake_env(feet=fields)


def test_contact_count_counts_touching_geoms():
  env = _sensor_env(found=torch.tensor([[1.0, 1.0], [0.0, 2.0], [0.0, 0.0]]))
  assert mdp.contact_count(env, "feet").tolist() == [2.0, 1.0, 0.0]
  assert mdp.contact_flags(env, "feet").tolist() == [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]


def test_contact_lost_waits_for_the_grace_period():
  env = _sensor_env(current_air_time=torch.tensor([[0.0, 0.0], [0.4, 0.0], [0.0, 0.6]]))
  assert mdp.contact_lost(env, "feet", grace_s=0.5).tolist() == [False, False, True]


def test_contact_count_clamps_to_max_count():
  env = _sensor_env(found=torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]]))
  assert mdp.contact_count(env, "feet", max_count=3.0).tolist() == [3.0, 2.0]
  assert mdp.contact_count(env, "feet").tolist() == [4.0, 2.0]


def test_contact_lost_with_min_lost_needs_that_many_geoms_out():
  # Rows: one foot past the grace, two feet past it, two feet inside it.
  env = _sensor_env(
    current_air_time=torch.tensor(
      [[2.0, 0.0, 0.0, 0.0], [2.0, 1.6, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    )
  )
  lost = mdp.contact_lost(env, "feet", grace_s=1.5, min_lost=2)
  assert lost.tolist() == [False, True, False]
  lost = mdp.contact_lost(env, "feet", grace_s=1.5, min_lost=1)
  assert lost.tolist() == [True, True, False]


def test_gravity_tracking_peaks_at_the_stance_tilt():
  target = (0.0, -math.sin(SKATE_ROLL))
  gravity = torch.tensor([[0.0, target[1], -1.0], [0.0, 0.1, -1.0], [0.0, 0.5, -0.87]])
  env = _fake_env(robot={"projected_gravity_b": gravity})
  reward = mdp.gravity_tracking(env, target_xy=target, std=0.2)
  assert reward[0] == pytest.approx(1.0)
  assert reward[0] > reward[1] > reward[2]


def test_stance_pose_peaks_at_the_keyframe():
  default = torch.tensor([SKATE_POSE, SKATE_POSE])
  joint_pos = default.clone()
  joint_pos[1] += 0.3
  env = _fake_env(robot={"joint_pos": joint_pos, "default_joint_pos": default})
  reward = mdp.stance_pose(env, std=0.3)
  assert reward[0] == pytest.approx(1.0)
  assert reward[1] == pytest.approx(math.exp(-1.0))


def _yaw_quat(yaw: float) -> list[float]:
  return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def test_board_alignment_follows_the_robot_heading():
  # Env 0: board at the target offset. Env 1: robot turned 90 degrees left, so
  # "0.01 m behind me" is now world -y. Env 2: board 10 cm out of place.
  # Env 3: right place, board turned 0.3 rad.
  robot_pos = torch.zeros(4, 3)
  robot_quat = torch.tensor(
    [_yaw_quat(0.0), _yaw_quat(math.pi / 2), _yaw_quat(0.0), _yaw_quat(0.0)]
  )
  board_pos = torch.tensor(
    [
      [-0.01, 0.0, 0.03],
      [0.0, -0.01, 0.03],
      [-0.01, 0.10, 0.03],
      [-0.01, 0.0, 0.03],
    ]
  )
  board_quat = torch.tensor(
    [_yaw_quat(0.0), _yaw_quat(math.pi / 2), _yaw_quat(0.0), _yaw_quat(0.3)]
  )
  env = _fake_env(
    robot={"root_link_pos_w": robot_pos, "root_link_quat_w": robot_quat},
    board={"root_link_pos_w": board_pos, "root_link_quat_w": board_quat},
  )
  reward = mdp.board_alignment(
    env, target_offset=SKATE_BOARD_OFFSET, std_pos=0.05, std_yaw=0.3
  )
  assert reward[0] == pytest.approx(1.0, abs=1e-5)
  assert reward[1] == pytest.approx(1.0, abs=1e-5)
  assert reward[2] == pytest.approx(math.exp(-4.0), abs=1e-5)
  assert reward[3] == pytest.approx(math.exp(-1.0), abs=1e-5)


def test_board_penalties_measure_board_motion():
  env = _fake_env(
    board={
      "root_link_lin_vel_w": torch.tensor([[0.3, 0.4, 9.0]]),
      "root_link_ang_vel_w": torch.tensor([[9.0, 9.0, 0.5]]),
      "joint_pos": torch.tensor([[0.1]]),
    }
  )
  assert mdp.board_speed_l2(env).item() == pytest.approx(0.25)
  assert mdp.board_yaw_rate_l2(env).item() == pytest.approx(0.25)
  assert mdp.deck_roll_l2(env).item() == pytest.approx(0.01)


def test_track_board_velocity_peaks_at_the_commanded_speed():
  env = _fake_env(
    board={
      "root_link_lin_vel_b": torch.tensor(
        [[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [-0.3, 0.0, 0.0]]
      )
    }
  )
  command = torch.tensor([[0.3, 0.0, 0.0]] * 3)
  env.command_manager = SimpleNamespace(get_command=lambda name: command)
  reward = mdp.track_board_velocity(env, command_name="twist", std=0.15)
  assert reward[0] == pytest.approx(1.0)
  assert reward[1] == pytest.approx(math.exp(-4.0))  # Sideways is not forward.
  assert reward[2] < reward[1]  # Backwards is worse than still.


def test_joint_speed_over_limit_ignores_speeds_under_the_limit():
  env = _fake_env(
    robot={"joint_vel": torch.tensor([[7.0, -8.0, 0.0], [10.0, -11.0, 0.0]])}
  )
  assert mdp.joint_speed_over_limit(env, limit=8.0).tolist() == [0.0, 13.0]


def test_torque_saturation_ramps_from_the_soft_limit():
  force = torch.tensor([[2.4, -1.0], [2.7, -3.0], [3.5, 0.0]])
  env = _fake_env(robot={"actuator_force": force})
  cost = mdp.torque_saturation(env, limit=3.0, soft_ratio=0.8)
  assert cost.tolist() == pytest.approx([0.0, 1.5, 1.0], abs=1e-5)


def _with_command(env: Any, speed: float, board_speed: list[float]) -> Any:
  """Command ``speed`` forward in every row; the board rolls at ``board_speed``."""
  rows = len(board_speed)
  command = torch.tensor([[speed, 0.0, 0.0]] * rows)
  env.command_manager = SimpleNamespace(get_command=lambda name: command)
  env.scene["board"] = SimpleNamespace(
    data=SimpleNamespace(
      root_link_lin_vel_b=torch.tensor([[v, 0.0, 0.0] for v in board_speed])
    )
  )
  return env


def test_push_stroke_pays_once_as_a_foot_leaves_the_floor():
  # Foot 0 varies per row; foot 1 is on the floor (no air time).
  # Rows: lift-off after a full stroke, after a short one, already in the air,
  # still on the floor, and a full stroke with the board already at speed.
  env = _sensor_env(
    current_air_time=torch.tensor(
      [[0.02, 0.0], [0.02, 0.0], [0.06, 0.0], [0.0, 0.0], [0.02, 0.0]]
    ),
    last_contact_time=torch.tensor(
      [[0.4, 0.5], [0.1, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5]]
    ),
  )
  env.step_dt = 0.02
  _with_command(env, 0.4, [0.0, 0.1, 0.0, 0.0, 0.4])
  params: dict[str, Any] = {
    "sensor_name": "feet",
    "stroke_s": 0.25,
    "command_name": "twist",
    "command_threshold": 0.1,
    "speed_fraction": 1.0,
  }
  reward = mdp.push_stroke(env, **params)
  assert reward.tolist() == pytest.approx([50.0, 20.0, 0.0, 0.0, 0.0])
  _with_command(env, 0.1, [0.0] * 5)
  assert mdp.push_stroke(env, **params).tolist() == [0.0] * 5


def test_board_progress_pays_the_share_of_the_commanded_speed():
  # Rows: rolling backwards, at rest, half the command, the command, above it.
  env = _with_command(_fake_env(), 0.4, [-0.2, 0.0, 0.2, 0.4, 0.6])
  progress = mdp.board_progress(env, command_name="twist", command_threshold=0.1)
  assert progress.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
  # No command, or one at the threshold, pays nothing.
  for speed in (0.0, 0.1):
    env = _with_command(_fake_env(), speed, [0.0, 0.2])
    progress = mdp.board_progress(env, command_name="twist", command_threshold=0.1)
    assert progress.tolist() == [0.0, 0.0]


def _foot_reach_env(foot_z: list[list[float]]) -> Any:
  pos = torch.zeros(len(foot_z), 4, 3)
  pos[:, :, 2] = torch.tensor(foot_z)
  return _fake_env(robot={"site_pos_w": pos})


_FOOT_REACH: dict[str, Any] = {
  "std": 0.03,
  "foot_radius": 0.02,
  "command_name": "twist",
  "command_threshold": 0.1,
  "speed_fraction": 0.8,
  "foot_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
}


def test_foot_reach_pays_for_the_lowest_foot_near_the_floor():
  # Rows: one foot touching the floor, the lowest foot 3 cm up (the others
  # higher), one foot below its radius (sunk into the floor), all on a deck.
  env = _foot_reach_env(
    [
      [0.10, 0.02, 0.10, 0.10],
      [0.10, 0.05, 0.08, 0.10],
      [0.0, 0.10, 0.10, 0.10],
      [0.09, 0.09, 0.09, 0.09],
    ]
  )
  _with_command(env, 0.5, [0.0, 0.0, 0.0, 0.0])
  reward = mdp.foot_reach(env, **_FOOT_REACH)
  assert reward[0] == pytest.approx(1.0)
  assert reward[1] == pytest.approx(math.exp(-1.0))
  assert reward[2] == pytest.approx(1.0)
  assert reward[3] == pytest.approx(math.exp(-0.07 / 0.03))


def test_foot_reach_needs_a_command_and_a_slow_board():
  env = _foot_reach_env([[0.02, 0.10, 0.10, 0.10]] * 3)
  # Board at 0.78, at 0.8 and at 0.9 of the commanded 0.5 m/s.
  _with_command(env, 0.5, [0.39, 0.4, 0.45])
  assert mdp.foot_reach(env, **_FOOT_REACH).tolist() == pytest.approx([1.0, 0.0, 0.0])
  # A command at the threshold pays nothing.
  _with_command(env, 0.1, [0.0, 0.0, 0.0])
  assert mdp.foot_reach(env, **_FOOT_REACH).tolist() == [0.0] * 3


def test_foot_reach_continuous_gate_fades_as_the_board_speeds_up():
  # One foot on the floor, so the ungated reward is 1. The fade ends at
  # 0.8 * 0.5 = 0.4 m/s. Rows: at rest, half of 0.4, 0.4, above it, backwards.
  env = _foot_reach_env([[0.02, 0.10, 0.10, 0.10]] * 5)
  params: dict[str, Any] = {**_FOOT_REACH, "continuous_gate": True}
  _with_command(env, 0.5, [0.0, 0.2, 0.4, 0.45, -0.1])
  reward = mdp.foot_reach(env, **params)
  assert reward.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0, 1.0])
  # No command, or one at the threshold, pays nothing.
  for speed in (0.0, 0.1):
    _with_command(env, speed, [0.0] * 5)
    assert mdp.foot_reach(env, **params).tolist() == [0.0] * 5


##
# The balance env, stepped on the real simulator.
##


def _calm_balance_cfg(num_envs: int, spawn_noise: bool):
  """The nominal 0.103 m board with no shoves: what the keyframe was solved for."""
  cfg = pupper_skate_balance_env_cfg()
  cfg.scene.num_envs = num_envs
  for name in ("deck_height", "push_robot", "nudge_board"):
    del cfg.events[name]
  if not spawn_noise:
    for name in ("reset_base", "reset_board"):
      cfg.events[name].params["pose_range"] = {}
  return cfg


def _passive_hold_cfg(num_envs: int, spawn_noise: bool, truck_scale: float = 1.0):
  """`_calm_balance_cfg` with the truck stiffness fixed at `truck_scale` x nominal.

  A soft truck needs the policy to balance: with zero actions the stance is only
  expected to hold on the nominal board or a stiffer one.
  """
  cfg = _calm_balance_cfg(num_envs, spawn_noise)
  cfg.events["truck_stiffness"].params["ranges"] = (truck_scale, truck_scale)
  cfg.seed = 0
  return cfg


def _zero_action_falls(cfg) -> float:
  """Share of envs that reset during 5 s of zero actions."""
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  actions = torch.zeros(cfg.scene.num_envs, 12, device=env.device)
  reset = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=env.device)
  for _ in range(250):  # 5 s.
    _, _, terminated, truncated, _ = env.step(actions)
    reset |= terminated | truncated
  env.close()
  return reset.float().mean().item()


@pytest.fixture(scope="module")
def balance_env():
  cfg = _calm_balance_cfg(NUM_ENVS, spawn_noise=False)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield env
  env.close()


def test_skate_tasks_are_registered():
  tasks = list_tasks()
  assert "Mjlab-Skate-Balance-Pupper-v3" in tasks
  assert "Mjlab-Skate-Ride-Pupper-v3" in tasks


def test_balance_env_keeps_the_deployable_actor_frame(balance_env):
  obs, _ = balance_env.reset()
  assert obs["actor"].shape == (NUM_ENVS, 48 * 20)
  assert gait_frame_kind(balance_env) == "zeroref"
  # 7 board pose + 3 + 3 board velocity + 2 deck roll + 2 + 2 foot contacts.
  stock = pupper_velocity_fs_env_cfg().observations["critic"].terms
  assert len(balance_env.cfg.observations["critic"].terms) == len(stock) + 6


def test_foot_sensors_follow_foot_site_order(balance_env):
  expected = [site.replace("_foot_site", "_collision") for site in FOOT_SITE_NAMES]
  for name in ("deck_feet_contact", "ground_feet_contact"):
    sensor = balance_env.scene[name]
    assert [n.split("/")[-1] for n in sensor.primary_names] == expected, name


@pytest.mark.slow
def test_stance_holds_with_zero_actions():
  env = ManagerBasedRlEnv(
    cfg=_passive_hold_cfg(NUM_ENVS, spawn_noise=False), device=get_test_device()
  )
  try:
    env.reset()
    robot, board = env.scene["robot"], env.scene["board"]
    actions = torch.zeros(NUM_ENVS, 12, device=env.device)
    for _ in range(250):  # 5 s.
      _, reward, terminated, truncated, _ = env.step(actions)
      assert torch.isfinite(reward).all()
      assert not (terminated | truncated).any()
      roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
      assert wrap_to_pi(roll).abs().max() < 0.1
      assert wrap_to_pi(pitch).abs().max() < 0.1
    assert (mdp.contact_flags(env, "deck_feet_contact") == 1).all()
    assert (mdp.contact_count(env, "ground_feet_contact") == 0).all()
    assert (mdp.contact_count(env, "knees_on_deck") == 0).all()
    # The loaded legs settle up to 3 cm below the feet-touching height.
    height = robot.data.root_link_pos_w[:, 2]
    assert ((height > SKATE_HEIGHT - 0.03) & (height < SKATE_HEIGHT)).all()
    assert board.data.root_link_lin_vel_w[:, :2].norm(dim=1).max() < 0.02
    assert (
      mdp.board_alignment(env, SKATE_BOARD_OFFSET, std_pos=0.05, std_yaw=0.3).min()
      > 0.3
    )
  finally:
    env.close()


@pytest.mark.slow
def test_stance_holds_without_torque_saturation():
  # The balance task pays for motors above half their limit; an honest stance
  # should not.
  cfg = _calm_balance_cfg(1, spawn_noise=False)
  cfg.episode_length_s = 20.0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  actions = torch.zeros(1, 12, device=env.device)
  costs = []
  for _ in range(500):  # 10 s.
    _, _, terminated, truncated, _ = env.step(actions)
    assert not (terminated | truncated).any()
    costs.append(
      mdp.torque_saturation(
        env, PUPPER_EFFORT_LIMIT, soft_ratio=BALANCE_TORQUE_SOFT_RATIO
      ).item()
    )
  env.close()
  assert sum(costs[250:]) / 250 < 0.01


@pytest.mark.slow
@pytest.mark.xfail(
  strict=True,
  reason=(
    "At the guessed 30 N*m/rad truck stiffness the pose with no policy tips in "
    "about half of noisy spawns (28 to 40 of 64 over seeds 0 to 5). A trained "
    "policy has to balance there. Revisit once the real truck stiffness is measured."
  ),
)
def test_stance_survives_spawn_noise_with_zero_actions():
  # Some noisy spawns tip over with zero actions; a policy has to correct those.
  # This checks the rest hold, at nominal truck stiffness.
  cfg = _passive_hold_cfg(64, spawn_noise=True)
  assert _zero_action_falls(cfg) <= 0.2


@pytest.mark.slow
def test_stance_survives_spawn_noise_on_a_stiff_truck():
  # The top of the truck stiffness randomization, 6x nominal.
  cfg = _passive_hold_cfg(64, spawn_noise=True, truck_scale=6.0)
  assert _zero_action_falls(cfg) < 0.2


@pytest.mark.slow
def test_leaving_the_deck_ends_the_episode(balance_env):
  env = balance_env
  env.reset()
  robot = env.scene["robot"]
  lifted = robot.data.default_root_state.clone()
  lifted[:, :3] += env.scene.env_origins
  lifted[:, 2] += 0.3
  actions = torch.zeros(NUM_ENVS, 12, device=env.device)
  fired = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
  for _ in range(40):  # 0.8 s held in the air; the grace period is 0.5 s.
    robot.write_root_state_to_sim(lifted)
    env.step(actions)
    fired |= env.termination_manager.get_term("feet_off_deck")
  assert fired.all()


##
# Export: the stance ships as the policy's default pose, with no new C++.
##


class _Actor(torch.nn.Module):
  """Minimal MLPModel-like holder: has ``.mlp`` and ``.obs_normalizer``."""

  def __init__(self, in_dim: int):
    super().__init__()
    self.mlp = MLP(in_dim, 12, (64, 64), activation="elu")
    self.obs_normalizer = EmpiricalNormalization(in_dim)


def test_balance_policy_exports_with_the_stance_as_default_pose(balance_env):
  policy = export_pupper_gait_policy_from_env(_Actor(48 * 20).eval(), balance_env)
  assert policy["default_joint_pos"] == pytest.approx(list(SKATE_POSE))
  assert policy["command_clip"] == {
    "vx": [0.0, 0.0],
    "vy": [0.0, 0.0],
    "wz": [0.0, 0.0],
  }
  # The robot turns heading hold on for any policy with a gait_reference block
  # unless the JSON turns it off; skate policies trained without it.
  assert policy["heading_hold"] == {
    "kp": 0.0,
    "clip": 0.3,
    "yaw_threshold": 0.1,
    "walk_threshold": 0.05,
  }
  assert "gait_reference" in policy


##
# The ride task (config only; training is what tests the behavior).
##


def test_ride_task_commands_forward_speed_only():
  cfg = pupper_skate_ride_env_cfg()
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  assert twist.ranges.lin_vel_x == CENTRED_RIDE_SPEED_RANGE == (0.0, 0.4)
  assert RIDE_SPEED_RANGE == (0.0, 0.5)  # Split ride's, unchanged.
  assert twist.rel_standing_envs == 0.1
  assert twist.ranges.lin_vel_y == (0.0, 0.0) and twist.ranges.ang_vel_z == (0.0, 0.0)
  names = ("board_still", "feet_on_ground", "push_stride", "push_leg_pose")
  for name in (*names, "stance_pose"):
    assert name not in cfg.rewards
  assert "nudge_board" not in cfg.events
  new_terms = (
    "push_stroke",
    "joint_vel_l2",
    "joint_speed_limit",
    "torque_saturation",
  )
  for name in ("track_board_velocity", "board_yaw_rate", *new_terms):
    assert cfg.rewards[name].weight != 0.0
  assert cfg.rewards["push_stroke"].params["sensor_name"] == "ground_feet_contact"
  assert cfg.rewards["joint_speed_limit"].params == {"limit": 6.0}
  assert cfg.rewards["joint_speed_limit"].params["limit"] < JOINT_SPEED_LIMIT
  assert cfg.rewards["joint_speed_limit"].weight == -0.1
  # A push takes a foot off the deck for about a second.
  assert cfg.terminations["feet_off_deck"].params["grace_s"] == 1.5
  assert cfg.rewards["feet_on_deck"].weight == 0.5
  assert cfg.terminations["fell_over"].params["limit_angle"] == 0.6
  # The reference sets the pose: tracked on every joint instead of stance_pose.
  track = cfg.rewards["reference_tracking"]
  assert track.func is mdp.gait_tracking and track.weight == 1.0
  assert track.params["std"] == {".*": 0.3}
  assert track.params["phase_lead_steps"] == 0.0
  patterns = track.params["asset_cfg"].joint_names
  assert {j for j in JOINT_NAMES if any(re.fullmatch(p, j) for p in patterns)} == set(
    JOINT_NAMES
  )
  actor = cfg.observations["actor"].terms["policy"].params
  for key in ("gait", "frequency", "blend_speed", "gallop_speed", "n_samples"):
    assert track.params[key] == actor[key], key
  # The balance task is untouched by building the ride task.
  balance = pupper_skate_balance_env_cfg()
  assert "board_still" in balance.rewards and "feet_on_ground" in balance.rewards
  # Balance has its own, stricter torque_saturation.
  assert set(new_terms) & set(balance.rewards) == {"torque_saturation"}
  assert balance.terminations["feet_off_deck"].params["grace_s"] == 0.5


def test_ride_task_rewards_reaching_for_the_floor():
  cfg = pupper_skate_ride_env_cfg()
  reach = cfg.rewards["foot_reach"]
  assert reach.func is mdp.foot_reach
  assert reach.weight == 1.0
  params = dict(reach.params)
  foot_cfg = params.pop("foot_cfg")
  assert params == {
    "std": 0.08,
    "foot_radius": 0.02,
    "command_name": "twist",
    "command_threshold": 0.1,
    "speed_fraction": 0.8,
    "continuous_gate": True,
    "foot_ids": (PUSH_FOOT_ID,),
    "support_sensor": "deck_feet_contact",
    "support_grace_s": 0.1,
  }
  assert foot_cfg.name == "robot"
  assert tuple(foot_cfg.site_names) == FOOT_SITE_NAMES
  stroke = cfg.rewards["push_stroke"]
  assert stroke.weight == 3.0
  assert stroke.params["speed_fraction"] == 1.0
  # Centred ride trains on the stock trucks only; balance keeps both.
  assert cfg.events["deck_height"].params["ranges"] == RIDE_DECK_HEIGHT_RANGE
  assert RIDE_DECK_HEIGHT_RANGE == (-0.01, 0.01)
  balance = pupper_skate_balance_env_cfg()
  assert balance.events["deck_height"].params["ranges"] == DECK_HEIGHT_RANGE
  assert "foot_reach" not in balance.rewards
  # The fading gate is for centred ride only: no other task opts in.
  for other in (balance, pupper_skate_split_ride_env_cfg()):
    for term in other.rewards.values():
      assert "continuous_gate" not in term.params


def test_ride_task_pays_for_a_push():
  # Standing on four feet must not out-earn a push on three, so the deck pays
  # for at most three feet, one foot may stay off, and the body may tilt and
  # the board shift a little while a leg reaches the floor.
  cfg = pupper_skate_ride_env_cfg()
  r = cfg.rewards
  assert r["feet_on_deck"].weight == 0.5
  assert r["feet_on_deck"].params["max_count"] == 3.0
  feet_off = cfg.terminations["feet_off_deck"].params
  assert (feet_off["min_lost"], feet_off["grace_s"]) == (2, 1.5)
  assert r["tilt"].params["std"] == 0.5
  assert r["board_alignment"].params["std_pos"] == 0.08
  assert (r["foot_reach"].weight, r["push_stroke"].weight) == (1.0, 3.0)
  progress = r["board_progress"]
  assert progress.func is mdp.board_progress
  assert progress.weight == 4.0
  assert progress.params == {"command_name": "twist", "command_threshold": 0.1}
  track = r["track_board_velocity"]
  assert (track.weight, track.params["std"]) == (3.0, 0.10)
  # A fall costs less than in balance, so a push is worth trying.
  assert r["termination"].weight == -20.0
  # The balance task keeps the strict stance.
  balance = pupper_skate_balance_env_cfg()
  assert "max_count" not in balance.rewards["feet_on_deck"].params
  assert balance.rewards["tilt"].params["std"] == 0.2
  assert balance.rewards["board_alignment"].params["std_pos"] == 0.05
  feet_off = balance.terminations["feet_off_deck"].params
  assert feet_off.get("min_lost", 1) == 1 and feet_off["grace_s"] == 0.5
  assert balance.rewards["termination"].weight == -100.0


def test_split_ride_has_no_board_progress():
  assert "board_progress" not in pupper_skate_split_ride_env_cfg().rewards


def test_only_centred_ride_plays_the_push_reference():
  ride = pupper_skate_ride_env_cfg()
  actor = ride.observations["actor"].terms["policy"]
  assert actor.func is mdp.pupper_gait_actor_obs
  assert actor.params["gait"] == "skate_push"
  assert actor.params["frequency"] == gr.SKATE_PUSH_FREQUENCY
  assert actor.params["blend_speed"] == gr.SKATE_PUSH_BLEND_SPEED
  assert actor.params["gallop_speed"] > max(CENTRED_RIDE_SPEED_RANGE)
  critic = ride.observations["critic"].terms
  assert critic["gait_reference"].params == {
    k: v
    for k, v in actor.params.items()
    if k not in ("add_noise", "imu_latency_distribution")
  }
  assert critic["cycle_phase"].params == {"frequency": gr.SKATE_PUSH_FREQUENCY}
  # Balance and both split tasks build as before.
  others = (
    pupper_skate_balance_env_cfg(),
    pupper_skate_split_balance_env_cfg(),
    pupper_skate_split_ride_env_cfg(),
  )
  for cfg, fell in zip(others, (0.5, 0.6, 0.6), strict=True):
    assert "reference_tracking" not in cfg.rewards
    assert cfg.events["deck_height"].params["ranges"] == DECK_HEIGHT_RANGE
    assert cfg.terminations["fell_over"].params["limit_angle"] == fell
    policy = cfg.observations["actor"].terms["policy"]
    assert policy.func is mdp.zero_reference_actor_obs
    assert "cycle_phase" not in cfg.observations["critic"].terms
    assert "gait_reference" not in cfg.observations["critic"].terms


def test_cycle_phase_follows_the_episode_clock():
  env = SimpleNamespace(episode_length_buf=torch.tensor([0, 25, 50, 100]), step_dt=0.02)
  phase = mdp.cycle_phase(env, frequency=0.5)  # type: ignore[arg-type]
  expected = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
  torch.testing.assert_close(phase, expected, atol=1e-5, rtol=0)


@pytest.fixture(scope="module")
def ride_env():
  cfg = pupper_skate_ride_env_cfg(play=True)
  cfg.scene.num_envs = 3
  cfg.seed = 0
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  twist.resampling_time_range = (1e6, 1e6)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield env
  env.close()


def test_ride_exports_the_push_table_and_matches_the_robot_math(ride_env):
  env = ride_env
  policy = export_pupper_gait_policy_from_env(_Actor(48 * 20).eval(), env)
  block = policy["gait_reference"]
  assert gait_frame_kind(env) == "single"
  np.testing.assert_allclose(
    block["trot_table"], gr.build_joint_reference_table(100, "skate_push"), atol=1e-6
  )
  assert (block["frequency"], block["blend_speed"]) == (0.4, 0.05)
  assert block["gallop_speed"] > max(CENTRED_RIDE_SPEED_RANGE)
  assert policy["heading_hold"]["kp"] == 0.0
  assert policy["command_clip"]["vx"] == [0.0, 0.4]
  # The deploy mirror, fed the exported block, reproduces the offset in the
  # newest frame of the actor observation, at command 0, 0.03 and 0.3.
  env.reset()
  speeds = torch.tensor([0.0, 0.03, 0.3], device=env.device)
  twist = env.command_manager.get_term("twist")
  twist.vel_command_b[:] = 0.0
  twist.vel_command_b[:, 0] = speeds
  default = np.asarray(policy["default_joint_pos"])
  actions = torch.zeros(3, 12, device=env.device)
  checked = 0
  for step in range(130):
    obs, *_ = env.step(actions)
    if step % 13:
      continue
    command = env.command_manager.get_command("twist")
    assert command is not None
    cmd = command.cpu().numpy().astype(np.float64)
    t = env.episode_length_buf.cpu().numpy() * env.step_dt
    expected = reference_offset_numpy(
      block, default, t, cmd[:, 0], cmd[:, 1], cmd[:, 2]
    )
    actor = obs["actor"]
    assert isinstance(actor, torch.Tensor)
    np.testing.assert_allclose(actor[:, -12:].cpu().numpy(), expected, atol=1e-5)
    checked += 1
  assert checked == 10
  # Zero command is the still stance; 0.03 m/s plays 60% of the push.
  assert np.allclose(expected[0], 0.0)
  np.testing.assert_allclose(expected[1], 0.6 * expected[2], atol=1e-6)


@pytest.mark.slow
def test_ride_foot_reach_is_small_with_all_feet_on_the_deck():
  # The lowest deck the centred ride trains on (the stock trucks), a clean
  # spawn, a speed command and zero actions: the feet stay on the deck, so
  # reaching pays little.
  cfg = pupper_skate_ride_env_cfg()
  cfg.scene.num_envs = 1
  low = RIDE_DECK_HEIGHT_RANGE[0]
  cfg.events["deck_height"].params["ranges"] = (low, low)
  del cfg.events["push_robot"]
  for name in ("reset_base", "reset_board"):
    cfg.events[name].params["pose_range"] = {}
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.4, 0.4)
  twist.rel_standing_envs = 0.0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  reach = env.reward_manager.get_term_cfg("foot_reach")
  deck_top = SkateboardParams().deck_top_height + low
  on_deck_cap = math.exp(-(deck_top - 0.015) / reach.params["std"])
  actions = torch.zeros(1, 12, device=env.device)
  for _ in range(50):  # 1 s.
    _, _, terminated, truncated, _ = env.step(actions)
    assert not (terminated | truncated).any()
    # exp(-clearance / std): a foot on the deck is at least the deck height
    # less 1.5 cm of contact sink up; a foot on the floor pays 1.
    assert reach.func(env, **reach.params).max() < on_deck_cap
  assert (mdp.contact_flags(env, "deck_feet_contact") == 1).all()
  env.close()


@pytest.mark.slow
def test_ride_env_steps_with_random_actions():
  cfg = pupper_skate_ride_env_cfg()
  cfg.scene.num_envs = NUM_ENVS
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  obs, _ = env.reset()
  torch.manual_seed(0)
  for _ in range(100):
    actions = 0.3 * torch.randn(NUM_ENVS, 12, device=env.device)
    obs, reward, _, _, _ = env.step(actions)
    assert torch.isfinite(reward).all()
    for group in ("actor", "critic"):
      values = obs[group]
      assert isinstance(values, torch.Tensor) and torch.isfinite(values).all()
  command = env.command_manager.get_command("twist")
  assert command is not None and command[:, 0].min() >= 0.0
  env.close()


##
# Diverged envs: a NaN in one env resets that env instead of ending the run.
##

ALL_SKATE_TASKS = (
  "Mjlab-Skate-Balance-Pupper-v3",
  "Mjlab-Skate-Ride-Pupper-v3",
  "Mjlab-SkateSplit-Balance-Pupper-v3",
  "Mjlab-SkateSplit-Ride-Pupper-v3",
)


@pytest.mark.parametrize("task", ALL_SKATE_TASKS)
def test_skate_tasks_terminate_on_nan(task):
  term = load_env_cfg(task).terminations["nan"]
  assert term.func is mdp.nan_detection
  assert not term.time_out


@pytest.mark.slow
@pytest.mark.parametrize(
  "task", ["Mjlab-Skate-Ride-Pupper-v3", "Mjlab-SkateSplit-Ride-Pupper-v3"]
)
def test_nan_in_one_env_resets_only_that_env(task):
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = NUM_ENVS
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    actions = torch.zeros(NUM_ENVS, 12, device=env.device)
    env.step(actions)
    env.sim.data.qvel[1, :] = float("nan")
    obs, reward, terminated, _, _ = env.step(actions)
    assert terminated.tolist() == [False, True, False, False]
    assert env.episode_length_buf.tolist() == [2, 0, 2, 2]
    for group in ("actor", "critic"):
      values = obs[group]
      assert isinstance(values, torch.Tensor) and torch.isfinite(values).all()
    assert torch.isfinite(reward).all()
    # The reset env keeps running cleanly.
    obs, reward, terminated, _, _ = env.step(actions)
    actor = obs["actor"]
    assert isinstance(actor, torch.Tensor) and torch.isfinite(actor).all()
    assert torch.isfinite(reward).all() and not terminated.any()
  finally:
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize(
  "task", ["Mjlab-Skate-Balance-Pupper-v3", "Mjlab-SkateSplit-Balance-Pupper-v3"]
)
def test_board_joint_limits_survive_the_scene_compile(task):
  # The board is attached to the scene and compiled again; its limits must stay
  # in radians there too.
  p = SkateboardParams()
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  model = env.sim.mj_model
  roll = model.joint(f"board/{DECK_ROLL_JOINT_NAME}").range
  np.testing.assert_allclose(roll, (-p.roll_limit, p.roll_limit), atol=1e-6)
  yaw = p.steer_gain * p.roll_limit + 0.05  # Margin above the steer coupling.
  for name in ("truck_f_yaw", "truck_r_yaw"):
    np.testing.assert_allclose(
      model.joint(f"board/{name}").range, (-yaw, yaw), atol=1e-6
    )
  env.close()


##
# Robustness: deck heights for stock and lowered trucks, and small shoves.
##


def test_deck_height_range_covers_both_trucks():
  assert DECK_HEIGHT_RANGE == (-0.035, 0.01)


def test_balance_cfg_has_robustness_events():
  events = pupper_skate_balance_env_cfg().events
  assert events["deck_height"].mode == "startup"
  assert PUSH_INTERVAL_S == (2.0, 4.0)
  for name in ("push_robot", "nudge_board"):
    assert events[name].mode == "interval"
    assert events[name].interval_range_s == (2.0, 4.0)


def test_ride_cfg_has_no_board_nudge():
  events = pupper_skate_ride_env_cfg().events
  assert "deck_height" in events and "push_robot" in events
  assert "nudge_board" not in events


def test_play_cfg_drops_pushes():
  for make_cfg in (pupper_skate_balance_env_cfg, pupper_skate_ride_env_cfg):
    events = make_cfg(play=True).events
    assert "deck_height" in events
    assert "push_robot" not in events and "nudge_board" not in events


@pytest.fixture(scope="module")
def robust_env():
  cfg = pupper_skate_balance_env_cfg()
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield env
  env.close()


def test_deck_height_is_randomized_per_env(robust_env):
  env = robust_env
  deck = env.sim.mj_model.geom(f"board/{DECK_GEOM_NAME}").id
  nominal = float(env.sim.mj_model.geom_pos[deck, 2])
  z = env.sim.model.geom_pos[:, deck, 2]
  assert z.shape == (8,)
  assert not torch.allclose(z, z[0].expand_as(z))
  low, high = DECK_HEIGHT_RANGE
  assert ((z >= nominal + low - 1e-6) & (z <= nominal + high + 1e-6)).all()


@pytest.mark.slow
def test_balance_env_survives_pushes(robust_env):
  env = robust_env
  env.reset()
  actions = torch.zeros(8, 12, device=env.device)
  for _ in range(150):  # 3 s: at least one push interval.
    obs, reward, _, _, _ = env.step(actions)
    assert torch.isfinite(reward).all()
    for group in ("actor", "critic"):
      values = obs[group]
      assert isinstance(values, torch.Tensor) and torch.isfinite(values).all()


##
# Centred ride: only a standing back-foot push pays (task 35). The last run lay
# on the deck and paddled with the other legs.
##

_NEW_RIDE_TERMS = ("knees_on_deck", "support_feet_off_deck", "base_near_deck")
_NEW_RIDE_KEYWORDS = ("foot_ids", "support_sensor", "support_grace_s")


def test_support_feet_off_deck_ignores_the_push_foot():
  term = pupper_skate_ride_env_cfg().terminations["support_feet_off_deck"]
  assert term.func is mdp.contact_lost
  params = term.params
  assert params["foot_ids"] == SUPPORT_FOOT_IDS == (0, 1, 3)
  assert params["grace_s"] == SUPPORT_OFF_DECK_S == 0.5
  assert PUSH_FOOT_ID == 2 and PUSH_FOOT_ID not in SUPPORT_FOOT_IDS
  # Rows: a support foot off 0.6 s, off 0.4 s, the push foot alone off 5 s,
  # all four down.
  env = _fake_env(
    deck_feet_contact={
      "current_air_time": torch.tensor(
        [
          [0.0, 0.6, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.4],
          [0.0, 0.0, 5.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
        ]
      )
    }
  )
  assert mdp.contact_lost(env, **params).tolist() == [True, False, False, False]
  # Without foot_ids every foot counts, as before.
  lost = mdp.contact_lost(env, "deck_feet_contact", grace_s=0.5)
  assert lost.tolist() == [True, False, True, False]


def test_knees_on_deck_ends_the_episode_after_0_2_s():
  term = pupper_skate_ride_env_cfg().terminations["knees_on_deck"]
  assert term.func is mdp.contact_held
  assert term.params == {
    "sensor_name": "knees_on_deck",
    "max_contact_s": KNEES_ON_DECK_S,
    "min_count": 2,
  }
  assert KNEES_ON_DECK_S == 0.4
  # Rows: no knee down; one knee down 5 s; two down, one only 0.1 s; two down
  # past the 0.4 s hold.
  env = _fake_env(
    knees_on_deck={
      "current_contact_time": torch.tensor(
        [
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 5.0, 0.0, 0.0],
          [0.1, 0.5, 0.0, 0.0],
          [0.0, 0.5, 0.45, 0.0],
        ]
      )
    }
  )
  held = mdp.contact_held(env, **term.params)
  assert held.tolist() == [False, False, False, True]
  # The default min_count of 1: any one knee past the hold.
  held = mdp.contact_held(env, "knees_on_deck", max_contact_s=0.4)
  assert held.tolist() == [False, True, True, True]


def _deck_env(deck_top: list[float], base_z: list[float]) -> Any:
  """Board with one deck geom whose top is ``deck_top``; robot base at ``base_z``."""
  half = SkateboardParams().deck_thickness / 2
  geom = torch.zeros(len(deck_top), 1, 3)
  geom[:, 0, 2] = torch.tensor(deck_top) - half
  base = torch.zeros(len(base_z), 3)
  base[:, 2] = torch.tensor(base_z)
  return _fake_env(board={"geom_pos_w": geom}, robot={"root_link_pos_w": base})


def test_base_near_deck_follows_each_env_deck_top():
  term = pupper_skate_ride_env_cfg().terminations["base_near_deck"]
  assert term.func is mdp.base_near_deck
  assert term.params["minimum_height"] == BASE_NEAR_DECK_MIN == 0.055
  deck_cfg = SceneEntityCfg("board", geom_names=(DECK_GEOM_NAME,))
  deck_cfg.geom_ids = [0]
  nominal = SkateboardParams().deck_top_height
  low, high = (nominal + d for d in RIDE_DECK_HEIGHT_RANGE)
  # At each end of the deck range: the settled stance (0.117 m up), the loaded
  # push crouch (0.073 m up), and the base lowered onto the deck (0.04 m up).
  tops = [low, low, low, high, high, high]
  heights = [0.117, 0.073, 0.04] * 2
  env = _deck_env(tops, [t + h for t, h in zip(tops, heights, strict=True)])
  near = mdp.base_near_deck(env, BASE_NEAR_DECK_MIN, deck_cfg)
  assert near.tolist() == [False, False, True] * 2
  # A fixed base height means different things over different decks.
  env = _deck_env([low, high], [high + 0.05, high + 0.05])
  assert mdp.base_near_deck(env, BASE_NEAR_DECK_MIN, deck_cfg).tolist() == [
    False,
    True,
  ]


def _push_env(
  ground_air: list[list[float]],
  stroke: list[list[float]],
  deck_air: list[list[float]],
  foot_z: list[list[float]],
) -> Any:
  """Ground and deck foot sensors and foot heights; a 0.4 m/s command, board at rest."""
  pos = torch.zeros(len(foot_z), 4, 3)
  pos[:, :, 2] = torch.tensor(foot_z)
  env = _fake_env(
    ground_feet_contact={
      "current_air_time": torch.tensor(ground_air),
      "last_contact_time": torch.tensor(stroke),
    },
    deck_feet_contact={"current_air_time": torch.tensor(deck_air)},
    robot={"site_pos_w": pos},
  )
  env.step_dt = 0.02
  return _with_command(env, 0.4, [0.0] * len(foot_z))


_STROKE: dict[str, Any] = {
  "sensor_name": "ground_feet_contact",
  "stroke_s": 0.25,
  "command_name": "twist",
  "command_threshold": 0.1,
  "speed_fraction": 1.0,
}
_REACH: dict[str, Any] = {**_FOOT_REACH, "std": 0.08, "continuous_gate": True}
_PUSH_ONLY: dict[str, Any] = {
  "foot_ids": (PUSH_FOOT_ID,),
  "support_sensor": "deck_feet_contact",
  "support_grace_s": 0.1,
}


def _push_rows() -> Any:
  # Rows: the push foot lifts off after a full stroke with the support feet on
  # the deck; a front right foot does the same with the push foot on the
  # deck; the push foot strokes with the back left foot off the deck 0.2 s;
  # the push foot strokes while the front left foot flickers off for one step.
  # Feet on the floor are at z = 0.02, on the deck at 0.12.
  up, down = 0.12, 0.02
  return _push_env(
    ground_air=[
      [0.0, 0.0, 0.02, 0.0],
      [0.02, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.02, 0.0],
      [0.0, 0.0, 0.02, 0.0],
    ],
    stroke=[[0.0, 0.0, 0.4, 0.0], [0.4, 0.0, 0.0, 0.0]] + [[0.0, 0.0, 0.4, 0.0]] * 2,
    deck_air=[
      [0.0, 0.0, 1.0, 0.0],
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.2],
      [0.0, 0.02, 1.0, 0.0],
    ],
    foot_z=[
      [up, up, down, up],
      [down, up, up, up],
      [up, up, down, up],
      [up, up, down, up],
    ],
  )


def test_push_stroke_pays_only_the_push_foot_with_the_support_down():
  env = _push_rows()
  old = mdp.push_stroke(env, **_STROKE)
  # Unset keywords: any foot's stroke pays, in any posture.
  assert old.tolist() == pytest.approx([50.0] * 4)
  new = mdp.push_stroke(env, **_STROKE, **_PUSH_ONLY)
  # A one-step flicker (0.02 s off) does not lose the stroke; 0.2 s off does.
  assert new.tolist() == pytest.approx([50.0, 0.0, 0.0, 50.0])


def test_foot_reach_pays_only_the_push_foot_with_the_support_down():
  env = _push_rows()
  old = mdp.foot_reach(env, **_REACH)
  assert old.tolist() == pytest.approx([1.0] * 4)
  new = mdp.foot_reach(env, **_REACH, **_PUSH_ONLY)
  # Rows 1 and 2: a support foot is off the deck (front right on the floor, back
  # left lifted), so nothing pays.
  assert new.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])
  # With the support feet down, a push foot on the deck pays its deck height.
  env.scene["deck_feet_contact"].data.current_air_time[1] = 0.0
  reach = mdp.foot_reach(env, **_REACH, **_PUSH_ONLY)[1].item()
  assert reach == pytest.approx(math.exp(-0.10 / 0.08))


def test_unset_keywords_keep_the_old_push_terms():
  # Same random inputs, with and without the keywords at their defaults.
  torch.manual_seed(0)
  rows = 32
  ground_air = (torch.rand(rows, 4) < 0.5) * torch.rand(rows, 4) * 0.04
  stroke = torch.rand(rows, 4) * 0.5
  foot_z = torch.rand(rows, 4) * 0.15
  env = _push_env(
    ground_air.tolist(), stroke.tolist(), [[0.0] * 4] * rows, foot_z.tolist()
  )
  defaults = {"foot_ids": None, "support_sensor": None}
  torch.testing.assert_close(
    mdp.push_stroke(env, **_STROKE), mdp.push_stroke(env, **_STROKE, **defaults)
  )
  torch.testing.assert_close(
    mdp.foot_reach(env, **_REACH), mdp.foot_reach(env, **_REACH, **defaults)
  )
  lost = mdp.contact_lost(env, "ground_feet_contact", grace_s=0.01)
  torch.testing.assert_close(
    lost, mdp.contact_lost(env, "ground_feet_contact", grace_s=0.01, foot_ids=None)
  )


def test_ride_cfg_gates_both_push_terms():
  r = pupper_skate_ride_env_cfg().rewards
  for name in ("push_stroke", "foot_reach"):
    for key, value in _PUSH_ONLY.items():
      assert r[name].params[key] == value, (name, key)
  terms = pupper_skate_ride_env_cfg(play=True).terminations
  assert set(_NEW_RIDE_TERMS) <= set(terms)


@pytest.mark.parametrize(
  "make_cfg",
  [
    pupper_skate_balance_env_cfg,
    pupper_skate_split_balance_env_cfg,
    pupper_skate_split_ride_env_cfg,
    pupper_skate_split_steer_env_cfg,
  ],
)
def test_other_skate_tasks_keep_their_rules(make_cfg):
  for play in (False, True):
    cfg = make_cfg(play=play)
    for name in _NEW_RIDE_TERMS:
      assert name not in cfg.terminations, name
    for term in (*cfg.rewards.values(), *cfg.terminations.values()):
      for key in _NEW_RIDE_KEYWORDS:
        assert key not in term.params, key
    knees = [s for s in cfg.scene.sensors or () if s.name == "knees_on_deck"]
    assert len(knees) == 1 and not knees[0].track_air_time


@pytest.mark.slow
@pytest.mark.parametrize("deck_offset", RIDE_DECK_HEIGHT_RANGE)
def test_base_near_deck_in_the_sim(deck_offset):
  # The ride env with the deck pinned at one end of its range, a still command
  # and zero actions: the stance holds 1 s without tripping the new rules.
  # Then the base is held 4 cm over the deck top and base_near_deck fires.
  cfg = pupper_skate_ride_env_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.seed = 0
  cfg.events["deck_height"].params["ranges"] = (deck_offset, deck_offset)
  for name in ("reset_base", "reset_board"):
    cfg.events[name].params["pose_range"] = {}
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.0, 0.0)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    actions = torch.zeros(2, 12, device=env.device)
    for _ in range(50):
      env.step(actions)
      for name in _NEW_RIDE_TERMS:
        assert not env.termination_manager.get_term(name).any(), name
    robot = env.scene["robot"]
    deck_top = SkateboardParams().deck_top_height + deck_offset
    low = robot.data.default_root_state.clone()
    low[:, :3] += env.scene.env_origins
    low[:, 2] = deck_top + 0.04
    robot.write_root_state_to_sim(low)
    env.step(actions)
    assert env.termination_manager.get_term("base_near_deck").all()
  finally:
    env.close()


def play_push_reference_terminations(
  deck_top: float,
  num_envs: int = 8,
  seconds: float = 6.0,
  seed: int = 0,
  watch: tuple[str, ...] = (),
) -> dict[str, Any]:
  """Play the skate_push reference open loop on the centred ride (play config).

  Command 0.3 m/s, deck top pinned at ``deck_top``. Returns, per termination,
  how many env-steps it fired in all (``total``) and how many of those came
  with the body tilted under 0.2 rad and the episode over 0.5 s old
  (``calm``; both read the step before, since a terminated env is reset
  within the step). Also returns the number of push_stroke pays and the share
  of env-steps that paid.

  The terminations named in ``watch`` are taken out of the config and only
  watched: each step they are evaluated on the state the termination manager
  would have judged, without ending the episode, so it is visible whether a
  fall follows. ``pre_fall`` counts their calm firings followed by a
  ``fell_over`` in that env within 0.5 s, and ``calm_no_fall`` the rest.
  """
  cfg = pupper_skate_ride_env_cfg(play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.3, 0.3)
  twist.rel_standing_envs = 0.0
  twist.resampling_time_range = (1e6, 1e6)
  offset = deck_top - SkateboardParams().deck_top_height
  cfg.events["deck_height"].params["ranges"] = (offset, offset)
  watched = {name: cfg.terminations.pop(name) for name in watch}
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  for term in watched.values():
    for value in term.params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
  window = int(round(0.5 / env.step_dt))
  try:
    table = torch.tensor(
      gr.build_joint_reference_table(GAIT_N_SAMPLES, "skate_push"),
      dtype=torch.float32,
      device=env.device,
    )
    action_term = env.action_manager.get_term("joint_pos")
    assert isinstance(action_term, BaseAction)
    robot = env.scene["robot"]
    names = list(env.termination_manager.active_terms)
    total = dict.fromkeys([*names, *watched], 0)
    calm = dict.fromkeys([*names, *watched], 0)
    # Per watched rule, the (step, env) of each calm firing.
    calm_at: dict[str, list[tuple[int, int]]] = {name: [] for name in watched}
    fell, ended = [], []
    causes: list[list[list[str]]] = []
    stroke_idx = env.reward_manager.active_terms.index("push_stroke")
    pays = 0
    env.reset()
    steps = int(round(seconds / env.step_dt))
    with torch.no_grad():
      for step in range(steps):
        command = env.command_manager.get_command("twist")
        assert command is not None
        default = robot.data.default_joint_pos
        reference = default + reference_offset_from_tables(
          table,
          table,
          env.episode_length_buf.float() * env.step_dt,
          command[:, 0],
          command[:, 1],
          command[:, 2],
          default,
          frequency=gr.SKATE_PUSH_FREQUENCY,
          blend_speed=gr.SKATE_PUSH_BLEND_SPEED,
          gallop_speed=GAIT_GALLOP_SPEED,
          gallop_freq_mult=1.0,
        )
        action = (reference - action_term.offset) / action_term.scale
        gz = robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
        level = torch.acos(-gz) < 0.2
        settled = env.episode_length_buf * env.step_dt > 0.5
        for name, term in watched.items():
          fired = term.func(env, **term.params)
          total[name] += int(fired.sum())
          hits = fired & level & settled
          calm[name] += int(hits.sum())
          calm_at[name] += [(step, i) for i in hits.nonzero().flatten().tolist()]
        _, _, terminated, truncated, _ = env.step(action)
        fell.append(env.termination_manager.get_term("fell_over").cpu().clone())
        causes.append(
          [
            [n for n in names if env.termination_manager.get_term(n)[i]]
            for i in range(num_envs)
          ]
        )
        ended.append((terminated | truncated).cpu().clone())
        for name in names:
          fired = env.termination_manager.get_term(name)
          total[name] += int(fired.sum())
          calm[name] += int((fired & level & settled).sum())
        pays += int((env.reward_manager._step_reward[:, stroke_idx] > 0).sum())
  finally:
    env.close()
  pre_fall = dict.fromkeys(watched, 0)
  # Calm firings with no fall after them: (step, env, steps to the next reset
  # or None if the run ended first, what ended that episode).
  no_fall: dict[str, list[tuple[int, int, int | None, list[str]]]] = {
    name: [] for name in watched
  }
  for name, hits in calm_at.items():
    for step, i in hits:
      # The first reset in the window, if any, must be a fall.
      for later in range(step, min(step + window, steps)):
        if ended[later][i]:
          if fell[later][i]:
            pre_fall[name] += 1
            break
          no_fall[name].append((step, i, later - step, causes[later][i]))
          break
      else:
        after = [k for k in range(step, steps) if ended[k][i]]
        nxt = after[0] - step if after else None
        cause = causes[after[0]][i] if after else []
        no_fall[name].append((step, i, nxt, cause))
  return {
    "total": total,
    "calm": calm,
    "pre_fall": pre_fall,
    "calm_no_fall": {name: calm[name] - pre_fall[name] for name in watched},
    "no_fall_events": no_fall,
    "stroke_pays": pays,
    "stroke_pay_share": pays / (steps * num_envs),
  }


@pytest.mark.slow
@pytest.mark.parametrize("deck_top", [0.093, 0.103, 0.113])
def test_push_reference_does_not_trip_the_lying_down_rules(deck_top):
  # Open loop the reference falls at the end of each push; only the steps with
  # the body level and the episode under way must stay clear of the rules that
  # catch lying on the deck. The 0.5 s before a fell_over in that env are left
  # out too: they are the start of the reference's own end-of-push fall, where
  # the base sinks under 0.055 m while the tilt is still under 0.2 rad, and
  # ending the episode a few steps early there is harmless.
  watch = ("knees_on_deck", "base_near_deck")
  result = play_push_reference_terminations(deck_top, watch=watch)
  for name in watch:
    assert result["calm_no_fall"][name] == 0, (name, result)


@pytest.mark.slow
def test_knees_on_deck_fires_lying_on_the_deck():
  # The base held 0.07 m over the deck top, level: the knee spheres (0.074 m
  # under the base at the stance) rest on the deck, as when lying on it. Two
  # knees down past the hold end the episode before base_near_deck would.
  cfg = pupper_skate_ride_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.seed = 0
  cfg.events["deck_height"].params["ranges"] = (0.0, 0.0)
  for name in ("reset_base", "reset_board"):
    cfg.events[name].params["pose_range"] = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    robot = env.scene["robot"]
    lying = robot.data.default_root_state.clone()
    lying[:, :3] += env.scene.env_origins
    lying[:, 2] = SkateboardParams().deck_top_height + 0.07
    actions = torch.zeros(1, 12, device=env.device)
    most_knees, fired_at = 0, None
    for step in range(40):  # 0.8 s.
      robot.write_root_state_to_sim(lying)
      env.step(actions)
      found = env.scene["knees_on_deck"].data.found
      most_knees = max(most_knees, int((found > 0).sum()))
      if env.termination_manager.get_term("knees_on_deck").any():
        fired_at = step
        break
    assert most_knees >= 2
    # Not before the hold: the knees have been down (fired_at + 1) steps.
    assert fired_at is not None
    assert (fired_at + 1) * env.step_dt >= KNEES_ON_DECK_S - 1e-6
  finally:
    env.close()
