"""The split-stance skateboard tasks: left feet on the deck, right on the floor."""

from __future__ import annotations

import dataclasses
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

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
  DECK_GEOM_NAME,
  SkateboardParams,
)
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  JOINT_NAMES,
  PUPPER_EFFORT_LIMIT,
  PUPPER_MUJOCO_OPTIONS,
  SKATE_SPLIT_BOARD_OFFSET,
  SKATE_SPLIT_HEIGHT,
  SKATE_SPLIT_KEYFRAME,
  SKATE_SPLIT_POSE,
  SKATE_SPLIT_ROLL,
  get_spec,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim import MujocoCfg
from mjlab.tasks.pupper_gait.export import (
  export_pupper_gait_policy_from_env,
  heading_hold_metadata,
)
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  GAIT_HEADING_HOLD_CLIP,
  GAIT_HEADING_HOLD_KP,
  pupper_velocity_fs_env_cfg,
)
from mjlab.tasks.pupper_skate import mdp
from mjlab.tasks.pupper_skate.mdp.rewards import stance_target
from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import (
  BALANCE_TORQUE_SOFT_RATIO,
  DECK_HEIGHT_RANGE,
  RIDE_DECK_HEIGHT_RANGE,
  RIDE_SPEED_RANGE,
  pupper_skate_balance_env_cfg,
  pupper_skate_ride_env_cfg,
)
from mjlab.tasks.pupper_skate.pupper_skate_split_env_cfg import (
  BOARD_CREEP_CAP,
  BOARD_CREEP_TAU,
  BOARD_HEADING_HOLD_STD,
  BOARD_HEADING_HOLD_WEIGHT,
  BOARD_MAX_SLIDE,
  BOARD_MAX_SLIDE_ABS,
  BOARD_SIDE_CREEP_WEIGHT,
  BOARD_SLIDE_STD,
  DECK_BODY_FRICTION_RANGE,
  DECK_FEET_SLIDE_WEIGHT,
  DECK_FOOT_FRICTION_RANGE,
  FLOOR_FOOT_FRICTION_RANGE,
  HOLD_BOARD_GAIN,
  SPLIT_CONTACT_CONE,
  SPLIT_CONTACT_IMPRATIO,
  SPLIT_RIDE_DECK_HEIGHT_RANGE,
  STEER_BOARD_ALIGN_STD_YAW,
  STEER_CARVE_COMMAND_RANGE,
  STEER_CARVE_GAIN,
  STEER_CARVE_THRESHOLD,
  STEER_DECK_FEET_SLIDE_WEIGHT,
  STEER_HEADING_HOLD_CLIP,
  STEER_STRAIGHT_SHARE,
  STEER_TRUCK_STIFFNESS_SCALE,
  STEER_YAW_MIN_SPEED,
  WHEEL_DRAG_SCALE_RANGE,
  pupper_skate_split_balance_env_cfg,
  pupper_skate_split_ride_env_cfg,
  pupper_skate_split_steer_env_cfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_from_euler_xyz,
  quat_mul,
  wrap_to_pi,
)

NUM_ENVS = 4
SPLIT_TASKS = ("Mjlab-SkateSplit-Balance-Pupper-v3", "Mjlab-SkateSplit-Ride-Pupper-v3")
CENTRED_TASKS = ("Mjlab-Skate-Balance-Pupper-v3", "Mjlab-Skate-Ride-Pupper-v3")
STEER_TASK = "Mjlab-SkateSplit-Steer-Pupper-v3"

##
# Keyframe.
##


def _foot_in_base(data: mujoco.MjData, leg: str) -> np.ndarray:
  return data.site(f"leg_{leg}_3_foot_site").xpos - data.body("base_link").xpos


@pytest.fixture(scope="module")
def stance() -> tuple[mujoco.MjModel, mujoco.MjData]:
  """The robot alone, base at the origin and level, joints at SKATE_SPLIT_POSE."""
  model = get_spec().compile()
  data = mujoco.MjData(model)
  data.qpos[3] = 1.0
  for name, angle in zip(JOINT_NAMES, SKATE_SPLIT_POSE, strict=True):
    data.qpos[model.joint(name).qposadr[0]] = angle
  mujoco.mj_kinematics(model, data)
  return model, data


def test_split_pose_is_inside_joint_limits(stance):
  model, _ = stance
  for name, angle in zip(JOINT_NAMES, SKATE_SPLIT_POSE, strict=True):
    low, high = model.joint(name).range
    assert low + 0.2 < angle < high - 0.2, name


def test_split_pose_lifts_the_left_feet_over_the_deck(stance):
  _, data = stance
  left = [_foot_in_base(data, leg) for leg in ("front_l", "back_l")]
  right = [_foot_in_base(data, leg) for leg in ("front_r", "back_r")]
  for foot in left:
    assert foot[1] == pytest.approx(0.14, abs=0.005)
    assert foot[2] == pytest.approx(-0.064, abs=0.005)
  for foot in right:
    assert foot[1] < -0.08
    assert foot[2] == pytest.approx(-0.145, abs=0.005)
  # The deck sits midway between the left feet, fore and aft.
  assert (left[0][0] + left[1][0]) / 2 == pytest.approx(
    SKATE_SPLIT_BOARD_OFFSET[0], abs=0.005
  )


def test_split_pose_centres_the_left_feet_on_the_deck(stance):
  # Rolled by SKATE_SPLIT_ROLL, each left foot sits within 1 cm of the deck's
  # centre line and at least 2.5 cm in from both edges.
  _, data = stance
  cos, sin = math.cos(SKATE_SPLIT_ROLL), math.sin(SKATE_SPLIT_ROLL)
  half_width = SkateboardParams().deck_width / 2
  for leg in ("front_l", "back_l"):
    _, y, z = _foot_in_base(data, leg)
    off_centre = y * cos - z * sin - SKATE_SPLIT_BOARD_OFFSET[1]
    assert abs(off_centre) < 0.01, leg
    assert abs(off_centre) <= half_width - 0.025, leg


def test_split_keyframe_spawns_the_pose_already_tilted():
  assert SKATE_SPLIT_KEYFRAME.joint_pos is not None
  assert list(SKATE_SPLIT_KEYFRAME.joint_pos.values()) == list(SKATE_SPLIT_POSE)
  w, x, y, z = SKATE_SPLIT_KEYFRAME.rot
  assert 2 * math.atan2(x, w) == pytest.approx(SKATE_SPLIT_ROLL)
  assert SKATE_SPLIT_ROLL > 0.0  # The floor (right) side is lower.
  assert (y, z) == (0.0, 0.0)
  assert SKATE_SPLIT_KEYFRAME.pos[2] == pytest.approx(SKATE_SPLIT_HEIGHT + 0.01)


##
# push_stride on a hand-built state.
##


def _fake_env(sensor_data: SimpleNamespace, get_command) -> Any:
  """Enough of an env for push_stride: one sensor, the step and the command."""
  return SimpleNamespace(
    scene={"feet": SimpleNamespace(data=sensor_data)},
    device="cpu",
    step_dt=0.02,
    command_manager=SimpleNamespace(get_command=get_command),
  )


def test_push_stride_pays_a_planted_stride_once_at_touchdown():
  # Foot 0 varies per row; foot 1 has been standing a while.
  # Rows: planted stride, a tap, too quick a swing, foot held up, landed earlier.
  data = SimpleNamespace(
    current_contact_time=torch.tensor(
      [[0.02, 1.0], [0.02, 1.0], [0.02, 1.0], [0.02, 1.0], [0.06, 1.0]]
    ),
    last_air_time=torch.tensor(
      [[0.2, 0.0], [0.2, 0.0], [0.04, 0.0], [0.8, 0.0], [0.2, 0.0]]
    ),
    last_contact_time=torch.tensor(
      [[0.4, 0.0], [0.1, 0.0], [0.4, 0.0], [0.4, 0.0], [0.4, 0.0]]
    ),
  )
  command = torch.tensor([[0.4, 0.0, 0.0]] * 5)
  env = _fake_env(data, lambda name: command)
  params: dict[str, Any] = {
    "sensor_name": "feet",
    "stance_s": 0.4,
    "air_range": (0.1, 0.5),
    "command_name": "twist",
    "command_threshold": 0.1,
  }
  reward = mdp.push_stride(env, **params)
  assert reward.tolist() == pytest.approx([50.0, 12.5, 0.0, 0.0, 0.0])
  command = torch.tensor([[0.05, 0.0, 0.0]] * 5)
  assert mdp.push_stride(env, **params).tolist() == [0.0] * 5


def test_short_stride_charges_a_landing_after_a_short_stance():
  # Foot 0 varies per row; foot 1 has been standing a while.
  # Rows: landed after 0.1 s of stance, after 0.4 s, after 0.6 s, landed
  # earlier, in the air, spawned on the floor (first step after a reset, no
  # stance on record).
  data = SimpleNamespace(
    current_contact_time=torch.tensor(
      [[0.02, 1.0], [0.02, 1.0], [0.02, 1.0], [0.06, 1.0], [0.0, 1.0], [0.02, 1.0]]
    ),
    last_contact_time=torch.tensor(
      [[0.1, 0.0], [0.4, 0.0], [0.6, 0.0], [0.1, 0.0], [0.1, 0.0], [0.0, 0.0]]
    ),
  )
  expected = [0.75 / 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
  # Not gated on the command: a tap costs the same when told to stand still.
  for speed in (0.4, 0.0):
    command = torch.tensor([[speed, 0.0, 0.0]] * 6)
    env = _fake_env(data, lambda name, command=command: command)
    reward = mdp.short_stride(env, sensor_name="feet", stance_s=0.4)
    assert reward.tolist() == pytest.approx(expected)


def test_stance_target_scales_a_10_cm_stroke_to_the_command():
  speed = torch.tensor([0.0, 0.2, 0.3, 0.4, 0.5])
  target = stance_target(speed, stroke_m=0.10, stance_range=(0.25, 0.5))
  assert target.tolist() == pytest.approx([0.5, 0.5, 1 / 3, 0.25, 0.25])


def test_push_stride_with_a_stroke_scales_the_stance_to_the_command():
  # One foot that just landed after a 0.25 s stance and a 0.3 s swing.
  data = SimpleNamespace(
    current_contact_time=torch.tensor([[0.02], [0.02]]),
    last_air_time=torch.tensor([[0.3], [0.3]]),
    last_contact_time=torch.tensor([[0.25], [0.25]]),
  )
  # Row 0 at 0.4 m/s wants a 0.25 s stance, row 1 at 0.2 m/s wants 0.5 s.
  command = torch.tensor([[0.4, 0.0, 0.0], [0.2, 0.0, 0.0]])
  env = _fake_env(data, lambda name: command)
  reward = mdp.push_stride(
    env,
    sensor_name="feet",
    stroke_m=0.10,
    stance_range=(0.25, 0.5),
    air_range=(0.2, 0.5),
    command_name="twist",
    command_threshold=0.1,
  )
  assert reward.tolist() == pytest.approx([1.0 / 0.02, 0.5 / 0.02])
  cost = mdp.short_stride(
    env,
    sensor_name="feet",
    stroke_m=0.10,
    stance_range=(0.25, 0.5),
    command_name="twist",
  )
  assert cost.tolist() == pytest.approx([0.0, 0.5 / 0.02])


def _fake_height_env(found: list[float], foot_z: list[float]) -> Any:
  """Enough of an env for foot_height_over: one env, a sensor and foot sites."""
  site_pos = torch.zeros(1, len(foot_z), 3)
  site_pos[0, :, 2] = torch.tensor(foot_z)
  return SimpleNamespace(
    scene={
      "feet": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([found]))),
      "robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=site_pos)),
    },
    device="cpu",
  )


def test_foot_height_over_charges_swing_feet_above_3_cm():
  def over(found: float, underside_z: float) -> float:
    # The site is the foot's centre, a foot radius above its underside.
    env = _fake_height_env([found], [underside_z + 0.02])
    return mdp.foot_height_over(
      env,
      max_height=0.03,
      foot_radius=0.02,
      sensor_name="feet",
      foot_cfg=SceneEntityCfg("robot"),
    ).item()

  assert over(1.0, 0.05) == 0.0  # In contact: never charged.
  assert over(0.0, 0.02) == 0.0
  assert over(0.0, 0.05) == pytest.approx(0.02)
  # Feet add up.
  env = _fake_height_env([0.0, 0.0], [0.07, 0.09])
  cost = mdp.foot_height_over(env, 0.03, 0.02, "feet", SceneEntityCfg("robot"))
  assert cost.tolist() == pytest.approx([0.06])


def test_foot_idle_air_counts_feet_up_too_long():
  # Rows: both feet planted, foot 0 up briefly, foot 0 up too long, both up too
  # long, foot 1 exactly at the limit.
  data = SimpleNamespace(
    current_air_time=torch.tensor(
      [[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.8, 1.2], [0.0, 0.5]]
    )
  )
  env = _fake_env(data, lambda name: None)
  cost = mdp.foot_idle_air(env, sensor_name="feet", max_air_s=0.5)
  assert cost.tolist() == [0.0, 0.0, 1.0, 2.0, 0.0]


def _fake_slide_env(found: list[float], foot_vel: list[list[float]]) -> Any:
  """Enough of an env for feet_slide: one env, a contact sensor and foot sites."""
  return SimpleNamespace(
    scene={
      "feet": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([found]))),
      "robot": SimpleNamespace(
        data=SimpleNamespace(site_lin_vel_w=torch.tensor([foot_vel]))
      ),
    },
    device="cpu",
  )


def test_feet_slide_charges_a_planted_foot_moving_over_the_floor():
  # Two feet: foot 0 moves at 0.5 m/s (0.3, 0.4) in xy, foot 1 is still. The z
  # speed does not count.
  foot_vel = [[0.3, 0.4, 1.0], [0.0, 0.0, 1.0]]

  def slide(found: list[float]) -> float:
    env = _fake_slide_env(found, foot_vel)
    return mdp.feet_slide(env, "feet", SceneEntityCfg("robot")).item()

  assert slide([1.0, 1.0]) == pytest.approx(0.25)
  assert slide([0.0, 1.0]) == 0.0


def _fake_deck_slide_env(found: list[float], foot_vel: list[list[float]]) -> Any:
  """Two deck feet on a deck rolling at 0.3 m/s and turning at 0.5 rad/s.

  The deck link is at the origin; foot 0 sits 0.1 m ahead of it, foot 1
  0.1 m behind, so the deck carries them at (0.3, 0.05) and (0.3, -0.05).
  """
  robot = SimpleNamespace(
    site_pos_w=torch.tensor([[[0.1, 0.0, 0.05], [-0.1, 0.0, 0.05]]]),
    site_lin_vel_w=torch.tensor([foot_vel]),
  )
  board = SimpleNamespace(
    body_link_pos_w=torch.zeros(1, 1, 3),
    body_link_lin_vel_w=torch.tensor([[[0.3, 0.0, 0.0]]]),
    body_link_ang_vel_w=torch.tensor([[[0.0, 0.0, 0.5]]]),
  )
  return SimpleNamespace(
    scene={
      "deck_feet": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([found]))),
      "robot": SimpleNamespace(data=robot),
      "board": SimpleNamespace(data=board),
    },
    device="cpu",
  )


def test_deck_feet_slide_charges_slip_relative_to_the_deck():
  def slide(found: list[float], foot_vel: list[list[float]]) -> float:
    env = _fake_deck_slide_env(found, foot_vel)
    return mdp.deck_feet_slide(
      env, "deck_feet", SceneEntityCfg("robot"), SceneEntityCfg("board")
    ).item()

  carried = [[0.3, 0.05, 0.0], [0.3, -0.05, 0.0]]
  # Riding along with the rolling, turning deck: no slip, though the feet move
  # at 0.3 m/s over the floor.
  assert slide([1.0, 1.0], carried) == pytest.approx(0.0, abs=1e-9)
  # Foot 1 creeps sideways 0.1 m/s over the deck: 0.01. Height changes do not
  # count.
  creep = [[0.3, 0.05, 0.2], [0.3, 0.05, 0.0]]
  assert slide([1.0, 1.0], creep) == pytest.approx(0.01)
  # The same creep with foot 1 off the deck costs nothing.
  assert slide([1.0, 0.0], creep) == pytest.approx(0.0, abs=1e-9)


##
# Rewards: the a110b84 split configs, name for name and weight for weight, plus
# the balance torque penalty (Task 25), the ride shuffle fix (Task 29), the
# ride motor limits and idle-foot charge (Task 30), the smoother push
# (Task 31) and the deck-foot slip and firmer board alignment (Task 42).
##

SPLIT_BALANCE_WEIGHTS = {
  "feet_on_deck": 1.0,
  "feet_on_ground": 0.5,
  "deck_feet_on_ground": -2.0,
  "ground_feet_on_deck": -2.0,
  "knees_on_deck": -1.0,
  "tilt": 1.0,
  "base_height": 1.0,
  "stance_pose": 1.0,
  "board_alignment": 1.0,
  "deck_flat": -5.0,
  "board_still": -2.0,
  "action_rate_l2": -0.01,
  "joint_torques_l2": -2e-4,
  "termination": -100.0,
  "torque_saturation": -1.0,
}

SPLIT_RIDE_WEIGHTS = {
  **{k: v for k, v in SPLIT_BALANCE_WEIGHTS.items() if k != "board_still"},
  "feet_on_ground": 0.1,
  "track_board_velocity": 3.0,
  "board_yaw_rate": -0.5,
  "track_yaw_rate": 1.0,
  "push_stride": 0.5,
  "short_stride": -0.3,
  "feet_slide": -2.0,
  "deck_feet_slide": -20.0,
  "board_alignment": 2.0,
  "push_leg_pose": 0.3,
  "joint_vel_l2": -1e-3,
  "joint_speed_limit": -0.3,
  "torque_saturation": -1.5,
  "push_foot_idle": -0.5,
  "action_rate_l2": -0.05,
  "action_acc": -0.02,
  "joint_acc": -1e-6,
  "swing_height": -5.0,
  "board_fore_aft": 1.0,
  "board_creep": -1.0,
}

# The centred configs as they were before the split tasks came back, plus the
# balance torque penalty (Task 25) and the ride push weights (Task 28).
CENTRED_BALANCE_WEIGHTS = {
  "feet_on_deck": 1.0,
  "feet_on_ground": -1.0,
  "knees_on_deck": -1.0,
  "tilt": 1.0,
  "stance_pose": 1.0,
  "board_alignment": 1.0,
  "deck_flat": -5.0,
  "board_still": -2.0,
  "action_rate_l2": -0.01,
  "joint_torques_l2": -2e-4,
  "termination": -100.0,
  "torque_saturation": -1.0,
}

# The tripod ride (task 38) dropped knees_on_deck and added feet_slide, and
# charges push-foot taps.
CENTRED_RIDE_WEIGHTS = {
  "feet_on_deck": 0.5,
  "tilt": 1.0,
  "board_alignment": 1.0,
  "deck_flat": -5.0,
  "action_rate_l2": -0.05,
  "joint_torques_l2": -2e-4,
  "termination": -20.0,
  "reference_tracking": 1.0,
  "track_board_velocity": 3.0,
  "board_yaw_rate": -0.5,
  "foot_reach": 1.0,
  "push_stroke": 3.0,
  "short_stride": -3.0,
  "feet_slide": -2.0,
  "board_progress": 4.0,
  "action_acc": -0.02,
  "joint_acc": -1e-6,
  "joint_vel_l2": -1e-3,
  "joint_speed_limit": -0.1,
  "torque_saturation": -1.0,
}


def _weights(cfg) -> dict[str, float]:
  return {name: term.weight for name, term in cfg.rewards.items()}


def test_split_balance_rewards_match_a110b84_plus_torque_penalty():
  cfg = pupper_skate_split_balance_env_cfg()
  assert _weights(cfg) == pytest.approx(SPLIT_BALANCE_WEIGHTS)
  r = cfg.rewards
  assert r["tilt"].params["target_xy"] == pytest.approx(
    (0.0, -math.sin(SKATE_SPLIT_ROLL))
  )
  assert r["base_height"].params["target_height"] == SKATE_SPLIT_HEIGHT
  assert r["board_alignment"].params["target_offset"] == SKATE_SPLIT_BOARD_OFFSET


def test_split_ride_rewards_match_a110b84():
  cfg = pupper_skate_split_ride_env_cfg()
  assert _weights(cfg) == pytest.approx(SPLIT_RIDE_WEIGHTS)
  for name in ("push_stroke", "foot_reach"):
    assert name not in cfg.rewards
  stride = cfg.rewards["push_stride"]
  assert stride.func is mdp.push_stride
  assert stride.params == {
    "sensor_name": "ground_feet_contact",
    "stroke_m": 0.10,
    "stance_range": (0.25, 0.5),
    "air_range": (0.2, 0.5),
    "command_name": "twist",
    "command_threshold": 0.1,
  }
  assert cfg.rewards["joint_speed_limit"].params == {"limit": 5.0}
  short = cfg.rewards["short_stride"]
  assert short.func is mdp.short_stride
  assert short.params == {
    "sensor_name": "ground_feet_contact",
    "stroke_m": 0.10,
    "stance_range": (0.25, 0.5),
    "command_name": "twist",
  }
  assert cfg.rewards["action_rate_l2"].func is mdp.action_rate_l2
  assert cfg.rewards["action_acc"].func is mdp.action_acc_l2
  joint_acc = cfg.rewards["joint_acc"]
  assert joint_acc.func is mdp.joint_acc_l2
  assert joint_acc.params["asset_cfg"].name == "robot"
  assert joint_acc.params["asset_cfg"].joint_names == (".*",)
  swing = cfg.rewards["swing_height"]
  assert swing.func is mdp.foot_height_over
  assert {k: v for k, v in swing.params.items() if k != "foot_cfg"} == {
    "max_height": 0.03,
    "foot_radius": 0.02,
    "sensor_name": "ground_feet_contact",
  }
  assert swing.params["foot_cfg"].site_names == (
    "leg_front_r_3_foot_site",
    "leg_back_r_3_foot_site",
  )
  slide = cfg.rewards["feet_slide"]
  assert slide.func is mdp.feet_slide
  assert slide.params["sensor_name"] == "ground_feet_contact"
  assert slide.params["asset_cfg"].site_names == (
    "leg_front_r_3_foot_site",
    "leg_back_r_3_foot_site",
  )

  def matched(term: str) -> set[str]:
    patterns = cfg.rewards[term].params["asset_cfg"].joint_names
    return {j for j in JOINT_NAMES if any(re.fullmatch(p, j) for p in patterns)}

  # Firm on the deck-side legs, loose on the pushing legs.
  assert matched("stance_pose") == {j for j in JOINT_NAMES if "_l_" in j}
  assert matched("push_leg_pose") == {j for j in JOINT_NAMES if "_r_" in j}
  pose = cfg.rewards["push_leg_pose"]
  assert pose.func is mdp.stance_pose and pose.params["std"] == 0.6


def test_split_ride_commands_forward_speed_only():
  cfg = pupper_skate_split_ride_env_cfg()
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  assert twist.ranges.lin_vel_x == RIDE_SPEED_RANGE
  assert twist.ranges.lin_vel_y == (0.0, 0.0) and twist.ranges.ang_vel_z == (0.0, 0.0)
  assert "nudge_board" not in cfg.events
  assert "push_robot" in cfg.events


def test_split_balance_keeps_the_deck_height_range_and_the_rides_reach_higher():
  # Deck top 6.8 to 11.8 cm for the ride and steer, 6.8 to 11.3 for balance.
  assert SPLIT_RIDE_DECK_HEIGHT_RANGE == (-0.035, 0.015)
  for make_cfg, ranges in (
    (pupper_skate_split_balance_env_cfg, DECK_HEIGHT_RANGE),
    (pupper_skate_split_ride_env_cfg, SPLIT_RIDE_DECK_HEIGHT_RANGE),
    (pupper_skate_split_steer_env_cfg, SPLIT_RIDE_DECK_HEIGHT_RANGE),
  ):
    for play in (False, True):
      events = make_cfg(play=play).events
      assert events["deck_height"].mode == "startup"
      assert events["deck_height"].params["ranges"] == ranges
      assert ("push_robot" in events) is not play


def test_split_terminations_match_a110b84():
  terms = pupper_skate_split_balance_env_cfg().terminations
  assert terms["fell_over"].params["limit_angle"] == 0.6
  assert terms["feet_off_deck"].func is mdp.contact_lost
  assert terms["feet_off_deck"].params == {
    "sensor_name": "deck_feet_contact",
    "grace_s": 0.5,
  }
  assert terms["base_too_low"].params["minimum_height"] == 0.08
  assert terms["board_tipped"].params["limit_angle"] == 0.3
  ride = pupper_skate_split_ride_env_cfg().terminations
  assert ride["feet_off_deck"].params["grace_s"] == 0.5


_DECK_FEET_GEOMS = ("leg_front_l_3_collision", "leg_back_l_3_collision")
_FLOOR_FEET_GEOMS = ("leg_front_r_3_collision", "leg_back_r_3_collision")


def _matched_geoms(term) -> set[str]:
  pattern = term.params["asset_cfg"].geom_names
  patterns = [pattern] if isinstance(pattern, str) else list(pattern)
  feet = _DECK_FEET_GEOMS + _FLOOR_FEET_GEOMS
  return {g for g in feet if any(re.fullmatch(p, g) for p in patterns)}


@pytest.mark.parametrize("play", [False, True])
def test_split_ride_trains_for_long_rides_grip_drag_and_placement(play):
  cfg = pupper_skate_split_ride_env_cfg(play=play)
  # 20 s episodes in training; play keeps the base config's unlimited ones.
  if play:
    assert cfg.episode_length_s >= 20.0
  else:
    assert cfg.episode_length_s == 20.0
  events = cfg.events
  # Grip per side: two draws that split the four feet between them.
  assert "foot_friction" not in events
  deck, floor = events["deck_foot_friction"], events["floor_foot_friction"]
  for term in (deck, floor):
    assert term.func is dr.geom_friction
    assert term.mode == "startup"
    assert term.params["operation"] == "abs"
    assert term.params["shared_random"] is True
  assert deck.params["ranges"] == (0.5, 1.4)
  assert floor.params["ranges"] == (0.5, 1.6)
  assert _matched_geoms(deck) == set(_DECK_FEET_GEOMS)
  assert _matched_geoms(floor) == set(_FLOOR_FEET_GEOMS)
  # The deck geom's own friction, which reaches only body, knee and shin
  # contacts (the feet take their own).
  body = events["deck_friction"]
  assert body.func is dr.geom_friction and body.mode == "startup"
  assert body.params["operation"] == "abs"
  assert body.params["ranges"] == (0.3, 1.2)
  assert body.params["asset_cfg"].name == "board"
  assert body.params["asset_cfg"].geom_names == (DECK_GEOM_NAME,)
  # Wheel drag: one factor per env for all four wheels' bearings, log-uniform
  # from hard floor to carpet.
  drag = events["wheel_drag"]
  assert drag.func is mdp.scale_joint_drag and drag.mode == "startup"
  assert drag.params["scale_range"] == (0.5, 10.0)
  assert drag.params["log_uniform"] is True
  assert drag.params["asset_cfg"].name == "board"
  assert drag.params["asset_cfg"].joint_names == (r"wheel_.*_spin",)
  # Placement and shoves.
  assert events["reset_board"].params["pose_range"] == {
    "x": (-0.03, 0.03),
    "y": (-0.02, 0.02),
    "yaw": (-0.15, 0.15),
  }
  if play:
    assert "push_robot" not in events
  else:
    assert events["push_robot"].params["velocity_range"] == {
      "x": (-0.35, 0.35),
      "y": (-0.35, 0.35),
    }
  # Deck feet must not slide on the deck. The weight was the floor feet's
  # (-2); at that it cost about 1e-3 a step, so it is ten times that now.
  slide = cfg.rewards["deck_feet_slide"]
  assert slide.func is mdp.deck_feet_slide
  assert slide.weight == DECK_FEET_SLIDE_WEIGHT == -20.0
  assert cfg.rewards["feet_slide"].weight == -2.0
  assert slide.params["sensor_name"] == "deck_feet_contact"
  assert slide.params["asset_cfg"].site_names == (
    "leg_front_l_3_foot_site",
    "leg_back_l_3_foot_site",
  )
  assert slide.params["deck_cfg"].body_names == ("deck",)
  alignment = cfg.rewards["board_alignment"]
  assert alignment.weight == 2.0
  assert alignment.params == {
    "target_offset": SKATE_SPLIT_BOARD_OFFSET,
    "std_pos": 0.03,
    "std_yaw": 0.1,
  }
  drifted = cfg.terminations["board_drifted"]
  assert drifted.func is mdp.board_drifted
  assert drifted.params == {"max_drift": 0.06}
  assert not drifted.time_out
  slid = cfg.terminations["board_slid"]
  assert slid.func is mdp.board_slid
  assert slid.params == {"max_slide": 0.08, "max_slide_abs": 0.12}
  assert BOARD_MAX_SLIDE == 0.08 and BOARD_MAX_SLIDE_ABS == 0.12
  assert not slid.time_out
  fore_aft = cfg.rewards["board_fore_aft"]
  assert fore_aft.func is mdp.board_fore_aft and fore_aft.weight == 1.0
  assert fore_aft.params == {"std": 0.03} and BOARD_SLIDE_STD == 0.03
  creep = cfg.rewards["board_creep"]
  assert creep.func is mdp.board_creep and creep.weight == -1.0
  assert creep.params == {"tau": 1.0, "rate_cap": 0.02}
  assert BOARD_CREEP_TAU == 1.0 and BOARD_CREEP_CAP == 0.02
  # The termination penalty stays where it was.
  assert cfg.rewards["termination"].weight == -100.0


def test_carpet_constants():
  assert WHEEL_DRAG_SCALE_RANGE == (0.5, 10.0)
  assert FLOOR_FOOT_FRICTION_RANGE == (0.5, 1.6)
  assert DECK_FOOT_FRICTION_RANGE == (0.5, 1.4)
  assert DECK_BODY_FRICTION_RANGE == (0.3, 1.2)
  assert SPLIT_RIDE_DECK_HEIGHT_RANGE == (-0.035, 0.015)


@pytest.mark.parametrize(
  "make_cfg", [pupper_skate_split_ride_env_cfg, pupper_skate_split_steer_env_cfg]
)
@pytest.mark.parametrize("play", [False, True])
def test_split_ride_and_steer_train_on_carpet(make_cfg, play):
  events = make_cfg(play=play).events
  assert events["wheel_drag"].params["scale_range"] == WHEEL_DRAG_SCALE_RANGE
  assert events["wheel_drag"].params["log_uniform"] is True
  assert events["floor_foot_friction"].params["ranges"] == FLOOR_FOOT_FRICTION_RANGE
  assert events["deck_foot_friction"].params["ranges"] == DECK_FOOT_FRICTION_RANGE
  assert events["deck_friction"].params["ranges"] == DECK_BODY_FRICTION_RANGE
  assert events["deck_height"].params["ranges"] == SPLIT_RIDE_DECK_HEIGHT_RANGE


PROTECTED_TASKS = (
  pupper_skate_balance_env_cfg,
  pupper_skate_ride_env_cfg,
  pupper_skate_split_balance_env_cfg,
)


@pytest.mark.parametrize("make_cfg", PROTECTED_TASKS)
def test_other_skate_tasks_have_none_of_the_long_ride_changes(make_cfg):
  for play in (False, True):
    cfg = make_cfg(play=play)
    assert "deck_feet_slide" not in cfg.rewards
    assert "board_drifted" not in cfg.terminations
    assert "board_slid" not in cfg.terminations
    assert "board_fore_aft" not in cfg.rewards
    assert "board_creep" not in cfg.rewards
    for name in ("wheel_drag", "deck_foot_friction", "floor_foot_friction"):
      assert name not in cfg.events, name
    events = cfg.events
    assert events["foot_friction"].params["ranges"] == (0.6, 1.4)
    assert _matched_geoms(events["foot_friction"]) == set(
      _DECK_FEET_GEOMS + _FLOOR_FEET_GEOMS
    )
    assert events["deck_friction"].params["ranges"] == (0.6, 1.2)
    assert events["deck_height"].params["ranges"] == (
      RIDE_DECK_HEIGHT_RANGE
      if make_cfg is pupper_skate_ride_env_cfg
      else DECK_HEIGHT_RANGE
    )
    assert events["reset_board"].params["pose_range"] == {
      "x": (-0.03, 0.03),
      "yaw": (-0.05, 0.05),
    }
    if play:
      assert cfg.episode_length_s == int(1e9)
    else:
      assert cfg.episode_length_s == 10.0
      assert events["push_robot"].params["velocity_range"] == {
        "x": (-0.2, 0.2),
        "y": (-0.2, 0.2),
      }
    alignment = cfg.rewards["board_alignment"]
    assert alignment.weight == 1.0
    expected_std = 0.08 if make_cfg is pupper_skate_ride_env_cfg else 0.05
    assert alignment.params["std_pos"] == expected_std


@pytest.mark.parametrize(
  "make_cfg", [pupper_skate_balance_env_cfg, pupper_skate_split_balance_env_cfg]
)
def test_balance_penalizes_torque_above_half_the_limit(make_cfg):
  term = make_cfg().rewards["torque_saturation"]
  assert term.func is mdp.torque_saturation
  assert term.weight == -1.0
  assert term.params == {"limit": PUPPER_EFFORT_LIMIT, "soft_ratio": 0.5}
  assert BALANCE_TORQUE_SOFT_RATIO == 0.5


@pytest.mark.parametrize(
  "make_cfg, weight",
  [(pupper_skate_ride_env_cfg, -1.0), (pupper_skate_split_ride_env_cfg, -1.5)],
)
def test_ride_keeps_its_own_torque_penalty(make_cfg, weight):
  term = make_cfg().rewards["torque_saturation"]
  assert term.func is mdp.torque_saturation
  assert term.weight == weight
  assert term.params["limit"] == PUPPER_EFFORT_LIMIT
  assert term.params.get("soft_ratio", 0.8) == 0.8


def test_centred_cfgs_are_unchanged():
  assert _weights(pupper_skate_balance_env_cfg()) == pytest.approx(
    CENTRED_BALANCE_WEIGHTS
  )
  assert _weights(pupper_skate_ride_env_cfg()) == pytest.approx(CENTRED_RIDE_WEIGHTS)


@pytest.mark.parametrize(
  "make_cfg",
  [
    pupper_skate_split_balance_env_cfg,
    pupper_skate_balance_env_cfg,
    pupper_skate_ride_env_cfg,
  ],
)
def test_only_the_rides_charge_taps_and_sliding_feet(make_cfg):
  rewards = make_cfg().rewards
  assert "push_foot_idle" not in rewards
  # The tripod ride charges its push foot's taps and sliding on the floor;
  # balance does neither.
  if make_cfg is not pupper_skate_ride_env_cfg:
    assert "short_stride" not in rewards
    assert "feet_slide" not in rewards


@pytest.mark.parametrize(
  "make_cfg",
  [pupper_skate_split_balance_env_cfg, pupper_skate_balance_env_cfg],
)
def test_balance_has_no_smoothing_terms(make_cfg):
  rewards = make_cfg().rewards
  for name in ("action_acc", "joint_acc", "swing_height"):
    assert name not in rewards, name
  assert rewards["action_rate_l2"].weight == -0.01


def test_both_rides_share_the_smoothing_terms():
  centred, split = pupper_skate_ride_env_cfg(), pupper_skate_split_ride_env_cfg()
  for name in ("action_rate_l2", "action_acc", "joint_acc"):
    assert centred.rewards[name].func is split.rewards[name].func, name
    assert centred.rewards[name].weight == split.rewards[name].weight, name
  assert "swing_height" not in centred.rewards


def test_split_ride_charges_an_idle_push_foot():
  cfg = pupper_skate_split_ride_env_cfg()
  term = cfg.rewards["push_foot_idle"]
  assert term.func is mdp.foot_idle_air
  assert term.weight == -0.5
  assert term.params == {"sensor_name": "ground_feet_contact", "max_air_s": 0.5}
  assert cfg.rewards["joint_speed_limit"].weight == -0.3
  assert cfg.rewards["torque_saturation"].weight == -1.5


##
# Registration: both pairs of tasks build.
##


def test_split_tasks_are_registered():
  tasks = list_tasks()
  for task in SPLIT_TASKS + CENTRED_TASKS:
    assert task in tasks


@pytest.mark.slow
@pytest.mark.parametrize("task", SPLIT_TASKS + CENTRED_TASKS)
def test_registered_skate_tasks_build(task):
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  obs, _ = env.reset()
  _, reward, _, _, _ = env.step(torch.zeros(1, 12, device=env.device))
  actor = obs["actor"]
  assert isinstance(actor, torch.Tensor) and actor.shape == (1, 48 * 20)
  assert torch.isfinite(reward).all()
  env.close()


##
# Contact options: strict friction on the split tasks only.
##

_SPLIT_CFGS = (
  pupper_skate_split_balance_env_cfg,
  pupper_skate_split_ride_env_cfg,
  pupper_skate_split_steer_env_cfg,
)


def test_split_contact_constants():
  assert SPLIT_CONTACT_CONE == "elliptic"
  assert SPLIT_CONTACT_IMPRATIO == 10.0


@pytest.mark.parametrize("make_cfg", _SPLIT_CFGS)
def test_split_tasks_use_the_elliptic_cone(make_cfg):
  opts = make_cfg().sim.mujoco
  assert opts.cone == "elliptic"
  assert opts.impratio == 10.0


@pytest.mark.parametrize("make_cfg", _SPLIT_CFGS)
def test_split_tasks_keep_the_other_mujoco_options(make_cfg):
  got = dataclasses.asdict(make_cfg().sim.mujoco)
  want = dataclasses.asdict(MujocoCfg(**PUPPER_MUJOCO_OPTIONS))
  for key in ("cone", "impratio"):
    got.pop(key)
    want.pop(key)
  assert got == want


@pytest.mark.parametrize(
  "make_cfg", [pupper_skate_balance_env_cfg, pupper_skate_ride_env_cfg]
)
def test_centred_skate_tasks_keep_the_default_contacts(make_cfg):
  assert make_cfg().sim.mujoco == MujocoCfg(**PUPPER_MUJOCO_OPTIONS)
  assert make_cfg().sim.mujoco.cone == "pyramidal"


def test_walking_task_keeps_the_default_contacts():
  opts = pupper_velocity_fs_env_cfg().sim.mujoco
  assert opts.cone == PUPPER_MUJOCO_OPTIONS["cone"] == "pyramidal"
  assert opts.impratio == PUPPER_MUJOCO_OPTIONS["impratio"]


def test_foot_sites_follow_the_ground_sensor_feet():
  cfg = pupper_skate_split_ride_env_cfg()
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  sensor = env.scene["ground_feet_contact"]
  feet = [n.split("/")[-1] for n in sensor.primary_names]
  sites = {}
  for term_name, param in (("feet_slide", "asset_cfg"), ("swing_height", "foot_cfg")):
    site_ids = env.reward_manager.get_term_cfg(term_name).params[param].site_ids
    sites[term_name] = [env.scene["robot"].site_names[i] for i in site_ids]
  env.close()
  assert feet == ["leg_front_r_3_collision", "leg_back_r_3_collision"]
  for term_name, names in sites.items():
    assert [s.replace("_foot_site", "") for s in names] == [
      f.replace("_collision", "") for f in feet
    ], term_name


##
# The split balance env, stepped on the real simulator.
##


def _calm_split_cfg(num_envs: int, spawn_noise: bool, deck_offset: float = 0.0):
  """No shoves, a fixed deck height offset, and episodes longer than the test."""
  cfg = pupper_skate_split_balance_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 20.0
  cfg.events["deck_height"].params["ranges"] = (deck_offset, deck_offset)
  for name in ("push_robot", "nudge_board"):
    del cfg.events[name]
  if not spawn_noise:
    for name in ("reset_base", "reset_board"):
      cfg.events[name].params["pose_range"] = {}
  return cfg


@pytest.fixture(scope="module")
def split_env():
  env = ManagerBasedRlEnv(
    cfg=_calm_split_cfg(NUM_ENVS, spawn_noise=False), device=get_test_device()
  )
  yield env
  env.close()


def test_split_sensors_split_feet_by_side(split_env):
  def names(sensor: str) -> list[str]:
    return [n.split("/")[-1] for n in split_env.scene[sensor].primary_names]

  left = ["leg_front_l_3_collision", "leg_back_l_3_collision"]
  right = ["leg_front_r_3_collision", "leg_back_r_3_collision"]
  assert names("deck_feet_contact") == left
  assert names("deck_feet_on_ground") == left
  assert names("ground_feet_contact") == right
  assert names("ground_feet_on_deck") == right
  assert len(names("knees_on_deck")) == 4
  # 7 board pose + 3 + 3 board velocity + 2 deck roll + 2 + 2 foot contacts.
  stock = pupper_velocity_fs_env_cfg().observations["critic"].terms
  assert len(split_env.cfg.observations["critic"].terms) == len(stock) + 6


@pytest.mark.slow
def test_split_stance_holds_with_zero_actions(split_env):
  env = split_env
  env.reset()
  robot, board = env.scene["robot"], env.scene["board"]
  actions = torch.zeros(NUM_ENVS, 12, device=env.device)
  for _ in range(150):  # 3 s.
    _, reward, terminated, truncated, _ = env.step(actions)
    assert torch.isfinite(reward).all()
    assert not (terminated | truncated).any()
  assert (mdp.contact_count(env, "deck_feet_contact") == 2).all()
  assert (mdp.contact_count(env, "ground_feet_contact") == 2).all()
  for wrong in ("deck_feet_on_ground", "ground_feet_on_deck", "knees_on_deck"):
    assert (mdp.contact_count(env, wrong) == 0).all()
  roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
  assert torch.allclose(roll, torch.full_like(roll, SKATE_SPLIT_ROLL), atol=0.05)
  assert pitch.abs().max() < 0.08
  height = robot.data.root_link_pos_w[:, 2]
  assert torch.allclose(height, torch.full_like(height, SKATE_SPLIT_HEIGHT), atol=0.015)
  assert board.data.root_link_lin_vel_w[:, :2].norm(dim=1).max() < 0.02
  assert board.data.joint_pos[:, 0].abs().max() < 0.05
  alignment = mdp.board_alignment(
    env, SKATE_SPLIT_BOARD_OFFSET, std_pos=0.05, std_yaw=0.3
  )
  assert alignment.min() > 0.3


@pytest.mark.slow
# +1.5 cm is the top of the split ride's deck height range.
@pytest.mark.parametrize("deck_offset", [-0.035, 0.0, 0.01, 0.015])
def test_split_stance_holds_for_10_s_across_deck_heights(deck_offset):
  cfg = _calm_split_cfg(1, spawn_noise=False, deck_offset=deck_offset)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  actions = torch.zeros(1, 12, device=env.device)
  for _ in range(500):  # 10 s.
    _, _, terminated, truncated, _ = env.step(actions)
    assert not (terminated | truncated).any()
  env.close()


@pytest.mark.slow
def test_split_stance_holds_without_torque_saturation():
  # The balance task pays for motors above half their limit; an honest stance
  # should not.
  env = ManagerBasedRlEnv(
    cfg=_calm_split_cfg(1, spawn_noise=False), device=get_test_device()
  )
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
def test_split_stance_survives_spawn_noise_with_zero_actions():
  num_envs = 64
  cfg = _calm_split_cfg(num_envs, spawn_noise=True)
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  actions = torch.zeros(num_envs, 12, device=env.device)
  reset = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  for _ in range(500):  # 10 s.
    _, _, terminated, truncated, _ = env.step(actions)
    reset |= terminated | truncated
  env.close()
  assert reset.float().mean() <= 0.2


@pytest.mark.slow
def test_leaving_the_deck_ends_the_split_episode(split_env):
  env = split_env
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
# The split ride env: long rides, grip per side, wheel drag, board drift.
##


def _calm_ride_cfg(num_envs: int, deck_offset: float = 0.0):
  """The split ride with no shoves, no spawn noise and a fixed deck height."""
  cfg = pupper_skate_split_ride_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  cfg.events["deck_height"].params["ranges"] = (deck_offset, deck_offset)
  del cfg.events["push_robot"]
  for name in ("reset_base", "reset_board"):
    cfg.events[name].params["pose_range"] = {}
  return cfg


@pytest.fixture(scope="module")
def ride_env():
  cfg = pupper_skate_split_ride_env_cfg()
  cfg.scene.num_envs = 8
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()


@pytest.mark.slow
def test_split_ride_observations_and_actions_keep_their_size(ride_env):
  obs = ride_env.observation_manager.compute()
  # The stock 48-dim actor frame with 20 steps of history, and the critic as
  # it was before Task 42 (no new observation terms).
  assert obs["actor"].shape == (8, 48 * 20)
  assert obs["critic"].shape == (8, 91)
  assert ride_env.action_manager.total_action_dim == 12


def _drag_env(num_envs: int) -> tuple[Any, SceneEntityCfg]:
  """A fake env for scale_joint_drag: 5 dofs, the 4 wheels at dofs 1 to 4."""
  damping = torch.tensor([7.0, 0.1, 0.2, 0.3, 0.4])
  friction = torch.tensor([9.0, 0.01, 0.02, 0.03, 0.04])
  defaults = {"dof_damping": damping, "dof_frictionloss": friction}
  model = SimpleNamespace(
    dof_damping=damping.repeat(num_envs, 1),
    dof_frictionloss=friction.repeat(num_envs, 1),
  )
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene={
      "board": SimpleNamespace(indexing=SimpleNamespace(joint_v_adr=torch.arange(5)))
    },
    sim=SimpleNamespace(model=model, get_default_field=defaults.__getitem__),
  )
  asset_cfg = SceneEntityCfg("board")
  asset_cfg.joint_ids = [1, 2, 3, 4]
  return env, asset_cfg


def _drag_factors(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
  """Each env's damping and frictionloss factors on the four wheels."""
  model = env.sim.model
  damping = model.dof_damping[:, 1:] / torch.tensor([0.1, 0.2, 0.3, 0.4])
  friction = model.dof_frictionloss[:, 1:] / torch.tensor([0.01, 0.02, 0.03, 0.04])
  return damping, friction


def test_scale_joint_drag_log_uniform_spans_hard_floor_to_carpet():
  env, asset_cfg = _drag_env(4096)
  torch.manual_seed(0)
  mdp.scale_joint_drag(env, None, (0.5, 10.0), asset_cfg, log_uniform=True)
  damping, friction = _drag_factors(env)
  # One factor per env for all four wheels and both fields.
  torch.testing.assert_close(damping, damping[:, :1].expand_as(damping))
  torch.testing.assert_close(friction, damping)
  factor = damping[:, 0]
  assert torch.all((factor >= 0.5 - 1e-5) & (factor <= 10.0 + 1e-5))
  # Uniform in the log: the median is the geometric mean, not the midpoint.
  median = factor.median().item()
  assert abs(median / math.sqrt(0.5 * 10.0) - 1.0) < 0.15
  # The other dof is untouched.
  assert torch.all(env.sim.model.dof_damping[:, 0] == 7.0)
  assert torch.all(env.sim.model.dof_frictionloss[:, 0] == 9.0)


def test_scale_joint_drag_uniform_is_unchanged_for_a_fixed_seed():
  env, asset_cfg = _drag_env(64)
  torch.manual_seed(3)
  mdp.scale_joint_drag(env, None, (0.5, 3.0), asset_cfg, log_uniform=False)
  damping, friction = _drag_factors(env)
  torch.manual_seed(3)
  expected = torch.rand(64, 1) * (3.0 - 0.5) + 0.5
  torch.testing.assert_close(damping, expected.expand(64, 4))
  torch.testing.assert_close(friction, damping)
  # And the default draw is the uniform one.
  env, asset_cfg = _drag_env(64)
  torch.manual_seed(3)
  mdp.scale_joint_drag(env, None, (0.5, 3.0), asset_cfg)
  torch.testing.assert_close(_drag_factors(env)[0], expected.expand(64, 4))


@pytest.mark.slow
def test_wheel_drag_shares_one_factor_per_env(ride_env):
  board = ride_env.scene["board"]
  model = ride_env.sim.model
  spin = [i for i, n in enumerate(board.joint_names) if n.endswith("_spin")]
  assert len(spin) == 4
  dofs = board.indexing.joint_v_adr[spin]
  params = SkateboardParams()
  damping = model.dof_damping[:, dofs] / params.wheel_damping
  friction = model.dof_frictionloss[:, dofs] / params.wheel_frictionloss
  # All four wheels and both fields share each env's factor.
  torch.testing.assert_close(damping, damping[:, :1].expand_as(damping))
  torch.testing.assert_close(friction, damping)
  factor = damping[:, 0]
  assert torch.all((factor >= 0.5) & (factor <= 10.0))
  assert factor.max() - factor.min() > 0.1


@pytest.mark.slow
def test_split_ride_draws_grip_per_side(ride_env):
  robot = ride_env.scene["robot"]
  mu = ride_env.sim.model.geom_friction[:, :, 0]

  def side(names: tuple[str, ...]) -> torch.Tensor:
    ids = [robot.indexing.geom_ids[robot.geom_names.index(n)].item() for n in names]
    return mu[:, ids]

  deck, floor = side(_DECK_FEET_GEOMS), side(_FLOOR_FEET_GEOMS)
  # One draw per side and env, each in its own range.
  torch.testing.assert_close(deck[:, 1], deck[:, 0])
  torch.testing.assert_close(floor[:, 1], floor[:, 0])
  assert torch.all((deck >= 0.5) & (deck <= 1.4))
  assert torch.all((floor >= 0.5) & (floor <= 1.6))
  assert not torch.allclose(deck[:, 0], floor[:, 0])
  # The deck geom draws its own friction for the body, knee and shin contacts.
  board = ride_env.scene["board"]
  deck_geom = board.indexing.geom_ids[board.geom_names.index(DECK_GEOM_NAME)].item()
  body = mu[:, deck_geom]
  assert torch.all((body >= 0.3) & (body <= 1.2))
  assert body.max() - body.min() > 0.05


@pytest.mark.slow
@pytest.mark.parametrize("deck_offset", SPLIT_RIDE_DECK_HEIGHT_RANGE)
def test_board_drifted_allows_placement_noise_and_ends_a_7_cm_drift(deck_offset):
  env = ManagerBasedRlEnv(cfg=_calm_ride_cfg(5, deck_offset), device=get_test_device())
  env.reset()
  actions = torch.zeros(5, 12, device=env.device)
  # The keyframe, held with zero actions for 1 s, never counts as drifted.
  for _ in range(50):
    env.step(actions)
    assert not env.termination_manager.get_term("board_drifted").any()
  env.reset()
  # Envs: keyframe; the two corners of the board's reset range (2 cm aside,
  # 0.15 rad and 3 cm along); the board 7 cm to either side.
  board = env.scene["board"]
  pose = board.data.root_link_pose_w.clone()
  zeros = torch.zeros(5, device=env.device)
  shift = torch.tensor(
    [[0.0, 0.0], [0.03, 0.02], [-0.03, -0.02], [0.0, 0.07], [0.0, -0.07]],
    device=env.device,
  )
  yaw = torch.tensor([0.0, 0.15, -0.15, 0.0, 0.0], device=env.device)
  pose[:, :2] += shift
  pose[:, 3:7] = quat_mul(pose[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))
  board.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  assert mdp.board_drifted(env, max_drift=0.06).tolist() == [
    False,
    False,
    False,
    True,
    True,
  ]
  # And through the termination manager, on the next step.
  env.step(actions)
  fired = env.termination_manager.get_term("board_drifted")
  assert fired.tolist() == [False, False, False, True, True]
  env.close()


def test_board_lateral_offset_is_in_the_board_heading_frame():
  # Board 0.146 m to the robot's left; turning the board 0.5 rad about its own
  # centre moves the offset along the board's left axis only by the cosine.
  board_pos = torch.tensor([[0.0, 0.146, 0.0], [0.0, 0.146, 0.0]])
  robot_pos = torch.zeros(2, 3)
  zeros = torch.zeros(2)
  quat = quat_from_euler_xyz(zeros, zeros, torch.tensor([0.0, 0.5]))
  lateral = mdp.board_lateral_offset(board_pos, quat, robot_pos)
  assert lateral.tolist() == pytest.approx([0.146, 0.146 * math.cos(0.5)])


def _fake_board_env(shift: list[list[float]], yaw: list[float]) -> Any:
  """The split keyframe with the board moved by ``shift`` (x, y) and turned
  ``yaw`` about its own centre, world frame. The robot stays at its default."""
  n = len(shift)
  robot_default = torch.zeros(n, 13)
  robot_default[:, 2] = SKATE_SPLIT_HEIGHT
  robot_default[:, 3] = 1.0
  board_default = torch.zeros(n, 13)
  board_default[:, :2] = torch.tensor(SKATE_SPLIT_BOARD_OFFSET)
  board_default[:, 3] = 1.0
  zeros = torch.zeros(n)
  board_pos = board_default[:, :3].clone()
  board_pos[:, :2] += torch.tensor(shift)
  board_quat = quat_from_euler_xyz(zeros, zeros, torch.tensor(yaw))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=robot_default[:, :3], default_root_state=robot_default
    )
  )
  board = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=board_pos,
      root_link_quat_w=board_quat,
      default_root_state=board_default,
    )
  )
  return SimpleNamespace(scene={"robot": robot, "board": board})


_STEP_DT = 0.02


def _moving_board_env(n: int) -> tuple[Any, Any]:
  """A fake env at the keyframe and a setter that puts the robot ``x`` [m]
  along the deck from the keyframe in each env."""
  env = _fake_board_env([[0.0, 0.0]] * n, [0.0] * n)
  env.num_envs, env.device, env.step_dt = n, "cpu", _STEP_DT
  board_pos = env.scene["board"].data.root_link_pos_w
  start = board_pos[:, 0].clone()

  def set_slide(x: list[float]) -> None:
    board_pos[:, 0] = start + torch.tensor(x)

  return env, set_slide


def _slid_term(env: Any) -> Any:
  return mdp.board_slid(cfg=None, env=env)  # type: ignore[arg-type]


def _slid(term: Any, env: Any) -> torch.Tensor:
  return term(env, max_slide=BOARD_MAX_SLIDE, max_slide_abs=BOARD_MAX_SLIDE_ABS)


def test_board_slid_ends_a_slide_beyond_8_cm_from_the_start_either_way():
  # Changed: the 8 cm now counts from where the episode started, not from the
  # keyframe. Here they start at the keyframe, so the numbers are the old ones.
  env, set_slide = _moving_board_env(5)
  term = _slid_term(env)
  assert not _slid(term, env).any()
  along = [0.0, 0.079, -0.079, 0.081, -0.081]
  set_slide(along)
  assert mdp.board_slide(env).tolist() == pytest.approx(along, abs=1e-6)
  assert _slid(term, env).tolist() == [False, False, False, True, True]
  # A slide along the deck does not trip the sideways check.
  assert not mdp.board_drifted(env, max_drift=0.06).any()


def test_board_slid_ignores_a_sideways_offset():
  env = _fake_board_env([[0.0, 0.07], [0.0, -0.07], [0.0, 0.2]], [0.0] * 3)
  env.num_envs, env.device = 3, "cpu"
  assert not _slid(_slid_term(env), env).any()
  assert mdp.board_slide(env).abs().max() < 1e-6
  assert mdp.board_drifted(env, max_drift=0.06).all()


def test_board_slid_never_fires_on_a_fresh_reset():
  # Every corner of the split ride's resets: board +-3 cm along, +-2 cm aside,
  # +-0.15 rad; robot +-1 cm both ways (moved here onto the board instead).
  corners = [
    (dx, dy, yaw)
    for dx in (-0.04, 0.04)
    for dy in (-0.03, 0.03)
    for yaw in (-0.15, 0.15)
  ]
  env = _fake_board_env([[dx, dy] for dx, dy, _ in corners], [y for *_, y in corners])
  env.num_envs, env.device = len(corners), "cpu"
  slide = mdp.board_slide(env)
  # The largest is 6.6 cm: 4 cm along plus the turn swinging the 0.146 m
  # sideways offset onto the board's forward axis. Under the 12 cm backstop,
  # and the first step only records the start.
  assert slide.abs().max().item() == pytest.approx(0.066, abs=5e-4)
  assert not _slid(_slid_term(env), env).any()


def test_a_robot_reset_5_cm_off_that_holds_still_is_not_slid_and_paid_in_full():
  env, set_slide = _moving_board_env(2)
  slid, fore_aft = _slid_term(env), _fore_aft_term(env)
  set_slide([0.05, -0.05])
  for _ in range(100):
    assert not _slid(slid, env).any()
    reward = fore_aft(env, std=BOARD_SLIDE_STD)
    assert reward.tolist() == pytest.approx([1.0, 1.0])
  # board_slide itself still reads from the keyframe.
  assert mdp.board_slide(env).tolist() == pytest.approx([0.05, -0.05], abs=1e-6)


def test_board_slid_fires_on_9_cm_from_the_start_wherever_it_started():
  # Env 0 starts at the keyframe and moves 9 cm; env 1 starts 5 cm back and
  # moves 9 cm forward, only 4 cm from the keyframe; env 2 starts 5 cm back
  # and moves 7 cm.
  env, set_slide = _moving_board_env(3)
  term = _slid_term(env)
  set_slide([0.0, -0.05, -0.05])
  assert not _slid(term, env).any()
  set_slide([0.09, 0.04, 0.02])
  assert _slid(term, env).tolist() == [True, True, False]


def test_a_robot_5_cm_off_that_moves_on_hits_the_12_cm_backstop_first():
  # Starts 5 cm off and moves 8 cm more the same way, 1 mm a step. The
  # absolute 12 cm from the keyframe comes at 7 cm moved, before the 8 cm
  # from the start would, so the backstop is what ends it.
  for sign in (1.0, -1.0):
    env, set_slide = _moving_board_env(1)
    term = _slid_term(env)
    fired_at = None
    for mm in range(81):
      set_slide([sign * (0.05 + mm * 1e-3)])
      if _slid(term, env).item():
        fired_at = mm
        break
    # 12 cm from the keyframe, 7 cm from the start (70 or 71 by rounding).
    assert fired_at in (70, 71)
    assert fired_at * 1e-3 < BOARD_MAX_SLIDE


def test_board_slid_start_resets_per_env():
  env, set_slide = _moving_board_env(3)
  term = _slid_term(env)
  set_slide([0.0, 0.0, 0.0])
  _slid(term, env)
  set_slide([0.07, 0.07, 0.07])
  assert not _slid(term, env).any()
  # Env 1 resets; on its next step it stands 4 cm further and that is its
  # new start. The others still count from the keyframe.
  term.reset(env_ids=torch.tensor([1]))
  set_slide([0.11, 0.11, 0.11])
  assert _slid(term, env).tolist() == [True, False, True]
  set_slide([0.11, 0.18, 0.11])
  assert _slid(term, env)[1].item()  # 7 cm on, but 18 cm from the keyframe.
  # A full reset clears every env: 10 cm from the keyframe is a new start.
  term.reset(env_ids=None)
  set_slide([0.1, 0.1, 0.1])
  assert not _slid(term, env).any()


def _fore_aft_term(env: Any) -> Any:
  return mdp.board_fore_aft(cfg=None, env=env)  # type: ignore[arg-type]


def test_board_fore_aft_pays_1_at_the_start_and_falls_with_the_slide():
  # Changed: the slide counts from the episode's start, not the keyframe.
  along = [0.0, 0.03, -0.03, 0.06, 0.08]
  env, set_slide = _moving_board_env(5)
  term = _fore_aft_term(env)
  set_slide([-0.04] * 5)
  assert term(env, std=BOARD_SLIDE_STD).tolist() == pytest.approx([1.0] * 5)
  set_slide([x - 0.04 for x in along])
  reward = term(env, std=BOARD_SLIDE_STD)
  expected = [math.exp(-((x / 0.03) ** 2)) for x in along]
  assert reward.tolist() == pytest.approx(expected, rel=1e-4)
  # The start resets per env.
  term.reset(env_ids=torch.tensor([3]))
  assert term(env, std=BOARD_SLIDE_STD)[3].item() == pytest.approx(1.0)
  # A sideways offset costs nothing here (board_alignment pays for that).
  env = _fake_board_env([[0.0, 0.05]], [0.0])
  env.num_envs, env.device = 1, "cpu"
  assert _fore_aft_term(env)(env, std=BOARD_SLIDE_STD).item() == pytest.approx(1.0)


def _creep_env(n: int) -> tuple[Any, Any, Any]:
  """A fake env at the keyframe, the board_creep term on it, and a setter that
  puts the robot ``x`` [m] along the deck from the keyframe in each env."""
  env, set_slide = _moving_board_env(n)
  return env, mdp.board_creep(cfg=None, env=env), set_slide  # type: ignore[arg-type]


def _creep(term: Any, env: Any) -> torch.Tensor:
  return term(env, tau=BOARD_CREEP_TAU, rate_cap=BOARD_CREEP_CAP)


def test_board_creep_is_0_for_a_robot_that_holds_its_place():
  env, term, set_slide = _creep_env(2)
  set_slide([0.0, 0.03])  # Held still, at the keyframe or 3 cm off it.
  for _ in range(200):
    assert _creep(term, env).abs().max().item() < 1e-6


def test_board_creep_averages_out_the_stride():
  # Env 0 rocks 1 cm either way at 2 Hz, a peak speed of 12.6 cm/s; env 1 creeps
  # steadily at that speed. Two averages leave about amplitude / (omega tau^2)
  # of the rocking (0.8 mm/s, 0.04 of the cap; one 2 s average left about
  # 0.25), swinging both ways around 0.
  env, term, set_slide = _creep_env(2)
  omega = 2 * math.pi * 2.0
  rocking, cost = [], torch.zeros(2)
  for i in range(1, 501):  # 10 s.
    t = i * _STEP_DT
    set_slide([0.01 * math.sin(omega * t), 0.01 * omega * t])
    cost = _creep(term, env)
    if t > 6.0:
      rocking.append(cost[0].item())
  assert max(rocking) < 0.08
  assert cost[1].item() == pytest.approx(1.0)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_board_creep_converges_to_the_creep_over_the_cap(sign):
  env, term, set_slide = _creep_env(1)
  cost = torch.zeros(1)
  for i in range(1, 501):  # 10 s, five tau.
    set_slide([sign * 0.005 * i * _STEP_DT])
    cost = _creep(term, env)
  assert cost.item() == pytest.approx(0.005 / 0.02, abs=0.005)


def test_board_creep_is_capped_at_1():
  env, term, set_slide = _creep_env(1)
  cost = torch.zeros(1)
  for i in range(1, 501):
    set_slide([0.1 * i * _STEP_DT])  # 10 cm/s, five times the cap.
    cost = _creep(term, env)
  assert cost.item() == pytest.approx(1.0)


def test_board_creep_reset_clears_only_that_env_and_ignores_the_jump():
  env, term, set_slide = _creep_env(3)
  before = torch.zeros(3)
  for i in range(1, 301):
    set_slide([0.005 * i * _STEP_DT] * 3)
    before = _creep(term, env)
  assert before.min().item() > 0.2
  x = 0.005 * 300 * _STEP_DT
  term.reset(env_ids=torch.tensor([1]))
  # The reset moved env 1's robot 5 cm along the deck; the others creep on.
  step = 0.005 * _STEP_DT
  set_slide([x + step, x - 0.05, x + step])
  after = _creep(term, env)
  assert after[1].item() == 0.0
  assert after[0].item() == pytest.approx(before[0].item(), abs=1e-3)
  assert after[2].item() == pytest.approx(before[2].item(), abs=1e-3)
  # From there env 1 starts from nothing: one step of creep costs a little.
  set_slide([x + 2 * step, x - 0.05 + step, x + 2 * step])
  # One step through both averages: (dt / tau)^2 of the rate.
  gain = _STEP_DT / BOARD_CREEP_TAU
  assert _creep(term, env)[1].item() == pytest.approx(
    gain**2 * 0.005 / BOARD_CREEP_CAP, rel=1e-3
  )
  # A full reset clears every env.
  term.reset(env_ids=None)
  set_slide([0.0, 0.0, 0.0])
  assert _creep(term, env).abs().max().item() == 0.0


def _side_creep_env(n: int) -> tuple[Any, Any, Any]:
  """A fake env at the keyframe, the board_side_creep term on it, and a setter
  that moves the board ``y`` [m] sideways from the keyframe in each env."""
  env, _ = _moving_board_env(n)
  board_pos = env.scene["board"].data.root_link_pos_w
  start = board_pos[:, 1].clone()

  def set_side(y: list[float]) -> None:
    board_pos[:, 1] = start + torch.tensor(y)

  term = mdp.board_side_creep(cfg=None, env=env)  # type: ignore[arg-type]
  return env, term, set_side


def test_board_side_creep_is_0_once_a_lean_is_held():
  # Env 0 holds the keyframe. Env 1 leans: the board moves 4 cm sideways in
  # 0.5 s, then stays there. Once both averages settle the lean is free.
  env, term, set_side = _side_creep_env(2)
  cost = torch.zeros(2)
  for i in range(1, 501):  # 10 s.
    t = i * _STEP_DT
    set_side([0.0, 0.04 * min(t / 0.5, 1.0)])
    cost = _creep(term, env)
    assert cost[0].item() == 0.0
  assert cost[1].item() < 0.01


def test_board_side_creep_charges_a_steady_slide_in_full():
  # At the cap, either way, and at 2.5 times it.
  env, term, set_side = _side_creep_env(3)
  cost = torch.zeros(3)
  for i in range(1, 501):  # 10 s.
    t = i * _STEP_DT
    set_side([BOARD_CREEP_CAP * t, -BOARD_CREEP_CAP * t, 0.05 * t])
    cost = _creep(term, env)
  torch.testing.assert_close(cost, torch.ones(3), atol=0.01, rtol=0.0)


def test_board_side_creep_averages_out_the_stride_sway():
  # The stride sways the board 1 cm either way at 2 Hz. Two averages leave
  # about amplitude / (omega tau^2) of it, 0.8 mm/s or 0.04 of the cap.
  env, term, set_side = _side_creep_env(1)
  omega = 2 * math.pi * 2.0
  sway = []
  for i in range(1, 501):  # 10 s.
    t = i * _STEP_DT
    set_side([0.01 * math.sin(omega * t)])
    cost = _creep(term, env)
    if t > 6.0:
      sway.append(cost[0].item())
  assert max(sway) < 0.15


def test_board_side_creep_reads_only_the_sideways_offset():
  # A slide along the deck is board_creep's; this term ignores it.
  env, term, _ = _side_creep_env(1)
  board_pos = env.scene["board"].data.root_link_pos_w
  start = board_pos[:, 0].clone()
  for i in range(1, 201):
    board_pos[:, 0] = start + 0.05 * i * _STEP_DT
    assert _creep(term, env).item() == 0.0


def test_board_side_creep_reset_clears_only_that_env():
  env, term, set_side = _side_creep_env(2)
  before = torch.zeros(2)
  for i in range(1, 301):
    set_side([0.005 * i * _STEP_DT] * 2)
    before = _creep(term, env)
  assert before.min().item() > 0.2
  y = 0.005 * 300 * _STEP_DT
  term.reset(env_ids=torch.tensor([1]))
  # The reset moved env 1's board 3 cm the other way; env 0 slides on.
  step = 0.005 * _STEP_DT
  set_side([y + step, y - 0.03])
  after = _creep(term, env)
  assert after[1].item() == 0.0
  assert after[0].item() == pytest.approx(before[0].item(), abs=1e-3)


def test_board_fore_aft_offset_is_in_the_board_heading_frame():
  # Board 0.146 m to the robot's left, turned 0.5 rad: the sideways offset
  # swings onto the board's forward axis by the sine.
  board_pos = torch.tensor([[0.0, 0.146, 0.0], [0.0, 0.146, 0.0]])
  robot_pos = torch.zeros(2, 3)
  zeros = torch.zeros(2)
  quat = quat_from_euler_xyz(zeros, zeros, torch.tensor([0.0, 0.5]))
  along = mdp.board_fore_aft_offset(board_pos, quat, robot_pos)
  assert along.tolist() == pytest.approx([0.0, 0.146 * math.sin(0.5)], abs=1e-7)


@pytest.mark.slow
def test_board_slid_through_the_termination_manager():
  env = ManagerBasedRlEnv(cfg=_calm_ride_cfg(4), device=get_test_device())
  env.reset()
  actions = torch.zeros(4, 12, device=env.device)
  for _ in range(50):  # The keyframe held for 1 s never counts as slid.
    env.step(actions)
    assert not env.termination_manager.get_term("board_slid").any()
  env.reset()
  # Changed: the start is recorded on the first step after the reset, so a
  # board moved 9 cm by the reset alone does not count. Only past the 12 cm
  # backstop does it end the episode. Envs: keyframe; 6 cm; 9 cm; 13 cm.
  board = env.scene["board"]
  pose = board.data.root_link_pose_w.clone()
  pose[:, 0] += torch.tensor([0.0, 0.06, 0.09, 0.13], device=env.device)
  board.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  env.step(actions)
  fired = env.termination_manager.get_term("board_slid")
  assert fired.tolist() == [False, False, False, True]
  env.reset()
  env.step(actions)  # Records the start at the keyframe.
  # Then the boards move: none; 6 cm along; 9 cm either way.
  pose = board.data.root_link_pose_w.clone()
  pose[:, 0] += torch.tensor([0.0, 0.06, 0.09, -0.09], device=env.device)
  board.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  env.step(actions)
  fired = env.termination_manager.get_term("board_slid")
  assert fired.tolist() == [False, False, True, True]
  assert not env.termination_manager.get_term("board_drifted").any()
  env.close()


@pytest.mark.slow
def test_board_creep_through_the_reward_manager():
  env = ManagerBasedRlEnv(cfg=_calm_ride_cfg(4), device=get_test_device())
  env.reset()
  actions = torch.zeros(4, 12, device=env.device)
  names = env.reward_manager.active_terms
  column = names.index("board_creep")
  for _ in range(20):
    env.step(actions)
  # A reset that moves env 1's board 6 cm along the deck pays 0 on the step
  # after it; reward_manager.reset runs before the scene is forwarded.
  env.reset(env_ids=torch.tensor([1], device=env.device))
  board = env.scene["board"]
  pose = board.data.root_link_pose_w.clone()
  pose[1, 0] += 0.06
  board.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  env.step(actions)
  step_cost = env.reward_manager._step_reward[:, column]
  assert step_cost[1].item() == 0.0
  assert torch.isfinite(step_cost).all()
  assert (step_cost <= 0.0).all() and (step_cost >= -1.0).all()
  env.close()


##
# The steer task: the split ride plus a carve command.
##


def _fake_yaw_env(command: list[list[float]], board_yaw_rate: list[float]) -> Any:
  """Enough of an env for track_board_yaw_rate: the board and the command."""
  ang_vel = torch.zeros(len(board_yaw_rate), 3)
  ang_vel[:, 2] = torch.tensor(board_yaw_rate)
  return SimpleNamespace(
    scene={"board": SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_w=ang_vel))},
    command_manager=SimpleNamespace(get_command=lambda name: torch.tensor(command)),
  )


def test_track_board_yaw_rate_peaks_at_the_command():
  # Rows: on the command, turning the other way, not turning, half the command.
  command = [[0.3, 0.0, 0.2]] * 4
  env = _fake_yaw_env(command, [0.2, -0.2, 0.0, 0.1])
  reward = mdp.track_board_yaw_rate(env, command_name="twist", std=0.15)
  assert reward[0] == pytest.approx(1.0)
  assert torch.all(reward[1:] < 1.0)
  assert reward[3] == pytest.approx(math.exp(-(0.1**2) / 0.15**2))
  assert reward[1] < reward[2] < reward[3]


def test_steer_constants():
  assert STEER_CARVE_COMMAND_RANGE == (0.1, 0.6)
  assert STEER_CARVE_GAIN == 0.1
  assert STEER_CARVE_THRESHOLD == 0.1
  assert STEER_HEADING_HOLD_CLIP == 0.1
  assert (STEER_STRAIGHT_SHARE, STEER_YAW_MIN_SPEED) == (0.4, 0.1)
  # The steer hold's clip, the carve threshold and the smallest carve are one
  # value, the hold's yaw threshold: |command| <= 0.1 is a correction, above
  # it the stick. The ride keeps the gait tasks' clip.
  twist = pupper_skate_split_steer_env_cfg().commands["twist"]
  assert isinstance(twist, mdp.SteerVelocityCommandCfg)
  hold = twist.heading_hold_yaw_threshold
  assert STEER_HEADING_HOLD_CLIP == STEER_CARVE_THRESHOLD == hold
  assert STEER_CARVE_COMMAND_RANGE[0] == STEER_CARVE_THRESHOLD
  assert GAIT_HEADING_HOLD_CLIP == 0.3
  # 9 to 18 N*m/rad on the 30 N*m/rad nominal truck.
  assert STEER_TRUCK_STIFFNESS_SCALE == (0.3, 0.6)
  assert SkateboardParams().roll_stiffness == 30.0


def test_carve_yaw_target_scales_only_carves():
  cmd = torch.tensor([0.0, 0.05, -0.1, 0.1, 0.2, -0.15, 0.5, -0.4, -0.6])
  target = mdp.carve_yaw_target(cmd, STEER_CARVE_THRESHOLD, STEER_CARVE_GAIN)
  # Hold-range commands (up to 0.1) pass through; carves, partial or full, are
  # scaled, sign kept.
  expected = torch.tensor([0.0, 0.05, -0.1, 0.1, 0.02, -0.015, 0.05, -0.04, -0.06])
  torch.testing.assert_close(target, expected)
  # No mapping by default.
  assert mdp.carve_yaw_target(cmd, None, STEER_CARVE_GAIN) is cmd


_CARVE = {"carve_threshold": STEER_CARVE_THRESHOLD, "carve_gain": STEER_CARVE_GAIN}


def test_track_board_yaw_rate_pays_only_carves_against_the_scaled_target():
  # Rows: a 0.5 carve with the board at its 0.05 target, at 0 and the wrong
  # way; a -0.4 carve on target; a partial 0.2 carve on its 0.02 target and at
  # the full carve's 0.06; hold corrections of 0.1 and 0 with the board on the
  # command, which pay nothing.
  wz = (0.5, 0.5, 0.5, -0.4, 0.2, 0.2, 0.1, 0.0)
  command = [[0.3, 0.0, w] for w in wz]
  env = _fake_yaw_env(command, [0.05, 0.0, -0.05, -0.04, 0.02, 0.06, 0.1, 0.0])
  reward = mdp.track_board_yaw_rate(env, command_name="twist", std=0.02, **_CARVE)
  expected = [1.0, math.exp(-(0.05**2) / 0.02**2), math.exp(-(0.1**2) / 0.02**2)]
  expected += [1.0, 1.0, math.exp(-(0.04**2) / 0.02**2), 0.0, 0.0]
  torch.testing.assert_close(reward, torch.tensor(expected))


def test_track_board_yaw_rate_pays_a_straight_board_when_asked():
  # Rows, (speed, yaw) command and board yaw rate: straight at 0.3 m/s with
  # the board not turning, pulling 0.013 rad/s either way; straight at the
  # 0.1 m/s minimum; a hold correction of 0.08, whose target is still 0;
  # standing, and under the minimum, which pay 0; a 0.5 carve on its 0.05
  # target and a partial 0.2 carve at 0.06, as before.
  rows = [
    ((0.3, 0.0), 0.0),
    ((0.3, 0.0), 0.013),
    ((0.3, 0.0), -0.013),
    ((0.1, 0.0), 0.0),
    ((0.3, 0.08), 0.0),
    ((0.0, 0.0), 0.0),
    ((0.05, 0.0), 0.0),
    ((0.3, 0.5), 0.05),
    ((0.3, 0.2), 0.06),
  ]
  command = [[vx, 0.0, wz] for (vx, wz), _ in rows]
  env = _fake_yaw_env(command, [rate for _, rate in rows])
  straight = mdp.track_board_yaw_rate(
    env,
    command_name="twist",
    std=0.02,
    straight_min_speed=STEER_YAW_MIN_SPEED,
    **_CARVE,
  )
  pull = math.exp(-((0.013 / 0.02) ** 2))
  expected = [1.0, pull, pull, 1.0, 1.0, 0.0, 0.0, 1.0]
  expected += [math.exp(-(0.04**2) / 0.02**2)]
  torch.testing.assert_close(straight, torch.tensor(expected), atol=5e-4, rtol=0.0)
  # Carves pay what they did; None is the old term, carves only.
  old = mdp.track_board_yaw_rate(env, command_name="twist", std=0.02, **_CARVE)
  assert torch.equal(straight[7:], old[7:])
  assert old[:7].abs().max().item() == 0.0
  unset = mdp.track_board_yaw_rate(
    env, command_name="twist", std=0.02, straight_min_speed=None, **_CARVE
  )
  assert torch.equal(unset, old)


def test_track_board_yaw_rate_answers_the_hold_outside_carves():
  # Rows, (speed, yaw) command and board yaw rate: the hold's largest 0.1 with
  # the board on its 0.03 target and not turning; a quiet 0 with the board
  # still; standing and under the minimum, which pay 0; a 0.5 carve on its 0.05
  # target and a partial 0.2 carve at 0.06.
  rows = [
    ((0.3, 0.1), 0.03),
    ((0.3, 0.1), 0.0),
    ((0.3, 0.0), 0.0),
    ((0.0, 0.0), 0.0),
    ((0.05, 0.1), 0.03),
    ((0.3, 0.5), 0.05),
    ((0.3, 0.2), 0.06),
  ]
  command = [[vx, 0.0, wz] for (vx, wz), _ in rows]
  env = _fake_yaw_env(command, [rate for _, rate in rows])
  std = 0.02

  def reward(**kwargs: Any) -> torch.Tensor:
    return mdp.track_board_yaw_rate(
      env, command_name="twist", std=std, **_CARVE, **kwargs
    )

  hold = reward(straight_min_speed=STEER_YAW_MIN_SPEED, straight_gain=0.3)
  assert hold[0].item() == pytest.approx(1.0)
  assert hold[1].item() == pytest.approx(math.exp(-((0.03 / std) ** 2)), abs=5e-4)
  assert hold[2].item() == pytest.approx(1.0)
  assert hold[3].item() == 0.0 and hold[4].item() == 0.0
  # Carves pay exactly what they do today.
  today = reward()
  assert torch.equal(hold[5:], today[5:])
  # A zero gain is the target-zero term (v8's); no minimum speed is today's.
  zero = reward(straight_min_speed=STEER_YAW_MIN_SPEED, straight_gain=0.0)
  v8 = reward(straight_min_speed=STEER_YAW_MIN_SPEED)
  assert torch.equal(zero, v8)
  assert zero[0].item() == pytest.approx(math.exp(-((0.03 / std) ** 2)), abs=5e-4)
  assert zero[1].item() == pytest.approx(1.0)
  assert torch.equal(reward(straight_min_speed=None, straight_gain=0.3), today)


def _fake_hold_env(rows: list[tuple[float, float, bool, float, float]]) -> Any:
  """Enough of an env for board_heading_hold, from rows of (speed command, yaw
  command, hold active, hold target, board heading)."""
  command = torch.tensor([[vx, 0.0, wz] for vx, wz, *_ in rows])
  active = torch.tensor([row[2] for row in rows])
  target = torch.tensor([row[3] for row in rows])
  heading = torch.tensor([row[4] for row in rows])
  term = SimpleNamespace(heading_hold_state=lambda: (active, target))
  return SimpleNamespace(
    scene={"board": SimpleNamespace(data=SimpleNamespace(heading_w=heading))},
    command_manager=SimpleNamespace(
      get_command=lambda name: command, get_term=lambda name: term
    ),
  )


def test_board_heading_hold_pays_the_board_on_the_hold_target():
  # Rows: the board on the target; 0.03 rad off either way; a 0.02 rad error
  # across the +-pi wrap; a carve (the hold lets go), a standing env (hold
  # off), and a hold-active env under the 0.1 m/s minimum, which pay 0.
  wrap = math.pi - 0.01
  rows = [
    (0.3, 0.0, True, 0.2, 0.2),
    (0.3, 0.05, True, 0.2, 0.23),
    (0.1, 0.0, True, 0.2, 0.17),
    (0.3, 0.0, True, wrap, -wrap),
    (0.3, 0.5, False, 0.2, 0.2),
    (0.0, 0.0, False, 0.2, 0.2),
    (0.06, 0.0, True, 0.2, 0.2),
  ]
  env = _fake_hold_env(rows)
  reward = mdp.board_heading_hold(
    env, command_name="twist", std=0.03, min_speed=STEER_YAW_MIN_SPEED
  )
  assert reward.shape == (len(rows),)
  assert reward[0].item() == pytest.approx(1.0)
  assert reward[1].item() == pytest.approx(math.exp(-1.0), abs=1e-3)
  assert reward[2].item() == pytest.approx(math.exp(-1.0), abs=1e-3)
  assert reward[3].item() == pytest.approx(math.exp(-((0.02 / 0.03) ** 2)), abs=1e-3)
  assert reward[4:].abs().max().item() == 0.0


def test_board_turn_progress_pays_only_carves_against_the_scaled_target():
  # Rows: half, all, more and the wrong way of a 0.5 carve's 0.05 target, half
  # of a -0.6 carve's -0.06, half of a partial 0.2 carve's 0.02, then hold
  # corrections of 0.1 and -0.1 with the board on the command.
  wz = (0.5, 0.5, 0.5, 0.5, -0.6, 0.2, 0.1, -0.1)
  command = [[0.3, 0.0, w] for w in wz]
  env = _fake_yaw_env(command, [0.025, 0.05, 0.08, -0.02, -0.03, 0.01, 0.1, -0.1])
  reward = mdp.board_turn_progress(
    env, command_name="twist", command_threshold=STEER_CARVE_THRESHOLD, **_CARVE
  )
  expected = torch.tensor([0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0])
  torch.testing.assert_close(reward, expected)


def test_board_yaw_rate_penalty_skips_carves_only_when_asked():
  # Carves of 0.5, -0.4 and a partial 0.2 pay nothing; a 0.1 hold correction
  # and a quiet 0 pay the ride's penalty.
  command = [[0.3, 0.0, w] for w in (0.5, -0.4, 0.2, 0.1, 0.0)]
  env = _fake_yaw_env(command, [0.05, -0.04, 0.02, 0.1, 0.02])
  plain = mdp.board_yaw_rate_l2(env)
  rates = torch.tensor([0.05, -0.04, 0.02, 0.1, 0.02])
  torch.testing.assert_close(plain, rates**2)
  gated = mdp.board_yaw_rate_l2(
    env, command_name="twist", carve_threshold=STEER_CARVE_THRESHOLD
  )
  torch.testing.assert_close(gated, torch.tensor([0.0, 0.0, 0.0, 0.01, 0.0004]))


def _fake_robot_yaw_env(command: list[list[float]], yaw_rate: list[float]) -> Any:
  """Enough of an env for track_yaw_velocity: the robot and the command."""
  ang_vel = torch.zeros(len(yaw_rate), 3)
  ang_vel[:, 2] = torch.tensor(yaw_rate)
  return SimpleNamespace(
    scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=ang_vel))},
    command_manager=SimpleNamespace(get_command=lambda name: torch.tensor(command)),
  )


def test_track_yaw_rate_asks_a_carve_for_the_board_rate():
  # A 0.5 carve with the body at 0.05 and at 0.5; a partial 0.2 carve with the
  # body at its 0.02 and at 0.2; a 0.1 hold correction on it.
  command = [[0.3, 0.0, w] for w in (0.5, 0.5, 0.2, 0.2, 0.1)]
  env = _fake_robot_yaw_env(command, [0.05, 0.5, 0.02, 0.2, 0.1])
  reward = mdp.track_yaw_velocity(env, sigma=0.05, command_name="twist", **_CARVE)
  expected = [1.0, math.exp(-(0.45**2) / 0.05), 1.0, math.exp(-(0.18**2) / 0.05)]
  expected = torch.tensor([*expected, 1.0])
  torch.testing.assert_close(reward, expected)


def test_new_yaw_parameters_leave_the_ride_rewards_unchanged():
  # The defaults are the old formulas, bit for bit, on carve-sized commands.
  command = [[0.3, 0.0, w] for w in (0.5, -0.4, 0.3, 0.0, 0.08)]
  rates = [0.05, -0.2, 0.1, 0.02, 0.5]
  board = _fake_yaw_env(command, rates)
  robot = _fake_robot_yaw_env(command, rates)
  cmd_wz, wz = torch.tensor(command)[:, 2], torch.tensor(rates)
  assert torch.equal(
    mdp.track_yaw_velocity(robot, sigma=0.05, command_name="twist"),
    torch.exp(-torch.square(cmd_wz - wz) / 0.05),
  )
  assert torch.equal(
    mdp.track_board_yaw_rate(board, command_name="twist", std=0.04),
    torch.exp(-torch.square(cmd_wz - wz) / 0.04**2),
  )
  assert torch.equal(mdp.board_yaw_rate_l2(board), torch.square(wz))


def test_only_the_steer_trains_on_loosened_trucks():
  for play in (False, True):
    steer = pupper_skate_split_steer_env_cfg(play=play)
    ride = pupper_skate_split_ride_env_cfg(play=play)
    balance = pupper_skate_split_balance_env_cfg(play=play)
    term = steer.events["truck_stiffness"]
    assert term.func is dr.joint_stiffness
    assert term.params["ranges"] == STEER_TRUCK_STIFFNESS_SCALE
    for cfg in (ride, balance):
      assert cfg.events["truck_stiffness"].params["ranges"] == (0.5, 6.0)
    assert "board_turn_progress" not in ride.rewards
    assert "board_turn_progress" not in balance.rewards


@pytest.mark.parametrize(
  "make_cfg",
  [
    pupper_skate_balance_env_cfg,
    pupper_skate_ride_env_cfg,
    pupper_skate_split_balance_env_cfg,
    pupper_skate_split_ride_env_cfg,
  ],
)
def test_only_the_steer_charges_side_creep_and_a_straight_pull(make_cfg):
  cfg = make_cfg()
  assert "board_side_creep" not in cfg.rewards
  for name, term in cfg.rewards.items():
    assert "straight_min_speed" not in term.params, name
    assert "straight_gain" not in term.params, name


def test_steer_pays_the_board_for_answering_the_hold():
  # v13: outside carves the board's yaw rate tracks HOLD_BOARD_GAIN times the
  # hold's command, and board alignment's yaw width is tighter than the ride's.
  # Every weight stays what it was.
  assert HOLD_BOARD_GAIN == 0.3
  assert STEER_BOARD_ALIGN_STD_YAW == 0.04
  for play in (False, True):
    steer = pupper_skate_split_steer_env_cfg(play=play)
    ride = pupper_skate_split_ride_env_cfg(play=play)
    yaw = steer.rewards["track_board_yaw_rate"]
    assert yaw.weight == 2.0 and yaw.params["std"] == 0.02
    assert yaw.params["straight_min_speed"] == STEER_YAW_MIN_SPEED
    assert yaw.params["straight_gain"] == 0.3
    assert steer.rewards["board_alignment"].params["std_yaw"] == 0.04
    assert ride.rewards["board_alignment"].params["std_yaw"] == 0.1
    assert "track_board_yaw_rate" not in ride.rewards


def test_steer_pays_the_board_for_holding_the_hold_heading():
  # v14: the board is paid for staying on the hold's target heading. At the
  # 1.5 degree rotation v13 measured, the term pays 0.47.
  assert BOARD_HEADING_HOLD_WEIGHT == 1.0
  assert BOARD_HEADING_HOLD_STD == 0.03
  rotated = math.radians(1.5) / BOARD_HEADING_HOLD_STD
  assert math.exp(-(rotated**2)) == pytest.approx(0.47, abs=0.005)
  for play in (False, True):
    steer = pupper_skate_split_steer_env_cfg(play=play)
    term = steer.rewards["board_heading_hold"]
    assert term.func is mdp.board_heading_hold
    assert term.weight == 1.0
    assert term.params == {
      "command_name": "twist",
      "std": 0.03,
      "min_speed": STEER_YAW_MIN_SPEED,
    }
    # Everything v13 set stays.
    assert HOLD_BOARD_GAIN == 0.3 and STEER_BOARD_ALIGN_STD_YAW == 0.04
    assert steer.rewards["board_side_creep"].weight == 0.0
    assert steer.rewards["deck_feet_slide"].weight == -2.0


@pytest.mark.parametrize(
  "make_cfg",
  [
    pupper_skate_balance_env_cfg,
    pupper_skate_ride_env_cfg,
    pupper_skate_split_balance_env_cfg,
    pupper_skate_split_ride_env_cfg,
  ],
)
def test_only_the_steer_pays_the_board_heading_hold(make_cfg):
  for play in (False, True):
    rewards = make_cfg(play=play).rewards
    assert "board_heading_hold" not in rewards
    assert all(t.func is not mdp.board_heading_hold for t in rewards.values())


def test_steer_trains_on_steer_v6s_rewards():
  # v9 resumes from steer v6 and changes only the carpet ranges, so the steer
  # task's weights are v6's (commit 9743be3): the deck-foot scuff at -2, not
  # v7's -20, and no sideways creep charge (v8's). The split ride keeps the -20
  # it has trained on since v7. v13's yaw-rate and alignment params are below.
  for play in (False, True):
    steer = pupper_skate_split_steer_env_cfg(play=play)
    ride = pupper_skate_split_ride_env_cfg(play=play)
    slide = steer.rewards["deck_feet_slide"]
    assert slide.weight == STEER_DECK_FEET_SLIDE_WEIGHT == -2.0
    assert slide.params == ride.rewards["deck_feet_slide"].params
    assert steer.rewards["board_side_creep"].weight == 0.0
    assert ride.rewards["deck_feet_slide"].weight == DECK_FEET_SLIDE_WEIGHT == -20.0
    assert _weights(ride) == pytest.approx(SPLIT_RIDE_WEIGHTS)
    # v7's slide from the episode start stays: the stricter check.
    slid = steer.terminations["board_slid"].params
    assert slid == {"max_slide": BOARD_MAX_SLIDE, "max_slide_abs": BOARD_MAX_SLIDE_ABS}


def test_steer_is_the_split_ride_plus_a_carve_command():
  for play in (False, True):
    ride = pupper_skate_split_ride_env_cfg(play=play)
    steer = pupper_skate_split_steer_env_cfg(play=play)
    expected = _weights(ride)
    expected["track_board_yaw_rate"] = 2.0
    expected["board_turn_progress"] = 1.0
    expected["board_side_creep"] = 0.0
    expected["deck_feet_slide"] = -2.0
    expected["board_heading_hold"] = 1.0
    assert _weights(steer) == pytest.approx(expected)
    term = steer.rewards["track_board_yaw_rate"]
    assert term.func is mdp.track_board_yaw_rate
    assert term.params == {
      "command_name": "twist",
      "std": 0.02,
      **_CARVE,
      "straight_min_speed": STEER_YAW_MIN_SPEED,
      "straight_gain": HOLD_BOARD_GAIN,
    }
    # The sideways twin of board_creep, at its rate and averages, off by default
    # (steer v8's experiment, turned back on from the notebook table).
    term = steer.rewards["board_side_creep"]
    assert term.func is mdp.board_side_creep
    assert term.weight == BOARD_SIDE_CREEP_WEIGHT == 0.0
    assert term.params == {"tau": BOARD_CREEP_TAU, "rate_cap": BOARD_CREEP_CAP}
    term = steer.rewards["board_turn_progress"]
    assert term.func is mdp.board_turn_progress
    assert term.params == {
      "command_name": "twist",
      "command_threshold": STEER_CARVE_THRESHOLD,
      **_CARVE,
    }
    # The ride's robot yaw term stays, with the carve mapping on.
    term = steer.rewards["track_yaw_rate"]
    assert term.func is mdp.track_yaw_velocity and term.weight == 1.0
    assert term.params == {"sigma": 0.05, "command_name": "twist", **_CARVE}
    # The ride's board yaw penalty stays, outside carves only.
    term = steer.rewards["board_yaw_rate"]
    assert term.func is mdp.board_yaw_rate_l2 and term.weight == -0.5
    assert term.params == {
      "command_name": "twist",
      "carve_threshold": STEER_CARVE_THRESHOLD,
    }
    assert ride.rewards["board_yaw_rate"].params == {}
    # Board alignment twice as wide sideways and tighter in yaw; the ride
    # keeps 3 cm and 0.1 rad.
    align = steer.rewards["board_alignment"].params
    ride_align = ride.rewards["board_alignment"].params
    assert align["std_pos"] == 0.06 and ride_align["std_pos"] == 0.03
    assert align["std_yaw"] == 0.04 and ride_align["std_yaw"] == 0.1
    assert {**align, "std_pos": 0.03, "std_yaw": 0.1} == ride_align
    changed = ("board_yaw_rate", "track_yaw_rate", "board_alignment")
    for name in ride.rewards:
      if name not in changed:
        assert steer.rewards[name].func is ride.rewards[name].func, name
        assert steer.rewards[name].params == ride.rewards[name].params, name
    assert steer.terminations.keys() == ride.terminations.keys()
    assert steer.terminations["board_drifted"].params == {"max_drift": 0.06}
    assert steer.terminations["board_slid"].func is mdp.board_slid
    assert steer.terminations["board_slid"].params == {
      "max_slide": 0.08,
      "max_slide_abs": 0.12,
    }
    assert steer.rewards["board_fore_aft"].params == {"std": 0.03}
    assert steer.rewards["board_creep"].func is mdp.board_creep
    assert steer.rewards["board_creep"].weight == -1.0
    assert steer.rewards["board_creep"].params == {"tau": 1.0, "rate_cap": 0.02}
    assert steer.events.keys() == ride.events.keys()
    assert steer.observations["critic"].terms.keys() == (
      ride.observations["critic"].terms.keys()
    )
    twist = steer.commands["twist"]
    assert isinstance(twist, mdp.SteerVelocityCommandCfg)
    assert isinstance(twist, mdp.BoardHeadingHoldCommandCfg)
    assert twist.carve_range == STEER_CARVE_COMMAND_RANGE
    assert twist.carve_left_share == 0.5
    assert (twist.rel_straight_envs, twist.yaw_min_speed) == (0.4, 0.1)
    # The yaw range is the export clip only: it covers the carves.
    assert twist.ranges.ang_vel_z == (-0.6, 0.6)
    # Every other command field is the split ride's, heading hold included,
    # except its clip.
    ride_twist = ride.commands["twist"]
    assert isinstance(ride_twist, mdp.BoardHeadingHoldCommandCfg)
    for field in dataclasses.fields(mdp.BoardHeadingHoldCommandCfg):
      if field.name not in ("ranges", "heading_hold_clip"):
        expected = getattr(ride_twist, field.name)
        assert getattr(twist, field.name) == expected, field.name
    assert twist.ranges == dataclasses.replace(ride_twist.ranges, ang_vel_z=(-0.6, 0.6))
    assert twist.ranges.lin_vel_x == RIDE_SPEED_RANGE
    assert twist.rel_forward_envs == 0.0 and twist.rel_turn_envs == 0.0
    assert twist.heading_hold_kp == GAIT_HEADING_HOLD_KP
    assert twist.heading_hold_clip == STEER_HEADING_HOLD_CLIP
    assert ride_twist.heading_hold_clip == GAIT_HEADING_HOLD_CLIP


@pytest.mark.parametrize(
  "make_cfg",
  [
    pupper_skate_balance_env_cfg,
    pupper_skate_ride_env_cfg,
    pupper_skate_split_balance_env_cfg,
    pupper_skate_split_ride_env_cfg,
  ],
)
def test_existing_skate_tasks_keep_a_zero_yaw_command(make_cfg):
  for play in (False, True):
    cfg = make_cfg(play=play)
    twist = cfg.commands["twist"]
    # The stock command class (the split ride's adds only the board heading
    # hold): the steer task's sampling rules are not in it.
    stock = (mdp.PupperVelocityCommandCfg, mdp.BoardHeadingHoldCommandCfg)
    assert type(twist) in stock
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    assert twist.rel_forward_envs == 0.0 and twist.rel_turn_envs == 0.0
    # Heading hold only in the split ride (its own tests below).
    if make_cfg is not pupper_skate_split_ride_env_cfg:
      assert twist.heading_hold_kp is None
    assert "track_board_yaw_rate" not in cfg.rewards


@pytest.mark.parametrize(
  "make_cfg", [pupper_skate_ride_env_cfg, pupper_skate_split_ride_env_cfg]
)
def test_ride_tasks_keep_their_yaw_rate_penalty(make_cfg):
  term = make_cfg().rewards["board_yaw_rate"]
  assert term.func is mdp.board_yaw_rate_l2 and term.weight == -0.5


def test_steer_task_is_registered():
  assert STEER_TASK in list_tasks()


@pytest.mark.slow
@pytest.mark.parametrize("play", [False, True])
def test_steer_task_builds_and_steps_with_zero_actions(play):
  cfg = load_env_cfg(STEER_TASK, play=play)
  cfg.scene.num_envs = 4
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  obs, _ = env.reset()
  actor = obs["actor"]
  assert isinstance(actor, torch.Tensor) and actor.shape == (4, 48 * 20)
  for _ in range(100):  # 2 s
    _, reward, _, _, _ = env.step(torch.zeros(4, 12, device=env.device))
    assert torch.isfinite(reward).all()
  command = env.command_manager.get_command("twist")
  assert command is not None
  # Each emitted yaw is the raw carve, or a hold correction within the steer
  # clip where the raw yaw is quiet.
  term = env.command_manager.get_term("twist")
  assert isinstance(term, mdp.SteerVelocityCommand)
  raw = term.vel_command_b[:, 2]
  carve = raw != 0.0
  torch.testing.assert_close(command[carve, 2], raw[carve])
  assert torch.all(command[~carve, 2].abs() <= STEER_HEADING_HOLD_CLIP + 1e-6)
  env.close()


class _Actor(torch.nn.Module):
  """Minimal MLPModel-like holder: has ``.mlp`` and ``.obs_normalizer``."""

  def __init__(self, in_dim: int):
    super().__init__()
    self.mlp = MLP(in_dim, 12, (64, 64), activation="elu")
    self.obs_normalizer = EmpiricalNormalization(in_dim)


@pytest.mark.slow
def test_steer_policy_exports_the_carve_clip_with_heading_hold_on():
  cfg = load_env_cfg(STEER_TASK, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  policy = export_pupper_gait_policy_from_env(_Actor(48 * 20).eval(), env)
  env.close()
  assert policy["command_clip"] == {
    "vx": list(RIDE_SPEED_RANGE),
    "vy": [0.0, 0.0],
    "wz": [-0.6, 0.6],
  }
  assert policy["heading_hold"] == {
    "kp": GAIT_HEADING_HOLD_KP,
    "clip": 0.1,
    "yaw_threshold": 0.1,
    "walk_threshold": 0.05,
  }
  assert "gait_reference" in policy


@pytest.fixture(scope="module")
def steer_env():
  cfg = pupper_skate_split_steer_env_cfg(play=True)
  cfg.scene.num_envs = 8
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()


@pytest.mark.slow
def test_board_turn_progress_pays_the_share_of_the_commanded_turn(steer_env):
  env = steer_env
  env.reset()
  board = env.scene["board"]
  # Rows: half, all and more of a 0.5 carve's 0.05 rad/s target, the wrong
  # way, a 0.1 hold correction (not a carve), then half, all and the wrong way
  # of a -0.4 carve's -0.04.
  cmd_wz = [0.5, 0.5, 0.5, 0.5, 0.1, -0.4, -0.4, -0.4]
  board_wz = [0.025, 0.05, 0.08, -0.02, 0.1, -0.02, -0.06, 0.02]
  vel = torch.zeros(env.num_envs, 6, device=env.device)
  vel[:, 5] = torch.tensor(board_wz, device=env.device)
  board.write_root_link_velocity_to_sim(vel)
  env.sim.forward()
  wz = board.data.root_link_ang_vel_w[:, 2]
  torch.testing.assert_close(wz, vel[:, 5], atol=1e-5, rtol=0)
  command = env.command_manager.get_command("twist")
  assert command is not None
  command[:, 2] = torch.tensor(cmd_wz, device=env.device)
  term = env.reward_manager.get_term_cfg("board_turn_progress")
  reward = term.func(env, **term.params)
  expected = torch.tensor([0.5, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0], device=env.device)
  torch.testing.assert_close(reward, expected, atol=1e-4, rtol=0)


##
# Split ride heading hold: the yaw command holds the board's line.
##


def test_split_ride_holds_heading_with_gait_gains():
  for play in (False, True):
    ride = pupper_skate_split_ride_env_cfg(play=play).commands["twist"]
    assert isinstance(ride, mdp.BoardHeadingHoldCommandCfg)
    assert ride.board_name == "board"
    assert ride.heading_hold_kp == GAIT_HEADING_HOLD_KP
    assert ride.heading_hold_clip == GAIT_HEADING_HOLD_CLIP
    balance = pupper_skate_split_balance_env_cfg(play=play).commands["twist"]
    assert isinstance(balance, mdp.PupperVelocityCommandCfg)
    assert balance.heading_hold_kp is None
    # The steer task keeps the ride's hold, so a quiet yaw is the ride.
    steer = pupper_skate_split_steer_env_cfg(play=play).commands["twist"]
    assert type(steer) is mdp.SteerVelocityCommandCfg
    assert isinstance(steer, mdp.BoardHeadingHoldCommandCfg)
    assert steer.heading_hold_kp == GAIT_HEADING_HOLD_KP
    # Its clip is the carve threshold, so a correction never looks like a carve.
    assert steer.heading_hold_clip == STEER_HEADING_HOLD_CLIP


def test_split_ride_pays_for_following_the_hold_yaw_command():
  for play in (False, True):
    term = pupper_skate_split_ride_env_cfg(play=play).rewards["track_yaw_rate"]
    assert term.func is mdp.track_yaw_velocity
    assert term.weight == 1.0
    assert term.params == {"sigma": 0.05, "command_name": "twist"}
    # The steer task keeps it, with carves mapped to the board's rate.
    steer = pupper_skate_split_steer_env_cfg(play=play).rewards["track_yaw_rate"]
    assert steer.func is mdp.track_yaw_velocity and steer.weight == 1.0


@pytest.mark.slow
def test_track_yaw_rate_peaks_at_the_commanded_yaw_rate(ride_env):
  env = ride_env
  env.reset()
  robot = env.scene["robot"]
  # Spin the base about world z, then command its own body-frame yaw rate in
  # the first half of the envs and 0.3 rad/s more in the second half.
  vel = torch.zeros(env.num_envs, 6, device=env.device)
  vel[:, 5] = 0.2
  robot.write_root_link_velocity_to_sim(vel)
  env.sim.forward()
  yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
  assert torch.all(yaw_rate.abs() > 0.1)
  half = env.num_envs // 2
  command = env.command_manager.get_command("twist")
  assert command is not None
  command[:half, 2] = yaw_rate[:half]
  command[half:, 2] = yaw_rate[half:] + 0.3
  term = env.reward_manager.get_term_cfg("track_yaw_rate")
  reward = term.func(env, **term.params)
  torch.testing.assert_close(reward[:half], torch.ones_like(reward[:half]))
  expected = torch.full_like(reward[half:], math.exp(-0.09 / 0.05))
  torch.testing.assert_close(reward[half:], expected)


def test_split_ride_alignment_yaw_width():
  ride = pupper_skate_split_ride_env_cfg().rewards["board_alignment"]
  balance = pupper_skate_split_balance_env_cfg().rewards["board_alignment"]
  assert ride.params["std_yaw"] == 0.1
  assert balance.params["std_yaw"] == 0.3


def test_split_heading_hold_exports_the_gait_gains_on_the_ride_and_steer():
  balance = SimpleNamespace(cfg=pupper_skate_split_balance_env_cfg(play=True))
  balance_block = heading_hold_metadata(balance)
  assert balance_block is not None and balance_block["kp"] == 0.0
  assert balance_block["clip"] == 0.3
  # The ride stamps the gait tasks' 0.3 clip, the steer its own 0.1; kp and
  # the thresholds are the same.
  blocks = {}
  for make_cfg, clip in (
    (pupper_skate_split_ride_env_cfg, 0.3),
    (pupper_skate_split_steer_env_cfg, 0.1),
  ):
    block = heading_hold_metadata(SimpleNamespace(cfg=make_cfg(play=True)))
    assert block is not None
    assert block["kp"] == GAIT_HEADING_HOLD_KP
    assert block["clip"] == clip
    blocks[clip] = block
  assert blocks[0.3] == {
    "kp": GAIT_HEADING_HOLD_KP,
    "clip": 0.3,
    "yaw_threshold": 0.1,
    "walk_threshold": 0.05,
  }
  assert {**blocks[0.1], "clip": 0.3} == blocks[0.3]


@pytest.fixture(scope="module")
def hold_env():
  """The split ride in play, two envs, a fixed walking command and no noise."""
  cfg = pupper_skate_split_ride_env_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.seed = 0
  cfg.events["deck_height"].params["ranges"] = (0.0, 0.0)
  cfg.events.pop("push_robot", None)
  for name in ("reset_base", "reset_board"):
    cfg.events[name].params["pose_range"] = {}
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.3, 0.3)
  twist.rel_standing_envs = 0.0
  twist.resampling_time_range = (1e6, 1e6)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()


@pytest.mark.slow
def test_split_ride_yaw_command_points_at_board_heading(hold_env):
  env = hold_env
  env.reset()
  term = env.command_manager.get_term("twist")
  assert isinstance(term, mdp.BoardHeadingHoldCommand)
  board = env.scene["board"]
  # The board yawed +0.1 rad from the robot in env 0 and -0.1 rad in env 1.
  pose = board.data.root_link_pose_w.clone()
  zeros = torch.zeros(2, device=env.device)
  yaw = torch.tensor([0.1, -0.1], device=env.device)
  pose[:, 3:7] = quat_mul(pose[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))
  board.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  # Re-arm the hold (reset already captured the board's old heading), so it
  # captures on the next step as at a fresh walk command.
  term._hh_prev_active[:] = False
  env.step(torch.zeros(2, 12, device=env.device))
  command = env.command_manager.get_command("twist")
  assert command is not None
  expected = GAIT_HEADING_HOLD_KP * yaw
  torch.testing.assert_close(command[:, 2], expected, atol=0.02, rtol=0)
  torch.testing.assert_close(command[:, 0], term.vel_command_b[:, 0])
  torch.testing.assert_close(command[:, 0], torch.full_like(zeros, 0.3))


@pytest.mark.slow
def test_split_ride_clips_the_hold_correction_at_0_3(hold_env):
  env = hold_env
  env.reset()
  term = env.command_manager.get_term("twist")
  assert isinstance(term, mdp.BoardHeadingHoldCommand)
  board = env.scene["board"]
  # The board yawed 0.5 rad from the robot: past the ride's 0.3 clip.
  pose = board.data.root_link_pose_w.clone()
  zeros = torch.zeros(2, device=env.device)
  yaw = torch.tensor([0.5, -0.5], device=env.device)
  pose[:, 3:7] = quat_mul(pose[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))
  board.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  term._hh_prev_active[:] = False
  env.step(torch.zeros(2, 12, device=env.device))
  command = env.command_manager.get_command("twist")
  assert command is not None
  expected = torch.tensor([0.3, -0.3], device=env.device)
  torch.testing.assert_close(command[:, 2], expected)


@pytest.mark.slow
def test_heading_hold_reference_defaults_to_robot_heading(hold_env):
  term = hold_env.command_manager.get_term("twist")
  assert isinstance(term, UniformVelocityCommand)
  robot = hold_env.scene["robot"]
  board = hold_env.scene["board"]
  base = UniformVelocityCommand._heading_hold_reference(term)
  torch.testing.assert_close(base, robot.data.heading_w)
  # The ride's own hook reads the board.
  torch.testing.assert_close(term._heading_hold_reference(), board.data.heading_w)


@pytest.mark.slow
def test_heading_hold_state_is_the_hold_that_made_the_command(hold_env):
  # Rewards run before the step's command update, so after a step the state
  # is the one the next step's rewards read: the hold that made the command
  # the policy acts on.
  env = hold_env
  env.reset()
  term = env.command_manager.get_term("twist")
  assert isinstance(term, mdp.BoardHeadingHoldCommand)
  board = env.scene["board"]
  term._hh_prev_active[:] = False
  env.step(torch.zeros(2, 12, device=env.device))
  active, target = term.heading_hold_state()
  assert active.dtype == torch.bool and bool(active.all())
  torch.testing.assert_close(target, board.data.heading_w, atol=0.01, rtol=0)
  assert torch.equal(target, term._hh_target)
  reward = mdp.board_heading_hold(
    env, command_name="twist", std=0.03, min_speed=STEER_YAW_MIN_SPEED
  )
  assert bool((reward > 0.9).all())


##
# The heading hold's state accessor, on a bare command term (no simulator).
##


def _bare_hold_term(heading_hold_kp: float | None, num_envs: int):
  """A velocity command term with just what the heading hold reads."""
  cfg = dataclasses.replace(
    pupper_skate_split_ride_env_cfg().commands["twist"],
    heading_hold_kp=heading_hold_kp,
  )
  term = object.__new__(UniformVelocityCommand)
  term.cfg = cfg
  term._env = SimpleNamespace(num_envs=num_envs, device="cpu")
  heading = torch.zeros(num_envs)
  term.robot = SimpleNamespace(data=SimpleNamespace(heading_w=heading))
  term.vel_command_b = torch.zeros(num_envs, 3)
  term._hh_cmd_b = torch.zeros(num_envs, 3)
  term._hh_target = torch.zeros(num_envs)
  term._hh_prev_active = torch.zeros(num_envs, dtype=torch.bool)
  term._joystick_enabled = None
  return term


def _run_hold(term, steps: int, seed: int, read_state: bool) -> list[torch.Tensor]:
  """Drive the hold with random commands and headings; returns the commands
  emitted, and checks the accessor against an independent copy of the hold."""
  gen = torch.Generator().manual_seed(seed)
  n = term.vel_command_b.shape[0]
  cfg = term.cfg
  prev = torch.zeros(n, dtype=torch.bool)
  target = torch.zeros(n)
  emitted = []
  for _ in range(steps):
    raw = torch.rand(n, 3, generator=gen) * 0.6 - 0.3
    # Quiet yaw in half the envs, standing in a quarter.
    raw[torch.rand(n, generator=gen) < 0.5, 2] = 0.0
    raw[torch.rand(n, generator=gen) < 0.25, :2] = 0.0
    heading = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
    term.vel_command_b[:] = raw
    term.robot.data.heading_w[:] = heading
    term.compute(0.02)
    emitted.append(term.command.clone())
    if not read_state:
      continue
    active, held = term.heading_hold_state()
    if cfg.heading_hold_kp is None:
      assert not bool(active.any())
      continue
    want = (raw[:, 2].abs() < cfg.heading_hold_yaw_threshold) & (
      torch.norm(raw[:, :2], dim=1) > cfg.heading_hold_walk_threshold
    )
    target = torch.where(want & ~prev, heading, target)
    prev = want
    assert torch.equal(active, want)
    assert torch.equal(held, target)
    err = wrap_to_pi(target - heading)
    clip = cfg.heading_hold_clip
    corr = torch.clamp(cfg.heading_hold_kp * err, -clip, clip)
    expected = raw.clone()
    expected[:, 2] = torch.where(want, corr, raw[:, 2])
    assert torch.equal(term.command, expected)
    # The accessor hands out copies: writing to them leaves the hold alone.
    active[:] = ~active
    held += 1.0
  return emitted


@pytest.mark.parametrize("kp", [GAIT_HEADING_HOLD_KP, None])
def test_heading_hold_state_reads_the_hold_without_changing_it(monkeypatch, kp):
  # Only the hold itself runs: the base compute (resampling) is stubbed.
  monkeypatch.setattr(CommandTerm, "compute", lambda self, dt: None)
  read = _run_hold(_bare_hold_term(kp, 64), steps=20, seed=0, read_state=True)
  blind = _run_hold(_bare_hold_term(kp, 64), steps=20, seed=0, read_state=False)
  for a, b in zip(read, blind, strict=True):
    assert torch.equal(a, b)


##
# Steer command sampling, on a bare command term (no simulator).
##


def _bare_command(cls, cfg, num_envs: int):
  """A command term with just the buffers ``_resample_command`` touches."""
  term = object.__new__(cls)
  term.cfg = cfg
  term._env = SimpleNamespace(num_envs=num_envs, device="cpu")
  term.vel_command_b = torch.zeros(num_envs, 3)
  term.vel_command_w = torch.zeros(num_envs, 3)
  flags = torch.zeros(num_envs, dtype=torch.bool)
  for name in (
    "is_heading_env",
    "is_standing_env",
    "is_world_env",
    "is_forward_env",
    "is_turn_env",
    "_capture_heading",
  ):
    setattr(term, name, flags.clone())
  return term


def _sample_steer(num_envs: int = 20000, seed: int = 0) -> torch.Tensor:
  cfg = pupper_skate_split_steer_env_cfg().commands["twist"]
  term = _bare_command(mdp.SteerVelocityCommand, cfg, num_envs)
  torch.manual_seed(seed)
  term._resample_command(torch.arange(num_envs))
  # Standing envs are zeroed on update, not at sampling: drop them here.
  return term.vel_command_b[~term.is_standing_env]


def test_steer_carves_or_rides_straight_across_the_speed_range():
  command = _sample_steer()
  vx, wz = command[:, 0], command[:, 2]
  lo, hi = STEER_CARVE_COMMAND_RANGE
  # Every yaw is 0 or a carve, of either sign in about equal shares, partial
  # turns as well as full ones: above the 0.1 threshold, up to 0.6.
  carve = wz != 0.0
  assert (lo, hi) == (0.1, 0.6)
  assert torch.all((wz.abs() > lo) & (wz.abs() < hi) | ~carve)
  size = wz[carve].abs()
  assert size.min() < 0.11 and size.max() > 0.59
  partial = (size < 0.35).float().mean()
  assert partial.item() == pytest.approx(0.5, abs=0.02)
  left = (wz > 0.0).sum() / carve.sum()
  assert left.item() == pytest.approx(0.5, abs=0.02)
  rolling = vx >= 0.1
  straight = rolling & ~carve
  share = straight.sum() / rolling.sum()
  assert share.item() == pytest.approx(0.4, abs=0.02)
  # Their speeds fill the rolling range, not pinned at 0.3 m/s.
  speeds = vx[straight]
  assert speeds.min() < 0.11 and speeds.max() > 0.49
  assert (speeds == 0.3).float().mean() < 0.01
  for band in ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)):
    in_band = ((speeds >= band[0]) & (speeds < band[1])).float().mean()
    assert in_band.item() == pytest.approx(0.25, abs=0.04), band


def test_steer_gives_no_yaw_below_0_1_m_s():
  command = _sample_steer()
  slow = command[:, 0] < 0.1
  assert slow.any()
  assert torch.all(command[slow, 2] == 0.0)
  assert torch.any(command[~slow, 2] != 0.0)


def test_steer_command_draws_the_ride_command_then_the_carve():
  # The speed, lateral and every share are the ride's, draw for draw; only
  # the yaw channel is new.
  cfg = pupper_skate_split_steer_env_cfg().commands["twist"]
  assert isinstance(cfg, mdp.SteerVelocityCommandCfg)
  ride_cfg = mdp.BoardHeadingHoldCommandCfg(
    **{
      f.name: getattr(cfg, f.name)
      for f in dataclasses.fields(mdp.BoardHeadingHoldCommandCfg)
    }
  )
  terms = []
  for cls, c in (
    (mdp.SteerVelocityCommand, cfg),
    (mdp.BoardHeadingHoldCommand, ride_cfg),
  ):
    term = _bare_command(cls, c, 1000)
    torch.manual_seed(3)
    term._resample_command(torch.arange(1000))
    terms.append(term)
  steer, ride = terms
  torch.testing.assert_close(steer.vel_command_b[:, :2], ride.vel_command_b[:, :2])
  assert torch.equal(steer.is_standing_env, ride.is_standing_env)


def test_fixed_carve_fixes_the_sign():
  cfg = pupper_skate_split_steer_env_cfg().commands["twist"]
  assert isinstance(cfg, mdp.SteerVelocityCommandCfg)
  cfg.rel_straight_envs = 0.0
  cfg.yaw_min_speed = 0.0
  for share, sign in ((1.0, 1.0), (0.0, -1.0)):
    cfg.carve_left_share = share
    term = _bare_command(mdp.SteerVelocityCommand, cfg, 1000)
    term._resample_command(torch.arange(1000))
    assert torch.all(torch.sign(term.vel_command_b[:, 2]) == sign)


@pytest.fixture(scope="module")
def steer_hold_env():
  """The steer task in play, two envs, a fixed walking command and no noise."""
  cfg = pupper_skate_split_steer_env_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.seed = 0
  cfg.events["deck_height"].params["ranges"] = (0.0, 0.0)
  cfg.events.pop("push_robot", None)
  for name in ("reset_base", "reset_board"):
    cfg.events[name].params["pose_range"] = {}
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.SteerVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.3, 0.3)
  twist.rel_standing_envs = 0.0
  twist.rel_straight_envs = 1.0
  twist.resampling_time_range = (1e6, 1e6)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()


@pytest.mark.slow
def test_steer_emits_the_hold_correction_when_quiet_and_the_carve_raw(steer_hold_env):
  env = steer_hold_env
  env.reset()
  term = env.command_manager.get_term("twist")
  assert isinstance(term, mdp.SteerVelocityCommand)
  board = env.scene["board"]
  # The board yawed 0.5 rad from the robot in both envs: past the hold's clip.
  pose = board.data.root_link_pose_w.clone()
  zeros = torch.zeros(2, device=env.device)
  yaw = torch.full_like(zeros, 0.5)
  pose[:, 3:7] = quat_mul(pose[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))
  board.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  # Env 0 quiet, env 1 carving at -0.5 rad/s; re-arm the hold so it captures
  # the board's new heading on the next step.
  term.vel_command_b[:, 2] = torch.tensor([0.0, -0.5], device=env.device)
  term._hh_prev_active[:] = False
  env.step(torch.zeros(2, 12, device=env.device))
  command = env.command_manager.get_command("twist")
  assert command is not None
  # Quiet: the correction toward the board's line, clipped at the steer
  # task's 0.1. Carve: the raw command.
  assert command[0, 2].item() == pytest.approx(STEER_HEADING_HOLD_CLIP)
  assert command[1, 2].item() == pytest.approx(-0.5)
  torch.testing.assert_close(command[:, 0], torch.full_like(zeros, 0.3))
