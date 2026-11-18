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

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import SkateboardParams
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  JOINT_NAMES,
  PUPPER_EFFORT_LIMIT,
  SKATE_SPLIT_BOARD_OFFSET,
  SKATE_SPLIT_HEIGHT,
  SKATE_SPLIT_KEYFRAME,
  SKATE_SPLIT_POSE,
  SKATE_SPLIT_ROLL,
  get_spec,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper_gait.export import export_pupper_gait_policy_from_env
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import pupper_velocity_fs_env_cfg
from mjlab.tasks.pupper_skate import mdp
from mjlab.tasks.pupper_skate.mdp.rewards import stance_target
from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import (
  BALANCE_TORQUE_SOFT_RATIO,
  DECK_HEIGHT_RANGE,
  RIDE_SPEED_RANGE,
  pupper_skate_balance_env_cfg,
  pupper_skate_ride_env_cfg,
)
from mjlab.tasks.pupper_skate.pupper_skate_split_env_cfg import (
  STEER_YAW_RATE_RANGE,
  pupper_skate_split_balance_env_cfg,
  pupper_skate_split_ride_env_cfg,
  pupper_skate_split_steer_env_cfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

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


##
# Rewards: the a110b84 split configs, name for name and weight for weight, plus
# the balance torque penalty (Task 25), the ride shuffle fix (Task 29), the
# ride motor limits and idle-foot charge (Task 30) and the smoother push
# (Task 31).
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
  "push_stride": 0.5,
  "short_stride": -0.3,
  "feet_slide": -2.0,
  "push_leg_pose": 0.3,
  "joint_vel_l2": -1e-3,
  "joint_speed_limit": -0.3,
  "torque_saturation": -1.5,
  "push_foot_idle": -0.5,
  "action_rate_l2": -0.05,
  "action_acc": -0.02,
  "joint_acc": -1e-6,
  "swing_height": -5.0,
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

# The tripod ride (task 38) dropped knees_on_deck and added feet_slide.
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
  "feet_slide": -2.0,
  "board_progress": 4.0,
  "action_acc": -0.02,
  "joint_acc": -1e-6,
  "joint_vel_l2": -1e-3,
  "joint_speed_limit": -0.1,
  "torque_saturation": -0.5,
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


def test_split_cfgs_share_the_deck_height_range():
  for make_cfg in (pupper_skate_split_balance_env_cfg, pupper_skate_split_ride_env_cfg):
    for play in (False, True):
      events = make_cfg(play=play).events
      assert events["deck_height"].mode == "startup"
      assert events["deck_height"].params["ranges"] == DECK_HEIGHT_RANGE
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
  [(pupper_skate_ride_env_cfg, -0.5), (pupper_skate_split_ride_env_cfg, -1.5)],
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
def test_only_split_ride_charges_taps_sliding_and_idle_feet(make_cfg):
  rewards = make_cfg().rewards
  assert "short_stride" not in rewards
  assert "push_foot_idle" not in rewards
  # The tripod ride charges its push foot sliding on the floor; balance does not.
  if make_cfg is not pupper_skate_ride_env_cfg:
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
@pytest.mark.parametrize("deck_offset", [-0.035, 0.0, 0.01])
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
# The steer task: the split ride plus a board yaw-rate command.
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


def test_steer_is_the_split_ride_plus_a_yaw_command():
  for play in (False, True):
    ride = pupper_skate_split_ride_env_cfg(play=play)
    steer = pupper_skate_split_steer_env_cfg(play=play)
    expected = _weights(ride)
    del expected["board_yaw_rate"]
    expected["track_board_yaw_rate"] = 2.0
    assert _weights(steer) == pytest.approx(expected)
    term = steer.rewards["track_board_yaw_rate"]
    assert term.func is mdp.track_board_yaw_rate
    assert term.params == {"command_name": "twist", "std": 0.15}
    for name in ride.rewards:
      if name != "board_yaw_rate":
        assert steer.rewards[name].func is ride.rewards[name].func, name
        assert steer.rewards[name].params == ride.rewards[name].params, name
    assert steer.terminations.keys() == ride.terminations.keys()
    assert steer.events.keys() == ride.events.keys()
    assert steer.observations["critic"].terms.keys() == (
      ride.observations["critic"].terms.keys()
    )
    twist = steer.commands["twist"]
    assert isinstance(twist, mdp.SteerVelocityCommandCfg)
    assert STEER_YAW_RATE_RANGE == (-0.15, 0.15)
    assert twist.ranges.ang_vel_z == STEER_YAW_RATE_RANGE
    assert (twist.rel_straight_envs, twist.yaw_min_speed) == (0.3, 0.1)
    # Every other command field is the split ride's.
    ride_twist = ride.commands["twist"]
    assert isinstance(ride_twist, mdp.PupperVelocityCommandCfg)
    for field in dataclasses.fields(ride_twist):
      if field.name != "ranges":
        expected = getattr(ride_twist, field.name)
        assert getattr(twist, field.name) == expected, field.name
    assert twist.ranges == dataclasses.replace(
      ride_twist.ranges, ang_vel_z=STEER_YAW_RATE_RANGE
    )
    assert twist.ranges.lin_vel_x == RIDE_SPEED_RANGE
    assert twist.rel_forward_envs == 0.0 and twist.rel_turn_envs == 0.0
    assert twist.heading_hold_kp is None


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
    # The stock command class: the steer task's sampling rules are not in it.
    assert type(twist) is mdp.PupperVelocityCommandCfg
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    assert twist.rel_forward_envs == 0.0 and twist.rel_turn_envs == 0.0
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
  lo, hi = STEER_YAW_RATE_RANGE
  assert torch.all((command[:, 2] >= lo) & (command[:, 2] <= hi))
  env.close()


class _Actor(torch.nn.Module):
  """Minimal MLPModel-like holder: has ``.mlp`` and ``.obs_normalizer``."""

  def __init__(self, in_dim: int):
    super().__init__()
    self.mlp = MLP(in_dim, 12, (64, 64), activation="elu")
    self.obs_normalizer = EmpiricalNormalization(in_dim)


@pytest.mark.slow
def test_steer_policy_exports_the_yaw_clip_with_heading_hold_off():
  cfg = load_env_cfg(STEER_TASK, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  policy = export_pupper_gait_policy_from_env(_Actor(48 * 20).eval(), env)
  env.close()
  assert policy["command_clip"] == {
    "vx": list(RIDE_SPEED_RANGE),
    "vy": [0.0, 0.0],
    "wz": [-0.15, 0.15],
  }
  assert policy["heading_hold"] == {
    "kp": 0.0,
    "clip": 0.3,
    "yaw_threshold": 0.1,
    "walk_threshold": 0.05,
  }
  assert "gait_reference" in policy


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
    "is_straight_env",
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


def test_steer_rides_straight_in_30_percent_across_the_speed_range():
  command = _sample_steer()
  vx, wz = command[:, 0], command[:, 2]
  lo, hi = STEER_YAW_RATE_RANGE
  assert torch.all((wz >= lo) & (wz <= hi))
  rolling = vx >= 0.1
  straight = rolling & (wz == 0.0)
  share = straight.sum() / rolling.sum()
  assert share.item() == pytest.approx(0.3, abs=0.02)
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


def test_steer_command_with_its_rules_off_samples_like_the_stock_command():
  # The four existing tasks use the stock class; this checks the subclass adds
  # nothing (and draws nothing extra) with both rules off.
  cfg = pupper_skate_split_steer_env_cfg().commands["twist"]
  assert isinstance(cfg, mdp.SteerVelocityCommandCfg)
  cfg.rel_straight_envs = 0.0
  cfg.yaw_min_speed = 0.0
  stock_cfg = mdp.PupperVelocityCommandCfg(
    **{
      f.name: getattr(cfg, f.name)
      for f in dataclasses.fields(mdp.PupperVelocityCommandCfg)
    }
  )
  commands = []
  for cls, c in (
    (mdp.SteerVelocityCommand, cfg),
    (mdp.PupperVelocityCommand, stock_cfg),
  ):
    term = _bare_command(cls, c, 1000)
    torch.manual_seed(3)
    term._resample_command(torch.arange(1000))
    commands.append(term.vel_command_b)
  torch.testing.assert_close(commands[0], commands[1])
