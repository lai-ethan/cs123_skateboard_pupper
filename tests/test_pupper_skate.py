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

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
  DECK_GEOM_NAME,
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
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper_gait.export import (
  export_pupper_gait_policy_from_env,
  gait_frame_kind,
)
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import pupper_velocity_fs_env_cfg
from mjlab.tasks.pupper_skate import mdp
from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import (
  BALANCE_TORQUE_SOFT_RATIO,
  DECK_HEIGHT_RANGE,
  JOINT_SPEED_LIMIT,
  PUSH_INTERVAL_S,
  RIDE_DECK_HEIGHT_RANGE,
  RIDE_SPEED_RANGE,
  pupper_skate_balance_env_cfg,
  pupper_skate_ride_env_cfg,
)
from mjlab.tasks.registry import list_tasks
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
def test_stance_holds_with_zero_actions(balance_env):
  env = balance_env
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
    mdp.board_alignment(env, SKATE_BOARD_OFFSET, std_pos=0.05, std_yaw=0.3).min() > 0.3
  )


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
def test_stance_survives_spawn_noise_with_zero_actions():
  # About 9% of noisy spawns tip over with zero actions (30 of 320 envs measured;
  # seed 0 gives 5 of 64). A policy has to correct those; this checks the rest
  # hold.
  num_envs = 64
  cfg = _calm_balance_cfg(num_envs, spawn_noise=True)
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  actions = torch.zeros(num_envs, 12, device=env.device)
  reset = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  for _ in range(250):  # 5 s.
    _, _, terminated, truncated, _ = env.step(actions)
    reset |= terminated | truncated
  env.close()
  assert reset.float().mean() <= 0.2


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
  assert "heading_hold" not in policy
  assert "gait_reference" in policy


##
# The ride task (config only; training is what tests the behavior).
##


def test_ride_task_commands_forward_speed_only():
  cfg = pupper_skate_ride_env_cfg()
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  assert twist.ranges.lin_vel_x == RIDE_SPEED_RANGE
  assert twist.ranges.lin_vel_y == (0.0, 0.0) and twist.ranges.ang_vel_z == (0.0, 0.0)
  for name in ("board_still", "feet_on_ground", "push_stride", "push_leg_pose"):
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
  assert cfg.rewards["joint_speed_limit"].params["limit"] < JOINT_SPEED_LIMIT
  # A push takes a foot off the deck for about a second.
  assert cfg.terminations["feet_off_deck"].params["grace_s"] == 1.5
  assert cfg.rewards["feet_on_deck"].weight == 0.5

  # Either side may push, so the whole stance is held loosely.
  pose = cfg.rewards["stance_pose"]
  patterns = pose.params["asset_cfg"].joint_names
  assert {j for j in JOINT_NAMES if any(re.fullmatch(p, j) for p in patterns)} == set(
    JOINT_NAMES
  )
  assert (pose.weight, pose.params["std"]) == (0.5, 0.6)
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
  assert reach.weight == 0.5
  params = dict(reach.params)
  foot_cfg = params.pop("foot_cfg")
  assert params == {
    "std": 0.03,
    "foot_radius": 0.02,
    "command_name": "twist",
    "command_threshold": 0.1,
    "speed_fraction": 0.8,
  }
  assert foot_cfg.name == "robot"
  assert tuple(foot_cfg.site_names) == FOOT_SITE_NAMES
  stroke = cfg.rewards["push_stroke"]
  assert stroke.weight == 0.3
  assert stroke.params["speed_fraction"] == 1.0
  # Ride trains on the lowered trucks only; balance keeps both.
  assert cfg.events["deck_height"].params["ranges"] == RIDE_DECK_HEIGHT_RANGE
  balance = pupper_skate_balance_env_cfg()
  assert balance.events["deck_height"].params["ranges"] == DECK_HEIGHT_RANGE
  assert "foot_reach" not in balance.rewards


@pytest.mark.slow
def test_ride_foot_reach_is_small_with_all_feet_on_the_deck():
  # The lowest deck the ride task trains on, a clean spawn, a speed command and
  # zero actions: the feet stay on the deck, so reaching pays little.
  cfg = pupper_skate_ride_env_cfg()
  cfg.scene.num_envs = 1
  cfg.events["deck_height"].params["ranges"] = (-0.035, -0.035)
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
  actions = torch.zeros(1, 12, device=env.device)
  for _ in range(50):  # 1 s.
    _, _, terminated, truncated, _ = env.step(actions)
    assert not (terminated | truncated).any()
    assert reach.func(env, **reach.params).max() < 0.2
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
