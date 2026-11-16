"""Reward terms for the Pupper skateboard tasks.

Every term returns one value per env, shape ``[B]``. The sign lives in the
weight, so "penalty" terms here return a non-negative cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_BOARD = SceneEntityCfg("board")


def contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """How many of the sensor's primary geoms are touching its secondary."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return (found > 0).float().sum(dim=1)


def gravity_tracking(
  env: ManagerBasedRlEnv,
  target_xy: tuple[float, float],
  std: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward holding a target tilt, exp(-|g_xy - target|^2 / std^2).

  ``g_xy`` is the x, y of gravity in the body frame (what the IMU reports). The
  split stance is not level, so the target is the stance's tilt rather than zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor(target_xy, device=env.device)
  error = asset.data.projected_gravity_b[:, :2] - target
  return torch.exp(-torch.sum(torch.square(error), dim=1) / std**2)


def stance_pose(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward joints near the keyframe pose, exp(-mean(error^2) / std^2)."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  error = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
  return torch.exp(-torch.mean(torch.square(error), dim=1) / std**2)


def board_alignment(
  env: ManagerBasedRlEnv,
  target_offset: tuple[float, float],
  std_pos: float,
  std_yaw: float,
  robot_cfg: SceneEntityCfg = _ROBOT,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """Reward the board sitting where the stance expects it, pointing our way.

  ``target_offset`` is the board centre in the robot's heading frame (x forward,
  y left). The error is that offset plus the yaw between board and robot.
  """
  robot: Entity = env.scene[robot_cfg.name]
  board: Entity = env.scene[board_cfg.name]
  _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_link_quat_w)
  _, _, board_yaw = euler_xyz_from_quat(board.data.root_link_quat_w)
  delta = board.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  cos, sin = torch.cos(robot_yaw), torch.sin(robot_yaw)
  offset_x = cos * delta[:, 0] + sin * delta[:, 1]
  offset_y = -sin * delta[:, 0] + cos * delta[:, 1]
  pos_error = torch.square(offset_x - target_offset[0]) + torch.square(
    offset_y - target_offset[1]
  )
  yaw_error = torch.square(wrap_to_pi(board_yaw - robot_yaw))
  return torch.exp(-pos_error / std_pos**2 - yaw_error / std_yaw**2)


def board_speed_l2(
  env: ManagerBasedRlEnv, board_cfg: SceneEntityCfg = _BOARD
) -> torch.Tensor:
  """Penalize the board rolling (balance stage): planar speed squared."""
  board: Entity = env.scene[board_cfg.name]
  return torch.sum(torch.square(board.data.root_link_lin_vel_w[:, :2]), dim=1)


def board_yaw_rate_l2(
  env: ManagerBasedRlEnv, board_cfg: SceneEntityCfg = _BOARD
) -> torch.Tensor:
  """Penalize the board turning: yaw rate squared."""
  board: Entity = env.scene[board_cfg.name]
  return torch.square(board.data.root_link_ang_vel_w[:, 2])


def deck_roll_l2(
  env: ManagerBasedRlEnv,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """Penalize the deck leaning on its trucks (leaning steers the board)."""
  board: Entity = env.scene[board_cfg.name]
  return torch.sum(torch.square(board.data.joint_pos[:, board_cfg.joint_ids]), dim=1)


def track_board_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """Reward the board rolling at the commanded forward speed.

  exp(-(cmd_vx - board_vx)^2 / std^2), with ``board_vx`` along the board's own
  long axis, so rolling backwards or being dragged sideways earns nothing.
  """
  board: Entity = env.scene[board_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = command[:, 0] - board.data.root_link_lin_vel_b[:, 0]
  return torch.exp(-torch.square(error) / std**2)


def joint_speed_over_limit(
  env: ManagerBasedRlEnv, limit: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Penalize joint speed past what the motors can do: sum((|qd| - limit)+^2)."""
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
  return torch.sum(torch.square(torch.clamp(speed - limit, min=0.0)), dim=1)


def torque_saturation(
  env: ManagerBasedRlEnv,
  limit: float,
  soft_ratio: float = 0.8,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Penalize motors working near their torque limit.

  Per joint: 0 below ``soft_ratio * limit``, rising linearly to 1 at ``limit``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ratio = torch.abs(asset.data.actuator_force[:, asset_cfg.actuator_ids]) / limit
  over = (ratio - soft_ratio) / (1.0 - soft_ratio)
  return torch.sum(torch.clamp(over, 0.0, 1.0), dim=1)


def push_stride(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  stance_s: float,
  air_range: tuple[float, float],
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Reward slow, planted push strides, paid once per touchdown.

  A stride is the stance that ended at lift-off plus the swing that just
  landed. It scores min(stance / stance_s, 1) when the swing lasted within
  ``air_range`` seconds, and 0 otherwise, so quick taps and feet held in the air
  earn little. The score is divided by the step so the weight is the reward for
  one full stride. Paid only above ``command_threshold`` m/s of commanded speed.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  contact_time, air_time = data.current_contact_time, data.last_air_time
  stance_time = data.last_contact_time
  assert contact_time is not None and air_time is not None and stance_time is not None
  landed = (contact_time > 0.0) & (contact_time < env.step_dt + 1.0e-6)
  swing_ok = (air_time >= air_range[0]) & (air_time <= air_range[1])
  score = torch.clamp(stance_time / stance_s, max=1.0)
  reward = torch.sum(score * (landed & swing_ok).float(), dim=1) / env.step_dt
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return reward * (command[:, 0] > command_threshold).float()
