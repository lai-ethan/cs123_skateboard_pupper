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


def contact_count(
  env: ManagerBasedRlEnv, sensor_name: str, max_count: float | None = None
) -> torch.Tensor:
  """How many of the sensor's primary geoms are touching its secondary.

  With ``max_count`` the count stops there, so contacts past it earn nothing.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  count = (found > 0).float().sum(dim=1)
  return count if max_count is None else torch.clamp(count, max=max_count)


def gravity_tracking(
  env: ManagerBasedRlEnv,
  target_xy: tuple[float, float],
  std: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward holding a target tilt, exp(-|g_xy - target|^2 / std^2).

  ``g_xy`` is the x, y of gravity in the body frame (what the IMU reports). The
  target is the stance's tilt, which is zero for the level centred stance.
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


def board_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """Reward the board's share of the commanded speed, clamp(board_vx / cmd, 0, 1).

  ``board_vx`` is along the board's own long axis, as in
  ``track_board_velocity``. Unlike that term, this one grows from the first
  centimetre per second, so a small push already pays. Rolling backwards pays
  nothing, and so does a command at or under ``command_threshold`` m/s.
  """
  board: Entity = env.scene[board_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  speed = command[:, 0]
  active = speed > command_threshold
  share = board.data.root_link_lin_vel_b[:, 0] / torch.where(active, speed, 1.0)
  return torch.clamp(share, 0.0, 1.0) * active.float()


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
  """Reward slow, planted push strides, paid once per touchdown (split stance).

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


def push_stroke(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  stroke_s: float,
  command_name: str,
  command_threshold: float,
  speed_fraction: float,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """Reward planted pushes against the floor, paid once as the foot lifts off.

  A stroke is the floor contact that just ended. It scores
  min(stroke / stroke_s, 1), so quick taps earn little. The score is divided by
  the step so the weight is the reward for one full stroke. Paid only above
  ``command_threshold`` m/s of commanded speed, and only while the board's
  forward speed is under ``speed_fraction`` of the command, so tapping on a
  board that already rolls fast enough earns nothing.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air_time, stroke_time = sensor.data.current_air_time, sensor.data.last_contact_time
  assert air_time is not None and stroke_time is not None
  lifted = (air_time > 0.0) & (air_time < env.step_dt + 1.0e-6)
  score = torch.clamp(stroke_time / stroke_s, max=1.0)
  reward = torch.sum(score * lifted.float(), dim=1) / env.step_dt
  return reward * _needs_push(
    env, command_name, command_threshold, speed_fraction, board_cfg
  )


def foot_reach(
  env: ManagerBasedRlEnv,
  std: float,
  foot_radius: float,
  command_name: str,
  command_threshold: float,
  speed_fraction: float,
  foot_cfg: SceneEntityCfg,
  board_cfg: SceneEntityCfg = _BOARD,
  continuous_gate: bool = False,
) -> torch.Tensor:
  """Reward the lowest foot getting close to the floor, exp(-clearance / std).

  ``clearance`` is the gap between the lowest foot's underside and the floor
  (z = 0): the smallest foot site height minus ``foot_radius``, at least 0. A
  foot on the deck is a deck height up, so this pays little until a foot goes
  over the edge and down. Paid only when a push is needed: the command is
  above ``command_threshold`` m/s and the board's forward speed is under
  ``speed_fraction`` of it.

  With ``continuous_gate`` the pay fades instead of switching off: it is scaled
  by clamp(1 - board_vx / (speed_fraction * cmd_vx), 0, 1), so it is full at
  rest and gone by ``speed_fraction`` of the command. A foot parked on the
  floor then cannot out-earn rolling at the commanded speed.
  """
  robot: Entity = env.scene[foot_cfg.name]
  foot_z = robot.data.site_pos_w[:, foot_cfg.site_ids, 2]
  clearance = torch.clamp(torch.min(foot_z, dim=1).values - foot_radius, min=0.0)
  reward = torch.exp(-clearance / std)
  if continuous_gate:
    return reward * _push_share_missing(
      env, command_name, command_threshold, speed_fraction, board_cfg
    )
  return reward * _needs_push(
    env, command_name, command_threshold, speed_fraction, board_cfg
  )


def _needs_push(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  speed_fraction: float,
  board_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """1.0 where a speed is commanded and the board rolls slower than its share."""
  board: Entity = env.scene[board_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  speed = command[:, 0]
  board_vx = board.data.root_link_lin_vel_b[:, 0]
  return ((speed > command_threshold) & (board_vx < speed_fraction * speed)).float()


def _push_share_missing(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  speed_fraction: float,
  board_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """clamp(1 - board_vx / (speed_fraction * cmd), 0, 1) where a speed is commanded.

  1.0 with the board at rest, 0.0 from ``speed_fraction`` of the command up,
  and 0.0 everywhere when the command is at or under ``command_threshold``.
  """
  board: Entity = env.scene[board_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  speed = command[:, 0]
  board_vx = board.data.root_link_lin_vel_b[:, 0]
  # The floor on the target only avoids dividing by zero; those rows are
  # inactive and zeroed below.
  target = torch.clamp(speed_fraction * speed, min=1.0e-6)
  share = torch.clamp(1.0 - board_vx / target, min=0.0, max=1.0)
  return share * (speed > command_threshold).float()
