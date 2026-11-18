"""Critic-only observations: the board, which the robot cannot sense, and the clock."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.pupper_gait.mdp.gait import gait_phase
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_BOARD = SceneEntityCfg("board")


def board_pose_in_robot_frame(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """Board position (3) and orientation quaternion (4) in the robot base frame."""
  robot: Entity = env.scene[robot_cfg.name]
  board: Entity = env.scene[board_cfg.name]
  robot_quat = robot.data.root_link_quat_w
  pos = quat_apply_inverse(
    robot_quat, board.data.root_link_pos_w - robot.data.root_link_pos_w
  )
  quat = quat_mul(quat_conjugate(robot_quat), board.data.root_link_quat_w)
  return torch.cat([pos, quat], dim=1)


def deck_roll_state(
  env: ManagerBasedRlEnv, board_cfg: SceneEntityCfg = _BOARD
) -> torch.Tensor:
  """Deck lean angle and rate on the trucks, shape [B, 2]."""
  board: Entity = env.scene[board_cfg.name]
  ids = board_cfg.joint_ids
  return torch.cat([board.data.joint_pos[:, ids], board.data.joint_vel[:, ids]], dim=1)


def contact_flags(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1.0 per sensor geom that is in contact, else 0.0."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return (found > 0).float()


def cycle_phase(env: ManagerBasedRlEnv, frequency: float) -> torch.Tensor:
  """The reference clock's phase as ``(sin, cos)``, shape [B, 2].

  The same episode clock the gait reference plays on. The actor sees only the
  reference offset, which is constant through the glide; this tells the critic
  how long until the next push.
  """
  angle = 2.0 * torch.pi * gait_phase(env, frequency)
  return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
