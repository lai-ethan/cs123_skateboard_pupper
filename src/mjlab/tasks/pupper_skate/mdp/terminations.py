"""Termination terms for the Pupper skateboard tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import SkateboardParams
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_BOARD = SceneEntityCfg("board")


def contact_lost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  grace_s: float,
  min_lost: int = 1,
  foot_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
  """Terminate when the sensor's geoms have been out of contact too long.

  Ends the episode once at least ``min_lost`` geoms have each been out of
  contact for longer than ``grace_s``. The sensor must be built with
  ``track_air_time=True``. With ``foot_ids`` only those of the sensor's geoms
  (by index, in sensor order) count; the others may stay off for good.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  assert air_time is not None
  if foot_ids is not None:
    air_time = air_time[:, list(foot_ids)]
  return torch.sum(air_time > grace_s, dim=1) >= min_lost


def contact_held(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contact_s: float,
  min_count: int = 1,
  sensor_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
  """Terminate when the sensor's geoms have been in contact too long.

  Ends the episode once at least ``min_count`` geoms have each touched for
  longer than ``max_contact_s`` without a break, so with ``min_count`` 2 they
  have all been down together for that long. The sensor must be built with
  ``track_air_time=True``. With ``sensor_ids`` only those of the sensor's geoms
  (by index, in sensor order) count; the others may rest for good.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  contact_time = sensor.data.current_contact_time
  assert contact_time is not None
  if sensor_ids is not None:
    contact_time = contact_time[:, list(sensor_ids)]
  return torch.sum(contact_time > max_contact_s, dim=1) >= min_count


def base_near_deck(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  deck_cfg: SceneEntityCfg,
  deck_half_thickness: float = SkateboardParams().deck_thickness / 2,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Terminate when the robot base is less than ``minimum_height`` above the deck.

  The deck top is that env's own: the deck geom's centre height (``deck_cfg``
  names the one deck geom, which the deck height event moves per env) plus
  ``deck_half_thickness``.
  """
  robot: Entity = env.scene[asset_cfg.name]
  board: Entity = env.scene[deck_cfg.name]
  deck_z = board.data.geom_pos_w[:, deck_cfg.geom_ids, 2]
  deck_top = deck_z.reshape(deck_z.shape[0], -1)[:, 0] + deck_half_thickness
  return robot.data.root_link_pos_w[:, 2] - deck_top < minimum_height


def board_lateral_offset(
  board_pos: torch.Tensor, board_quat: torch.Tensor, robot_pos: torch.Tensor
) -> torch.Tensor:
  """The board centre's sideways offset from the robot base [m], shape ``[B]``.

  Board position minus robot position, along the board's own left axis (its
  heading frame), so the value does not change as the board turns.
  """
  rel = board_pos - robot_pos
  return quat_apply_inverse(yaw_quat(board_quat), rel)[:, 1]


def board_fore_aft_offset(
  board_pos: torch.Tensor, board_quat: torch.Tensor, robot_pos: torch.Tensor
) -> torch.Tensor:
  """The board centre's fore-aft offset from the robot base [m], shape ``[B]``.

  As ``board_lateral_offset``, along the board's own forward axis instead.
  """
  rel = board_pos - robot_pos
  return quat_apply_inverse(yaw_quat(board_quat), rel)[:, 0]


def _offset_from_nominal(
  env: ManagerBasedRlEnv,
  offset: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
  robot_cfg: SceneEntityCfg,
  board_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """``offset`` of the board from the robot, minus its value in the keyframe.

  The nominal is the same offset between the two entities' default root states,
  the keyframe the scene spawns, so it follows the stance instead of a typed
  number.
  """
  robot: Entity = env.scene[robot_cfg.name]
  board: Entity = env.scene[board_cfg.name]
  current = offset(
    board.data.root_link_pos_w,
    board.data.root_link_quat_w,
    robot.data.root_link_pos_w,
  )
  board_default = board.data.default_root_state
  nominal = offset(
    board_default[:, :3], board_default[:, 3:7], robot.data.default_root_state[:, :3]
  )
  return current - nominal


def board_drifted(
  env: ManagerBasedRlEnv,
  max_drift: float,
  robot_cfg: SceneEntityCfg = _ROBOT,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """Terminate when the board has slid sideways under the robot.

  Fires when ``board_lateral_offset`` is more than ``max_drift`` [m] from its
  nominal value, the keyframe's (see ``_offset_from_nominal``).
  """
  drift = _offset_from_nominal(env, board_lateral_offset, robot_cfg, board_cfg)
  return torch.abs(drift) > max_drift


def board_slide(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """How far the robot has slid along the deck [m], shape ``[B]``.

  ``board_fore_aft_offset`` minus the keyframe's. Negative when the robot has
  moved towards the nose (the board centre is then further behind it).
  """
  return _offset_from_nominal(env, board_fore_aft_offset, robot_cfg, board_cfg)


def board_slid(
  env: ManagerBasedRlEnv,
  max_slide: float,
  robot_cfg: SceneEntityCfg = _ROBOT,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """Terminate when the robot has slid along the deck, either way.

  The fore-aft twin of ``board_drifted``: fires when ``board_slide`` is more
  than ``max_slide`` [m].
  """
  return torch.abs(board_slide(env, robot_cfg, board_cfg)) > max_slide
