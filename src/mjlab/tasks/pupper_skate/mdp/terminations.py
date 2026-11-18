"""Termination terms for the Pupper skateboard tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import SkateboardParams
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


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
