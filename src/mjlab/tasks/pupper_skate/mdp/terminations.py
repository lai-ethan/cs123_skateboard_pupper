"""Termination terms for the Pupper skateboard tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def contact_lost(
  env: ManagerBasedRlEnv, sensor_name: str, grace_s: float, min_lost: int = 1
) -> torch.Tensor:
  """Terminate when the sensor's geoms have been out of contact too long.

  Ends the episode once at least ``min_lost`` geoms have each been out of
  contact for longer than ``grace_s``. The sensor must be built with
  ``track_air_time=True``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  assert air_time is not None
  return torch.sum(air_time > grace_s, dim=1) >= min_lost
