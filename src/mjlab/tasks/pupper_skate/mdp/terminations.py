"""Termination terms for the Pupper skateboard tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def contact_lost(
  env: ManagerBasedRlEnv, sensor_name: str, grace_s: float
) -> torch.Tensor:
  """Terminate when any of the sensor's geoms has been out of contact too long.

  The sensor must be built with ``track_air_time=True``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  assert air_time is not None
  return torch.any(air_time > grace_s, dim=1)
