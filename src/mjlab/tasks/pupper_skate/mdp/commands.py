"""The skate tasks' commands: the Pupper twist with board-specific rules."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.pupper.mdp.velocity_command import (
  PupperVelocityCommand,
  PupperVelocityCommandCfg,
)


class SteerVelocityCommand(PupperVelocityCommand):
  """``PupperVelocityCommand`` with two extra rules on the yaw rate (dim 2).

  After the usual sampling, a ``rel_straight_envs`` share of envs get a yaw
  rate of exactly 0 and keep their sampled forward speed (unlike
  ``rel_forward_envs``, which also lifts it to at least 0.3 m/s). Then any env
  whose forward speed command is under ``yaw_min_speed`` m/s gets yaw 0 too: a
  board that is not rolling cannot carve. With both at 0 the commands are the
  parent's, draw for draw.
  """

  cfg: "SteerVelocityCommandCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: "SteerVelocityCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.is_straight_env = torch.zeros_like(self.is_standing_env)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    if self.cfg.rel_straight_envs > 0.0:
      r = torch.empty(len(env_ids), device=self.device)
      straight = r.uniform_(0.0, 1.0) <= self.cfg.rel_straight_envs
      self.is_straight_env[env_ids] = straight
      self.vel_command_b[env_ids[straight], 2] = 0.0
    if self.cfg.yaw_min_speed > 0.0:
      slow = self.vel_command_b[env_ids, 0] < self.cfg.yaw_min_speed
      self.vel_command_b[env_ids[slow], 2] = 0.0


@dataclass
class SteerVelocityCommandCfg(PupperVelocityCommandCfg):
  rel_straight_envs: float = 0.0
  """Fraction of envs given a zero yaw rate, their sampled speed kept."""
  yaw_min_speed: float = 0.0
  """Forward speed command [m/s] under which the yaw rate is set to 0."""

  def build(self, env: ManagerBasedRlEnv) -> SteerVelocityCommand:
    return SteerVelocityCommand(self, env)


class BoardHeadingHoldCommand(PupperVelocityCommand):
  """``PupperVelocityCommand`` whose heading hold captures the board's heading.

  At the hold's activation edge the target is the board's heading, not the
  robot's, so the emitted yaw command is a clipped P-correction of the robot's
  yaw error against the board's line. On the robot the same loop captures the
  IMU yaw with the robot placed parallel to the board.
  """

  cfg: "BoardHeadingHoldCommandCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: "BoardHeadingHoldCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.board: Entity = env.scene[cfg.board_name]

  def _heading_hold_reference(self) -> torch.Tensor:
    return self.board.data.heading_w


@dataclass
class BoardHeadingHoldCommandCfg(PupperVelocityCommandCfg):
  board_name: str = "board"
  """Scene entity whose heading the heading hold captures."""

  def build(self, env: ManagerBasedRlEnv) -> BoardHeadingHoldCommand:
    return BoardHeadingHoldCommand(self, env)
