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


class SteerVelocityCommand(BoardHeadingHoldCommand):
  """The split ride's board heading-hold command, with carve commands on yaw.

  The yaw rate (dim 2) is a carve signal, not a yaw rate to track. After the
  parent's sampling every env draws a carve: a magnitude from ``carve_range``
  and a sign, positive with probability ``carve_left_share``. Then a
  ``rel_straight_envs`` share of envs get yaw 0 and keep their sampled speed
  (unlike ``rel_forward_envs``, which also lifts it to at least 0.3 m/s), and
  so does any env whose speed command is under ``yaw_min_speed`` m/s: a board
  that is not rolling cannot carve. Standing envs are zeroed by the parent.

  A quiet yaw (0) is exactly the ride: the heading hold is on and the emitted
  yaw is its clipped correction toward the board's line. A carve is above the
  hold's yaw threshold, so the hold lets go and the raw command reaches the
  policy, and above the hold's clip, so the policy can tell a carve from a
  correction. When the carve ends, the hold recaptures the board's heading.
  The robot runs the same hold, so a carve needs no robot-side change.
  """

  cfg: "SteerVelocityCommandCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    n = len(env_ids)
    magnitude = torch.empty(n, device=self.device).uniform_(*self.cfg.carve_range)
    left = torch.rand(n, device=self.device) < self.cfg.carve_left_share
    yaw = torch.where(left, magnitude, -magnitude)
    if self.cfg.rel_straight_envs > 0.0:
      straight = torch.rand(n, device=self.device) <= self.cfg.rel_straight_envs
      yaw[straight] = 0.0
    if self.cfg.yaw_min_speed > 0.0:
      yaw[self.vel_command_b[env_ids, 0] < self.cfg.yaw_min_speed] = 0.0
    self.vel_command_b[env_ids, 2] = yaw


@dataclass
class SteerVelocityCommandCfg(BoardHeadingHoldCommandCfg):
  carve_range: tuple[float, float] = (0.0, 0.0)
  """Magnitude range [rad/s] of the carve command drawn for every env."""
  carve_left_share: float = 0.5
  """Probability that a carve is positive (a left turn)."""
  rel_straight_envs: float = 0.0
  """Fraction of envs given a zero yaw rate, their sampled speed kept."""
  yaw_min_speed: float = 0.0
  """Forward speed command [m/s] under which the yaw rate is set to 0."""

  def build(self, env: ManagerBasedRlEnv) -> SteerVelocityCommand:
    return SteerVelocityCommand(self, env)
