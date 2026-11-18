"""Reset events for the Pupper skateboard tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import SkateboardParams
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _interp_rows(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
  """Row-wise linear interpolation of ``fp`` (len(xp), D) at ``x`` (N,), clamped."""
  x = x.clamp(xp[0], xp[-1])
  i = torch.clamp(torch.searchsorted(xp, x, right=True) - 1, 0, len(xp) - 2)
  w = ((x - xp[i]) / (xp[i + 1] - xp[i])).unsqueeze(1)
  return fp[i] + w * (fp[i + 1] - fp[i])


def reset_tripod_layout(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  deck_tops: tuple[float, ...],
  base_poses: tuple[tuple[float, ...], ...],
  joint_poses: tuple[tuple[float, ...], ...],
  speed_range: tuple[float, float],
  deck_cfg: SceneEntityCfg,
  deck_half_thickness: float = SkateboardParams().deck_thickness / 2,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Place the robot in a pose solved per deck height, on its env's own board.

  Runs after the board's own reset. ``base_poses`` rows are (x, y) of the base
  from the board centre in the board's heading frame, base height above the
  deck top, roll and pitch; ``joint_poses`` rows are joint angles. Both are
  interpolated linearly on that env's deck top (the deck geom's height, which
  the deck height event moves per env) between the ``deck_tops`` they were
  solved at. The board and the robot then roll forward together at a speed
  drawn from ``speed_range`` [m/s].
  """
  env_ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene[asset_cfg.name]
  board: Entity = env.scene[deck_cfg.name]
  # The board was just written: bring its derived poses up to date.
  env.sim.forward()
  dev = env.device
  xp = torch.tensor(deck_tops, device=dev)
  deck_z = board.data.geom_pos_w[env_ids][:, deck_cfg.geom_ids, 2]
  deck_top = deck_z.reshape(len(env_ids), -1)[:, 0] + deck_half_thickness
  base = _interp_rows(deck_top, xp, torch.tensor(base_poses, device=dev))
  joints = _interp_rows(deck_top, xp, torch.tensor(joint_poses, device=dev))

  heading = yaw_quat(board.data.root_link_quat_w[env_ids])
  offset = torch.zeros(len(env_ids), 3, device=dev)
  offset[:, :2] = base[:, :2]
  pos = board.data.root_link_pos_w[env_ids] + quat_apply(heading, offset)
  pos[:, 2] = deck_top + base[:, 2]
  zeros = torch.zeros(len(env_ids), device=dev)
  quat = quat_mul(heading, quat_from_euler_xyz(base[:, 3], base[:, 4], zeros))

  speed = sample_uniform(*speed_range, (len(env_ids),), dev)
  forward = torch.zeros(len(env_ids), 3, device=dev)
  forward[:, 0] = speed
  vel = torch.zeros(len(env_ids), 6, device=dev)
  vel[:, :3] = quat_apply(heading, forward)

  robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
  robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
  board.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
