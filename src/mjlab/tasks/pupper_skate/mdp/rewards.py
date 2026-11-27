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
from mjlab.tasks.pupper.mdp.rewards import carve_yaw_target
from mjlab.tasks.pupper_skate.mdp.terminations import (
  SlideSinceStart,
  board_drift,
  board_slide,
)
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

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


class board_fore_aft:
  """Reward the robot staying put along the deck: ``exp(-(slide / std)^2)``.

  ``slide`` is ``board_slide`` since the episode's start (``SlideSinceStart``),
  the quantity ``board_slid`` ends the episode on. It counts from the start, not
  the keyframe, because the policy cannot see where a reset put it. So a robot
  that holds still where it started is paid in full.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg  # Unused.
    self._since_start = SlideSinceStart(env.num_envs, env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._since_start.reset(env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    robot_cfg: SceneEntityCfg = _ROBOT,
    board_cfg: SceneEntityCfg = _BOARD,
  ) -> torch.Tensor:
    slide = self._since_start(board_slide(env, robot_cfg, board_cfg))
    return torch.exp(-torch.square(slide / std))


class _OffsetCreep:
  """Cost of a slow, steady change in one board offset, in [0, 1].

  The rate is the per-step change of ``_offset`` over the control step,
  smoothed by two exponential averages in a row, each with time constant
  ``tau`` [s], so the stride's own back-and-forth cancels out and only a steady
  drift is left. One average would leave about amplitude / tau of the stride;
  two leave about amplitude / (omega * tau^2). The cost is
  ``min(|average|, rate_cap) / rate_cap``.

  Keeps the last offset and both averages per env. A reset zeroes them and
  marks the env, and the env's first call after it only records the offset and
  costs 0: the reward manager resets before the scene is forwarded, so the
  fresh pose is not readable in ``reset``, and the reset jump must not read as
  a velocity.
  """

  def _offset(
    self, env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg, board_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    raise NotImplementedError

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg  # Unused.
    self._ema1 = torch.zeros(env.num_envs, device=env.device)
    self._ema2 = torch.zeros(env.num_envs, device=env.device)
    self._prev = torch.zeros(env.num_envs, device=env.device)
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._ema1[env_ids] = 0.0
    self._ema2[env_ids] = 0.0
    self._fresh[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    tau: float,
    rate_cap: float,
    robot_cfg: SceneEntityCfg = _ROBOT,
    board_cfg: SceneEntityCfg = _BOARD,
  ) -> torch.Tensor:
    offset = self._offset(env, robot_cfg, board_cfg)
    prev = torch.where(self._fresh, offset, self._prev)
    rate = (offset - prev) / env.step_dt
    gain = env.step_dt / tau
    self._ema1 = self._ema1 + gain * (rate - self._ema1)
    self._ema2 = self._ema2 + gain * (self._ema1 - self._ema2)
    self._prev = offset.clone()
    self._fresh[:] = False
    return torch.clamp(self._ema2.abs(), max=rate_cap) / rate_cap


class board_creep(_OffsetCreep):
  """Cost of a slow, steady slide along the deck, in [0, 1].

  The rate of ``board_slide``, averaged twice (see ``_OffsetCreep``).
  ``board_fore_aft`` pays for where the robot is; this pays for where it is
  heading, while it is heading there, long before ``board_slid`` ends the
  episode.
  """

  def _offset(
    self, env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg, board_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    return board_slide(env, robot_cfg, board_cfg)


class board_side_creep(_OffsetCreep):
  """Cost of a slow, steady sideways slide of the board, in [0, 1].

  The sideways twin of ``board_creep``: the rate of ``board_drift``, the offset
  ``board_drifted`` ends the episode on, averaged twice (see ``_OffsetCreep``).
  The stride's side-to-side sway cancels out. A lean is a constant offset, so
  once it is held it costs nothing; only a slide that keeps going does.
  """

  def _offset(
    self, env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg, board_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    return board_drift(env, robot_cfg, board_cfg)


def board_speed_l2(
  env: ManagerBasedRlEnv, board_cfg: SceneEntityCfg = _BOARD
) -> torch.Tensor:
  """Penalize the board rolling (balance stage): planar speed squared."""
  board: Entity = env.scene[board_cfg.name]
  return torch.sum(torch.square(board.data.root_link_lin_vel_w[:, :2]), dim=1)


def board_yaw_rate_l2(
  env: ManagerBasedRlEnv,
  board_cfg: SceneEntityCfg = _BOARD,
  command_name: str | None = None,
  carve_threshold: float | None = None,
) -> torch.Tensor:
  """Penalize the board turning: yaw rate squared.

  With ``command_name`` and ``carve_threshold`` set (steer task), envs whose
  yaw command is a carve, above ``carve_threshold`` in magnitude, pay nothing.
  """
  board: Entity = env.scene[board_cfg.name]
  cost = torch.square(board.data.root_link_ang_vel_w[:, 2])
  if command_name is None or carve_threshold is None:
    return cost
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return cost * (command[:, 2].abs() <= carve_threshold).float()


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


def track_board_yaw_rate(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  board_cfg: SceneEntityCfg = _BOARD,
  carve_threshold: float | None = None,
  carve_gain: float = 1.0,
  straight_min_speed: float | None = None,
  straight_gain: float = 0.0,
) -> torch.Tensor:
  """Reward the board turning at the commanded yaw rate (steer task).

  exp(-(target - board_wz)^2 / std^2), with ``board_wz`` the board's yaw rate
  about world z, the quantity ``board_yaw_rate_l2`` penalizes, and ``target``
  the command's yaw rate (dim 2). How the board is made to turn is left open.

  With ``carve_threshold`` set, the target is the carve mapping's (see
  ``carve_yaw_target``) and only carves pay: an env whose command is at or
  under the threshold, the heading hold's correction, gets 0. The correction
  turns the robot, not the board, and a quiet env then pays what the ride pays.

  With ``straight_min_speed`` set as well, an env outside a carve whose speed
  command (dim 0) is at least that is paid for a board that answers the hold:
  the target is ``straight_gain`` times the yaw command (the hold's correction),
  with the same ``std``. The default gain of 0 pays a board that does not turn.
  Standing envs, under the speed, still get 0.
  """
  board: Entity = env.scene[board_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_wz = command[:, 2]
  board_wz = board.data.root_link_ang_vel_w[:, 2]
  target = carve_yaw_target(cmd_wz, carve_threshold, carve_gain)
  reward = torch.exp(-torch.square(target - board_wz) / std**2)
  if carve_threshold is None:
    return reward
  carving = cmd_wz.abs() > carve_threshold
  if straight_min_speed is None:
    return reward * carving.float()
  rolling = command[:, 0] >= straight_min_speed
  hold = straight_gain * cmd_wz
  straight = torch.exp(-torch.square(hold - board_wz) / std**2) * rolling.float()
  return torch.where(carving, reward, straight)


def board_heading_hold(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  min_speed: float,
  board_cfg: SceneEntityCfg = _BOARD,
) -> torch.Tensor:
  """Reward the board for staying on the heading hold's target (steer task).

  exp(-(wrap(board heading - target) / std)^2) in envs where the command's
  heading hold is active and the speed command (dim 0) is at least
  ``min_speed``; 0 elsewhere (carves, standing, under the speed). The skate
  tasks' hold captures the board's heading, but measures the error from the
  robot's, so a board rotated against the robot drifts sideways unseen. This
  term sees it.
  """
  board: Entity = env.scene[board_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  term = env.command_manager.get_term(command_name)
  active, target = term.heading_hold_state()  # type: ignore[attr-defined]
  error = wrap_to_pi(board.data.heading_w - target)
  reward = torch.exp(-torch.square(error / std))
  paid = active & (command[:, 0] >= min_speed)
  return reward * paid.float()


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


def board_turn_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  board_cfg: SceneEntityCfg = _BOARD,
  carve_threshold: float | None = None,
  carve_gain: float = 1.0,
) -> torch.Tensor:
  """Reward the board's share of the commanded turn, clamp(board_wz / target, 0, 1).

  ``board_wz`` is the board's yaw rate about world z, as in
  ``track_board_yaw_rate``. That term is tight and flat far from the target;
  this one grows from the first bit of turn, so a small lean already pays.
  Turning the wrong way pays nothing, and so does a command at or under
  ``command_threshold`` rad/s in magnitude. ``target`` is the command's yaw
  rate, through the carve mapping when ``carve_threshold`` is set (see
  ``carve_yaw_target``).
  """
  board: Entity = env.scene[board_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_wz = command[:, 2]
  active = cmd_wz.abs() > command_threshold
  target = carve_yaw_target(cmd_wz, carve_threshold, carve_gain)
  share = board.data.root_link_ang_vel_w[:, 2] / torch.where(active, target, 1.0)
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


def stance_target(
  speed: torch.Tensor, stroke_m: float, stance_range: tuple[float, float]
) -> torch.Tensor:
  """Stance time [s] for a planted foot to cover ``stroke_m`` at ``speed`` m/s.

  clamp(stroke_m / max(speed, 0.05), stance_range): a fixed stroke takes less
  time on the floor at a higher command, and a command near zero asks for the
  longest stance in range.
  """
  target = stroke_m / torch.clamp(speed, min=0.05)
  return torch.clamp(target, stance_range[0], stance_range[1])


def _stance_seconds(
  env: ManagerBasedRlEnv,
  stance_s: float | None,
  stroke_m: float | None,
  stance_range: tuple[float, float] | None,
  command_name: str | None,
) -> torch.Tensor | float:
  """The fixed ``stance_s``, or with ``stroke_m`` a per-env [B, 1] target."""
  if stroke_m is None:
    if stance_s is None:
      raise ValueError("Give stance_s or stroke_m.")
    return stance_s
  if stance_s is not None:
    raise ValueError("Give stance_s or stroke_m, not both.")
  if stance_range is None or command_name is None:
    raise ValueError("stroke_m needs stance_range and command_name.")
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return stance_target(command[:, 0], stroke_m, stance_range).unsqueeze(1)


def push_stride(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  air_range: tuple[float, float],
  command_name: str,
  command_threshold: float,
  stance_s: float | None = None,
  stroke_m: float | None = None,
  stance_range: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Reward slow, planted push strides, paid once per touchdown (split stance).

  A stride is the stance that ended at lift-off plus the swing that just
  landed. It scores min(stance / stance_s, 1) when the swing lasted within
  ``air_range`` seconds, and 0 otherwise, so quick taps and feet held in the air
  earn little. The score is divided by the step so the weight is the reward for
  one full stride. Paid only above ``command_threshold`` m/s of commanded speed.

  With ``stroke_m`` the stance target is set per env by the command instead
  (see ``stance_target``), so the stroke the foot covers is the same at every
  speed; ``stance_s`` is then left unset.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  contact_time, air_time = data.current_contact_time, data.last_air_time
  stance_time = data.last_contact_time
  assert contact_time is not None and air_time is not None and stance_time is not None
  target = _stance_seconds(env, stance_s, stroke_m, stance_range, command_name)
  landed = (contact_time > 0.0) & (contact_time < env.step_dt + 1.0e-6)
  swing_ok = (air_time >= air_range[0]) & (air_time <= air_range[1])
  score = torch.clamp(stance_time / target, max=1.0)
  reward = torch.sum(score * (landed & swing_ok).float(), dim=1) / env.step_dt
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return reward * (command[:, 0] > command_threshold).float()


def short_stride(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  stance_s: float | None = None,
  stroke_m: float | None = None,
  stance_range: tuple[float, float] | None = None,
  command_name: str | None = None,
) -> torch.Tensor:
  """Penalize quick taps, charged once per touchdown (split stance).

  Each foot that just landed costs 1 - min(stance / stance_s, 1), where
  ``stance`` is the contact that ended at its last lift-off, so a tap with no
  stance before it costs 1 and a stance of ``stance_s`` or more costs nothing.
  The cost is divided by the step so the weight is the cost of one such tap.
  Not gated on the command: tapping while told to stand still costs the same.
  A foot with no stance on record has not lifted off since the reset (it
  spawned on the floor), so it is not charged.

  With ``stroke_m`` the target is the per-env ``stance_target`` of the
  ``command_name`` command, as in ``push_stride``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  contact_time, stance_time = data.current_contact_time, data.last_contact_time
  assert contact_time is not None and stance_time is not None
  target = _stance_seconds(env, stance_s, stroke_m, stance_range, command_name)
  landed = (contact_time > 0.0) & (contact_time < env.step_dt + 1.0e-6)
  landed &= stance_time > 0.0
  cost = 1.0 - torch.clamp(stance_time / target, max=1.0)
  return torch.sum(cost * landed.float(), dim=1) / env.step_dt


def foot_height_over(
  env: ManagerBasedRlEnv,
  max_height: float,
  foot_radius: float,
  sensor_name: str,
  foot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize swing feet lifted higher than ``max_height`` [m] (split stance).

  Per foot in the air: clamp(z - foot_radius - max_height, min=0) in metres,
  where ``z`` is the foot site height above the floor (z = 0, as in
  ``foot_reach``). Feet in contact on ``sensor_name`` add 0. ``foot_cfg`` names
  the foot sites in the sensor's order.
  """
  robot: Entity = env.scene[foot_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  foot_z = robot.data.site_pos_w[:, foot_cfg.site_ids, 2]
  over = torch.clamp(foot_z - foot_radius - max_height, min=0.0)
  return torch.sum(over * (found == 0).float(), dim=1)


def foot_idle_air(
  env: ManagerBasedRlEnv, sensor_name: str, max_air_s: float
) -> torch.Tensor:
  """Count the sensor's feet that have been in the air longer than ``max_air_s``.

  Charged every step while it lasts, so a pushing foot that hovers instead of
  stepping keeps paying. Not gated on the command.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  assert air_time is not None
  return (air_time > max_air_s).float().sum(dim=1)


def feet_slide(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize feet sliding while in contact: sum of foot xy speed squared.

  ``asset_cfg`` names the foot sites, in the same order as the sensor's feet.
  Not gated on the command, unlike the velocity tasks' ``feet_slip``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  in_contact = (found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  return torch.sum(torch.sum(torch.square(foot_vel_xy), dim=-1) * in_contact, dim=1)


def deck_feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  deck_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize feet sliding over the deck: sum of foot xy slip speed squared.

  The slip is the foot site's velocity minus the deck's own velocity at that
  point (the deck link's velocity plus its spin crossed with the lever from
  the link to the foot), so a foot carried along by a rolling, turning or
  leaning deck costs nothing. Only feet in contact on ``sensor_name`` count.
  ``asset_cfg`` names the foot sites in the sensor's order; ``deck_cfg`` names
  the deck body (the first of its bodies is used). Same units as
  ``feet_slide``.
  """
  robot: Entity = env.scene[asset_cfg.name]
  board: Entity = env.scene[deck_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  in_contact = (found > 0).float()
  foot_pos = robot.data.site_pos_w[:, asset_cfg.site_ids]
  foot_vel = robot.data.site_lin_vel_w[:, asset_cfg.site_ids]
  deck_pos = board.data.body_link_pos_w[:, deck_cfg.body_ids][:, :1]
  deck_vel = board.data.body_link_lin_vel_w[:, deck_cfg.body_ids][:, :1]
  deck_spin = board.data.body_link_ang_vel_w[:, deck_cfg.body_ids][:, :1]
  carried = deck_vel + torch.linalg.cross(
    deck_spin.expand_as(foot_pos), foot_pos - deck_pos, dim=-1
  )
  slip_xy = (foot_vel - carried)[..., :2]
  return torch.sum(torch.sum(torch.square(slip_xy), dim=-1) * in_contact, dim=1)


def push_stroke(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  stroke_s: float,
  command_name: str,
  command_threshold: float,
  speed_fraction: float,
  board_cfg: SceneEntityCfg = _BOARD,
  foot_ids: tuple[int, ...] | None = None,
  support_sensor: str | None = None,
  support_grace_s: float = 0.1,
) -> torch.Tensor:
  """Reward planted pushes against the floor, paid once as the foot lifts off.

  A stroke is the floor contact that just ended. It scores
  min(stroke / stroke_s, 1), so quick taps earn little. The score is divided by
  the step so the weight is the reward for one full stroke. Paid only above
  ``command_threshold`` m/s of commanded speed, and only while the board's
  forward speed is under ``speed_fraction`` of the command, so tapping on a
  board that already rolls fast enough earns nothing.

  With ``foot_ids`` only strokes by those feet (sensor order) count. With
  ``support_sensor`` it pays only while every other foot is down on that
  sensor, where a foot out of contact for under ``support_grace_s`` still
  counts as down (``_support_down``), so one step of contact flicker does not
  lose a stroke. Unset, any foot's stroke pays, in any posture.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air_time, stroke_time = sensor.data.current_air_time, sensor.data.last_contact_time
  assert air_time is not None and stroke_time is not None
  if foot_ids is not None:
    air_time = air_time[:, list(foot_ids)]
    stroke_time = stroke_time[:, list(foot_ids)]
  lifted = (air_time > 0.0) & (air_time < env.step_dt + 1.0e-6)
  score = torch.clamp(stroke_time / stroke_s, max=1.0)
  reward = torch.sum(score * lifted.float(), dim=1) / env.step_dt
  if support_sensor is not None:
    reward = reward * _support_down(env, support_sensor, foot_ids, support_grace_s)
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
  foot_ids: tuple[int, ...] | None = None,
  support_sensor: str | None = None,
  support_grace_s: float = 0.1,
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

  With ``foot_ids`` (indices into ``foot_cfg``'s sites) only those feet count
  as the lowest foot. With ``support_sensor`` it pays only while every other
  foot is down on that sensor, as in ``push_stroke``.
  """
  robot: Entity = env.scene[foot_cfg.name]
  foot_z = robot.data.site_pos_w[:, foot_cfg.site_ids, 2]
  if foot_ids is not None:
    foot_z = foot_z[:, list(foot_ids)]
  clearance = torch.clamp(torch.min(foot_z, dim=1).values - foot_radius, min=0.0)
  reward = torch.exp(-clearance / std)
  if support_sensor is not None:
    reward = reward * _support_down(env, support_sensor, foot_ids, support_grace_s)
  if continuous_gate:
    return reward * _push_share_missing(
      env, command_name, command_threshold, speed_fraction, board_cfg
    )
  return reward * _needs_push(
    env, command_name, command_threshold, speed_fraction, board_cfg
  )


def _support_down(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  foot_ids: tuple[int, ...] | None,
  grace_s: float,
) -> torch.Tensor:
  """1.0 where every foot of the sensor outside ``foot_ids`` is down.

  A foot is down while its air time is under ``grace_s`` (zero in contact), so
  a brief loss of contact still counts. The sensor must track air time.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  assert air_time is not None
  support = torch.ones(air_time.shape[1], dtype=torch.bool, device=air_time.device)
  if foot_ids is not None:
    support[list(foot_ids)] = False
  return (air_time[:, support] < grace_s).all(dim=1).float()


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
