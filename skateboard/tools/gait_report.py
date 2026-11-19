"""Roll out a saved skate policy and print its gait metrics and a verdict.

Takes a training checkpoint (`model_*.pt`) or an exported `policy.json`.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/gait_report.py \\
    Mjlab-Skate-Ride-Pupper-v3 skateboard/policies/ride_29p4j81t_model_1499.pt \\
    --speed 0.4 [--envs 16] [--seed 0] [--deck-height 0.103] [--steps 500] \\
    [--skip 100] [--yaw-rate 0.5]

Rolls out `--envs` environments at once, built from `--seed`, so two runs
give the same answer. Each env draws its own board (deck height, truck
stiffness, friction, gains) from the task's ranges. `--deck-height` pins the
deck top to that many metres above the floor in every env instead, for
example 0.103 for the stock trucks or the measured height of the lowered
ones; use it for the check before the robot.

Drops the first `--skip` steps (settling), then measures, per env, peak joint
speed (the largest per-joint 99th percentile), time at the torque limit, the
smallest share of time any foot touches the deck or the floor, the most often
any foot lands on the floor, and the board's mean forward speed as a share of
`--speed`, which must be from 0.5 to 1.5 (a robot that does not move fails,
and so does a board running away). Each limit is judged on the worst env, and
the line names that env.
It also prints the true largest joint speed of any env and each foot's share
of time on the floor (in the worst env for floor duty), so you can see which
foot pushes; those lines are not part of the verdict.
A split task (`Mjlab-SkateSplit-*`) counts only the two right feet, the ones
that push off the floor: their time on the floor and how often they land. The
left feet stay on the deck; the task ends the episode if they leave it.
A ride task needs a positive `--speed`. Without one (balance), the speed
share is reported as 1.0 and passes. Any episode reset in any env in the
measured steps fails the gait: a policy that falls is not a passing gait.
`--yaw-rate` is the yaw command in rad/s (default 0, the only yaw the other
tasks trained on). On the steer task 0 rides straight with the heading hold
on, as in the ride, and a carve command (0.4 to 0.6, either sign) asks the
board to turn at a tenth of it. The report always prints the board's mean yaw
rate; when `--yaw-rate` is not 0 it adds the ratio to the board yaw rate the
command asks for. That line is not part of the verdict. The steer task needs
a positive `--speed`, like a ride.
The last line is `GAIT: PASS` or `GAIT: FAIL`. The exit code is always 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# (name, "max" | "min", limit), judged on the worst env. A metric may have
# both. The peak joint speed limit must equal JOINT_SPEED_LIMIT in
# pupper_skate_env_cfg.py (an estimate, not a measured motor limit); main
# checks it.
GAIT_LIMITS: tuple[tuple[str, str, float], ...] = (
  ("peak_joint_speed", "max", 12.0),
  ("saturated_fraction", "max", 0.05),
  ("min_foot_duty", "min", 0.4),
  ("max_stride_rate", "max", 2.0),
  ("board_speed_ratio", "min", 0.5),
  ("board_speed_ratio", "max", 1.5),
)
# Column order of both foot contact sensors in the centred tasks.
FOOT_ORDER = ("FR", "FL", "BR", "BL")
# Column order of the split tasks' ground sensor: the two right feet, which push.
SPLIT_FLOOR_FEET = ("FR", "BR")


def is_split(task: str) -> bool:
  """True for the split-stance tasks (left feet on the deck, right on the floor)."""
  return "SkateSplit" in task


def floor_feet(task: str) -> tuple[str, ...]:
  """Names of the ground contact sensor's columns for this task."""
  return SPLIT_FLOOR_FEET if is_split(task) else FOOT_ORDER


def gait_metrics(
  joint_vel: np.ndarray,
  torque: np.ndarray,
  deck_contact: np.ndarray | None,
  ground_contact: np.ndarray,
  dt: float,
  effort_limit: float,
  *,
  board_speed: np.ndarray,
  command_speed: float | None,
) -> dict[str, float]:
  """Gait metrics from [T, 12] joint speeds and torques and [T, n] contacts.

  `deck_contact` and `ground_contact` are 1.0 while a foot is on the deck or
  on the floor, one column per foot in FR, FL, BR, BL order. For the split
  stance pass `deck_contact=None` and the two pushing feet's floor contacts:
  the metrics then cover only those feet.
  `min_foot_duty` is the smallest share of time any foot touches the deck or
  the floor (a pushing foot is on the floor for part of each stroke);
  `max_stride_rate` is the most landings on the floor per second by any one
  foot.
  `board_speed` is the board's [T] forward speed along its own long axis in
  m/s; `command_speed` is the commanded speed, and None or 0 (no speed
  command) makes `board_speed_ratio` 1.0.
  `max_joint_speed` is the largest joint speed on any step. It has no limit;
  it shows what the 99th percentile in `peak_joint_speed` leaves out.
  """
  on_floor = np.asarray(ground_contact) > 0.5
  contact = on_floor
  if deck_contact is not None:
    contact = (np.asarray(deck_contact) > 0.5) | on_floor
  saturated = np.abs(torque) >= 0.95 * effort_limit
  touchdowns = np.sum(on_floor[1:] & ~on_floor[:-1], axis=0)
  return {
    "peak_joint_speed": float(np.max(np.percentile(np.abs(joint_vel), 99, axis=0))),
    "max_joint_speed": float(np.max(np.abs(joint_vel))),
    "saturated_fraction": float(np.mean(np.any(saturated, axis=1))),
    "min_foot_duty": float(np.min(np.mean(contact, axis=0))),
    "max_stride_rate": float(np.max(touchdowns) / (len(contact) * dt)),
    "board_speed_ratio": (
      float(np.mean(board_speed) / command_speed) if command_speed else 1.0
    ),
  }


def floor_share(
  ground_contact: np.ndarray, feet: tuple[str, ...] = FOOT_ORDER
) -> dict[str, float]:
  """Each foot's share of time on the floor, from [T, n] contacts.

  `feet` names the columns. Not part of the verdict: it shows which foot
  pushes.
  """
  share = np.mean(np.asarray(ground_contact) > 0.5, axis=0)
  return {name: float(value) for name, value in zip(feet, share, strict=True)}


def gait_verdict(
  per_env: list[dict[str, float]],
) -> list[tuple[str, float, str, float, bool, int]]:
  """(name, value, kind, limit, passed, env) per row of GAIT_LIMITS.

  `per_env` holds one `gait_metrics` result per env. Each row takes the worst
  env for its limit: the largest value against a "max", the smallest against
  a "min". `env` is that env's index.
  """
  rows = []
  for name, kind, limit in GAIT_LIMITS:
    values = [metrics[name] for metrics in per_env]
    env = int(np.argmax(values) if kind == "max" else np.argmin(values))
    value = values[env]
    passed = value <= limit if kind == "max" else value >= limit
    rows.append((name, value, kind, limit, passed, env))
  return rows


def needs_speed(task: str) -> bool:
  """True for the tasks that command a board speed (ride and steer)."""
  return "Ride" in task or "Steer" in task


def yaw_rate_ratio(board_yaw_rate: np.ndarray, target: float) -> tuple[float, float]:
  """The board's mean yaw rate in rad/s and its ratio to a non-zero `target`.

  `board_yaw_rate` holds the board's yaw rate about world z, any shape (all
  steps of all envs); the ratio is 1.0 when the board turns at the target.
  """
  mean = float(np.mean(board_yaw_rate))
  return mean, mean / target


def deck_offset(deck_height: float, nominal: float) -> float:
  """The deck height event's offset in m for a deck top `deck_height` m up."""
  return deck_height - nominal


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
  parser.add_argument("task")
  parser.add_argument("policy", type=Path, help="model_*.pt or policy.json")
  parser.add_argument("--speed", type=float, help="board speed command, m/s")
  parser.add_argument("--envs", type=int, default=16, help="envs to roll out")
  parser.add_argument("--seed", type=int, default=0, help="env random seed")
  parser.add_argument(
    "--deck-height", type=float, help="pin the deck top to this many m up"
  )
  parser.add_argument("--steps", type=int, default=500, help="50 per second")
  parser.add_argument("--skip", type=int, default=100, help="settling steps")
  parser.add_argument(
    "--yaw-rate", type=float, default=0.0, help="yaw (steer: carve) command, rad/s"
  )
  args = parser.parse_args(argv)
  if args.steps <= args.skip:
    parser.error(f"--steps ({args.steps}) must be greater than --skip ({args.skip})")
  if needs_speed(args.task) and not (args.speed is not None and args.speed > 0):
    parser.error("a ride task needs a positive --speed, for example --speed 0.4")
  if args.envs < 1:
    parser.error(f"--envs ({args.envs}) must be at least 1")
  if args.deck_height is not None and args.deck_height <= 0:
    parser.error(f"--deck-height ({args.deck_height}) must be above the floor")

  # Imported here so the tests can load this module without the simulator.
  import torch
  from policy_loader import load_policy

  from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
    SkateboardParams,
  )
  from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import PUPPER_EFFORT_LIMIT
  from mjlab.tasks.pupper.mdp.rewards import carve_yaw_target
  from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import JOINT_SPEED_LIMIT
  from mjlab.tasks.pupper_skate.pupper_skate_split_env_cfg import (
    STEER_CARVE_GAIN,
    STEER_CARVE_THRESHOLD,
  )

  assert ("peak_joint_speed", "max", JOINT_SPEED_LIMIT) in GAIT_LIMITS

  configure = None
  if args.deck_height is not None:
    offset = deck_offset(args.deck_height, SkateboardParams().deck_top_height)

    def configure(env_cfg) -> None:
      env_cfg.events["deck_height"].params["ranges"] = (offset, offset)

  raw, env, policy = load_policy(
    args.task,
    args.policy,
    args.speed,
    num_envs=args.envs,
    seed=args.seed,
    configure=configure,
    yaw_rate=args.yaw_rate,
  )
  robot = raw.scene["robot"]
  deck_feet = raw.scene["deck_feet_contact"]
  ground_feet = raw.scene["ground_feet_contact"]
  board = raw.scene["board"]
  obs = env.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  # Each list gets one [envs, ...] array per measured step.
  joint_vel, torque, deck, ground, board_speed = [], [], [], [], []
  board_yaw_rate = []
  resets = np.zeros(args.envs, dtype=int)
  with torch.no_grad():
    for step in range(args.steps):
      obs, _, done, _ = env.step(policy(obs))
      if step < args.skip:
        continue
      resets += done.cpu().numpy().astype(int)
      joint_vel.append(robot.data.joint_vel.cpu().numpy())
      torque.append(robot.data.actuator_force.cpu().numpy())
      deck.append((deck_feet.data.found > 0).float().cpu().numpy())
      ground.append((ground_feet.data.found > 0).float().cpu().numpy())
      # Same frame as the ride reward track_board_velocity: the board's own axis.
      board_speed.append(board.data.root_link_lin_vel_b[:, 0].cpu().numpy())
      # Same quantity as the steer reward track_board_yaw_rate.
      board_yaw_rate.append(board.data.root_link_ang_vel_w[:, 2].cpu().numpy())

  split = is_split(args.task)
  joint_vel_a, torque_a = np.stack(joint_vel), np.stack(torque)
  deck_a, ground_a = np.stack(deck), np.stack(ground)
  board_speed_a = np.stack(board_speed)
  per_env = [
    gait_metrics(
      joint_vel_a[:, i],
      torque_a[:, i],
      None if split else deck_a[:, i],
      ground_a[:, i],
      raw.step_dt,
      PUPPER_EFFORT_LIMIT,
      board_speed=board_speed_a[:, i],
      command_speed=args.speed,
    )
    for i in range(args.envs)
  ]
  deck_text = "task range" if args.deck_height is None else f"{args.deck_height} m"
  print(f"task: {args.task}  policy: {args.policy}  speed: {args.speed}")
  print(f"envs: {args.envs}  seed: {args.seed}  deck height: {deck_text}")
  print(
    f"measured {len(joint_vel)} steps after {args.skip}; each line is the worst env"
  )
  if not args.speed:
    print("no speed command: board_speed_ratio is reported as 1.0")
  board_yaw_rate_a = np.stack(board_yaw_rate)
  yaw_line = (
    "board yaw rate (not part of the verdict): mean"
    f" {np.mean(board_yaw_rate_a):.3f} rad/s, mean size"
    f" {np.mean(np.abs(board_yaw_rate_a)):.3f} rad/s over all envs"
  )
  if args.yaw_rate:
    # The board yaw rate the command asks for: a tenth of a steer carve.
    target = args.yaw_rate
    if "Steer" in args.task:
      target = carve_yaw_target(
        torch.tensor(target), STEER_CARVE_THRESHOLD, STEER_CARVE_GAIN
      ).item()
    _, ratio = yaw_rate_ratio(board_yaw_rate_a, target)
    yaw_line += f", {ratio:.2f} of the {target:.3f} the command asks for"
  print(yaw_line)
  fastest = int(np.argmax([m["max_joint_speed"] for m in per_env]))
  print(
    "largest joint speed (not part of the verdict): "
    f"{per_env[fastest]['max_joint_speed']:.2f} rad/s in env {fastest}"
  )
  rows = gait_verdict(per_env)
  duty_env = next(row[5] for row in rows if row[0] == "min_foot_duty")
  floor = floor_share(ground_a[:, duty_env], floor_feet(args.task))
  print(
    f"time on the floor in env {duty_env} (not part of the verdict): "
    + "  ".join(f"{name} {value:.2f}" for name, value in floor.items())
  )
  for name, value, kind, limit, passed, worst in rows:
    op = "<=" if kind == "max" else ">="
    print(
      f"{'PASS' if passed else 'FAIL'}  {name} = {value:.3f} (need {op} {limit})"
      f"  env {worst}"
    )
  if resets.any():
    fell = ", ".join(str(i) for i in np.flatnonzero(resets))
    print(
      f"FAIL  {int(resets.sum())} resets while measuring (envs {fell}): a policy"
      " that falls does not pass"
    )
  else:
    print("PASS  no resets while measuring")
  passed = not resets.any() and all(row[4] for row in rows)
  print("GAIT:", "PASS" if passed else "FAIL")
  return 0


if __name__ == "__main__":
  sys.exit(main())
