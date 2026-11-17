"""Roll out a saved skate policy and print its gait metrics and a verdict.

Takes a training checkpoint (`model_*.pt`) or an exported `policy.json`.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/gait_report.py \\
    Mjlab-Skate-Ride-Pupper-v3 skateboard/policies/ride_29p4j81t_model_1499.pt \\
    --speed 0.4 [--steps 500] [--skip 100]

Drops the first `--skip` steps (settling), then measures peak joint speed
(the largest per-joint 99th percentile), time at the torque limit, the
smallest share of time any foot touches the deck or the floor, the most often
any foot lands on the floor, and the board's mean forward speed as a share of
`--speed` (a robot that does not move fails). It also prints each foot's share
of time on the floor, so you can see which foot pushes; that line is not part
of the verdict.
A split task (`Mjlab-SkateSplit-*`) counts only the two right feet, the ones
that push off the floor: their time on the floor and how often they land. The
left feet stay on the deck; the task ends the episode if they leave it.
Without `--speed`, or with 0 (balance), that share is reported as 1.0 and
passes. Any episode reset in the measured steps fails the gait: a policy that
falls is not a passing gait.
The last line is `GAIT: PASS` or `GAIT: FAIL`. The exit code is always 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# name -> ("max" | "min", limit). The peak joint speed limit must equal
# JOINT_SPEED_LIMIT in pupper_skate_env_cfg.py; main checks it.
GAIT_LIMITS: dict[str, tuple[str, float]] = {
  "peak_joint_speed": ("max", 8.0),
  "saturated_fraction": ("max", 0.05),
  "min_foot_duty": ("min", 0.4),
  "max_stride_rate": ("max", 2.0),
  "board_speed_ratio": ("min", 0.5),
}
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
  """
  on_floor = np.asarray(ground_contact) > 0.5
  contact = on_floor
  if deck_contact is not None:
    contact = (np.asarray(deck_contact) > 0.5) | on_floor
  saturated = np.abs(torque) >= 0.95 * effort_limit
  touchdowns = np.sum(on_floor[1:] & ~on_floor[:-1], axis=0)
  return {
    "peak_joint_speed": float(np.max(np.percentile(np.abs(joint_vel), 99, axis=0))),
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
  metrics: dict[str, float],
) -> list[tuple[str, float, str, float, bool]]:
  """(name, value, "max" | "min", limit, passed) per entry of GAIT_LIMITS."""
  rows = []
  for name, (kind, limit) in GAIT_LIMITS.items():
    value = metrics[name]
    passed = value <= limit if kind == "max" else value >= limit
    rows.append((name, value, kind, limit, passed))
  return rows


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
  parser.add_argument("task")
  parser.add_argument("policy", type=Path, help="model_*.pt or policy.json")
  parser.add_argument("--speed", type=float, help="board speed command, m/s")
  parser.add_argument("--steps", type=int, default=500, help="50 per second")
  parser.add_argument("--skip", type=int, default=100, help="settling steps")
  args = parser.parse_args(argv)
  if args.steps <= args.skip:
    parser.error(f"--steps ({args.steps}) must be greater than --skip ({args.skip})")

  # Imported here so the tests can load this module without the simulator.
  import torch
  from policy_loader import load_policy

  from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import PUPPER_EFFORT_LIMIT
  from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import JOINT_SPEED_LIMIT

  assert GAIT_LIMITS["peak_joint_speed"][1] == JOINT_SPEED_LIMIT

  raw, env, policy = load_policy(args.task, args.policy, args.speed)
  robot = raw.scene["robot"]
  deck_feet = raw.scene["deck_feet_contact"]
  ground_feet = raw.scene["ground_feet_contact"]
  board = raw.scene["board"]
  obs = env.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  joint_vel, torque, deck, ground, board_speed = [], [], [], [], []
  resets = 0
  with torch.no_grad():
    for step in range(args.steps):
      obs, _, done, _ = env.step(policy(obs))
      if step < args.skip:
        continue
      resets += int(done.any())
      joint_vel.append(robot.data.joint_vel[0].cpu().numpy())
      torque.append(robot.data.actuator_force[0].cpu().numpy())
      deck.append((deck_feet.data.found[0] > 0).float().cpu().numpy())
      ground.append((ground_feet.data.found[0] > 0).float().cpu().numpy())
      # Same frame as the ride reward track_board_velocity: the board's own axis.
      board_speed.append(float(board.data.root_link_lin_vel_b[0, 0]))

  split = is_split(args.task)
  metrics = gait_metrics(
    np.stack(joint_vel),
    np.stack(torque),
    None if split else np.stack(deck),
    np.stack(ground),
    raw.step_dt,
    PUPPER_EFFORT_LIMIT,
    board_speed=np.array(board_speed),
    command_speed=args.speed,
  )
  print(f"task: {args.task}  policy: {args.policy}  speed: {args.speed}")
  print(f"measured {len(joint_vel)} steps after {args.skip}")
  if not args.speed:
    print("no speed command: board_speed_ratio is reported as 1.0")
  floor = floor_share(np.stack(ground), floor_feet(args.task))
  print(
    "time on the floor (not part of the verdict): "
    + "  ".join(f"{name} {value:.2f}" for name, value in floor.items())
  )
  rows = gait_verdict(metrics)
  for name, value, kind, limit, passed in rows:
    op = "<=" if kind == "max" else ">="
    print(f"{'PASS' if passed else 'FAIL'}  {name} = {value:.3f} (need {op} {limit})")
  if resets:
    print(f"FAIL  {resets} resets while measuring: a policy that falls does not pass")
  else:
    print("PASS  no resets while measuring")
  passed = resets == 0 and all(row[4] for row in rows)
  print("GAIT:", "PASS" if passed else "FAIL")
  return 0


if __name__ == "__main__":
  sys.exit(main())
