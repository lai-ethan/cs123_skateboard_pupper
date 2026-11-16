"""Roll out a saved skate policy and print its gait metrics and a verdict.

Takes a training checkpoint (`model_*.pt`) or an exported `policy.json`.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/gait_report.py \\
    Mjlab-Skate-Ride-Pupper-v3 skateboard/policies/ride_29p4j81t_model_1499.pt \\
    --speed 0.4 [--steps 500] [--skip 100]

Drops the first `--skip` steps (settling), then measures peak joint speed
(the largest per-joint 99th percentile), time at the torque limit, how long
each floor-side foot stays down, how often it lands, and the board's mean
forward speed as a share of `--speed` (a robot that does not move fails).
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


def gait_metrics(
  joint_vel: np.ndarray,
  torque: np.ndarray,
  ground_contact: np.ndarray,
  dt: float,
  effort_limit: float,
  *,
  board_speed: np.ndarray,
  command_speed: float | None,
) -> dict[str, float]:
  """Gait metrics from [T, 12] joint speeds and torques and [T, 2] contacts.

  `ground_contact` is 1.0 while a floor-side foot is on the floor.
  `board_speed` is the board's [T] forward speed along its own long axis in
  m/s; `command_speed` is the commanded speed, and None or 0 (no speed
  command) makes `board_speed_ratio` 1.0.
  """
  contact = np.asarray(ground_contact) > 0.5
  saturated = np.abs(torque) >= 0.95 * effort_limit
  touchdowns = np.sum(contact[1:] & ~contact[:-1], axis=0)
  return {
    "peak_joint_speed": float(np.max(np.percentile(np.abs(joint_vel), 99, axis=0))),
    "saturated_fraction": float(np.mean(np.any(saturated, axis=1))),
    "min_foot_duty": float(np.min(np.mean(contact, axis=0))),
    "max_stride_rate": float(np.max(touchdowns) / (len(contact) * dt)),
    "board_speed_ratio": (
      float(np.mean(board_speed) / command_speed) if command_speed else 1.0
    ),
  }


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
  feet = raw.scene["ground_feet_contact"]
  board = raw.scene["board"]
  obs = env.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  joint_vel, torque, contact, board_speed = [], [], [], []
  resets = 0
  with torch.no_grad():
    for step in range(args.steps):
      obs, _, done, _ = env.step(policy(obs))
      if step < args.skip:
        continue
      resets += int(done.any())
      joint_vel.append(robot.data.joint_vel[0].cpu().numpy())
      torque.append(robot.data.actuator_force[0].cpu().numpy())
      contact.append((feet.data.found[0] > 0).float().cpu().numpy())
      # Same frame as the ride reward track_board_velocity: the board's own axis.
      board_speed.append(float(board.data.root_link_lin_vel_b[0, 0]))

  metrics = gait_metrics(
    np.stack(joint_vel),
    np.stack(torque),
    np.stack(contact),
    raw.step_dt,
    PUPPER_EFFORT_LIMIT,
    board_speed=np.array(board_speed),
    command_speed=args.speed,
  )
  print(f"task: {args.task}  policy: {args.policy}  speed: {args.speed}")
  print(f"measured {len(joint_vel)} steps after {args.skip}")
  if not args.speed:
    print("no speed command: board_speed_ratio is reported as 1.0")
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
