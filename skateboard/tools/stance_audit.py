"""Audit how much balance margin the split stance has when the sim is wrong.

Runs the balance policy under off-nominal conditions (deck height, truck
stiffness, deck grip, motor gain and torque, board placement, shoves), one
change per condition, and prints one PASS/FAIL row per condition. It trains
nothing and changes no task.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/stance_audit.py \\
    [skateboard/policies/balance_2ace1uqa_policy.json] [--conditions a,b,c]

Each condition is one env and 500 steps, measured after the first 100; a full
run takes about a minute per condition on a Mac. The last line is
`STANCE: PASS` or `STANCE: FAIL`. The exit code is always 0. A pass here says
nothing about the real board's size, its grip or the legs' backlash.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull

TASK = "Mjlab-Skate-Balance-Pupper-v3"
DEFAULT_POLICY = Path("skateboard/policies/balance_2ace1uqa_policy.json")
STEPS, SKIP, KICK_STEP = 500, 100, 200
WHEEL_FRICTION = 0.85
LIFT_FEET = {
  "lift_front_right": "leg_front_r_3_foot_site",
  "lift_rear_right": "leg_back_r_3_foot_site",
}

# Override key -> default. Lengths in m, angles in rad, speeds in m/s.
OVERRIDE_DEFAULTS: dict[str, float] = {
  "deck": 0.0,  # added to the deck height
  "truck": 1.0,  # truck stiffness scale
  "mu": 0.9,  # deck friction
  "kp": 1.0,  # position gain scale
  "effort": 3.0,  # actuator effort limit, N m
  "board_x": 0.0,  # board reset pose offsets
  "board_y": 0.0,
  "board_yaw": 0.0,
  "robot_yaw": 0.0,  # robot reset heading offset
  "joint_offset": 0.0,  # uniform +- joint offset at reset
  "push_y": 0.0,  # added to the robot's sideways speed at KICK_STEP
  "board_kick": 0.0,  # added to the board's forward speed at KICK_STEP
  "zero_action": 0.0,  # 1 replaces the policy with zero actions
}

CONDITIONS: dict[str, dict[str, float]] = {
  "baseline": {},
  "zero_action": {"zero_action": 1},
  "deck+3cm": {"deck": 0.03},
  "deck-3cm": {"deck": -0.03},
  "truck_x0.25": {"truck": 0.25},
  "deck_mu_0.3": {"mu": 0.3},
  "kp_x0.6": {"kp": 0.6},
  "effort_1.0": {"effort": 1.0},
  "board_y+4cm": {"board_y": 0.04},
  "board_y-4cm": {"board_y": -0.04},
  "board_yaw_0.2": {"board_yaw": 0.2},
  "joint_offset_0.3": {"joint_offset": 0.3},
  "push_left_0.6": {"push_y": 0.6},
  "push_right_0.6": {"push_y": -0.6},
  "board_kick_0.8": {"board_kick": 0.8},
  "worst_combo": {"deck": 0.02, "truck": 0.25, "mu": 0.4, "kp": 0.7, "effort": 2.0},
}

# Resets must be zero; these are the other limits a condition must meet.
STANCE_LIMITS: dict[str, float] = {
  "min_deck_contact": 0.9,
  "max_torque_use": 0.6,
  "margin": 0.03,
}


def support_margin(feet_xy: Any, com_xy: Any) -> float:
  """Signed distance (m) from `com_xy` to the feet's convex hull, + inside.

  `feet_xy` is (N, 2) with N >= 3 not all on a line; `com_xy` is (2,).
  """
  points = np.asarray(feet_xy, dtype=float)
  point = np.asarray(com_xy, dtype=float)
  hull = ConvexHull(points)
  # Each facet row is (unit outward normal, offset): n . p + offset <= 0 inside.
  inside = -(hull.equations[:, :2] @ point + hull.equations[:, 2])
  if np.all(inside >= 0):
    return float(np.min(inside))
  corners = points[hull.vertices]
  starts, ends = corners, np.roll(corners, -1, axis=0)
  edges = ends - starts
  t = np.clip(np.sum((point - starts) * edges, axis=1) / np.sum(edges**2, axis=1), 0, 1)
  nearest = starts + t[:, None] * edges
  return -float(np.min(np.linalg.norm(point - nearest, axis=1)))


def lifted_foot_margins(
  feet_xy: Any, com_xy: Any, names: Sequence[str]
) -> dict[str, float]:
  """{foot name: support_margin of the other feet} for every foot."""
  points = np.asarray(feet_xy, dtype=float)
  return {
    name: support_margin(np.delete(points, i, axis=0), com_xy)
    for i, name in enumerate(names)
  }


def condition_verdict(row: dict[str, float]) -> bool:
  """True when a row has no resets and meets every entry of STANCE_LIMITS."""
  return (
    row["resets"] == 0
    and row["min_deck_contact"] >= STANCE_LIMITS["min_deck_contact"]
    and row["max_torque_use"] <= STANCE_LIMITS["max_torque_use"]
    and row["margin"] >= STANCE_LIMITS["margin"]
  )


def _configure(env_cfg: Any, o: dict[str, float]) -> None:
  """Pin every randomized event to a fixed value, then apply the overrides."""
  events = env_cfg.events
  events["deck_height"].params["ranges"] = (o["deck"], o["deck"])
  events["truck_stiffness"].params["ranges"] = (o["truck"], o["truck"])
  events["deck_friction"].params["ranges"] = (o["mu"], o["mu"])
  events["wheel_friction"].params["ranges"] = (WHEEL_FRICTION, WHEEL_FRICTION)
  events["board_inertial"].params["alpha_range"] = (0.0, 0.0)
  events["reset_board"].params["pose_range"] = {
    "x": (o["board_x"], o["board_x"]),
    "y": (o["board_y"], o["board_y"]),
    "yaw": (o["board_yaw"], o["board_yaw"]),
  }
  events["reset_base"].params["pose_range"] = {
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "yaw": (o["robot_yaw"], o["robot_yaw"]),
  }
  offset = o["joint_offset"]
  events["reset_robot_joints"].params["position_range"] = (-offset, offset)
  robot = env_cfg.scene.entities["robot"]
  articulation = robot.articulation
  actuator = dataclasses.replace(
    articulation.actuators[0],
    stiffness=articulation.actuators[0].stiffness * o["kp"],
    effort_limit=o["effort"],
  )
  robot.articulation = dataclasses.replace(articulation, actuators=(actuator,))


def _fmt(row: dict[str, float]) -> str:
  return (
    f"resets {row['resets']:d}  deck {row['min_deck_contact']:.2f}  "
    f"torque {row['max_torque_use']:.2f}  roll {row['roll_deg']:+.1f} deg  "
    f"travel {row['board_travel']:+.2f} m  margin {row['margin']:+.3f} m  "
    f"lift_fr {row['lift_front_right']:+.3f}  lift_rr {row['lift_rear_right']:+.3f}"
  )


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
  parser.add_argument(
    "policy", type=Path, nargs="?", default=DEFAULT_POLICY, help="policy.json"
  )
  parser.add_argument(
    "--conditions",
    default=",".join(CONDITIONS),
    help=f"comma-separated, from: {', '.join(CONDITIONS)}",
  )
  args = parser.parse_args(argv)
  names = [name for name in args.conditions.split(",") if name]
  unknown = [name for name in names if name not in CONDITIONS]
  if unknown or not names:
    parser.error(f"unknown condition(s) {unknown}; choose from {list(CONDITIONS)}")

  # Imported here so the tests can load this module without the simulator.
  import torch
  from policy_loader import load_policy

  print(f"task: {TASK}  policy: {args.policy}")
  print(f"{STEPS} steps per condition, measured after {SKIP}; limits {STANCE_LIMITS}")
  all_passed = True
  for name in names:
    o = {**OVERRIDE_DEFAULTS, **CONDITIONS[name]}
    raw, env, policy = load_policy(
      TASK, args.policy, None, configure=lambda cfg, o=o: _configure(cfg, o)
    )
    robot, board = raw.scene["robot"], raw.scene["board"]
    deck_feet = raw.scene["deck_feet_contact"]
    site_names = list(robot.site_names)
    foot_ids = [i for i, n in enumerate(site_names) if n.endswith("_foot_site")]
    foot_names = [site_names[i] for i in foot_ids]
    obs = env.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    x0 = board.data.root_link_pos_w[0, 0].item()
    torque, gravity, contact, com, feet = [], [], [], [], []
    resets = 0
    with torch.no_grad():
      for step in range(STEPS):
        if step == KICK_STEP and o["push_y"]:
          vel = robot.data.root_link_vel_w.clone()
          vel[:, 1] += o["push_y"]
          robot.write_root_link_velocity_to_sim(vel)
        if step == KICK_STEP and o["board_kick"]:
          vel = board.data.root_link_vel_w.clone()
          vel[:, 0] += o["board_kick"]
          board.write_root_link_velocity_to_sim(vel)
        if o["zero_action"]:
          action = torch.zeros(1, env.num_actions, device=raw.device)
        else:
          action = policy(obs)
        obs, _, done, _ = env.step(action)
        if step < SKIP:
          continue
        resets += int(done.any())
        data = robot.data
        torque.append(data.actuator_force[0].cpu().numpy())
        gravity.append(data.projected_gravity_b[0].cpu().numpy())
        contact.append((deck_feet.data.found[0] > 0).float().cpu().numpy().ravel())
        com.append(data.data.subtree_com[0, robot.indexing.root_body_id].cpu().numpy())
        feet.append(data.site_pos_w[0, foot_ids].cpu().numpy())
    travel = board.data.root_link_pos_w[0, 0].item() - x0
    env.close()

    feet_xy = np.mean(feet, axis=0)[:, :2]
    com_xy = np.mean(com, axis=0)[:2]
    lifted = lifted_foot_margins(feet_xy, com_xy, foot_names)
    roll = np.degrees(np.arcsin(np.clip(np.array(gravity)[:, 1], -1, 1)))
    row: dict[str, Any] = {
      "resets": resets,
      "min_deck_contact": float(np.min(np.mean(contact, axis=0))),
      "max_torque_use": float(np.max(np.mean(np.abs(torque), axis=0)) / o["effort"]),
      "roll_deg": float(np.mean(roll)),
      "board_travel": travel,
      "margin": support_margin(feet_xy, com_xy),
      **{key: lifted[site] for key, site in LIFT_FEET.items()},
    }
    passed = condition_verdict(row)
    all_passed &= passed
    print(f"{'PASS' if passed else 'FAIL'}  {name:18s} {_fmt(row)}", flush=True)
  print("STANCE:", "PASS" if all_passed else "FAIL")
  return 0


if __name__ == "__main__":
  sys.exit(main())
