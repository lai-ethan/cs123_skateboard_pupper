"""Audit how much balance margin a skate stance has when the sim is wrong.

Runs the balance policy under off-nominal conditions (deck height, truck
stiffness, deck grip, motor gain and torque, board placement, shoves), one
change per condition, and prints one PASS/FAIL row per condition. A row passes
when the deck feet stay on the deck at least 90% of the time, the motors stay
under 60% of their torque, the centre of mass stays 3 cm inside the feet and
nothing resets. Each row also shows the margin left on three feet with one
foot lifted. It trains nothing and changes no task.

`--task` picks the stance (default: centred balance):
* centred (`Mjlab-Skate-*`): all four feet are deck feet; the lift rows lift
  FR, FL, BR and BL in turn.
* split (`Mjlab-SkateSplit-*`): the two left feet are the deck feet; the lift
  rows lift the floor-side feet, FR and RR.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/stance_audit.py \\
    [skateboard/policies/balance_2ace1uqa_policy.json] [--conditions a,b,c] \\
    [--task Mjlab-SkateSplit-Balance-Pupper-v3]

Each condition is one env and 500 steps, measured after the first 100; a full
run takes about a minute per condition on a Mac. The last line is
`STANCE: PASS` or `STANCE: FAIL`. The exit code is always 0. A pass here says
nothing about the real board's size, its grip or the legs' backlash. The saved
policies were trained on the split stance, so audit them with a split task id.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull

DEFAULT_TASK = "Mjlab-Skate-Balance-Pupper-v3"
DEFAULT_POLICY = Path("skateboard/policies/balance_2ace1uqa_policy.json")
STEPS, SKIP, KICK_STEP = 500, 100, 200
WHEEL_FRICTION = 0.85
# Row key -> foot site, in the contact sensors' order (FR, FL, BR, BL).
LIFT_FEET = {
  "lift_fr": "leg_front_r_3_foot_site",
  "lift_fl": "leg_front_l_3_foot_site",
  "lift_br": "leg_back_r_3_foot_site",
  "lift_bl": "leg_back_l_3_foot_site",
}
# Split stance: lift the floor-side (right) feet, front and rear.
SPLIT_LIFT_FEET = {
  "lift_fr": "leg_front_r_3_foot_site",
  "lift_rr": "leg_back_r_3_foot_site",
}


def lift_feet(task: str) -> dict[str, str]:
  """The lift rows for the task's stance ("SkateSplit" in the id means split)."""
  return SPLIT_LIFT_FEET if "SkateSplit" in task else LIFT_FEET


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
  # Training randomizes the deck from -3.5 to +1 cm; +2 and +3 cm are outside.
  "deck-3.5cm": {"deck": -0.035},
  "deck-2cm": {"deck": -0.02},
  "deck+1cm": {"deck": 0.01},
  "deck+2cm": {"deck": 0.02},
  "deck+3cm": {"deck": 0.03},
  "truck_x0.25": {"truck": 0.25},
  "deck_mu_0.3": {"mu": 0.3},
  "kp_x0.6": {"kp": 0.6},
  "effort_1.0": {"effort": 1.0},
  # The feet sit 1.5 to 2 cm inside the deck edge.
  "board_y+1cm": {"board_y": 0.01},
  "board_y-1cm": {"board_y": -0.01},
  "board_y+2cm": {"board_y": 0.02},
  "board_y-2cm": {"board_y": -0.02},
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
  """{foot name: support_margin of the other feet} for every foot.

  Needs at least four feet, and the feet left after removing any one of them
  must not lie on a line.
  """
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


# Play-mode events that randomize nothing, so there is nothing to pin.
UNPINNED_EVENTS = ("reset_imu_latency",)


def _pin(events: dict[str, Any], name: str, key: str, value: Any) -> None:
  """Set `events[name].params[key]`; fail loudly if the event or key is gone."""
  if name not in events or key not in events[name].params:
    raise KeyError(f"event {name!r} with param {key!r} not in the env config")
  events[name].params[key] = value


def _configure(env_cfg: Any, o: dict[str, float]) -> None:
  """Pin every randomized event to a fixed value, then apply the overrides."""
  events = env_cfg.events
  # Board.
  _pin(events, "deck_height", "ranges", (o["deck"], o["deck"]))
  _pin(events, "truck_stiffness", "ranges", (o["truck"], o["truck"]))
  _pin(events, "deck_friction", "ranges", (o["mu"], o["mu"]))
  _pin(events, "wheel_friction", "ranges", (WHEEL_FRICTION, WHEEL_FRICTION))
  _pin(events, "board_inertial", "alpha_range", (0.0, 0.0))
  _pin(
    events,
    "reset_board",
    "pose_range",
    {
      "x": (o["board_x"], o["board_x"]),
      "y": (o["board_y"], o["board_y"]),
      "yaw": (o["board_yaw"], o["board_yaw"]),
    },
  )
  _pin(events, "reset_board", "velocity_range", {})
  # Robot. pd_gains scales on top of the actuator gains set below, so it must
  # be exactly 1 for kp to mean what the condition says.
  _pin(
    events,
    "reset_base",
    "pose_range",
    {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
      "roll": (0.0, 0.0),
      "yaw": (o["robot_yaw"], o["robot_yaw"]),
    },
  )
  _pin(events, "reset_base", "velocity_range", {})
  offset = o["joint_offset"]
  _pin(events, "reset_robot_joints", "position_range", (-offset, offset))
  _pin(events, "reset_robot_joints", "velocity_range", (0.0, 0.0))
  _pin(events, "pd_gains", "kp_range", (1.0, 1.0))
  _pin(events, "pd_gains", "kd_range", (1.0, 1.0))
  _pin(events, "inertial", "alpha_range", (0.0, 0.0))
  _pin(events, "encoder_bias", "bias_range", (0.0, 0.0))
  _pin(events, "foot_friction", "ranges", (1.0, 1.0))
  com_ranges = events["base_com"].params["ranges"] if "base_com" in events else {}
  _pin(events, "base_com", "ranges", {axis: (0.0, 0.0) for axis in com_ranges})
  pinned = {
    "deck_height",
    "truck_stiffness",
    "deck_friction",
    "wheel_friction",
    "board_inertial",
    "reset_board",
    "reset_base",
    "reset_robot_joints",
    "pd_gains",
    "inertial",
    "encoder_bias",
    "foot_friction",
    "base_com",
  }
  unknown = set(events) - pinned - set(UNPINNED_EVENTS)
  if unknown:
    raise KeyError(f"events {sorted(unknown)} are not pinned by the stance audit")
  robot = env_cfg.scene.entities["robot"]
  articulation = robot.articulation
  actuator = dataclasses.replace(
    articulation.actuators[0],
    stiffness=articulation.actuators[0].stiffness * o["kp"],
    effort_limit=o["effort"],
  )
  robot.articulation = dataclasses.replace(articulation, actuators=(actuator,))


def _fmt(row: dict[str, float], lift_keys: Sequence[str]) -> str:
  return (
    f"resets {row['resets']:d}  deck {row['min_deck_contact']:.2f}  "
    f"torque {row['max_torque_use']:.2f}  roll {row['roll_deg']:+.1f} deg  "
    f"travel {row['board_travel']:+.2f} m  margin {row['margin']:+.3f} m  "
    + "  ".join(f"{key} {row[key]:+.3f}" for key in lift_keys)
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
  parser.add_argument("--task", default=DEFAULT_TASK, help="balance task id")
  args = parser.parse_args(argv)
  names = [name for name in args.conditions.split(",") if name]
  unknown = [name for name in names if name not in CONDITIONS]
  if unknown or not names:
    parser.error(f"unknown condition(s) {unknown}; choose from {list(CONDITIONS)}")

  # Imported here so the tests can load this module without the simulator.
  import torch
  from policy_loader import load_policy

  lifts = lift_feet(args.task)
  print(f"task: {args.task}  policy: {args.policy}")
  print(f"{STEPS} steps per condition, measured after {SKIP}; limits {STANCE_LIMITS}")
  all_passed = True
  for name in names:
    o = {**OVERRIDE_DEFAULTS, **CONDITIONS[name]}
    raw, env, policy = load_policy(
      args.task, args.policy, None, configure=lambda cfg, o=o: _configure(cfg, o)
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
      # One column per foot: the share of time the least-planted foot is down.
      "min_deck_contact": float(np.min(np.mean(contact, axis=0))),
      "max_torque_use": float(np.max(np.mean(np.abs(torque), axis=0)) / o["effort"]),
      "roll_deg": float(np.mean(roll)),
      "board_travel": travel,
      "margin": support_margin(feet_xy, com_xy),
      **{key: lifted[site] for key, site in lifts.items()},
    }
    passed = condition_verdict(row)
    all_passed &= passed
    print(
      f"{'PASS' if passed else 'FAIL'}  {name:18s} {_fmt(row, list(lifts))}", flush=True
    )
  print("STANCE:", "PASS" if all_passed else "FAIL")
  return 0


if __name__ == "__main__":
  sys.exit(main())
