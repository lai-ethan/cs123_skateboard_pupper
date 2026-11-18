"""Audit how well a ride policy keeps riding when the sim is wrong.

Rolls out a ride policy at a fixed board speed under off-nominal conditions
(longer rides, grip on either side, board placement, shoves, wheel drag,
truck stiffness, deck height, motor gain), one change per row, and prints one
PASS/FAIL row per condition. Every row keeps the task's play-mode
randomization (training DR ranges, reset noise, observation noise, latency)
and overrides one thing. A row passes when at most 6.25% of the envs reset
(1 of 16), the board's mean speed is 0.7 to 1.3 times the command, and every
deck foot touches the deck at least 90% of the time in the worst env. It
trains nothing and changes no task.

Which feet are deck-side and floor-side comes from the task's contact sensors:
the geoms of `deck_feet_contact` (narrowed to the `foot_ids` of a termination
on that sensor, as in the tripod ride) are the deck feet, the other geoms of
`ground_feet_contact` the floor feet. Feet have geom priority 1, so a foot's
own friction is the friction of every contact it makes; the grip rows set it.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/ride_audit.py \\
    Mjlab-SkateSplit-Ride-Pupper-v3 \\
    skateboard/policies/split_ride_iznjwbka_policy.json --speed 0.3 \\
    [--num-envs 16] [--seed 0] [--rows nominal_10s,kp_x0.7] [--steps 500] \\
    [--json out.json]

Metrics start after step 100 (2 s). `--steps` replaces every row's step count
(for quick checks). A row that cannot be applied to the task prints `n/a` and
why. The last line is `RIDE AUDIT: PASS` or `RIDE AUDIT: FAIL (n rows)`, and
the exit code is 0 or 1. A pass here says nothing about the real board's
wheels, grip tape or the legs' backlash.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SKIP, KICK_STEP = 100, 250
# Board centre offset window at the end of a row: the last 2 s.
LATERAL_WINDOW = 100
DECK_SENSOR, GROUND_SENSOR = "deck_feet_contact", "ground_feet_contact"
WHEEL_SPIN_JOINTS = (r"wheel_.*_spin",)


@dataclass(frozen=True)
class Row:
  """One audit condition: `kind` names the override, `value` its size."""

  steps: int
  kind: str = "none"
  value: float = 0.0


ROWS: dict[str, Row] = {
  "nominal_10s": Row(500),
  "nominal_30s": Row(1500),
  "deck_grip_0.4": Row(500, "deck_grip", 0.4),
  "floor_grip_0.4": Row(500, "floor_grip", 0.4),
  "board_y_+2cm": Row(500, "board_y", 0.02),
  "board_y_-2cm": Row(500, "board_y", -0.02),
  "board_yaw_0.15": Row(500, "board_yaw", 0.15),
  "shove_side_0.5": Row(500, "shove_y", 0.5),
  "shove_fwd_0.5": Row(500, "shove_x", 0.5),
  "wheel_drag_x2": Row(500, "wheel_drag", 2.0),
  "wheel_drag_x3": Row(500, "wheel_drag", 3.0),
  "truck_x0.5": Row(500, "truck", 0.5),
  "truck_x4": Row(500, "truck", 4.0),
  "deck_93mm": Row(500, "deck_top", 0.093),
  "deck_113mm": Row(500, "deck_top", 0.113),
  "kp_x0.7": Row(500, "kp", 0.7),
}

RIDE_LIMITS: dict[str, float] = {
  "max_reset_share": 0.0625,  # of the envs; rounded up, so 1 of 16
  "min_speed_ratio": 0.7,
  "max_speed_ratio": 1.3,
  "min_deck_contact": 0.9,  # worst env, least-planted deck foot
}


class NotApplicable(Exception):
  """A row's override has nothing to act on in this task."""


def max_resets(num_envs: int) -> int:
  """Envs allowed to reset: RIDE_LIMITS' share of `num_envs`, rounded up."""
  # Rounded to 9 places first so 0.0625 * 16 is 1, not 1.0000000001 -> 2.
  return math.ceil(round(RIDE_LIMITS["max_reset_share"] * num_envs, 9))


def row_verdict(metrics: dict[str, Any], num_envs: int) -> bool:
  """True when a row's metrics meet every entry of RIDE_LIMITS."""
  return (
    metrics["envs_with_reset"] <= max_resets(num_envs)
    and RIDE_LIMITS["min_speed_ratio"]
    <= metrics["speed_ratio_mean"]
    <= RIDE_LIMITS["max_speed_ratio"]
    and metrics["deck_contact_worst_env"] >= RIDE_LIMITS["min_deck_contact"]
  )


@dataclass(frozen=True)
class FootSides:
  """Foot geom names by side, and the deck feet's slots in the deck sensor."""

  deck: tuple[str, ...]
  floor: tuple[str, ...]
  deck_slots: tuple[int, ...]


def _sensor_geoms(env_cfg: Any, sensor_name: str, geom_names: list[str]) -> list[str]:
  sensors = {s.name: s for s in env_cfg.scene.sensors or ()}
  if sensor_name not in sensors:
    raise NotApplicable(f"task has no {sensor_name} sensor")
  pattern = sensors[sensor_name].primary.pattern
  patterns = [pattern] if isinstance(pattern, str) else list(pattern)
  # Same order as the sensor's slots: the robot's geom order.
  return [n for n in geom_names if any(re.fullmatch(p, n) for p in patterns)]


def foot_sides(env_cfg: Any, geom_names: list[str] | None = None) -> FootSides:
  """Deck-side and floor-side feet from the task's contact sensor config.

  The deck feet are the geoms of the deck sensor; when a termination reads
  that sensor with `foot_ids` (the tripod ride's support feet), only those
  slots. The floor feet are the floor sensor's other geoms. `geom_names` is the
  robot's geom names in model order (read from its spec when not given).
  """
  if geom_names is None:
    spec = env_cfg.scene.entities["robot"].spec_fn()
    geom_names = [g.name for g in spec.geoms]
  deck = _sensor_geoms(env_cfg, DECK_SENSOR, geom_names)
  ground = _sensor_geoms(env_cfg, GROUND_SENSOR, geom_names)
  slots = tuple(range(len(deck)))
  for term in env_cfg.terminations.values():
    params = getattr(term, "params", None) or {}
    if params.get("sensor_name") == DECK_SENSOR and params.get("foot_ids"):
      slots = tuple(params["foot_ids"])
      break
  deck_feet = tuple(deck[i] for i in slots)
  floor_feet = tuple(n for n in ground if n not in deck_feet)
  if not deck_feet:
    raise NotApplicable(f"{DECK_SENSOR} matches no robot geom")
  return FootSides(deck_feet, floor_feet, slots)


def _event(events: dict[str, Any], name: str, key: str) -> Any:
  if name not in events or key not in events[name].params:
    raise NotApplicable(f"no {name!r} event with a {key!r} param")
  return events[name]


def _foot_grip(env_cfg: Any, name: str, geoms: tuple[str, ...], mu: float) -> None:
  """Pin `geoms`' friction to `mu`, after the task's own foot friction draw."""
  from mjlab.envs.mdp import dr
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  if not geoms:
    raise NotApplicable("task has no feet on that side")
  # Added last, so it runs after foot_friction and the other feet keep it.
  env_cfg.events[name] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geoms),
      "operation": "abs",
      "ranges": (mu, mu),
    },
  )


def _wheel_drag(env_cfg: Any, scale: float) -> None:
  """Scale the wheel spin joints' damping and frictionloss by `scale`."""
  from mjlab.envs.mdp import dr
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  if "board" not in env_cfg.scene.entities:
    raise NotApplicable("task has no board")
  for name, func in (
    ("wheel_damping", dr.joint_damping),
    ("wheel_drag", dr.joint_friction),
  ):
    env_cfg.events[f"audit_{name}"] = EventTermCfg(
      mode="startup",
      func=func,
      params={
        "asset_cfg": SceneEntityCfg("board", joint_names=WHEEL_SPIN_JOINTS),
        "operation": "scale",
        "ranges": (scale, scale),
      },
    )


def configure_row(env_cfg: Any, row: Row, sides: FootSides) -> None:
  """Apply `row`'s one override to a play-mode env config.

  Raises NotApplicable when the task has nothing for the override to act on.
  Shoves are applied during the rollout, not here.
  """
  events = env_cfg.events
  if row.kind in ("none", "shove_x", "shove_y"):
    return
  if row.kind == "deck_grip":
    _foot_grip(env_cfg, "audit_deck_grip", sides.deck, row.value)
  elif row.kind == "floor_grip":
    _foot_grip(env_cfg, "audit_floor_grip", sides.floor, row.value)
  elif row.kind == "board_y":
    term = _event(events, "reset_board", "pose_range")
    term.params["pose_range"] = {
      **term.params["pose_range"],
      "y": (row.value, row.value),
    }
  elif row.kind == "board_yaw":
    # On top of the task's own yaw noise.
    term = _event(events, "reset_board", "pose_range")
    lo, hi = term.params["pose_range"].get("yaw", (0.0, 0.0))
    term.params["pose_range"] = {
      **term.params["pose_range"],
      "yaw": (lo + row.value, hi + row.value),
    }
  elif row.kind == "wheel_drag":
    _wheel_drag(env_cfg, row.value)
  elif row.kind == "truck":
    _event(events, "truck_stiffness", "ranges").params["ranges"] = (
      row.value,
      row.value,
    )
  elif row.kind == "deck_top":
    from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
      SkateboardParams,
    )

    term = _event(events, "deck_height", "ranges")
    offset = row.value - SkateboardParams().deck_top_height
    term.params["ranges"] = (offset, offset)
  elif row.kind == "kp":
    _event(events, "pd_gains", "kp_range").params["kp_range"] = (row.value, row.value)
  else:
    raise ValueError(f"unknown row kind {row.kind!r}")


def _wrap(angle: Any) -> Any:
  return np.angle(np.exp(1j * np.asarray(angle)))


def summarize(
  done: np.ndarray,
  deck_contact: np.ndarray,
  board_speed: np.ndarray,
  lateral: np.ndarray,
  heading_step: np.ndarray,
  speed: float,
) -> dict[str, Any]:
  """Row metrics from the measured steps, each array [steps, envs, ...].

  `done` marks resets, `deck_contact` the deck feet's contact (1 on the deck),
  `board_speed` the board's speed along its own heading, `lateral` the board
  centre's offset from the robot in the robot's heading frame, `heading_step`
  the board's heading change over each step (ignored on reset steps).
  """
  resets = done.sum(axis=0)
  per_env = deck_contact.mean(axis=0)  # [envs, deck feet]
  env_speed = board_speed.mean(axis=0) / speed
  late = lateral[-LATERAL_WINDOW:].mean(axis=0)
  drift = np.abs(np.where(done, 0.0, heading_step).sum(axis=0))
  return {
    "envs_with_reset": int((resets > 0).sum()),
    "resets_total": int(resets.sum()),
    "deck_contact_mean": [float(x) for x in deck_contact.mean(axis=(0, 1))],
    "deck_contact_min_env": [float(x) for x in per_env.min(axis=0)],
    "deck_contact_worst_env": float(per_env.min()),
    "speed_ratio_mean": float(board_speed.mean() / speed),
    "speed_ratio_min_env": float(env_speed.min()),
    "speed_ratio_max_env": float(env_speed.max()),
    "lateral_offset_min": float(late.min()),
    "lateral_offset_max": float(late.max()),
    "heading_drift_max": float(drift.max()),
    "heading_drift_mean": float(drift.mean()),
  }


def run_row(
  task: str,
  policy_path: Path,
  speed: float,
  row: Row,
  num_envs: int,
  seed: int,
) -> dict[str, Any]:
  """Roll out one row and return its metrics (raises NotApplicable)."""
  import torch
  from policy_loader import load_policy

  from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

  found: dict[str, FootSides] = {}

  def configure(env_cfg: Any) -> None:
    found["sides"] = foot_sides(env_cfg)
    configure_row(env_cfg, row, found["sides"])

  raw, env, policy = load_policy(
    task, policy_path, speed, num_envs=num_envs, seed=seed, configure=configure
  )
  sides = found["sides"]
  robot, board = raw.scene["robot"], raw.scene["board"]
  deck_sensor = raw.scene[DECK_SENSOR]
  slot_names = [deck_sensor.primary_names[i].split("/")[-1] for i in sides.deck_slots]
  assert slot_names == list(sides.deck), (slot_names, sides.deck)
  terms = raw.termination_manager
  term_counts = {name: 0 for name in terms.active_terms}
  obs = env.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  # Shoves alternate in sign by env: even envs +, odd envs -.
  sign = torch.ones(num_envs, device=raw.device)
  sign[1::2] = -1.0
  shove_axis = {"shove_x": 0, "shove_y": 1}.get(row.kind)
  early_resets = 0
  done_l, deck_l, speed_l, lat_l, dyaw_l = [], [], [], [], []
  heading = board.data.heading_w.cpu().numpy()
  with torch.no_grad():
    for step in range(row.steps):
      if shove_axis is not None and step == KICK_STEP:
        vel = robot.data.root_link_vel_w.clone()
        vel[:, shove_axis] += row.value * sign
        robot.write_root_link_velocity_to_sim(vel)
      obs, _, done, _ = env.step(policy(obs))
      d = done.cpu().numpy().astype(bool)
      new_heading = board.data.heading_w.cpu().numpy()
      dyaw = _wrap(new_heading - heading)
      heading = new_heading
      if step < SKIP:
        early_resets += int(d.sum())
        continue
      if d.any():
        mask = torch.as_tensor(d, device=raw.device)
        for name in term_counts:
          term_counts[name] += int(terms.get_term(name)[mask].sum().item())
      found_deck = deck_sensor.data.found[:, list(sides.deck_slots)]
      rel = board.data.root_link_pos_w - robot.data.root_link_pos_w
      rel_b = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), rel)
      done_l.append(d)
      deck_l.append((found_deck > 0).float().cpu().numpy())
      speed_l.append(board.data.root_link_lin_vel_b[:, 0].cpu().numpy())
      lat_l.append(rel_b[:, 1].cpu().numpy())
      dyaw_l.append(dyaw)
  env.close()
  metrics = summarize(
    np.stack(done_l),
    np.stack(deck_l),
    np.stack(speed_l),
    np.stack(lat_l),
    np.stack(dyaw_l),
    speed,
  )
  metrics["terminations"] = {k: v for k, v in term_counts.items() if v}
  metrics["resets_before_skip"] = early_resets
  metrics["deck_feet"] = list(sides.deck)
  return metrics


def _fmt(m: dict[str, Any]) -> str:
  terms = ",".join(f"{k}:{v}" for k, v in m["terminations"].items()) or "-"
  deck = "/".join(
    f"{a:.3f}({b:.2f})"
    for a, b in zip(m["deck_contact_mean"], m["deck_contact_min_env"], strict=True)
  )
  return (
    f"resets {m['envs_with_reset']:2d}/{m['resets_total']:<2d} {terms:24s} "
    f"deck {deck}  speed {m['speed_ratio_mean']:.2f} "
    f"[{m['speed_ratio_min_env']:.2f},{m['speed_ratio_max_env']:.2f}]  "
    f"lat {m['lateral_offset_min']:.3f}..{m['lateral_offset_max']:.3f} m  "
    f"drift {m['heading_drift_max']:.2f} ({m['heading_drift_mean']:.2f}) rad"
  )


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
  parser.add_argument("task", help="ride task id")
  parser.add_argument("policy", type=Path, help="model_*.pt or policy.json")
  parser.add_argument("--speed", type=float, required=True, help="m/s")
  parser.add_argument("--num-envs", type=int, default=16, help="envs per row")
  parser.add_argument("--seed", type=int, default=0, help="env random seed")
  parser.add_argument(
    "--rows",
    default=",".join(ROWS),
    help=f"comma-separated, from: {', '.join(ROWS)}",
  )
  parser.add_argument("--steps", type=int, help="replace every row's step count")
  parser.add_argument("--json", type=Path, help="write all rows and metrics here")
  args = parser.parse_args(argv)
  names = [name for name in args.rows.split(",") if name]
  unknown = [name for name in names if name not in ROWS]
  if unknown or not names:
    parser.error(f"unknown row(s) {unknown}; choose from {list(ROWS)}")
  if args.speed <= 0:
    parser.error(f"--speed ({args.speed}) must be positive")
  if args.num_envs < 1:
    parser.error(f"--num-envs ({args.num_envs}) must be at least 1")
  if args.steps is not None and args.steps <= SKIP:
    parser.error(f"--steps ({args.steps}) must be greater than {SKIP}")

  print(f"task: {args.task}  policy: {args.policy}  speed: {args.speed} m/s")
  print(
    f"envs: {args.num_envs}  seed: {args.seed}  measured after step {SKIP}; "
    f"limits {RIDE_LIMITS} (at most {max_resets(args.num_envs)} envs reset)"
  )
  print("columns: envs reset/total resets, deck foot contact mean(worst env),")
  print("  speed ratio mean [min env, max env], board lateral offset over the")
  print("  last 2 s (env min..max), board heading drift max (mean)")
  results: dict[str, Any] = {}
  failed = 0
  for name in names:
    row = ROWS[name]
    if args.steps is not None:
      row = Row(args.steps, row.kind, row.value)
    start = time.time()
    try:
      metrics = run_row(
        args.task, args.policy, args.speed, row, args.num_envs, args.seed
      )
    except NotApplicable as reason:
      results[name] = {"verdict": "n/a", "reason": str(reason)}
      print(f"n/a   {name:15s} {reason}", flush=True)
      continue
    passed = row_verdict(metrics, args.num_envs)
    failed += not passed
    metrics.update(
      verdict="PASS" if passed else "FAIL",
      steps=row.steps,
      wall_s=round(time.time() - start, 1),
    )
    results[name] = metrics
    print(f"{metrics['verdict']}  {name:15s} {_fmt(metrics)}", flush=True)
  if args.json is not None:
    args.json.write_text(
      json.dumps(
        {
          "task": args.task,
          "policy": str(args.policy),
          "speed": args.speed,
          "num_envs": args.num_envs,
          "seed": args.seed,
          "limits": RIDE_LIMITS,
          "rows": results,
        },
        indent=2,
      )
    )
  print("RIDE AUDIT:", f"FAIL ({failed} rows)" if failed else "PASS")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
