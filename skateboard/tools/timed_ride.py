"""Measure what one timed voice command does in sim: travel, turn and coast.

Gemini Pupper has no distance sensing, so a voice command is a speed and a
yaw (carve) command held for a few seconds, then zero. This tool rolls that
out and prints one JSON line, so the voice tools can quote real numbers.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py \\
    skateboard/policies/split_steer_v6_max0.3_policy.json \\
    [--yaw-rate 0.3] [--ride-s 5] [--coast-s 4] [--envs 16] [--seed 0]

The ride phase commands `--speed` and `--yaw-rate` in every env. The coast
phase commands zero (every env is made a standing env), as the robot's
controller sees after the voice command ends. Distances are along the
board's path. An env that resets at any point is dropped and counted in
`resets`. Each env draws its own board and gains from the task's ranges (the
carpet ranges held at the hard floor, see `hold_hard_floor` in
policy_loader.py), so the mean and the max differ. `--wheel-drag X` pins
the wheel drag at X times nominal in every env instead, as the ride audit's
`wheel_drag_xN` rows do; `wheel_drag` in the output is X, or null without it.

Straightness, over the ride only (not the coast). Forward is the board's
heading at the first ride step and left is 90 degrees to its left.
- `side_m_mean`: the board's leftward displacement at the end of the ride, in
  m, signed (negative is right), mean over envs.
- `side_m_abs_max`: the largest size of that displacement in any env.
- `side_per_m_mean`: leftward over forward displacement at the end of the
  ride, mean over the envs that went at least 0.05 m forward (null if none).
  The tangent of the angle the board drifted off its first heading.
- `yaw_swing_deg_mean`: the nose swing. The board's yaw over the ride after
  its first 1.0 s, less its best straight line (so a steady turn counts as
  no swing), peak to peak in degrees, mean over envs.
- `side_swing_cm_mean`: the same for the board's leftward position, in cm.
Both swings are null for a ride shorter than 1.5 s.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from policy_loader import load_policy
from ride_audit import _wheel_drag

STEER = "Mjlab-SkateSplit-Steer-Pupper-v3"
# Swings skip the ride's first second (the push-off), and need half a second
# after it.
SWING_SKIP_S, SWING_MIN_RIDE_S = 1.0, 1.5
SIDE_PER_M_MIN_FORWARD = 0.05


def path_length(pos: np.ndarray) -> np.ndarray:
  """Distance along the path so far. `pos` is (T+1, E, 2); returns (T+1, E)."""
  steps = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
  return np.concatenate([np.zeros((1, pos.shape[1])), np.cumsum(steps, axis=0)])


def turn_so_far(yaw: np.ndarray) -> np.ndarray:
  """Heading change since step 0 in rad, unwrapped. `yaw` is (T+1, E)."""
  return np.unwrap(yaw, axis=0) - yaw[0]


def forward_and_left(pos: np.ndarray, yaw0: np.ndarray) -> np.ndarray:
  """Displacement since step 0 along heading `yaw0` and 90 degrees left of it.

  `pos` is (T+1, E, 2) and `yaw0` is (E,); returns (T+1, E, 2), forward first.
  """
  d = pos - pos[0]
  c, s = np.cos(yaw0), np.sin(yaw0)
  fwd = d[..., 0] * c + d[..., 1] * s
  left = -d[..., 0] * s + d[..., 1] * c
  return np.stack([fwd, left], axis=-1)


def detrended_swing(x: np.ndarray) -> np.ndarray:
  """Peak to peak of each column of `x` (N, E) less its least-squares line."""
  t = np.arange(x.shape[0], dtype=float)
  a = np.stack([t, np.ones_like(t)], axis=1)
  coef, *_ = np.linalg.lstsq(a, x, rcond=None)
  rest = x - a @ coef
  return rest.max(axis=0) - rest.min(axis=0)


def summarise(
  pos: np.ndarray, yaw: np.ndarray, ride_steps: int, step_dt: float, keep: np.ndarray
) -> dict:
  """Travel, turn and coast over the envs in `keep` (a bool mask over E).

  The ride is steps 0 to `ride_steps`; the coast is the rest.
  """
  if not keep.any():
    raise ValueError("every env reset; nothing to measure")
  dist = path_length(pos)[:, keep]
  turn = np.degrees(turn_so_far(yaw)[:, keep])
  ride_s = ride_steps * step_dt
  ride = dist[ride_steps]
  coast = dist[-1] - ride
  return {
    "envs": int(keep.sum()),
    "ride_m_mean": float(ride.mean()),
    "ride_m_max": float(ride.max()),
    "m_per_s": float(ride.mean() / ride_s),
    "turn_deg_mean": float(turn[ride_steps].mean()),
    "turn_deg_per_s": float(turn[ride_steps].mean() / ride_s),
    "coast_m_mean": float(coast.mean()),
    "coast_m_max": float(coast.max()),
    "coast_turn_deg_mean": float((turn[-1] - turn[ride_steps]).mean()),
    **straightness(pos[:, keep], yaw[:, keep], ride_steps, step_dt),
  }


def straightness(
  pos: np.ndarray, yaw: np.ndarray, ride_steps: int, step_dt: float
) -> dict:
  """Side drift and swings over the ride (steps 0 to `ride_steps`).

  See the module docstring for each field. `pos` is (T+1, E, 2), `yaw` is
  (T+1, E), every env kept.
  """
  pos, yaw = pos[: ride_steps + 1], yaw[: ride_steps + 1]
  frame = forward_and_left(pos, yaw[0])
  fwd, left = frame[-1, :, 0], frame[-1, :, 1]
  moved = fwd >= SIDE_PER_M_MIN_FORWARD
  out = {
    "side_m_mean": float(left.mean()),
    "side_m_abs_max": float(np.abs(left).max()),
    "side_per_m_mean": float((left[moved] / fwd[moved]).mean())
    if moved.any()
    else None,
    "yaw_swing_deg_mean": None,
    "side_swing_cm_mean": None,
  }
  if ride_steps * step_dt >= SWING_MIN_RIDE_S:
    skip = round(SWING_SKIP_S / step_dt)
    heading = np.degrees(np.unwrap(yaw, axis=0))[skip:]
    out["yaw_swing_deg_mean"] = float(detrended_swing(heading).mean())
    side_cm = 100.0 * frame[skip:, :, 1]
    out["side_swing_cm_mean"] = float(detrended_swing(side_cm).mean())
  return out


def configure_drag(env_cfg: Any, wheel_drag: float | None) -> None:
  """Pin the wheel drag at `wheel_drag` times nominal, or leave it (None).

  The ride audit's `wheel_drag_xN` code path: both wheel spin terms (damping
  and frictionloss) scaled by the same factor in every env, after the task's
  own draw.
  """
  if wheel_drag is not None:
    _wheel_drag(env_cfg, wheel_drag)


def main() -> None:
  ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
  ap.add_argument("policy", help="model_*.pt checkpoint or exported policy.json")
  ap.add_argument("--task", default=STEER)
  ap.add_argument("--speed", type=float, default=0.3, help="board speed, m/s")
  ap.add_argument("--yaw-rate", type=float, default=0.0, help="carve command, rad/s")
  ap.add_argument("--ride-s", type=float, default=5.0)
  ap.add_argument("--coast-s", type=float, default=4.0)
  ap.add_argument("--envs", type=int, default=16)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument(
    "--wheel-drag",
    type=float,
    default=None,
    help="pin the wheel drag at this many times nominal in every env",
  )
  args = ap.parse_args()

  raw, env, policy = load_policy(
    args.task,
    Path(args.policy),
    args.speed,
    num_envs=args.envs,
    seed=args.seed,
    yaw_rate=args.yaw_rate,
    configure=lambda cfg: configure_drag(cfg, args.wheel_drag),
  )
  term = raw.command_manager.get_term("twist")
  board = raw.scene["board"]
  ride_steps = round(args.ride_s / raw.step_dt)
  coast_steps = round(args.coast_s / raw.step_dt)

  def snapshot() -> tuple[np.ndarray, np.ndarray]:
    return (
      board.data.root_link_pos_w[:, :2].cpu().numpy().copy(),
      board.data.heading_w.cpu().numpy().copy(),
    )

  obs = env.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  pos, yaw = [], []
  p, y = snapshot()
  pos.append(p)
  yaw.append(y)
  keep = np.ones(args.envs, dtype=bool)
  with torch.no_grad():
    for k in range(ride_steps + coast_steps):
      if k >= ride_steps:
        # Zero command from here on: a standing env is zeroed every step, and
        # no resample may hand the speed back.
        term.is_standing_env[:] = True
        term.time_left[:] = 1e9
      obs, _, done, _ = env.step(policy(obs))
      keep &= ~done.cpu().numpy().astype(bool)
      if k >= ride_steps:
        command = term.vel_command_b.abs().amax(dim=1).cpu().numpy()
        assert (command[keep] == 0.0).all(), "the coast still has a command"
      p, y = snapshot()
      pos.append(p)
      yaw.append(y)

  row = summarise(np.stack(pos), np.stack(yaw), ride_steps, raw.step_dt, keep)
  out = {"stick": args.yaw_rate, "resets": int((~keep).sum())}
  out.update({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
  out["wheel_drag"] = args.wheel_drag
  print(json.dumps(out))


if __name__ == "__main__":
  main()
