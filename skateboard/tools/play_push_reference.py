"""Play the centred (tripod) ride's stroke reference open loop and print it.

No policy: every step the action is the reference itself (the `skate_push`
table in `gait_reference`, strokes of the back right foot from the tripod
layout, on its own clock and blend), converted through the action scale and
the default pose, so the PD target is the reference joint target. This checks
that the reference can roll the board before any training. Episodes start in
the tripod layout with the board already rolling at up to 0.2 m/s.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/play_push_reference.py \\
    [--envs 16] [--speed 0.3] [--deck-height 0.103] [--seconds 10] [--seed 0] \\
    [--zero-action] [--rewards] [--video out.mp4]

Builds `Mjlab-Skate-Ride-Pupper-v3` in play mode with the deck top pinned at
`--deck-height` and the command fixed at `--speed`, and prints: episode resets
and what ended them, the board's mean and final forward speed, the push foot's
(back right) share of time on the floor and the board's speed and
acceleration while it is down, the largest joint speed, the share of
steps with a motor at its torque limit, and the smallest margin of body tilt to
the `fell_over` limit. `--zero-action` holds the default pose instead (the tripod
layout, still), for comparison. `--rewards` also prints each reward term's
mean per step (weight included). `--video` saves env 0, rendered as
`render_policy.py` does.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

TASK = "Mjlab-Skate-Ride-Pupper-v3"
PUSH_FOOT = 2  # Back right, in the foot sensors' FR, FL, BR, BL order.


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
  ap.add_argument("--envs", type=int, default=16)
  ap.add_argument("--speed", type=float, default=0.3, help="board speed command")
  ap.add_argument("--deck-height", type=float, default=0.103, help="deck top, m")
  ap.add_argument("--seconds", type=float, default=10.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--zero-action", action="store_true", help="hold the layout instead")
  ap.add_argument("--rewards", action="store_true", help="print reward term means")
  ap.add_argument("--video", type=Path, help="save env 0 as this .mp4")
  args = ap.parse_args(argv)

  # Imported here so --help works without the simulator.
  import numpy as np
  import torch
  from policy_loader import fix_speed

  from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
    SkateboardParams,
  )
  from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import PUPPER_EFFORT_LIMIT
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs.mdp.actions.actions import BaseAction
  from mjlab.tasks.pupper_gait.mdp.gait import reference_offset_from_tables
  from mjlab.tasks.pupper_gait.mdp.gait_reference import (
    SKATE_PUSH_BLEND_SPEED,
    SKATE_PUSH_FREQUENCY,
    build_joint_reference_table,
  )
  from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
    GAIT_GALLOP_SPEED,
    GAIT_N_SAMPLES,
  )
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = args.envs
  cfg.seed = args.seed
  fix_speed(cfg, args.speed)
  offset = args.deck_height - SkateboardParams().deck_top_height
  cfg.events["deck_height"].params["ranges"] = (offset, offset)
  render_mode = None
  if args.video is not None:
    render_mode = "rgb_array"
    for key, value in {
      "width": 960,
      "height": 540,
      "azimuth": 50.0,
      "elevation": -18.0,
      "distance": 1.3,
    }.items():
      setattr(cfg.viewer, key, value)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)

  table = torch.tensor(
    build_joint_reference_table(GAIT_N_SAMPLES, "skate_push"),
    dtype=torch.float32,
    device=env.device,
  )
  action_term = env.action_manager.get_term("joint_pos")
  assert isinstance(action_term, BaseAction)
  robot = env.scene["robot"]
  board = env.scene["board"]
  ground = env.scene["ground_feet_contact"]
  fell_limit = float(cfg.terminations["fell_over"].params["limit_angle"])

  env.reset()
  steps = int(round(args.seconds / env.step_dt))
  resets = 0
  speed, push_floor, joint_speed, saturated, tilt = [], [], [], [], []
  reward_sums: dict[str, float] = {
    name: 0.0 for name in env.reward_manager.active_terms
  }
  reset_causes = {name: 0 for name in env.termination_manager.active_terms}
  frames = []
  with torch.no_grad():
    for _ in range(steps):
      if args.zero_action:
        action = torch.zeros(args.envs, 12, device=env.device)
      else:
        command = env.command_manager.get_command("twist")
        assert command is not None
        default = robot.data.default_joint_pos
        reference = default + reference_offset_from_tables(
          table,
          table,
          env.episode_length_buf.float() * env.step_dt,
          command[:, 0],
          command[:, 1],
          command[:, 2],
          default,
          frequency=SKATE_PUSH_FREQUENCY,
          blend_speed=SKATE_PUSH_BLEND_SPEED,
          gallop_speed=GAIT_GALLOP_SPEED,
          gallop_freq_mult=1.0,
        )
        action = (reference - action_term.offset) / action_term.scale
      _, _, terminated, truncated, _ = env.step(action)
      resets += int((terminated | truncated).sum())
      for name in reset_causes:
        reset_causes[name] += int(env.termination_manager.get_term(name).sum())
      speed.append(board.data.root_link_lin_vel_b[:, 0].cpu().numpy())
      push_floor.append((ground.data.found[:, PUSH_FOOT] > 0).cpu().numpy())
      joint_speed.append(robot.data.joint_vel.abs().max(dim=1).values.cpu().numpy())
      torque = robot.data.actuator_force.abs()
      saturated.append((torque >= 0.95 * PUPPER_EFFORT_LIMIT).any(dim=1).cpu().numpy())
      gz = robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
      tilt.append(torch.acos(-gz).cpu().numpy())
      if args.rewards:
        for i in range(args.envs):
          for name, values in env.reward_manager.get_active_iterable_terms(i):
            reward_sums[name] += values[0]
      if args.video is not None:
        frames.append(env.render())
  env.close()

  speed_a = np.stack(speed)
  print(f"task: {TASK}  {'zero action' if args.zero_action else 'push reference'}")
  print(
    f"envs: {args.envs}  seed: {args.seed}  deck top: {args.deck_height} m  "
    f"command: {args.speed} m/s  {steps} steps ({args.seconds} s)"
  )
  causes = ", ".join(f"{name} {n}" for name, n in reset_causes.items() if n)
  print(f"resets: {resets}" + (f" ({causes})" if causes else ""))
  print(
    f"board speed: mean {speed_a.mean():.3f} m/s, final {speed_a[-1].mean():.3f} m/s"
  )
  print(f"push foot (BR) on the floor: {np.mean(push_floor):.3f} of the time")
  # Thrust: how the board speed changes while the push foot is down. Steps
  # across a reset are left out.
  floor_a = np.stack(push_floor)
  accel = np.diff(speed_a, axis=0) / env.step_dt
  pushing = floor_a[1:] & floor_a[:-1]
  if pushing.any():
    print(
      f"while the push foot is down: board speed {speed_a[floor_a].mean():.3f} m/s,"
      f" acceleration {accel[pushing].mean():+.3f} m/s^2"
    )
  print(f"peak joint speed: {np.max(joint_speed):.2f} rad/s")
  print(f"saturated fraction: {np.mean(saturated):.3f}")
  margin = fell_limit - float(np.max(tilt))
  print(
    f"min tilt margin to fell_over: {margin:.3f} rad ({math.degrees(margin):.1f} deg;"
    f" limit {fell_limit} rad)"
  )
  if args.rewards:
    total = 0.0
    print("reward term means per step (weight included):")
    for name, value in reward_sums.items():
      mean = value / (steps * args.envs)
      total += mean
      print(f"  {name:22s} {mean:+.4f}")
    print(f"  {'total':22s} {total:+.4f}")
  if args.video is not None:
    import imageio.v2 as imageio

    args.video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.video, frames, fps=round(1 / env.step_dt), macro_block_size=1)
    print(f"wrote {args.video}: {len(frames)} frames")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
