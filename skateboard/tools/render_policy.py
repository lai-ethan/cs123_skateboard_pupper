"""Roll out a trained skate policy headless and save a video plus stills.

Takes a training checkpoint (`model_*.pt`) or an exported `policy.json`.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/render_policy.py \\
    Mjlab-Skate-Ride-Pupper-v3 skateboard/policies/ride_29p4j81t_model_1499.pt \\
    out/ride.mp4 --speed 0.4 [--yaw-rate 0.5]

`--yaw-rate` is the yaw command in rad/s; it defaults to 0, the only yaw the
other tasks trained on. On the steer task 0 rides straight with the heading
hold on, as in the ride, and a carve command (0.4 to 0.6, either sign) reaches
the policy as it is and asks the board to turn at a tenth of it.
"""

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import torch
from policy_loader import load_policy

ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
ap.add_argument("task")
ap.add_argument("policy", help="model_*.pt checkpoint or exported policy.json")
ap.add_argument("out", type=Path, help="output .mp4; stills are saved next to it")
ap.add_argument("--steps", type=int, default=500, help="50 per second of sim")
ap.add_argument("--speed", type=float, help="fixed board speed command, m/s (ride)")
ap.add_argument(
  "--yaw-rate", type=float, default=0.0, help="yaw (steer: carve) command, rad/s"
)
ap.add_argument("--azimuth", type=float, default=50.0)
ap.add_argument("--elevation", type=float, default=-18.0)
ap.add_argument("--distance", type=float, default=1.3)
args = ap.parse_args()

viewer = {
  "width": 960,
  "height": 540,
  "azimuth": args.azimuth,
  "elevation": args.elevation,
  "distance": args.distance,
}
raw, env, policy = load_policy(
  args.task,
  Path(args.policy),
  args.speed,
  render_mode="rgb_array",
  viewer=viewer,
  yaw_rate=args.yaw_rate,
)

obs = env.get_observations()
obs = obs[0] if isinstance(obs, tuple) else obs
board = raw.scene["board"]
x0 = board.data.root_link_pos_w[0, 0].item()
yaw0 = board.data.heading_w[0].item()
frames, resets = [], 0
with torch.no_grad():
  for _ in range(args.steps):
    obs, _, done, _ = env.step(policy(obs))
    resets += int(done.any())
    frames.append(raw.render())
travel = board.data.root_link_pos_w[0, 0].item() - x0
turned = board.data.heading_w[0].item() - yaw0

args.out.parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(args.out, frames, fps=50, macro_block_size=1)
for k in (args.steps // 4, args.steps // 2, 3 * args.steps // 4):
  imageio.imwrite(args.out.with_name(f"{args.out.stem}_{k}.png"), frames[k])
print(
  f"wrote {args.out}: {len(frames)} frames, board travel {travel:.2f} m, resets {resets}"
)
if args.yaw_rate:
  # Wrapped to (-pi, pi]: a turn of more than half a circle reads short.
  turned = (turned + math.pi) % (2 * math.pi) - math.pi
  print(f"board heading change {turned:.2f} rad (yaw command {args.yaw_rate})")
