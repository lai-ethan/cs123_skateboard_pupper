"""Roll out a trained skate policy headless and save a video plus stills.

Takes a training checkpoint (`model_*.pt`) or an exported `policy.json`.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/render_policy.py \\
    Mjlab-Skate-Ride-Pupper-v3 skateboard/policies/ride_29p4j81t_model_1499.pt \\
    out/ride.mp4 --speed 0.4
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.pupper.export import json_forward, to_robot_order
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("task")
ap.add_argument("policy", help="model_*.pt checkpoint or exported policy.json")
ap.add_argument("out", type=Path, help="output .mp4; stills are saved next to it")
ap.add_argument("--steps", type=int, default=500, help="50 per second of sim")
ap.add_argument("--speed", type=float, help="fixed board speed command, m/s (ride)")
ap.add_argument("--azimuth", type=float, default=50.0)
ap.add_argument("--elevation", type=float, default=-18.0)
ap.add_argument("--distance", type=float, default=1.3)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(args.task, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 960, 540
env_cfg.viewer.azimuth = args.azimuth
env_cfg.viewer.elevation = args.elevation
env_cfg.viewer.distance = args.distance
if args.speed is not None:
  env_cfg.commands["twist"].ranges.lin_vel_x = (args.speed, args.speed)
agent_cfg = load_rl_cfg(args.task)
raw = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)

if args.policy.endswith(".json"):
  exported = json.load(open(args.policy))
  history = exported["observation_history"]
  frame = exported["single_observation_size"]

  def policy(obs):
    x = to_robot_order(obs["actor"], history, frame)
    return torch.as_tensor(
      json_forward(exported, x), dtype=torch.float32, device=device
    )

else:
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.policy, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

obs = env.get_observations()
obs = obs[0] if isinstance(obs, tuple) else obs
board = raw.scene["board"]
x0 = board.data.root_link_pos_w[0, 0].item()
frames, resets = [], 0
with torch.no_grad():
  for _ in range(args.steps):
    obs, _, done, _ = env.step(policy(obs))
    resets += int(done.any())
    frames.append(raw.render())
travel = board.data.root_link_pos_w[0, 0].item() - x0

args.out.parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(args.out, frames, fps=50, macro_block_size=1)
for k in (args.steps // 4, args.steps // 2, 3 * args.steps // 4):
  imageio.imwrite(args.out.with_name(f"{args.out.stem}_{k}.png"), frames[k])
print(
  f"wrote {args.out}: {len(frames)} frames, board travel {travel:.2f} m, resets {resets}"
)
