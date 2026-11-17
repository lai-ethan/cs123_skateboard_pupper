"""Solve the centred-stance joint angles for a given foot spread and depth.

Usage (repo root):
  FORCE_CPU=1 uv run python skateboard/tools/solve_stance.py \
    FOOT_Y DEPTH [--back-y METRES] [--front-dx METRES]

FOOT_Y:     how far either side of the base's centre line each foot sits [m]
            (0.066 today; the deck is 0.161 m wide).
DEPTH:      how far below the base every foot sits [m] (0.115 today).
--back-y:   FOOT_Y for the back feet only (default FOOT_Y; 0.060 today).
--front-dx: shift of the front feet from their default x, forward positive
            (default 0; -0.02 today).

The back feet keep their default x. Prints, per leg, the solved angles, the
foot position they give, and each joint's margin to its limit (keep every
margin above 0.1 rad). The last line is the 12 angles in SKATE_POSE order,
ready for check_stance.py.
"""

import argparse
import itertools

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  DEFAULT_POSE,
  JOINT_NAMES,
  get_spec,
)

parser = argparse.ArgumentParser()
parser.add_argument("foot_y", type=float)
parser.add_argument("depth", type=float)
parser.add_argument("--back-y", type=float, default=None)
parser.add_argument("--front-dx", type=float, default=0.0)
args = parser.parse_args()
back_y = args.foot_y if args.back_y is None else args.back_y

model = get_spec().compile()
data = mujoco.MjData(model)
base = model.body("base_link").id


def foot_in_base(leg: str, angles: np.ndarray) -> np.ndarray:
  data.qpos[:] = 0.0
  data.qpos[3] = 1.0
  for i, angle in enumerate(angles):
    data.qpos[model.joint(f"leg_{leg}_{i + 1}").qposadr[0]] = angle
  mujoco.mj_kinematics(model, data)
  return data.site(f"leg_{leg}_3_foot_site").xpos - data.xpos[base]


pose: list[float] = []
for leg in ("front_r", "front_l", "back_r", "back_l"):
  names = [f"leg_{leg}_{i}" for i in (1, 2, 3)]
  default = np.array([DEFAULT_POSE[JOINT_NAMES.index(name)] for name in names])
  lower = np.array([model.joint(name).range[0] for name in names])
  upper = np.array([model.joint(name).range[1] for name in names])
  default_foot = foot_in_base(leg, default)
  side = 1.0 if leg.endswith("_l") else -1.0
  front = leg.startswith("front")
  target = np.array(
    [
      default_foot[0] + (args.front_dx if front else 0.0),
      side * (args.foot_y if front else back_y),
      -args.depth,
    ]
  )

  def residual(angles, leg=leg, target=target, default=default):
    # Reach the target; a small pull toward the default pose picks one solution.
    return np.concatenate(
      [foot_in_base(leg, angles) - target, 1e-3 * (angles - default)]
    )

  best = None
  for offset in itertools.product((-0.5, 0, 0.5), (-0.9, -0.4, 0, 0.3), (-0.3, 0, 0.6)):
    seed = np.clip(default + offset, lower + 1e-3, upper - 1e-3)
    result = least_squares(residual, seed, bounds=(lower, upper))
    if best is None or result.cost < best.cost - 1e-12:
      best = result
  assert best is not None
  angles = best.x.round(3)
  print(
    f"{leg}: angles {angles} foot {foot_in_base(leg, angles).round(4)} "
    f"margin {np.minimum(angles - lower, upper - angles).round(2)}"
  )
  pose += [float(a) for a in angles]

print(" ".join(f"{a + 0.0:g}" for a in pose))
