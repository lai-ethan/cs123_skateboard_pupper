"""Solve the split-stance joint angles for a given deck height.

Usage (repo root):
  FORCE_CPU=1 uv run python skateboard/tools/solve_split_stance.py \
    LEFT_Y LEFT_DEPTH RIGHT_DEPTH

LEFT_Y:      how far left of the base the deck-side feet sit [m] (0.14 today).
LEFT_DEPTH:  how far below the base the deck-side feet sit [m] (0.064 today, for
             the measured 0.103 m deck).
RIGHT_DEPTH: how far below the base the floor-side feet sit [m] (0.145 today).

Prints, per leg, the solved angles, the foot position they give, and each
joint's margin to its limit (keep every margin above 0.2 rad). The last line
is the 12 angles in SKATE_SPLIT_POSE order, ready for check_split_stance.py.
"""

import itertools
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  DEFAULT_POSE,
  JOINT_NAMES,
  get_spec,
)

left_y, left_depth, right_depth = (float(v) for v in sys.argv[1:4])

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
  on_deck = leg.endswith("_l")
  target = np.array(
    [
      default_foot[0],
      left_y if on_deck else default_foot[1],
      -left_depth if on_deck else -right_depth,
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
