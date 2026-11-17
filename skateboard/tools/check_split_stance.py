"""Drop Pupper on the board in a candidate split stance and watch whether it holds.

Usage (repo root):
  FORCE_CPU=1 uv run python skateboard/tools/check_split_stance.py \
    [--deck-height METRES] [BOARD_Y ROBOT_Z A1 ... A12]

BOARD_Y: board centre, metres left of the base (SKATE_SPLIT_BOARD_OFFSET).
ROBOT_Z: spawn height of the base [m] (SKATE_SPLIT_HEIGHT + 0.01).
A1..A12: the 12 angles printed by solve_split_stance.py (SKATE_SPLIT_POSE).
--deck-height: floor to top of deck [m]; defaults to the board model's
  measured height. Try both ends of the training range (0.068 and 0.113).
With no positional arguments it checks SKATE_SPLIT_KEYFRAME (spawned rolled).
Given angles, it spawns level.

Holds the pose with zero actions for 10 s on a free-rolling board and prints
the state once a second. Accept the stance if roll and height are steady over
the last prints, the board has not moved, both left feet were on the deck and
both right feet on the floor at least 95% of the time, and the deck roll ends
under 0.05 rad. The last print's roll and base z are the new SKATE_SPLIT_ROLL
and SKATE_SPLIT_HEIGHT. The last lines give each left foot's distance to both
deck edges and each right foot's position.
"""

import argparse
import dataclasses

import torch

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
  DECK_GEOM_NAME,
  SkateboardParams,
  get_skateboard_cfg,
)
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  JOINT_NAMES,
  SKATE_SPLIT_BOARD_OFFSET,
  SKATE_SPLIT_HEIGHT,
  SKATE_SPLIT_KEYFRAME,
  SKATE_SPLIT_POSE,
)
from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import pupper_velocity_fs_env_cfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse

parser = argparse.ArgumentParser()
parser.add_argument("--deck-height", type=float, default=None)
parser.add_argument("values", type=float, nargs="*")
args = parser.parse_args()
if args.values:
  assert len(args.values) == 14, "give BOARD_Y ROBOT_Z and 12 angles"
  board_y, robot_z, angles = args.values[0], args.values[1], args.values[2:]
  rot = (1.0, 0.0, 0.0, 0.0)
else:
  board_y, robot_z = SKATE_SPLIT_BOARD_OFFSET[1], SKATE_SPLIT_HEIGHT + 0.01
  angles = list(SKATE_SPLIT_POSE)
  rot = SKATE_SPLIT_KEYFRAME.rot
pose = dict(zip(JOINT_NAMES, angles, strict=True))
params = SkateboardParams()
if args.deck_height is not None:
  params = dataclasses.replace(params, deck_top_height=args.deck_height)

cfg = pupper_velocity_fs_env_cfg(play=True)
cfg.scene.num_envs = 2
robot = cfg.scene.entities["robot"]
robot.init_state = dataclasses.replace(
  robot.init_state, pos=(0.0, 0.0, robot_z), rot=rot, joint_pos=pose
)
board = get_skateboard_cfg(params)
board.init_state = dataclasses.replace(
  board.init_state, pos=(SKATE_SPLIT_BOARD_OFFSET[0], board_y, params.wheel_radius)
)
cfg.scene.entities = {"robot": robot, "board": board}


def _contact(name: str, feet: str, on_deck: bool) -> ContactSensorCfg:
  secondary = (
    ContactMatch(mode="geom", pattern=DECK_GEOM_NAME, entity="board")
    if on_deck
    else ContactMatch(mode="body", pattern="terrain")
  )
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=feet, entity="robot"),
    secondary=secondary,
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )


cfg.scene.sensors = (cfg.scene.sensors or ()) + (
  _contact("left_on_deck", r"^leg_(front|back)_l_3_collision$", on_deck=True),
  _contact("right_on_floor", r"^leg_(front|back)_r_3_collision$", on_deck=False),
)
# No randomization: spawn exactly in the pose, and never end the episode.
cfg.events["reset_base"].params["pose_range"] = {}
cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
cfg.events["reset_board"] = EventTermCfg(
  func=mdp.reset_root_state_uniform,
  mode="reset",
  params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("board")},
)
cfg.terminations.pop("fell_over", None)

env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
env.reset()
robot_entity, board_entity = env.scene["robot"], env.scene["board"]
left_sensor, right_sensor = env.scene["left_on_deck"], env.scene["right_on_floor"]
origin = env.scene.env_origins[0]
actions = torch.zeros(2, 12)
steps = 500  # 10 s.
left_ok = right_ok = 0
reset = False
for step in range(steps):
  _, _, terminated, truncated, _ = env.step(actions)
  reset |= bool((terminated | truncated)[0])
  left_ok += int((left_sensor.data.found[0] > 0).all())
  right_ok += int((right_sensor.data.found[0] > 0).all())
  if step % 50 == 49:
    roll, pitch, _ = euler_xyz_from_quat(robot_entity.data.root_link_quat_w)
    base = (robot_entity.data.root_link_pos_w[0] - origin).numpy().round(3)
    deck = (board_entity.data.root_link_pos_w[0] - origin).numpy().round(3)
    print(
      f"t {(step + 1) * 0.02:.0f} s  roll {roll[0]:.3f}  pitch {pitch[0]:.3f}  "
      f"base {base}  board {deck}  deck roll {board_entity.data.joint_pos[0, 0]:.3f}"
    )

# Feet in the board's frame: y is the distance from the deck's centre line.
half_width = params.deck_width / 2
board_pos = board_entity.data.root_link_pos_w[0]
board_quat = board_entity.data.root_link_quat_w[0]
for leg in ("front_l", "back_l", "front_r", "back_r"):
  site = robot_entity.find_sites((f"leg_{leg}_3_foot_site",))[0][0]
  foot_w = robot_entity.data.site_pos_w[0, site]
  foot_b = quat_apply_inverse(board_quat, foot_w - board_pos)
  if leg.endswith("_l"):
    y = float(foot_b[1])
    print(
      f"{leg}: {y:+.4f} m from the centre line, "
      f"{half_width - y:.4f} m to the left edge, {half_width + y:.4f} m to the right"
    )
  else:
    print(f"{leg}: world {(foot_w - origin).numpy().round(4)}")
print(
  f"reset {reset}  left feet on deck {left_ok / steps:.0%}  "
  f"right feet on floor {right_ok / steps:.0%}  "
  f"final deck roll {board_entity.data.joint_pos[0, 0]:.3f}"
)
env.close()
