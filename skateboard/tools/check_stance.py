"""Drop Pupper on the board in a candidate stance and watch whether it holds.

Usage (repo root):
  FORCE_CPU=1 uv run python skateboard/tools/check_stance.py BOARD_Y ROBOT_Z A1 ... A12

BOARD_Y: board centre, metres left of the base (0.14 today).
ROBOT_Z: spawn height of the base [m] (0.17 is fine).
A1..A12: the 12 angles printed by solve_stance.py.

Holds the pose with zero actions for 5 s on a free-rolling board and prints
the state once a second. Accept the stance if roll and height are steady over
the last three prints and the board has not moved. The last print's roll and
base z are the new SKATE_ROLL and SKATE_HEIGHT.
"""

import dataclasses
import sys

import torch

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
  SkateboardParams,
  get_skateboard_cfg,
)
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import pupper_velocity_fs_env_cfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

board_y, robot_z = float(sys.argv[1]), float(sys.argv[2])
pose = dict(zip(JOINT_NAMES, [float(v) for v in sys.argv[3:15]], strict=True))

cfg = pupper_velocity_fs_env_cfg(play=True)
cfg.scene.num_envs = 2
robot = cfg.scene.entities["robot"]
robot.init_state = dataclasses.replace(
  robot.init_state, pos=(0.0, 0.0, robot_z), joint_pos=pose
)
board = get_skateboard_cfg()
board.init_state = dataclasses.replace(
  board.init_state, pos=(-0.01, board_y, SkateboardParams().wheel_radius)
)
cfg.scene.entities = {"robot": robot, "board": board}
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
origin = env.scene.env_origins[0]
actions = torch.zeros(2, 12)
for step in range(250):
  env.step(actions)
  if step % 50 == 49:
    roll, pitch, _ = euler_xyz_from_quat(robot_entity.data.root_link_quat_w)
    base = (robot_entity.data.root_link_pos_w[0] - origin).numpy().round(3)
    deck = (board_entity.data.root_link_pos_w[0] - origin).numpy().round(3)
    print(
      f"t {(step + 1) * 0.02:.0f} s  roll {roll[0]:.3f}  pitch {pitch[0]:.3f}  "
      f"base {base}  board {deck}  deck roll {board_entity.data.joint_pos[0, 0]:.3f}"
    )
env.close()
