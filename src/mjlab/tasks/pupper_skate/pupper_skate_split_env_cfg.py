"""Pupper v3 on a mini cruiser skateboard, split stance.

One side of the robot stands on the deck, the other on the floor. This is the
second pair of skate tasks, next to the centred stance in
``pupper_skate_env_cfg.py``; the deck height range, joint speed limit, ride speeds
and shove interval are shared with it. Two tasks share one scene:

* balance -- hold the split stance on a board that is free to roll.
* ride    -- push with the floor-side legs so the board rolls at a commanded
             forward speed.

The actor is the stock 48-dim zero-reference frame (IMU, joints, command, last
action), so policies export and deploy like any gait policy. Everything about
the board goes to the critic only: the robot has no board sensor.

The board model follows the structure of HUSKY (Han et al., arXiv:2602.03205).
"""

from __future__ import annotations

import dataclasses
import math

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
  DECK_GEOM_NAME,
  DECK_ROLL_JOINT_NAME,
  WHEEL_GEOM_NAMES,
  SkateboardParams,
  get_skateboard_cfg,
)
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  PUPPER_EFFORT_LIMIT,
  PUPPER_MUJOCO_OPTIONS,
  SKATE_SPLIT_BOARD_OFFSET,
  SKATE_SPLIT_HEIGHT,
  SKATE_SPLIT_KEYFRAME,
  SKATE_SPLIT_ROLL,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import pupper_velocity_fs_env_cfg
from mjlab.tasks.pupper_skate import mdp
from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import (
  BALANCE_TORQUE_SOFT_RATIO,
  DECK_HEIGHT_RANGE,
  JOINT_SPEED_LIMIT,
  PUSH_INTERVAL_S,
  RIDE_SPEED_RANGE,
)

# Which side stands on the deck ("l" or "r"). SKATE_SPLIT_KEYFRAME is solved for
# "l".
DECK_SIDE = "l"
GROUND_SIDE = "r"

_DECK_FEET = rf"^leg_(front|back)_{DECK_SIDE}_3_collision$"
_GROUND_FEET = rf"^leg_(front|back)_{GROUND_SIDE}_3_collision$"
# The foot sites of the _GROUND_FEET geoms, in the sensor's order.
_GROUND_FOOT_SITES = tuple(
  f"leg_{end}_{GROUND_SIDE}_3_foot_site" for end in ("front", "back")
)
_KNEES = r"^leg_(front|back)_(r|l)_2_collision$"


def _contact(
  name: str, feet: str, on_deck: bool, track_air_time: bool = False
) -> ContactSensorCfg:
  secondary = (
    ContactMatch(mode="geom", pattern=DECK_GEOM_NAME, entity="board")
    if on_deck
    else ContactMatch(mode="body", pattern="terrain")
  )
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=feet, entity="robot"),
    secondary=secondary,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=track_air_time,
  )


def _balance_rewards() -> dict[str, RewardTermCfg]:
  return {
    # Feet where they belong: +1 per foot, two feet per sensor.
    "feet_on_deck": RewardTermCfg(
      func=mdp.contact_count, weight=1.0, params={"sensor_name": "deck_feet_contact"}
    ),
    "feet_on_ground": RewardTermCfg(
      func=mdp.contact_count, weight=0.5, params={"sensor_name": "ground_feet_contact"}
    ),
    # Feet (or knees) where they do not belong.
    "deck_feet_on_ground": RewardTermCfg(
      func=mdp.contact_count, weight=-2.0, params={"sensor_name": "deck_feet_on_ground"}
    ),
    "ground_feet_on_deck": RewardTermCfg(
      func=mdp.contact_count, weight=-2.0, params={"sensor_name": "ground_feet_on_deck"}
    ),
    "knees_on_deck": RewardTermCfg(
      func=mdp.contact_count, weight=-1.0, params={"sensor_name": "knees_on_deck"}
    ),
    # Body: the stance's tilt and height, joints near the keyframe.
    "tilt": RewardTermCfg(
      func=mdp.gravity_tracking,
      weight=1.0,
      params={"target_xy": (0.0, -math.sin(SKATE_SPLIT_ROLL)), "std": 0.2},
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height,
      weight=1.0,
      params={"target_height": SKATE_SPLIT_HEIGHT, "std": 0.03},
    ),
    "stance_pose": RewardTermCfg(
      func=mdp.stance_pose,
      weight=1.0,
      params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Board: under the deck-side feet, parallel to the robot, flat, not rolling.
    "board_alignment": RewardTermCfg(
      func=mdp.board_alignment,
      weight=1.0,
      params={
        "target_offset": SKATE_SPLIT_BOARD_OFFSET,
        "std_pos": 0.05,
        "std_yaw": 0.3,
      },
    ),
    "deck_flat": RewardTermCfg(
      func=mdp.deck_roll_l2,
      weight=-5.0,
      params={
        "board_cfg": SceneEntityCfg("board", joint_names=(DECK_ROLL_JOINT_NAME,))
      },
    ),
    "board_still": RewardTermCfg(func=mdp.board_speed_l2, weight=-2.0),
    # Smooth, cheap, alive.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2e-4),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-100.0),
    "torque_saturation": RewardTermCfg(
      func=mdp.torque_saturation,
      weight=-1.0,
      params={"limit": PUPPER_EFFORT_LIMIT, "soft_ratio": BALANCE_TORQUE_SOFT_RATIO},
    ),
  }


def pupper_skate_split_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Hold the split stance on a free-rolling board. Command is always zero."""
  cfg = pupper_velocity_fs_env_cfg(play=play)

  ##
  # Scene: robot in the split stance, board under its deck-side feet.
  ##
  robot = cfg.scene.entities["robot"]
  robot.init_state = dataclasses.replace(SKATE_SPLIT_KEYFRAME)
  board = get_skateboard_cfg()
  board.init_state = dataclasses.replace(
    board.init_state,
    pos=(
      SKATE_SPLIT_BOARD_OFFSET[0],
      SKATE_SPLIT_BOARD_OFFSET[1],
      SkateboardParams().wheel_radius,
    ),
  )
  cfg.scene.entities = {"robot": robot, "board": board}
  # Two bodies in contact with each other and the floor: more contact slots.
  cfg.sim = SimulationCfg(
    nconmax=80, njmax=800, mujoco=MujocoCfg(**PUPPER_MUJOCO_OPTIONS)
  )

  ##
  # Sensors. The stock feet_ground_contact (all four feet vs floor) stays for
  # the critic; these split contacts by side and by surface.
  ##
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    _contact("deck_feet_contact", _DECK_FEET, on_deck=True, track_air_time=True),
    _contact("ground_feet_contact", _GROUND_FEET, on_deck=False, track_air_time=True),
    _contact("deck_feet_on_ground", _DECK_FEET, on_deck=False),
    _contact("ground_feet_on_deck", _GROUND_FEET, on_deck=True),
    _contact("knees_on_deck", _KNEES, on_deck=True),
  )

  ##
  # Resets: robot and board start in the keyframe, a little apart from perfect.
  # The stock reset scatters the robot +-0.5 m with a random heading, which
  # would drop it off the board.
  ##
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.01, 0.01),
    "y": (-0.01, 0.01),
    "z": (0.0, 0.01),
    "yaw": (-0.05, 0.05),
  }
  cfg.events["reset_board"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-0.03, 0.03), "yaw": (-0.05, 0.05)},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("board"),
    },
  )

  ##
  # Board domain randomization (the robot's own randomization is inherited).
  ##
  # Mass and inertia together: alpha scales both by exp(2 * alpha), x[0.7, 1.3].
  cfg.events["board_inertial"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("board", body_names=(".*",)),
      "alpha_range": (-0.178, 0.131),
    },
  )
  cfg.events["deck_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("board", geom_names=(DECK_GEOM_NAME,)),
      "operation": "abs",
      "ranges": (0.6, 1.2),
    },
  )
  cfg.events["wheel_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("board", geom_names=WHEEL_GEOM_NAMES),
      "operation": "abs",
      "ranges": (0.6, 1.1),
      "shared_random": True,
    },
  )
  # Truck stiffness 15 to 180 N*m/rad. The real bushings are unmeasured and are
  # built for a rider about 20 times heavier than Pupper, so the real deck is
  # likely much stiffer than the 30 N*m/rad guess.
  cfg.events["truck_stiffness"] = EventTermCfg(
    mode="startup",
    func=dr.joint_stiffness,
    params={
      "asset_cfg": SceneEntityCfg("board", joint_names=(DECK_ROLL_JOINT_NAME,)),
      "operation": "scale",
      "ranges": (0.5, 6.0),
    },
  )
  cfg.events["deck_height"] = EventTermCfg(
    mode="startup",
    func=dr.geom_pos,
    params={
      "asset_cfg": SceneEntityCfg("board", geom_names=(DECK_GEOM_NAME,)),
      "operation": "add",
      "ranges": DECK_HEIGHT_RANGE,
      "axes": [2],
    },
  )

  ##
  # Shoves (training only). The stock base kick is off for Pupper; this one is
  # linear and small. Both add to the root velocity, so the deck roll joint is
  # untouched. Board yaw stays within +-0.05 rad, so world x is its heading.
  ##
  if not play:
    cfg.events["push_robot"] = EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=PUSH_INTERVAL_S,
      params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )
    cfg.events["nudge_board"] = EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=PUSH_INTERVAL_S,
      params={
        "velocity_range": {"x": (-0.15, 0.15)},
        "asset_cfg": SceneEntityCfg("board"),
      },
    )

  ##
  # Command: zero. The actor still sees the three command dims (as zeros).
  ##
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.0, 0.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.rel_turn_envs = 0.0
  twist.heading_hold_kp = None
  twist.debug_vis = False

  ##
  # Critic: add what the robot cannot sense.
  ##
  critic = cfg.observations["critic"]
  board_cfg = SceneEntityCfg("board")
  critic.terms["board_pose"] = ObservationTermCfg(func=mdp.board_pose_in_robot_frame)
  critic.terms["board_lin_vel"] = ObservationTermCfg(
    func=mdp.base_lin_vel, params={"asset_cfg": board_cfg}
  )
  critic.terms["board_ang_vel"] = ObservationTermCfg(
    func=mdp.base_ang_vel, params={"asset_cfg": board_cfg}
  )
  critic.terms["deck_roll"] = ObservationTermCfg(
    func=mdp.deck_roll_state,
    params={"board_cfg": SceneEntityCfg("board", joint_names=(DECK_ROLL_JOINT_NAME,))},
  )
  critic.terms["deck_feet_contact"] = ObservationTermCfg(
    func=mdp.contact_flags, params={"sensor_name": "deck_feet_contact"}
  )
  critic.terms["ground_feet_contact"] = ObservationTermCfg(
    func=mdp.contact_flags, params={"sensor_name": "ground_feet_contact"}
  )

  cfg.rewards = _balance_rewards()

  ##
  # Terminations. The stance itself is tilted by SKATE_SPLIT_ROLL, so the fall
  # limit is wider than the walking tasks' 30 degrees.
  ##
  cfg.terminations["fell_over"].params["limit_angle"] = 0.6
  cfg.terminations["feet_off_deck"] = TerminationTermCfg(
    func=mdp.contact_lost, params={"sensor_name": "deck_feet_contact", "grace_s": 0.5}
  )
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum, params={"minimum_height": 0.08}
  )
  cfg.terminations["board_tipped"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": 0.3, "asset_cfg": SceneEntityCfg("board")},
  )

  return cfg


def pupper_skate_split_ride_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Push with the floor-side legs so the board rolls at the commanded speed."""
  cfg = pupper_skate_split_balance_env_cfg(play=play)

  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  twist.ranges.lin_vel_x = RIDE_SPEED_RANGE
  twist.rel_standing_envs = 0.1
  # The board is meant to roll here: no shoving it.
  cfg.events.pop("nudge_board", None)

  r = cfg.rewards
  del r["board_still"]
  r["track_board_velocity"] = RewardTermCfg(
    func=mdp.track_board_velocity,
    weight=3.0,
    params={"command_name": "twist", "std": 0.15},
  )
  r["board_yaw_rate"] = RewardTermCfg(func=mdp.board_yaw_rate_l2, weight=-0.5)
  # Pushing: pay once per stride for a planted stance of 0.4 s followed by a
  # swing of 0.2 to 0.5 s (only when a speed is commanded). Short stances pay
  # proportionally less, and short_stride below charges for them outright, so a
  # fast tap costs more than it earns. Paying for air time taught a 3 Hz leg
  # windmill.
  r["push_stride"] = RewardTermCfg(
    func=mdp.push_stride,
    weight=0.5,
    params={
      "sensor_name": "ground_feet_contact",
      "stance_s": 0.4,
      "air_range": (0.2, 0.5),
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  # Taps and dragged feet cost something, at any command: with push_stride
  # alone, run vzs88cci shuffled the right feet at 5 strides a second.
  r["short_stride"] = RewardTermCfg(
    func=mdp.short_stride,
    weight=-0.3,
    params={"sensor_name": "ground_feet_contact", "stance_s": 0.4},
  )
  r["feet_slide"] = RewardTermCfg(
    func=mdp.feet_slide,
    weight=-2.0,
    params={
      "sensor_name": "ground_feet_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=_GROUND_FOOT_SITES, preserve_order=True
      ),
    },
  )
  r["feet_on_ground"].weight = 0.1
  # Hold the keyframe firmly on the deck-side legs and loosely on the pushing
  # legs, so they stride around the stance instead of flailing.
  r["stance_pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(rf"leg_(front|back)_{DECK_SIDE}_[123]",)
  )
  r["push_leg_pose"] = RewardTermCfg(
    func=mdp.stance_pose,
    weight=0.3,
    params={
      "std": 0.6,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(rf"leg_(front|back)_{GROUND_SIDE}_[123]",)
      ),
    },
  )
  # Keep the motion inside what the motors can do. The sim actuator limits
  # torque but not speed, so speed is limited here. The penalty starts at 75% of
  # the limit the gait check enforces (6 of 8 rad/s), so the best-paid gait
  # stays inside the check.
  r["joint_vel_l2"] = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3)
  r["joint_speed_limit"] = RewardTermCfg(
    func=mdp.joint_speed_over_limit,
    weight=-0.05,
    params={"limit": 0.75 * JOINT_SPEED_LIMIT},
  )
  r["torque_saturation"] = RewardTermCfg(
    func=mdp.torque_saturation, weight=-0.5, params={"limit": PUPPER_EFFORT_LIMIT}
  )
  return cfg
