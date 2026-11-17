"""Pupper v3 on a mini cruiser skateboard.

The robot stands centred on the deck, all four feet on it. Two tasks share one
scene:

* balance -- hold the stance on a board that is free to roll.
* ride    -- step a foot off either side, push against the floor and step back
             on, so the board rolls at a commanded forward speed.

The actor is the stock 48-dim zero-reference frame (IMU, joints, command, last
action), so policies export and deploy like any gait policy. Everything about
the board goes to the critic only: the robot has no board sensor.

The board model follows the structure of HUSKY (Han et al., arXiv:2602.03205).
"""

from __future__ import annotations

import dataclasses

from mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants import (
  DECK_GEOM_NAME,
  DECK_ROLL_JOINT_NAME,
  WHEEL_GEOM_NAMES,
  SkateboardParams,
  get_skateboard_cfg,
)
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  FOOT_SITE_NAMES,
  PUPPER_EFFORT_LIMIT,
  PUPPER_MUJOCO_OPTIONS,
  SKATE_BOARD_OFFSET,
  SKATE_KEYFRAME,
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

# All four feet, matched in FOOT_SITE_NAMES order (FR, FL, BR, BL).
_FEET = r"^leg_(front|back)_(r|l)_3_collision$"
_KNEES = r"^leg_(front|back)_(r|l)_2_collision$"

# Forward speed range for the ride task [m/s].
RIDE_SPEED_RANGE: tuple[float, float] = (0.0, 0.5)

# Joint speed limit [rad/s] the gait check enforces; the ride task's penalty
# starts at 75% of it. A design guess, not a datasheet number: replace it with
# the real motor's limit.
JOINT_SPEED_LIMIT: float = 8.0

# Deck height offset from the nominal [m], and the seconds between shoves to the
# robot and the board. Nominal is the measured 0.103 m; the range is 6.8 to
# 11.3 cm, which covers the stock trucks and the planned trucks that sit 2 to
# 3 cm lower. This range is for balance (both stances) and split ride; centred
# ride uses RIDE_DECK_HEIGHT_RANGE below.
DECK_HEIGHT_RANGE: tuple[float, float] = (-0.035, 0.01)
PUSH_INTERVAL_S: tuple[float, float] = (2.0, 4.0)

# Ride trains on the lowered trucks only (deck top 6.8 to 8.3 cm). The leg
# reaches 19.3 cm below the base; on the stock 10.3 cm deck the floor is
# 22 cm below it, so a push from the centred stance is out of reach there.
RIDE_DECK_HEIGHT_RANGE: tuple[float, float] = (-0.035, -0.02)

# The balance tasks penalize motors above this share of their torque limit
# (1.5 N*m). Standing with zero actions peaks near 0.5 N*m; run k4kci0oj instead
# held its legs on the joint stops at full torque, which a real motor cannot
# sustain.
BALANCE_TORQUE_SOFT_RATIO: float = 0.5


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
    # Feet on the deck: +1 per foot, so 4.0 with all four down.
    "feet_on_deck": RewardTermCfg(
      func=mdp.contact_count, weight=1.0, params={"sensor_name": "deck_feet_contact"}
    ),
    # Feet (or knees) where they do not belong. A foot on the floor is wrong here.
    "feet_on_ground": RewardTermCfg(
      func=mdp.contact_count, weight=-1.0, params={"sensor_name": "ground_feet_contact"}
    ),
    "knees_on_deck": RewardTermCfg(
      func=mdp.contact_count, weight=-1.0, params={"sensor_name": "knees_on_deck"}
    ),
    # Body: level, joints near the keyframe. No base height term: the deck
    # height is randomized, and the joint pose already sets the height above it.
    "tilt": RewardTermCfg(
      func=mdp.gravity_tracking,
      weight=1.0,
      params={"target_xy": (0.0, 0.0), "std": 0.2},
    ),
    "stance_pose": RewardTermCfg(
      func=mdp.stance_pose,
      weight=1.0,
      params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Board: centred under the feet, parallel to the robot, flat, not rolling.
    "board_alignment": RewardTermCfg(
      func=mdp.board_alignment,
      weight=1.0,
      params={"target_offset": SKATE_BOARD_OFFSET, "std_pos": 0.05, "std_yaw": 0.3},
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


def pupper_skate_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Hold the centred stance on a free-rolling board. Command is always zero."""
  cfg = pupper_velocity_fs_env_cfg(play=play)

  ##
  # Scene: robot in the centred stance, board under its feet.
  ##
  robot = cfg.scene.entities["robot"]
  robot.init_state = dataclasses.replace(SKATE_KEYFRAME)
  board = get_skateboard_cfg()
  board.init_state = dataclasses.replace(
    board.init_state,
    pos=(SKATE_BOARD_OFFSET[0], SKATE_BOARD_OFFSET[1], SkateboardParams().wheel_radius),
  )
  cfg.scene.entities = {"robot": robot, "board": board}
  # Two bodies in contact with each other and the floor: more contact slots.
  cfg.sim = SimulationCfg(
    nconmax=80, njmax=800, mujoco=MujocoCfg(**PUPPER_MUJOCO_OPTIONS)
  )

  ##
  # Sensors. The stock feet_ground_contact (all four feet vs floor) stays for
  # the critic; these split each foot's contacts by surface.
  ##
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    _contact("deck_feet_contact", _FEET, on_deck=True, track_air_time=True),
    _contact("ground_feet_contact", _FEET, on_deck=False, track_air_time=True),
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
  # Terminations. The centred stance is level, so the fall limit is a little
  # under the walking tasks' 30 degrees. Any foot off the deck past the grace
  # period ends the episode.
  ##
  cfg.terminations["fell_over"].params["limit_angle"] = 0.5
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
  # A diverged env (NaN or Inf in its physics state) is reset instead of taking
  # the whole run down at the next NaN check on the actor observation.
  cfg.terminations["nan"] = TerminationTermCfg(func=mdp.nan_detection)

  return cfg


def pupper_skate_ride_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Push off either side so the board rolls at the commanded speed."""
  cfg = pupper_skate_balance_env_cfg(play=play)

  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  twist.ranges.lin_vel_x = RIDE_SPEED_RANGE
  twist.rel_standing_envs = 0.1
  # The board is meant to roll here: no shoving it.
  cfg.events.pop("nudge_board", None)
  cfg.events["deck_height"].params["ranges"] = RIDE_DECK_HEIGHT_RANGE
  # A push takes a foot off the deck for about a second. One foot may stay off
  # for good: the episode ends only when two have been off past the grace.
  cfg.terminations["feet_off_deck"].params["grace_s"] = 1.5
  cfg.terminations["feet_off_deck"].params["min_lost"] = 2

  r = cfg.rewards
  del r["board_still"]
  del r["feet_on_ground"]
  # Standing on four feet must not out-earn pushing with one: the deck pays for
  # three feet at most, and the body may tilt and the board shift a little
  # while a leg reaches the floor.
  r["feet_on_deck"].weight = 0.5
  r["feet_on_deck"].params["max_count"] = 3.0
  r["tilt"].params["std"] = 0.35
  r["board_alignment"].params["std_pos"] = 0.08
  r["track_board_velocity"] = RewardTermCfg(
    func=mdp.track_board_velocity,
    weight=3.0,
    params={"command_name": "twist", "std": 0.15},
  )
  # track_board_velocity pays little until the board is near the command;
  # board_progress pays the share of it reached, so a first slow push pays too.
  r["board_progress"] = RewardTermCfg(
    func=mdp.board_progress,
    weight=1.0,
    params={"command_name": "twist", "command_threshold": 0.1},
  )
  r["board_yaw_rate"] = RewardTermCfg(func=mdp.board_yaw_rate_l2, weight=-0.5)
  # Pushing, in two steps. Either side may push, and both pay only while a
  # speed is commanded and the board is too slow.
  # foot_reach leads a foot to the floor: it pays more the closer the lowest
  # foot gets to the floor. A foot on the deck stays a deck height up and earns
  # little. Its pay fades as the board speeds up and is gone at 0.8 of the
  # command, so parking a foot on the floor just under that speed earns less
  # than rolling at the command.
  # push_stroke pays the push itself: once per stroke as a foot leaves the
  # floor, in full for a stroke of 0.25 s or more.
  r["foot_reach"] = RewardTermCfg(
    func=mdp.foot_reach,
    weight=1.5,
    params={
      "std": 0.03,
      "foot_radius": 0.02,
      "command_name": "twist",
      "command_threshold": 0.1,
      "speed_fraction": 0.8,
      "foot_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "continuous_gate": True,
    },
  )
  r["push_stroke"] = RewardTermCfg(
    func=mdp.push_stroke,
    weight=1.0,
    params={
      "sensor_name": "ground_feet_contact",
      "stroke_s": 0.25,
      "command_name": "twist",
      "command_threshold": 0.1,
      "speed_fraction": 1.0,
    },
  )
  # Hold the keyframe loosely on every leg, so one leg can stride off the deck
  # while the others stay put.
  r["stance_pose"].weight = 0.5
  r["stance_pose"].params["std"] = 0.6
  # Keep the motion inside what the motors can do. The sim actuator limits
  # torque but not speed, so speed is limited here. The penalty starts at 75% of
  # the limit the gait check enforces (6 of 8 rad/s), so the best-paid gait
  # stays inside the check.
  r["joint_vel_l2"] = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3)
  r["joint_speed_limit"] = RewardTermCfg(
    func=mdp.joint_speed_over_limit,
    weight=-0.02,
    params={"limit": 0.75 * JOINT_SPEED_LIMIT},
  )
  r["torque_saturation"] = RewardTermCfg(
    func=mdp.torque_saturation, weight=-0.5, params={"limit": PUPPER_EFFORT_LIMIT}
  )
  return cfg
