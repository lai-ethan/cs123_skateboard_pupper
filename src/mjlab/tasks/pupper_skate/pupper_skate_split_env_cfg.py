"""Pupper v3 on a mini cruiser skateboard, split stance.

One side of the robot stands on the deck, the other on the floor. This is the
second pair of skate tasks, next to the centred stance in
``pupper_skate_env_cfg.py``; the balance deck height range, ride speeds and
shove interval are shared with it. Two tasks share one scene:

* balance -- hold the split stance on a board that is free to roll.
* ride    -- push with the floor-side legs so the board rolls at a commanded
             forward speed.
* steer   -- the ride plus a carve command that turns the board.

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
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  GAIT_HEADING_HOLD_CLIP,
  GAIT_HEADING_HOLD_KP,
  pupper_velocity_fs_env_cfg,
)
from mjlab.tasks.pupper_skate import mdp
from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import (
  BALANCE_TORQUE_SOFT_RATIO,
  DECK_HEIGHT_RANGE,
  PUSH_INTERVAL_S,
  RIDE_SPEED_RANGE,
)

# Which side stands on the deck ("l" or "r"). SKATE_SPLIT_KEYFRAME is solved for
# "l".
DECK_SIDE = "l"
GROUND_SIDE = "r"

_DECK_FEET = rf"^leg_(front|back)_{DECK_SIDE}_3_collision$"
_GROUND_FEET = rf"^leg_(front|back)_{GROUND_SIDE}_3_collision$"
# The foot sites of the _DECK_FEET and _GROUND_FEET geoms, in the sensors' order.
_DECK_FOOT_SITES = tuple(
  f"leg_{end}_{DECK_SIDE}_3_foot_site" for end in ("front", "back")
)
_GROUND_FOOT_SITES = tuple(
  f"leg_{end}_{GROUND_SIDE}_3_foot_site" for end in ("front", "back")
)
_KNEES = r"^leg_(front|back)_(r|l)_2_collision$"

# Steer task: the yaw command is a carve signal that works with the robot's
# heading hold. Quiet (0) is the ride: the hold holds the board's line. The
# steer task clips the hold's correction at STEER_HEADING_HOLD_CLIP, equal to
# the hold's 0.1 rad/s yaw threshold, so the yaw channel cannot be misread:
# |command| <= 0.1 can only be a hold correction and |command| > 0.1 can only
# be the stick. The ride and balance keep the gait tasks' 0.3 clip.
STEER_HEADING_HOLD_CLIP = 0.1
# A carve command is drawn with a magnitude in this range and either sign:
# above the hold's 0.1 rad/s threshold, so the hold lets go and the command
# reaches the policy. Partial turns are trained as well as full ones, since the
# operator's stick can send any value.
STEER_CARVE_COMMAND_RANGE: tuple[float, float] = (0.1, 0.6)
# The board yaw rate [rad/s] a carve asks for, per unit of carve command: 0.01
# to 0.06 rad/s, proportional to the stick. The board turns only when the deck
# tilts.
STEER_CARVE_GAIN = 0.1
# |yaw command| above this is a carve, at or below it a hold correction. It
# equals the steer hold's clip and the hold's yaw threshold.
STEER_CARVE_THRESHOLD = 0.1
# Steer task: truck stiffness scale on the nominal 30 N*m/rad, so 9 to 18
# N*m/rad, loosened trucks that Pupper's weight can tilt. Measured board yaw
# rate at 0.3 m/s with the deck feet 4 cm off the centre line: 0.07 rad/s at 12
# N*m/rad, 0.04 at 20, 0.03 at 30, so stiffer trucks cannot reach a carve
# inside the deck width. The ride keeps 15 to 180.
STEER_TRUCK_STIFFNESS_SCALE: tuple[float, float] = (0.3, 0.6)
# Share of envs told to ride straight (yaw 0, sampled speed kept), and the
# speed command under which yaw is 0: a board that is not rolling cannot carve.
STEER_STRAIGHT_SHARE = 0.4
STEER_YAW_MIN_SPEED = 0.1


# Split ride and steer: the robot holds its place along the deck. The steer
# policy of run 188eygof fell in 4 of 16 envs over 30 s, every fall the same
# way: the deck feet crept forward along the deck at 0.6 to 2.4 cm/s, all of it
# in contact, until the front left foot ran off the nose (29.2 cm ahead of the
# board centre; episodes start with it 3 to 14 cm ahead) and the base pitched
# nose down. The falls came 17 to 22 cm from the start, a median 21 s in.
# board_drifted reads only the sideways offset, and deck_feet_slide charges
# about 1e-4 per step at these speeds. So the episode ends once the robot is
# 8 cm along the deck from the keyframe (a fresh reset starts within 6.6 cm),
# and a 3 cm width pays for staying put long before that.
# Steer run a9x8z66m trained 3000 iterations with both. The deck feet still
# crept forward in 14 of 16 envs, at 0 to 0.66 cm/s (mean about 0.3; the run
# before: 0.6 to 2.4). With board_slid off for 60 s none fell in the first
# 29 s, then 6 of 16 ran the front left foot off the nose. No randomised value
# explains the rate (|r| <= 0.25) and more solver iterations do not change
# it, so it is the gait, and the envs that start furthest forward creep
# fastest. The policy cannot see the board, and the 8 cm end comes 15 to 30 s
# after the creep starts, far past the 2 s the discount looks ahead. So
# board_creep charges the creep rate as it happens, with full cost at
# BOARD_CREEP_CAP. At weight -1, 0.3 cm/s costs 0.15 a step. The rate goes
# through two averages in a row, BOARD_CREEP_TAU each (about 2 s of lag in all),
# to cancel the stride's own back-and-forth (about 2 Hz): one average leaves
# about amplitude / tau of it, two leave about amplitude / (omega * tau^2),
# roughly six times less.
# Steer v6 (run hfd3qhwr) still crept in one env out of 16, the one with the
# softest deck leg gains. The policy cannot see where it stands along the deck,
# and a reset alone could start it 6.2 cm into the 8 cm. So the 8 cm and
# board_fore_aft now count from the episode's own start, and
# BOARD_MAX_SLIDE_ABS from the keyframe is only the backstop that keeps the
# front left foot on the deck.
BOARD_MAX_SLIDE = 0.08
BOARD_MAX_SLIDE_ABS = 0.12
BOARD_SLIDE_STD = 0.03
BOARD_CREEP_TAU = 1.0
BOARD_CREEP_CAP = 0.02
# Steer only: the sideways twin of board_creep, at the same tau and cap. Hard
# right carves under steer v7 end in board_drifted: the deck feet slide right
# across the deck while loaded (5 to 10 N, in contact 99% of steps), and that
# slide is 0.8 to 5.1 cm of the 6 cm in the last 2 s. No term charged it
# before the -100. A lean is a held offset, so it costs nothing once held.
# Steer v8 (69e4d6b) trained it at -1.0 and failed its pass bar. v9 resumes from
# steer v6, which had no such term, and changes one thing at a time (only the
# carpet ranges), so it is 0.0 here. The term stays, and the notebook table can
# turn it back on.
BOARD_SIDE_CREEP_WEIGHT = 0.0

# The creep in steer v6 is a ratchet: when one right foot pushes alone, the
# deck foot on the same end drops to 1 to 2 N and scuffs about 5 mm, and about
# 1 mm of that stays each stride. At weight -2 deck_feet_slide cost 0.0015 to
# 0.011 per step, against 0.07 to 0.33 for board_creep, so the scuff was all
# but free. Ten times that puts it at 0.015 to 0.11. The split ride has
# trained at -20 since then.
DECK_FEET_SLIDE_WEIGHT = -20.0
# Steer's own scuff weight. Steer v6 (9743be3) trained at -2. Steer v7 (5d32fdf)
# took the ride's -20 and pulled right at stick 0 and lost the board in hard
# right carves. v9 resumes from v6 and changes only the carpet ranges, so steer
# is back at v6's -2. v7's fore-aft slide from the episode start stays: it is
# the stricter check and the sim tools read it.
STEER_DECK_FEET_SLIDE_WEIGHT = -2.0

# Split tasks: strict friction on the contacts. MuJoCo's soft contacts let a
# loaded foot slip slowly under a steady sideways load; real grip tape holds.
# The a9x8z66m steer policy over 30 s at 0.3 m/s (16 envs): with the shared
# pyramidal cone, 12 board_slid resets and a mean creep of +0.29 cm/s; 30
# solver iterations, 12 and +0.27; the elliptic cone, 4 and -0.13. About half
# of the creep was this slip. PUPPER_MUJOCO_OPTIONS already sets impratio 10;
# it is pinned here too so the split tasks keep it if the shared value moves.
SPLIT_CONTACT_CONE = "elliptic"
SPLIT_CONTACT_IMPRATIO = 10.0

# Carpet (split ride and steer). On Gates carpet (hardware, 2026-11-24) steer
# v6's floor legs stroked but the board did not roll, and in one clip the robot
# crept from mid-deck to the nose in about 10 s. In sim, steer v6 at 0.3 m/s
# (16 boards, 10 s) with the wheel drag pinned made 0.94 of the asked speed
# with no resets at x3 (the old top), 0.70 and 4 resets at x6, 0.36 and 12 at
# x10, 0.18 and 25 at x15, 0.10 and 41 at x25, nearly all board_slid. x15 to
# x25 shows both hardware symptoms, so carpet was first taken to be x15 to x25
# of the nominal drag (4 to 7 N of rolling resistance against about 0.28 N).
# That is a sim estimate, not a measurement.
#
# The range is capped at x10. Steer v9 trained on x0.5 to x25 and learned to
# stand still above x10 (pinned drag, 16 boards, 10 s, 0.3 m/s; speed is the
# share of the asked speed):
#
#   drag     v6 speed  v6 resets  v9 speed  v9 resets
#   x1-x3    0.94-0.99     0        1.04       0
#   x6       0.70          4        0.95       4
#   x10      0.36         12        0.45       3
#   x15      0.18         25        0.07       1
#   x25      0.10         41        0.00       1
#
# Likely cause, not proven: the actor cannot see the board, and pushing at that
# drag ended in board_slid for v6 (25 and 41 resets at x15 and x25), so v9
# learned not to push; whether grip is the limit is untested. The x15 to x25
# estimate came from the 2026-11-24 videos, which the placement error explains;
# x5 to x10 for low-pile carpet is a textbook figure, not measured. The ride
# audit's wheel_drag_x15 and wheel_drag_x25 rows stay as stress rows. The draw
# is uniform in the log, so half the boards are under about x2.2 (the geometric
# mean) and the policy keeps its hard-floor ride.
WHEEL_DRAG_SCALE_RANGE: tuple[float, float] = (0.5, 10.0)
# Floor grip up to carpet: the TA's advice (2026-11-24) was to randomize the
# feet's friction on carpet, since Gates carpet differs from other carpets, so
# the floor feet's draw reaches 1.6, up from 1.2. Unmeasured. The deck feet's
# draw (the grip tape) is unchanged.
FLOOR_FOOT_FRICTION_RANGE: tuple[float, float] = (0.5, 1.6)
DECK_FOOT_FRICTION_RANGE: tuple[float, float] = (0.5, 1.4)
# The deck geom's own friction. Feet have geom priority 1 and keep their own
# draw, so this reaches only the body, knee and shin contacts with the deck,
# which were all at the compiled 1.0 before. The TA's advice was to randomize
# the torso's friction on the board.
DECK_BODY_FRICTION_RANGE: tuple[float, float] = (0.3, 1.2)
# Deck top 6.8 to 11.8 cm: balance's range (DECK_HEIGHT_RANGE) plus 5 mm on
# top, as the TA advised randomizing the board height. The split stance holds
# 10 s with zero actions at +1.5 cm (test_split_stance_holds_for_10_s_across_
# deck_heights; check_split_stance.py --deck-height 0.118: no reset, left feet
# on the deck and right feet on the floor 100% of the time).
SPLIT_RIDE_DECK_HEIGHT_RANGE: tuple[float, float] = (-0.035, 0.015)


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
  mujoco = MujocoCfg(
    **{
      **PUPPER_MUJOCO_OPTIONS,
      "cone": SPLIT_CONTACT_CONE,
      "impratio": SPLIT_CONTACT_IMPRATIO,
    }
  )
  cfg.sim = SimulationCfg(nconmax=80, njmax=800, mujoco=mujoco)

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
  # untouched. In balance the board yaw stays within +-0.05 rad, so world x is
  # its heading for the board nudge. The ride drops the nudge and places the
  # board at up to +-0.15 rad, and the steer task turns it, but the robot kick
  # is the same in x and y, so it does not depend on the heading.
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
  # A diverged env is reset instead of ending the run (see the centred config).
  cfg.terminations["nan"] = TerminationTermCfg(func=mdp.nan_detection)

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

  # Long rides: the 10 s policy lost the deck in a third of envs by 30 s, as
  # the board walked sideways under the feet. Play keeps its unlimited
  # episodes (the base play config), so play tools can ride past 20 s.
  if not play:
    cfg.episode_length_s = 20.0

  # Grip per side. Feet have geom priority 1, so a foot contact always takes
  # the foot's own friction: the deck feet's draw is the grip tape, the floor
  # feet's the floor. Each side shares one draw. The deck geom's own friction
  # then reaches only the body, knee and shin contacts. Those end the episode
  # or are charged anyway, so the event was dropped once; it is kept now, on
  # its own range, since a torso or shin resting on the deck can hold the
  # robot back on carpet.
  feet = cfg.events.pop("foot_friction")
  for name, pattern, ranges in (
    ("deck_foot_friction", _DECK_FEET, DECK_FOOT_FRICTION_RANGE),
    ("floor_foot_friction", _GROUND_FEET, FLOOR_FOOT_FRICTION_RANGE),
  ):
    cfg.events[name] = dataclasses.replace(
      feet,
      params={
        **feet.params,
        "asset_cfg": SceneEntityCfg("robot", geom_names=pattern),
        "ranges": ranges,
      },
    )
  deck = cfg.events["deck_friction"]
  deck.params = {**deck.params, "ranges": DECK_BODY_FRICTION_RANGE}
  # Wheel bearing drag: damping and frictionloss of all four wheels scaled by
  # one factor per env. The nominal values are guesses, and drag x2 already
  # cost the 10 s policy a fifth of its speed; the range reaches carpet (see
  # WHEEL_DRAG_SCALE_RANGE).
  cfg.events["wheel_drag"] = EventTermCfg(
    mode="startup",
    func=mdp.scale_joint_drag,
    params={
      "scale_range": WHEEL_DRAG_SCALE_RANGE,
      "asset_cfg": SceneEntityCfg("board", joint_names=(r"wheel_.*_spin",)),
      "log_uniform": True,
    },
  )
  height = cfg.events["deck_height"]
  height.params = {**height.params, "ranges": SPLIT_RIDE_DECK_HEIGHT_RANGE}
  # Hand placement: the board can start 2 cm to either side and 0.15 rad off
  # the robot's heading. No term reads the board's heading from world x (see
  # the steer docstring); the heading hold below reads it relative to the
  # robot's, so the placement error is what the hold corrects.
  cfg.events["reset_board"].params["pose_range"] = {
    "x": (-0.03, 0.03),
    "y": (-0.02, 0.02),
    "yaw": (-0.15, 0.15),
  }
  # Heading hold against the board's line: run cfcrepfu yawed 0.1 rad off the
  # board in the first 0.5 s and the feet crept off the deck at 1 cm/s, with a
  # yaw command that was always 0. Walking, the hold captures the board's
  # heading and the emitted yaw command is the clipped P-correction of the
  # robot's yaw error against it. On the robot the same loop captures the IMU
  # yaw with the robot placed parallel to the board, so the gains are the gait
  # tasks' (the robot-side controller's).
  fields = {f.name: getattr(twist, f.name) for f in dataclasses.fields(twist)}
  hold = mdp.BoardHeadingHoldCommandCfg(**fields)
  hold.heading_hold_kp = GAIT_HEADING_HOLD_KP
  hold.heading_hold_clip = GAIT_HEADING_HOLD_CLIP
  cfg.commands["twist"] = hold
  if not play:
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.35, 0.35),
      "y": (-0.35, 0.35),
    }

  r = cfg.rewards
  del r["board_still"]
  r["track_board_velocity"] = RewardTermCfg(
    func=mdp.track_board_velocity,
    weight=3.0,
    params={"command_name": "twist", "std": 0.15},
  )
  r["board_yaw_rate"] = RewardTermCfg(func=mdp.board_yaw_rate_l2, weight=-0.5)
  # Follow the heading hold's yaw command: the policy observed it but no term
  # paid for it, and error_vel_yaw stayed near 0.6 for 1500 iterations. sigma
  # 0.05 divides the squared error, so a 0.22 rad/s error keeps 37% of the term.
  r["track_yaw_rate"] = RewardTermCfg(
    func=mdp.track_yaw_velocity,
    weight=1.0,
    params={"sigma": 0.05, "command_name": "twist"},
  )
  # Pushing: pay once per stride for a planted stance followed by a swing of
  # 0.2 to 0.5 s (only when a speed is commanded). The stance target is the
  # time a 10 cm stroke takes at the commanded speed, 0.25 to 0.5 s; the back
  # right leg has about 10 cm of reach, so a fixed 0.4 s asked for a 16 cm
  # stroke at 0.4 m/s. Short stances pay proportionally less, and short_stride
  # below charges for them outright, so a fast tap costs more than it earns.
  # Paying for air time taught a 3 Hz leg windmill.
  r["push_stride"] = RewardTermCfg(
    func=mdp.push_stride,
    weight=0.5,
    params={
      "sensor_name": "ground_feet_contact",
      "stroke_m": 0.10,
      "stance_range": (0.25, 0.5),
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
    params={
      "sensor_name": "ground_feet_contact",
      "stroke_m": 0.10,
      "stance_range": (0.25, 0.5),
      "command_name": "twist",
    },
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
  # The deck feet ride the deck: charge their slip relative to the deck (not
  # the world, the deck moves). A deck foot that creeps lets the board walk out
  # from under the robot. Ten times the floor feet's weight (see the constant).
  r["deck_feet_slide"] = RewardTermCfg(
    func=mdp.deck_feet_slide,
    weight=DECK_FEET_SLIDE_WEIGHT,
    params={
      "sensor_name": "deck_feet_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=_DECK_FOOT_SITES, preserve_order=True
      ),
      "deck_cfg": SceneEntityCfg("board", body_names=("deck",)),
    },
  )
  # Board under the feet: twice the balance weight and a 3 cm width, so a
  # board 3 cm out of place already costs most of the term.
  r["board_alignment"].weight = 2.0
  r["board_alignment"].params["std_pos"] = 0.03
  # And pointing along the robot: a 0.1 rad yaw error costs most of the term.
  r["board_alignment"].params["std_yaw"] = 0.1
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
  # A pushing foot held up instead of stepping: run ffcxbwup rode on three
  # points with the back right foot never touching down.
  r["push_foot_idle"] = RewardTermCfg(
    func=mdp.foot_idle_air,
    weight=-0.5,
    params={"sensor_name": "ground_feet_contact", "max_air_s": 0.5},
  )
  # Keep the motion inside what the motors can do. The sim actuator limits
  # torque but not speed, so speed is limited here. The penalty starts at
  # 5 rad/s, well under the gait check's 12. A smooth push at 0.4 m/s needs
  # about 7, which costs little; the cost is squared, so snaps cost much more.
  # Both weights were raised after ffcxbwup peaked at 10.6 rad/s with 21% of
  # samples torque-saturated; the onset came down from 6 after 3qmtsv4o peaked
  # at 9.2 rad/s.
  r["joint_vel_l2"] = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3)
  r["joint_speed_limit"] = RewardTermCfg(
    func=mdp.joint_speed_over_limit,
    weight=-0.3,
    params={"limit": 5.0},
  )
  # Smoother lift-offs and touchdowns: 3qmtsv4o's speed peaks came from snaps
  # at contact changes, not from its cadence. Action rate is five times the
  # balance weight, action and joint acceleration are new, and a swing foot
  # pays for lifting its underside more than 3 cm off the floor.
  r["action_rate_l2"].weight = -0.05
  r["action_acc"] = RewardTermCfg(func=mdp.action_acc_l2, weight=-0.02)
  r["joint_acc"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-1e-6,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  r["swing_height"] = RewardTermCfg(
    func=mdp.foot_height_over,
    weight=-5.0,
    params={
      "max_height": 0.03,
      "foot_radius": 0.02,
      "sensor_name": "ground_feet_contact",
      "foot_cfg": SceneEntityCfg(
        "robot", site_names=_GROUND_FOOT_SITES, preserve_order=True
      ),
    },
  )
  r["torque_saturation"] = RewardTermCfg(
    func=mdp.torque_saturation, weight=-1.5, params={"limit": PUPPER_EFFORT_LIMIT}
  )

  # The board slid out from under the feet: its sideways offset from the robot,
  # in the board's heading frame, is more than 6 cm from the keyframe's. The
  # deck feet stand near the deck's centre line; the traced deck is at most
  # 7.9 cm from it to either edge and narrower toward the ends
  # (deck_half_width_at), so this ends the ride before they slip off. Ends the
  # episode in play too.
  cfg.terminations["board_drifted"] = TerminationTermCfg(
    func=mdp.board_drifted, params={"max_drift": 0.06}
  )
  # The fore-aft twin: the robot slid BOARD_MAX_SLIDE along the deck from its
  # start, either way, or is BOARD_MAX_SLIDE_ABS from the keyframe (see the
  # constants). board_fore_aft pays for staying put before that, with a
  # tighter width than board_alignment's, which also reads this axis.
  cfg.terminations["board_slid"] = TerminationTermCfg(
    func=mdp.board_slid,
    params={"max_slide": BOARD_MAX_SLIDE, "max_slide_abs": BOARD_MAX_SLIDE_ABS},
  )
  r["board_fore_aft"] = RewardTermCfg(
    func=mdp.board_fore_aft, weight=1.0, params={"std": BOARD_SLIDE_STD}
  )
  # And the slow slide that leads there, charged while it happens (see the
  # constants).
  r["board_creep"] = RewardTermCfg(
    func=mdp.board_creep,
    weight=-1.0,
    params={"tau": BOARD_CREEP_TAU, "rate_cap": BOARD_CREEP_CAP},
  )
  return cfg


def pupper_skate_split_steer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The split ride plus a carve command: push and lean to steer.

  The yaw command is a carve signal that works with the robot's heading hold
  (see ``SteerVelocityCommand``). At yaw 0 the task is the ride: the hold keeps
  the robot on the board's line. A carve command (0.1 to 0.6 rad/s, either
  sign) is above the hold's threshold, so it reaches the policy as it is, and
  asks for a board yaw rate of STEER_CARVE_GAIN times the command (0.01 to
  0.06 rad/s) with the robot turning along with the board. The hold's
  correction is clipped at STEER_HEADING_HOLD_CLIP (0.1), so it never looks
  like a carve. When the carve ends the hold recaptures the board's heading.

  The board turns only when the deck tilts, so this task trains on loosened
  trucks (STEER_TRUCK_STIFFNESS_SCALE) that Pupper's weight can tilt. The
  board's turn is paid, not how the lean is made: a tight tracking term plus
  board_turn_progress, which pays from the first bit of turn. The ride's robot
  yaw-rate term stays, with a carve asking the body for the board's rate.
  Board alignment is twice as wide sideways as the ride's, since a lean needs
  the deck feet about 4 cm off the centre line. The deck feet's scuff is
  charged at steer v6's weight, a tenth of the ride's, and board_side_creep
  (charging a sideways board slide as board_creep does along the deck) ships
  at weight 0. Every other term is the ride's: none of them reads a world-frame
  heading (forward speed is along the board's own axis, board alignment is in
  the robot's heading frame, board drift and slide in the board's, foot slide
  and the shoves are the same in every direction), so they hold once the board
  has turned.
  """
  cfg = pupper_skate_split_ride_env_cfg(play=play)

  # The ride's board heading-hold command with carves on the yaw channel.
  # STEER_STRAIGHT_SHARE of envs ride straight at their sampled speed
  # (rel_forward_envs would also lift it to 0.3 m/s, so it stays at the ride's
  # 0), and speed commands under STEER_YAW_MIN_SPEED get no carve. The standing
  # share is drawn on its own and zeroes everything. rel_turn_envs stays 0: it
  # would zero the speed. The yaw range is not drawn from; it is the export's
  # wz clip, so it covers the carves.
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.BoardHeadingHoldCommandCfg)
  fields = {f.name: getattr(twist, f.name) for f in dataclasses.fields(twist)}
  steer = mdp.SteerVelocityCommandCfg(
    **fields,
    carve_range=STEER_CARVE_COMMAND_RANGE,
    rel_straight_envs=STEER_STRAIGHT_SHARE,
    yaw_min_speed=STEER_YAW_MIN_SPEED,
  )
  top = STEER_CARVE_COMMAND_RANGE[1]
  steer.ranges = dataclasses.replace(twist.ranges, ang_vel_z=(-top, top))
  steer.rel_turn_envs = 0.0
  # The hold's correction stops at the carve threshold (see the constants).
  steer.heading_hold_clip = STEER_HEADING_HOLD_CLIP
  cfg.commands["twist"] = steer

  # Loosened trucks: the ride's stiffer range barely tilts under Pupper.
  stiffness = cfg.events["truck_stiffness"]
  stiffness.params = {**stiffness.params, "ranges": STEER_TRUCK_STIFFNESS_SCALE}

  carve = {
    "carve_threshold": STEER_CARVE_THRESHOLD,
    "carve_gain": STEER_CARVE_GAIN,
  }
  r = cfg.rewards
  # The ride's board yaw penalty, outside carves only.
  r["board_yaw_rate"].params = {
    "command_name": "twist",
    "carve_threshold": STEER_CARVE_THRESHOLD,
  }
  # The robot follows the hold's correction as in the ride, and in a carve
  # turns at the board's target rate.
  r["track_yaw_rate"].params = {**r["track_yaw_rate"].params, **carve}
  # The board's yaw rate. The std is tight so a partial carve and a full one
  # pay differently: a board turning at a 0.02 rad/s target under a 0.06
  # command keeps 2% of the term. Riding straight under a 0.01 to 0.06 rad/s
  # target pays 78 to 0%; board_turn_progress pays from the first bit of turn.
  # Outside carves it pays 0. Steer v8 (69e4d6b) also paid a rolling board for
  # not turning there (straight_min_speed at STEER_YAW_MIN_SPEED) and failed its
  # pass bar; v9 resumes from steer v6, which trained without it, and changes
  # only the carpet ranges. track_board_yaw_rate keeps the option.
  r["track_board_yaw_rate"] = RewardTermCfg(
    func=mdp.track_board_yaw_rate,
    weight=2.0,
    params={"command_name": "twist", "std": 0.02, **carve},
  )
  r["board_turn_progress"] = RewardTermCfg(
    func=mdp.board_turn_progress,
    weight=1.0,
    params={
      "command_name": "twist",
      "command_threshold": STEER_CARVE_THRESHOLD,
      **carve,
    },
  )
  # A lean needs the deck feet about 4 cm off the centre line, which the
  # ride's 3 cm width charges 83% of the term for. The yaw width is the ride's.
  r["board_alignment"].params = {**r["board_alignment"].params, "std_pos": 0.06}
  # The deck feet's scuff at steer v6's weight, not the ride's (see the
  # constants).
  r["deck_feet_slide"].weight = STEER_DECK_FEET_SLIDE_WEIGHT
  # The slow sideways slide that ends hard carves in board_drifted, off by
  # default (see the constants).
  r["board_side_creep"] = RewardTermCfg(
    func=mdp.board_side_creep,
    weight=BOARD_SIDE_CREEP_WEIGHT,
    params={"tau": BOARD_CREEP_TAU, "rate_cap": BOARD_CREEP_CAP},
  )
  return cfg
