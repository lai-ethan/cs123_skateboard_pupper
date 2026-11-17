"""Pupper skateboard task MDP terms."""

from mjlab.tasks.pupper_gait.mdp import *  # noqa: F401, F403

from .observations import (  # noqa: F401
  board_pose_in_robot_frame,
  contact_flags,
  deck_roll_state,
)
from .rewards import (  # noqa: F401
  board_alignment,
  board_progress,
  board_speed_l2,
  board_yaw_rate_l2,
  contact_count,
  deck_roll_l2,
  feet_slide,
  foot_reach,
  gravity_tracking,
  joint_speed_over_limit,
  push_stride,
  push_stroke,
  short_stride,
  stance_pose,
  torque_saturation,
  track_board_velocity,
)
from .terminations import contact_lost  # noqa: F401
