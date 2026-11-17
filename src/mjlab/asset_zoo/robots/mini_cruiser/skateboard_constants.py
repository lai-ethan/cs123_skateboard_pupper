"""Mini cruiser skateboard (Mammygol 23 inch) as a passive, free-floating entity.

Same structure as the HUSKY skateboard (arXiv 2602.03205): a deck that rolls on
a bushing spring, two trucks that yaw, and joint equalities that turn deck roll
into truck yaw (lean to steer). Nothing is actuated; the bushings are joint
stiffness and damping.

Deck length, deck width, wheel radius, deck height, wheelbase and track are
measured on the team's board. Mass and the truck, bushing and wheel numbers are
still guesses; change them in `SkateboardParams`.
"""

from dataclasses import dataclass

import mujoco

from mjlab.entity import EntityCfg

DECK_GEOM_NAME = "deck_collision"
WHEEL_GEOM_NAMES: tuple[str, ...] = (
  "wheel_fl_collision",
  "wheel_fr_collision",
  "wheel_rl_collision",
  "wheel_rr_collision",
)
DECK_ROLL_JOINT_NAME = "deck_roll"


@dataclass(frozen=True)
class SkateboardParams:
  deck_length: float = 0.584
  """Measured."""
  deck_width: float = 0.161
  """Measured."""
  wheel_radius: float = 0.029
  """Measured: 58 mm wheels."""
  deck_top_height: float = 0.103
  """Measured. Floor to top of deck with the board resting on its wheels."""
  deck_thickness: float = 0.03
  """Collision box thickness, not the real 12 mm: Pupper's soft foot contacts
  sink about 1 to 3 cm, so the box is thick enough to keep them from tunnelling
  through. The top face stays at `deck_top_height`."""
  wheelbase: float = 0.365
  """Measured, less accurately: front axle to rear axle, longer than the
  caliper reaches."""
  track: float = 0.104
  """Measured. Left wheel centre to right wheel centre."""
  mass: float = 1.8
  """Guess. Whole board."""
  roll_stiffness: float = 30.0
  """Guess. Bushing spring about the roll axis, N*m/rad."""
  roll_damping: float = 0.5
  roll_limit: float = 0.2
  """Deck roll range, rad (wheel bite on the real board)."""
  steer_gain: float = 0.577
  """Guess. Truck yaw per unit deck roll: tan(30 deg kingpin rake), as in HUSKY."""
  wheel_damping: float = 2e-4
  wheel_frictionloss: float = 2e-3
  """Guess for bearing drag. Fit to a coast-down of the real board."""
  deck_friction: tuple[float, float, float] = (1.0, 0.02, 0.01)
  wheel_friction: tuple[float, float, float] = (0.9, 0.005, 0.0001)


def get_skateboard_spec(
  params: SkateboardParams | None = None,
) -> mujoco.MjSpec:
  p = params or SkateboardParams()
  hinge = mujoco.mjtJoint.mjJNT_HINGE
  box = mujoco.mjtGeom.mjGEOM_BOX
  spec = mujoco.MjSpec()

  # Root frame sits at axle height, midway between the trucks.
  board = spec.worldbody.add_body(name="board", pos=(0, 0, p.wheel_radius))
  board.add_freejoint(name="board_free")
  board.add_geom(
    name="chassis",
    type=box,
    size=(p.wheelbase / 2, 0.01, 0.005),
    mass=0.1 * p.mass,
    contype=0,
    conaffinity=0,
  )

  # Deck: rolls about the axle line, held level by the bushing spring.
  deck_z = p.deck_top_height - p.deck_thickness / 2 - p.wheel_radius
  deck = board.add_body(name="deck", pos=(0, 0, deck_z))
  deck.add_joint(
    name=DECK_ROLL_JOINT_NAME,
    type=hinge,
    axis=(1, 0, 0),
    pos=(0, 0, -deck_z),
    stiffness=p.roll_stiffness,
    damping=p.roll_damping,
    range=(-p.roll_limit, p.roll_limit),
    limited=True,
  )
  deck.add_geom(
    name=DECK_GEOM_NAME,
    type=box,
    size=(p.deck_length / 2, p.deck_width / 2, p.deck_thickness / 2),
    mass=0.6 * p.mass,
    friction=p.deck_friction,
    rgba=(0.95, 0.45, 0.65, 1.0),
  )
  deck.add_site(name="deck_site", pos=(0, 0, p.deck_thickness / 2))

  # Trucks: front yaws against the lean and rear with it, so the board carves
  # towards the low side.
  yaw_limit = p.steer_gain * p.roll_limit + 0.05
  for end, x, sign in (("f", p.wheelbase / 2, -1.0), ("r", -p.wheelbase / 2, 1.0)):
    truck = board.add_body(name=f"truck_{end}", pos=(x, 0, 0))
    truck.add_joint(
      name=f"truck_{end}_yaw",
      type=hinge,
      axis=(0, 0, 1),
      range=(-yaw_limit, yaw_limit),
      limited=True,
      armature=1e-4,
      damping=0.01,
    )
    truck.add_geom(
      name=f"truck_{end}_geom",
      type=box,
      size=(0.005, p.track / 2, 0.005),
      mass=0.01,
      contype=0,
      conaffinity=0,
    )
    for side, y in (("l", p.track / 2), ("r", -p.track / 2)):
      wheel = truck.add_body(name=f"wheel_{end}{side}", pos=(0, y, 0))
      wheel.add_joint(
        name=f"wheel_{end}{side}_spin",
        type=hinge,
        axis=(0, 1, 0),
        damping=p.wheel_damping,
        frictionloss=p.wheel_frictionloss,
      )
      # Spheres, not cylinders: one contact per wheel and no edge catching.
      wheel.add_geom(
        name=f"wheel_{end}{side}_collision",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(p.wheel_radius, 0, 0),
        mass=0.3 * p.mass / 4,
        friction=p.wheel_friction,
        rgba=(0.9, 0.9, 0.95, 1.0),
      )
      # The thickened deck box overlaps the wheels at full lean.
      spec.add_exclude(bodyname1="deck", bodyname2=f"wheel_{end}{side}")
    steer = spec.add_equality(
      name=f"steer_{end}",
      type=mujoco.mjtEq.mjEQ_JOINT,
      name1=f"truck_{end}_yaw",
      name2=DECK_ROLL_JOINT_NAME,
      objtype=mujoco.mjtObj.mjOBJ_JOINT,
    )
    steer.data[:5] = [0.0, sign * p.steer_gain, 0.0, 0.0, 0.0]

  return spec


def get_skateboard_cfg(params: SkateboardParams | None = None) -> EntityCfg:
  p = params or SkateboardParams()
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, p.wheel_radius),
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
    spec_fn=lambda: get_skateboard_spec(p),
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  viewer.launch(Entity(get_skateboard_cfg()).spec.compile())
