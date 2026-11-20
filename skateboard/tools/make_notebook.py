"""Build notebooks/CS123_Pupper_Skate.ipynb from the course notebook.

Usage (repo root):
  python3 skateboard/tools/make_notebook.py notebooks/CS123_Pupper_mjlab.ipynb \\
    notebooks/CS123_Pupper_Skate.ipynb
"""

import json
import sys

src, dst = sys.argv[1], sys.argv[2]
nb = json.load(open(src))
cells = nb["cells"]


def md(text: str) -> dict:
  return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
  return {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": text.rstrip("\n").splitlines(True),
  }


def text(i: int) -> str:
  return "".join(cells[i]["source"])


def only_if(flag: str, body: str) -> str:
  """Wrap a cell body in `if flag:` so a settings switch can skip it."""
  lines = body.rstrip("\n").splitlines(True)
  return f"if {flag}:\n" + "".join("  " + ln if ln.strip() else ln for ln in lines)


setup = text(2)
setup = setup.replace(
  'REPO_URL = "https://github.com/cs123-stanford/pupper-mjlab.git"',
  'REPO_URL = "https://github.com/lai-ethan/cs123_skateboard_pupper.git"  # team repo',
)
assert "cs123_skateboard_pupper" in setup and 'BRANCH = "main"' in setup
# Rerunning the cell on a runtime that already has the repo pulls the latest
# commit instead of keeping the old one.
CLONE = """\
if not os.path.exists("/content/pupper-mjlab"):
  !git clone --branch {BRANCH} --depth 1 {REPO_URL} /content/pupper-mjlab
"""
assert CLONE in setup
setup = setup.replace(
  CLONE,
  CLONE
  + """\
else:
  !git -C /content/pupper-mjlab fetch --depth 1 origin {BRANCH}
  !git -C /content/pupper-mjlab reset --hard FETCH_HEAD
""",
)
assert "reset --hard FETCH_HEAD" in setup

watch_live = text(15)
assert watch_live.startswith("import subprocess")
watch_live_md = text(14).replace(
  "*This cell does **not** train — it only starts the viewer.*",
  "*This cell does **not** train — it only starts the viewer, and only when\n"
  "`WATCH_LIVE = True` in section 3.*",
)
assert "WATCH_LIVE" in watch_live_md
train = text(17)
assert 'agent_cfg.logger = "wandb"' in train
train = train.replace('agent_cfg.logger = "wandb"', "agent_cfg.logger = LOGGER")
# Warm start: RESUME_FROM set loads that W&B run's last checkpoint, like
# `train --agent.resume True --wandb-run-path ...`. Empty keeps the old call.
LAUNCH = "launch_training(TASK, TrainConfig(env=env_cfg, agent=agent_cfg))"
assert train.rstrip("\n").endswith(LAUNCH)
train = train.rstrip("\n")[: -len(LAUNCH)] + (
  "if RESUME_FROM:\n"
  "  # Checkpoint numbers carry on from the loaded one: resuming model_1499.pt\n"
  "  # for 1500 iterations ends at model_2998.pt, in a new run folder.\n"
  '  assert LOGGER == "wandb", "RESUME_FROM needs a W&B key (section 2)."\n'
  "  agent_cfg.resume = True\n"
  "  launch_training(\n"
  "    TASK, TrainConfig(env=env_cfg, agent=agent_cfg, wandb_run_path=RESUME_FROM)\n"
  "  )\n"
  "else:\n"
  f"  {LAUNCH}\n"
)
# The split ride draws the deck feet's and the floor feet's grip on their own
# and has no single foot_friction event; FRICTION_RANGE skips it.
FRICTION = 'env_cfg.events["foot_friction"].params["ranges"] = FRICTION_RANGE\n'
assert FRICTION in train
train = train.replace(
  FRICTION,
  'if "foot_friction" in env_cfg.events:  # split ride: grip per side\n  ' + FRICTION,
)
dr_md = text(11)
assert dr_md.startswith("## 6. Domain randomization")
dr_md = dr_md.replace("## 6.", "## 5.", 1)
viewer_md = text(18)
assert viewer_md.startswith("## 8. Watch it")
viewer_md = viewer_md.replace(
  "from Colab). Drive the robot with the sliders.",
  "from Colab). Drive the robot with the sliders. Skipped when\n"
  "`OPEN_VIEWER = False`; when it runs, the cell keeps running until you stop it.",
)
assert "OPEN_VIEWER" in viewer_md
# Re-wrap the lines that `if OPEN_VIEWER:` pushes past 88 columns.
viewer = text(19)
for long, wrapped in [
  (
    'runner.load(str(ckpts[-1]), load_cfg={"actor": True}, strict=True,'
    " map_location=device)",
    'runner.load(\n  str(ckpts[-1]), load_cfg={"actor": True}, strict=True,'
    " map_location=device\n)",
  ),
  (
    "ViserPlayViewer(env, runner.get_inference_policy(device=device),"
    " viser_server=server).run()",
    "ViserPlayViewer(\n  env, runner.get_inference_policy(device=device),"
    " viser_server=server\n).run()",
  ),
]:
  assert long in viewer, long
  viewer = viewer.replace(long, wrapped)

WANDB_CELL = """\
WANDB_API_KEY = ""  # optional: paste a key, or set the Colab secret WANDB_API_KEY

import os

if not WANDB_API_KEY:
  try:
    from google.colab import userdata

    WANDB_API_KEY = userdata.get("WANDB_API_KEY") or ""
  except Exception:
    WANDB_API_KEY = ""

if WANDB_API_KEY:
  import wandb

  # A pasted key often carries a stray space or newline; W&B rejects those.
  WANDB_API_KEY = "".join(c for c in WANDB_API_KEY if c.isalnum() or c == "_")

  os.environ["WANDB_API_KEY"] = WANDB_API_KEY
  wandb.login(key=WANDB_API_KEY, relogin=True)
  LOGGER = "wandb"
else:
  LOGGER = "tensorboard"
  print("No W&B key: logging to TensorBoard only. policy.json stays on this runtime.")
print("logger:", LOGGER)
"""

REWARD_CELL = """\
STEER = "-Steer-" in TASK
RIDE = "-Ride-" in TASK or STEER
SPLIT = "SkateSplit" in TASK

# These are the shipped defaults. Change a number, then rerun from here.
if SPLIT:
  REWARD_WEIGHTS = {
    "feet_on_deck": 1.0,
    "feet_on_ground": 0.1 if RIDE else 0.5,
    "deck_feet_on_ground": -2.0,
    "ground_feet_on_deck": -2.0,
    "knees_on_deck": -1.0,
    "tilt": 1.0,
    "base_height": 1.0,
    "stance_pose": 1.0,
    "board_alignment": 2.0 if RIDE else 1.0,
    "deck_flat": -5.0,
    "action_rate_l2": -0.05 if RIDE else -0.01,
    "joint_torques_l2": -0.0002,
    "termination": -100.0,
    "torque_saturation": -1.5 if RIDE else -1.0,
  }
  if RIDE:
    REWARD_WEIGHTS.update({
      "track_board_velocity": 3.0,
      "board_yaw_rate": -0.5,
      "track_yaw_rate": 1.0,
      "push_stride": 0.5,
      "short_stride": -0.3,
      "feet_slide": -2.0,
      "deck_feet_slide": -20.0,
      "push_leg_pose": 0.3,
      "joint_vel_l2": -0.001,
      "joint_speed_limit": -0.3,
      "push_foot_idle": -0.5,
      "action_acc": -0.02,
      "joint_acc": -1e-6,
      "swing_height": -5.0,
      "board_fore_aft": 1.0,
      "board_creep": -1.0,
    })
  else:
    REWARD_WEIGHTS["board_still"] = -2.0
  if STEER:
    REWARD_WEIGHTS.update({
      "track_board_yaw_rate": 2.0,
      "board_turn_progress": 1.0,
      "board_side_creep": -1.0,
    })
else:
  REWARD_WEIGHTS = {
    "feet_on_deck": 0.5 if RIDE else 1.0,
    "tilt": 1.0,
    "board_alignment": 1.0,
    "deck_flat": -5.0,
    "action_rate_l2": -0.05 if RIDE else -0.01,
    "joint_torques_l2": -0.0002,
    "termination": -20.0 if RIDE else -100.0,
    "torque_saturation": -1.0,
  }
  if RIDE:
    REWARD_WEIGHTS.update({
      "track_board_velocity": 3.0,
      "board_progress": 4.0,
      "board_yaw_rate": -0.5,
      "reference_tracking": 1.0,
      "foot_reach": 1.0,
      "push_stroke": 3.0,
      "short_stride": -3.0,
      "feet_slide": -2.0,
      "action_acc": -0.02,
      "joint_acc": -1e-6,
      "joint_vel_l2": -0.001,
      "joint_speed_limit": -0.1,
    })
  else:
    REWARD_WEIGHTS.update({
      "knees_on_deck": -1.0,
      "stance_pose": 1.0,
      "feet_on_ground": -1.0,
      "board_still": -2.0,
    })
"""

GAIT_CELL = """\
from pathlib import Path

runs = sorted(Path("logs/rsl_rl").glob("*/*"), key=lambda p: p.stat().st_mtime)
assert runs, "No runs yet -- train first."
ckpts = sorted(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
ckpt = ckpts[-1]
print("Gait check on:", ckpt)
# Ride is checked at 0.4 m/s, the centred (tripod) ride at its top command,
# 0.3 m/s; balance has no speed command.
SPEED_FLAG = ("--speed 0.4" if SPLIT else "--speed 0.3") if RIDE else ""
!python skateboard/tools/gait_report.py {TASK} {ckpt} {SPEED_FLAG}
"""

out = [
  md(
    "# Pupper on a skateboard 🛹\n\n"
    "Trains the skateboard tasks. There are two stances:\n\n"
    "- **Centred** (`Mjlab-Skate-*`): all four feet on the deck (balance), or\n"
    "  three on the deck and the back right foot on the floor (ride).\n"
    "- **Split** (`Mjlab-SkateSplit-*`): left feet on the deck, right feet on\n"
    "  the floor.\n\n"
    "Each stance has a balance task and a ride task. Built from\n"
    "`CS123_Pupper_mjlab.ipynb`; the tasks live in `src/mjlab/tasks/pupper_skate/`.\n\n"
    "*Runtime → Change runtime type → **G4 GPU**.* Run all works unattended; or\n"
    "run cells one at a time. Section 7 prints the verdict: the last line is\n"
    "`RESULT: PASS` or `RESULT: FAIL`.\n"
  ),
  cells[1],
  code(setup),
  md(
    "## 2. Weights & Biases (optional)\n\n"
    "Paste a key from [wandb.ai/authorize](https://wandb.ai/authorize) below, or\n"
    "store it once as the Colab secret `WANDB_API_KEY` (🔑 in the left bar). This\n"
    "cell never prompts.\n\n"
    "Without a key the run logs to TensorBoard only. You lose the W&B curves and\n"
    "the automatic `policy.json` upload that `deploy.sh` downloads; the policy\n"
    "stays in `logs/rsl_rl/` on this runtime, so download it before you\n"
    "disconnect. Never commit a key.\n"
  ),
  code(WANDB_CELL),
  md(
    "## 3. Choose your task\n\n"
    "| Task | What it is |\n| --- | --- |\n"
    "| `Mjlab-Skate-Balance-Pupper-v3` | Hold the centred stance on a board that"
    " is free to roll. Train this first. |\n"
    "| `Mjlab-Skate-Ride-Pupper-v3` | Tripod ride: start with three feet on"
    " the deck and the back right foot on the floor beside it, and stroke with"
    " that foot (a clocked stroke reference) so the board rolls at the"
    " commanded speed (0.1 to 0.3 m/s). The foot never steps back onto the"
    " deck. Built for the stock trucks. |\n"
    "| `Mjlab-SkateSplit-Balance-Pupper-v3` | Hold the split stance (left feet"
    " on the deck, right feet on the floor) on a board that is free to roll. |\n"
    "| `Mjlab-SkateSplit-Ride-Pupper-v3` | Push with the right legs so the board"
    " rolls at the commanded speed (0 to 0.5 m/s). |\n"
    "| `Mjlab-SkateSplit-Steer-Pupper-v3` | The split ride plus a carve command"
    " (0.1 to 0.6 rad/s, either sign) that asks the board to turn at a tenth of"
    " it, by leaning the deck. At yaw 0 it is the ride, heading hold on. Trains"
    " on loosened trucks (9 to 18 N m/rad). |\n\n"
    "Train a stance's balance task before its ride task.\n"
  ),
  code(
    "# Pick one:\n"
    "#   Mjlab-Skate-Balance-Pupper-v3       centred: hold the stance on a free board\n"
    "#   Mjlab-Skate-Ride-Pupper-v3          centred: tripod, stroke the BR foot\n"
    "#   Mjlab-SkateSplit-Balance-Pupper-v3  split: hold the stance, right feet down\n"
    "#   Mjlab-SkateSplit-Ride-Pupper-v3     split: push with the right legs to ride\n"
    "#   Mjlab-SkateSplit-Steer-Pupper-v3    split: ride plus a carve command"
    " to lean into a turn\n"
    'TASK = "Mjlab-Skate-Balance-Pupper-v3"\n'
    "NUM_ENVS = 4096  # halve it if you hit GPU OOM\n"
    "ITERATIONS = 1500\n"
    "WATCH_LIVE = False  # True: start the live viewer before training\n"
    "OPEN_VIEWER = True  # False: skip the interactive viewer at the end\n"
    "# A W&B run path, <entity>/mjlab/<run-id>, starts from that run's last\n"
    "# checkpoint; empty starts fresh.\n"
    'RESUME_FROM = ""\n'
  ),
  md(
    "## 4. Reward weights ✏️\n\n"
    "Unlike the course tasks, the skate tasks ship with working weights. The\n"
    "cell below lists every one for the chosen task; change a number to tune\n"
    "it. Shipped defaults, for reference.\n\n"
    "**Centred** (`Mjlab-Skate-*`):\n\n"
    "| Term | Balance | Ride | What it does |\n| --- | --- | --- | --- |\n"
    "| `feet_on_deck` | 1.0 | 0.5 | +1 per foot touching the deck (ride: the"
    " three support feet) |\n"
    "| `feet_on_ground` | -1.0 | n/a | a foot on the floor (balance only) |\n"
    "| `knees_on_deck` | -1.0 | n/a | knee resting on the deck (ride: the"
    " tripod layout rests the front left knee; lying down still ends the"
    " episode) |\n"
    "| `tilt` | 1.0 | 1.0 | body level (ride: at the tripod layout's lean,"
    " looser) |\n"
    "| `stance_pose` | 1.0 | n/a | joints near the keyframe (balance only) |\n"
    "| `reference_tracking` | n/a | 1.0 | joints follow the stroke reference"
    " (the held tripod layout at zero command) |\n"
    "| `board_alignment` | 1.0 | 1.0 | board centred under the feet, parallel to"
    " the robot (ride: where the tripod layout puts it, looser) |\n"
    "| `deck_flat` | -5.0 | -5.0 | deck leaning on its trucks |\n"
    "| `board_still` | -2.0 | n/a | board moving (balance only) |\n"
    "| `track_board_velocity` | n/a | 3.0 | board speed matches the command"
    " (ride: within about 0.1 m/s) |\n"
    "| `board_progress` | n/a | 4.0 | the share of the commanded speed the board"
    " reaches, so a first slow push pays |\n"
    "| `board_yaw_rate` | n/a | -0.5 | board turning |\n"
    "| `foot_reach` | n/a | 1.0 | the push foot is close to the floor; fades"
    " as the board speeds up, gone at 0.8 of the command |\n"
    "| `push_stroke` | n/a | 3.0 | the push foot pushes against the floor; paid"
    " once as it lifts, in full for a 0.75 s stroke |\n"
    "| `short_stride` | n/a | -3.0 | the push foot lands after less than 0.75 s"
    " on the floor; a tap costs about the full weight, so taps lose more than"
    " they earn |\n"
    "| `feet_slide` | n/a | -2.0 | the push foot moving while on the floor:"
    " landing still swinging forward brakes the board |\n"
    "| `joint_vel_l2` | n/a | -1e-3 | joint speed |\n"
    "| `joint_speed_limit` | n/a | -0.1 | joint speed above 6 rad/s |\n"
    "| `torque_saturation` | -1.0 | -1.0 | motors above half (balance) or 80% (ride)"
    " of their 3 Nm limit |\n"
    "| `action_rate_l2` | -0.01 | -0.05 | jerky actions |\n"
    "| `action_acc` | n/a | -0.02 | action acceleration |\n"
    "| `joint_acc` | n/a | -1e-6 | joint acceleration |\n"
    "| `joint_torques_l2` | -2e-4 | -2e-4 | effort |\n"
    "| `termination` | -100 | -20 | falling, leaving the deck (ride: a support"
    " foot off), lying on the deck, tipping the board (ride: cheaper, so a"
    " push is worth trying) |\n"
    "\n**Split** (`Mjlab-SkateSplit-*`):\n\n"
    "| Term | Balance | Ride | What it does |\n| --- | --- | --- | --- |\n"
    "| `feet_on_deck` | 1.0 | 1.0 | +1 per left foot touching the deck |\n"
    "| `feet_on_ground` | 0.5 | 0.1 | +1 per right foot touching the floor |\n"
    "| `deck_feet_on_ground` | -2.0 | -2.0 | left foot on the floor |\n"
    "| `ground_feet_on_deck` | -2.0 | -2.0 | right foot on the deck |\n"
    "| `knees_on_deck` | -1.0 | -1.0 | knee resting on the deck |\n"
    "| `tilt` | 1.0 | 1.0 | hold the stance's 0.083 rad roll, no pitch |\n"
    "| `base_height` | 1.0 | 1.0 | hold 0.154 m |\n"
    "| `stance_pose` | 1.0 | 1.0 | joints near the keyframe (ride: left legs"
    " only) |\n"
    "| `board_alignment` | 1.0 | 2.0 | board under the left feet, parallel to"
    " the robot (ride: tighter, about 3 cm) |\n"
    "| `board_fore_aft` | n/a | 1.0 | robot holds its place along the deck, measured"
    " from where the episode started (3 cm width) |\n"
    "| `board_creep` | n/a | -1.0 | robot creeping along the deck, averaged twice"
    " over 1 s (full cost at 2 cm/s) |\n"
    "| `deck_flat` | -5.0 | -5.0 | deck leaning on its trucks |\n"
    "| `board_still` | -2.0 | n/a | board moving (balance only) |\n"
    "| `track_board_velocity` | n/a | 3.0 | board speed matches the command |\n"
    "| `board_yaw_rate` | n/a | -0.5 | board turning |\n"
    "| `push_stride` | n/a | 0.5 | right feet take planted strides: the time a"
    " 10 cm stroke takes at the command (0.25 to 0.5 s) on the floor, then a"
    " 0.2 to 0.5 s swing |\n"
    "| `short_stride` | n/a | -0.3 | a right foot lands after a shorter stance"
    " than `push_stride` asks for; a tap with no stance costs the full weight |\n"
    "| `feet_slide` | n/a | -2.0 | a right foot slides while on the floor |\n"
    "| `deck_feet_slide` | n/a | -20.0 | a left foot slides over the deck (measured"
    " against the deck, which moves) |\n"
    "| `push_foot_idle` | n/a | -0.5 | a right foot held in the air for more than"
    " 0.5 s |\n"
    "| `push_leg_pose` | n/a | 0.3 | right legs stay loosely near the keyframe |\n"
    "| `joint_vel_l2` | n/a | -1e-3 | joint speed |\n"
    "| `joint_speed_limit` | n/a | -0.3 | joint speed above 5 rad/s |\n"
    "| `torque_saturation` | -1.0 | -1.5 | motors above half (balance) or 80% (ride)"
    " of their 3 Nm limit |\n"
    "| `action_rate_l2` | -0.01 | -0.05 | jerky actions |\n"
    "| `action_acc` | n/a | -0.02 | snaps in the actions (lift-off, touchdown) |\n"
    "| `joint_acc` | n/a | -1e-6 | joint acceleration |\n"
    "| `swing_height` | n/a | -5.0 | a right foot in the air lifted more than 3 cm"
    " (per metre over) |\n"
    "| `joint_torques_l2` | -2e-4 | -2e-4 | effort |\n"
    "| `termination` | -100 | -100 | falling, leaving the deck, tipping the board"
    " (ride: also the board 6 cm sideways out of place, or the robot 8 cm along"
    " the deck from its start or 12 cm from the keyframe) |\n"
  ),
  code(REWARD_CELL),
  md(dr_md),
  cells[12],
  md(
    "## 6. Train 🚀\n\n"
    "*Watch it live* starts a viewer when `WATCH_LIVE = True`; the training cell\n"
    "after it takes about 9 min for 1,500 iterations on the G4 (about 55 on a T4).\n"
    "The stop button is safe.\n\n"
    "What to watch on W&B (or TensorBoard without a key):\n\n"
    "- `Train/mean_episode_length`: 500 is a full 10 s episode.\n"
    "- `Episode_Reward/feet_on_deck`: centred, 4.0 means all four feet on the"
    " deck all episode (ride pays half for three feet at most, so 1.5); balance\n"
    "  passes at 3.6. Split, 2.0 means both left feet on the deck all episode;\n"
    "  balance passes at 1.8.\n"
    "- Centred balance: `Episode_Reward/feet_on_ground` should stay near 0.\n"
    "- Balance: `Episode_Reward/torque_saturation` should stay near 0; balance\n"
    "  passes at -0.05 or above. Lower means motors push near their limit.\n"
    "- `Episode_Termination/*`: which termination ends episodes.\n"
    "- Ride: `Episode_Reward/track_board_velocity` (3.0 is perfect).\n"
    "- Centred ride: `Episode_Reward/push_stroke` (above 0 means the push foot\n"
    "  is stroking). It should rise above zero within about 300 iterations. If\n"
    "  it stays at zero, stop the ride run: it is not learning to push.\n"
    "  (`foot_reach` pays from the start: the tripod layout already has the\n"
    "  push foot on the floor.)\n"
    "- Split ride: `Episode_Reward/push_stride` (above 0 means the right feet\n"
    "  take planted strides).\n"
  ),
  md(watch_live_md),
  code(only_if("WATCH_LIVE", watch_live)),
  md("### Train ▶️ — this is the cell that trains\n\nBlocks until training ends.\n"),
  code(train),
  md(
    "## 7. Did it pass?\n\n"
    "Averages the last 20 iterations of the newest run and checks them against\n"
    "the pass criteria in `skateboard/tools/run_summary.py`. The last line is\n"
    "`RESULT: PASS` or `RESULT: FAIL`.\n"
  ),
  code("!python skateboard/tools/run_summary.py {TASK}\n"),
  md(
    "### Gait check\n\n"
    "Curves can pass while the gait is one the robot cannot do. This rolls out\n"
    "the newest checkpoint on 16 boards drawn from the task's ranges (a fixed\n"
    "seed, so a rerun gives the same answer) and checks joint speed, torque,\n"
    "that each foot touches the deck or floor most of the time and how often a\n"
    "foot lands on the floor. Each limit is judged on the worst board.\n"
    "Ride runs at 0.4 m/s (centred ride: 0.3 m/s) and also checks that the\n"
    "board rolls at 0.5 to 1.5\n"
    "times the command; balance has no speed command, so it skips that check.\n"
    "In the split stance only the two right (pushing) feet are counted.\n"
    "It also prints each foot's time on the floor, so you can see which foot\n"
    "pushes. The last line is `GAIT: PASS` or `GAIT: FAIL`. Watch the video too\n"
    "before trusting a pass.\n"
  ),
  code(GAIT_CELL),
  md(viewer_md),
  code(only_if("OPEN_VIEWER", viewer)),
  md(
    "## 9. Deploy\n\n"
    "With W&B, `policy.json` uploads to the run when training ends; without it,\n"
    "download `policy.json` from the run folder in `logs/rsl_rl/`. **Do not run a\n"
    "skate policy on the robot without a TA sign-off**, and do the powered-off\n"
    "fit test on the real board first (see `skateboard/PLAN.md`, tasks 10 and 12).\n"
  ),
]
nb["cells"] = out
json.dump(nb, open(dst, "w"), indent=1, ensure_ascii=False)
open(dst, "a").write("\n")
print(f"wrote {dst}: {len(out)} cells")
