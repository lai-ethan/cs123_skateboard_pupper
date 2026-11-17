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

GAIT_CELL = """\
from pathlib import Path

if RIDE:
  runs = sorted(Path("logs/rsl_rl").glob("*/*"), key=lambda p: p.stat().st_mtime)
  assert runs, "No runs yet -- train first."
  ckpts = sorted(
    runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])
  )
  ckpt = ckpts[-1]
  print("Gait check on:", ckpt)
  !python skateboard/tools/gait_report.py {TASK} {ckpt} --speed 0.4
else:
  print("Gait check is for the ride task only.")
"""

out = [
  md(
    "# Pupper on a skateboard 🛹\n\n"
    "Trains the skateboard tasks. Pupper stands centred on the deck with all four\n"
    "feet on it. Built from `CS123_Pupper_mjlab.ipynb`; the tasks live in\n"
    "`src/mjlab/tasks/pupper_skate/`.\n\n"
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
    "| `Mjlab-Skate-Ride-Pupper-v3` | Push with one foot off either side so the"
    " board rolls at the commanded speed (0 to 0.5 m/s). |\n"
  ),
  code(
    'TASK = "Mjlab-Skate-Balance-Pupper-v3"\n'
    "NUM_ENVS = 4096  # halve it if you hit GPU OOM\n"
    "ITERATIONS = 1500\n"
    "WATCH_LIVE = False  # True: start the live viewer before training\n"
    "OPEN_VIEWER = True  # False: skip the interactive viewer at the end\n"
  ),
  md(
    "## 4. Reward weights ✏️\n\n"
    "Unlike the course tasks, the skate tasks ship with working weights. The\n"
    "cell below lists every one for the chosen task; change a number to tune\n"
    "it. Shipped defaults, for reference:\n\n"
    "| Term | Balance | Ride | What it does |\n| --- | --- | --- | --- |\n"
    "| `feet_on_deck` | 1.0 | 0.5 | +1 per foot touching the deck |\n"
    "| `feet_on_ground` | -1.0 | n/a | a foot on the floor (balance only) |\n"
    "| `knees_on_deck` | -1.0 | -1.0 | knee resting on the deck |\n"
    "| `tilt` | 1.0 | 1.0 | body level: no roll, no pitch |\n"
    "| `stance_pose` | 1.0 | 0.5 | joints near the keyframe (ride: looser) |\n"
    "| `board_alignment` | 1.0 | 1.0 | board centred under the feet, parallel to"
    " the robot |\n"
    "| `deck_flat` | -5.0 | -5.0 | deck leaning on its trucks |\n"
    "| `board_still` | -2.0 | n/a | board moving (balance only) |\n"
    "| `track_board_velocity` | n/a | 3.0 | board speed matches the command |\n"
    "| `board_yaw_rate` | n/a | -0.5 | board turning |\n"
    "| `push_stroke` | n/a | 0.3 | a foot pushes against the floor; paid once as"
    " it lifts, in full for a 0.25 s stroke |\n"
    "| `joint_vel_l2` | n/a | -1e-3 | joint speed |\n"
    "| `joint_speed_limit` | n/a | -0.02 | joint speed above 6 rad/s |\n"
    "| `torque_saturation` | n/a | -0.5 | motors above 80% of their 3 Nm limit |\n"
    "| `action_rate_l2` | -0.01 | -0.01 | jerky actions |\n"
    "| `joint_torques_l2` | -2e-4 | -2e-4 | effort |\n"
    "| `termination` | -100 | -100 | falling, leaving the deck, tipping the board |\n"
  ),
  code(
    'RIDE = TASK == "Mjlab-Skate-Ride-Pupper-v3"\n\n'
    "# These are the shipped defaults. Change a number, then rerun from here.\n"
    "REWARD_WEIGHTS = {\n"
    '  "feet_on_deck": 0.5 if RIDE else 1.0,\n'
    '  "knees_on_deck": -1.0,\n'
    '  "tilt": 1.0,\n'
    '  "stance_pose": 0.5 if RIDE else 1.0,\n'
    '  "board_alignment": 1.0,\n'
    '  "deck_flat": -5.0,\n'
    '  "action_rate_l2": -0.01,\n'
    '  "joint_torques_l2": -0.0002,\n'
    '  "termination": -100.0,\n'
    "}\n"
    "if RIDE:\n"
    "  REWARD_WEIGHTS.update({\n"
    '    "track_board_velocity": 3.0,\n'
    '    "board_yaw_rate": -0.5,\n'
    '    "push_stroke": 0.3,\n'
    '    "joint_vel_l2": -0.001,\n'
    '    "joint_speed_limit": -0.02,\n'
    '    "torque_saturation": -0.5,\n'
    "  })\n"
    "else:\n"
    "  REWARD_WEIGHTS.update({\n"
    '    "feet_on_ground": -1.0,\n'
    '    "board_still": -2.0,\n'
    "  })\n"
  ),
  md(dr_md),
  cells[12],
  md(
    "## 6. Train 🚀\n\n"
    "*Watch it live* starts a viewer when `WATCH_LIVE = True`; the training cell\n"
    "after it takes about 9 min for 1,500 iterations on the G4 (about 55 on a T4).\n"
    "The stop button is safe.\n\n"
    "What to watch on W&B (or TensorBoard without a key):\n\n"
    "- `Train/mean_episode_length`: 500 is a full 10 s episode.\n"
    "- `Episode_Reward/feet_on_deck`: 4.0 means all four feet on the deck all"
    " episode (ride pays half, so 2.0). Balance passes at 3.6.\n"
    "- Balance: `Episode_Reward/feet_on_ground` should stay near 0.\n"
    "- `Episode_Termination/*`: which termination ends episodes.\n"
    "- Ride only: `Episode_Reward/track_board_velocity` (3.0 is perfect) and\n"
    "  `Episode_Reward/push_stroke` (above 0 means a foot is pushing).\n"
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
    "### Gait check (ride)\n\n"
    "Curves can pass while the gait is one the robot cannot do. This rolls out the\n"
    "newest checkpoint at 0.4 m/s and checks joint speed, torque, how the feet\n"
    "step off and back onto the deck and that the board rolls at half the command\n"
    "or faster. It also prints each foot's time on the floor, so you can see which\n"
    "foot pushes. The last line is `GAIT: PASS` or `GAIT: FAIL`. Watch the video\n"
    "too before trusting a pass.\n"
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
