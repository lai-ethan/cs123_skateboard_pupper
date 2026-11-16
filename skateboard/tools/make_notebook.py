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

out = [
  md(
    "# Pupper on a skateboard 🛹\n\n"
    "Trains the split-stance skateboard tasks (left feet on the deck, right feet\n"
    "on the floor). Built from `CS123_Pupper_mjlab.ipynb`; the tasks live in\n"
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
    "| `Mjlab-Skate-Balance-Pupper-v3` | Hold the split stance on a board that is"
    " free to roll. Train this first. |\n"
    "| `Mjlab-Skate-Ride-Pupper-v3` | Push with the right legs so the board rolls"
    " at the commanded speed (0 to 0.5 m/s). |\n"
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
    "Unlike the course tasks, the skate tasks ship with working weights. Leave\n"
    "the dict empty to use them; add an entry to override one. Defaults:\n\n"
    "| Term | Balance | Ride | What it does |\n| --- | --- | --- | --- |\n"
    "| `feet_on_deck` | 1.0 | 1.0 | +1 per left foot touching the deck |\n"
    "| `feet_on_ground` | 0.5 | 0.1 | +1 per right foot touching the floor |\n"
    "| `deck_feet_on_ground` | -2.0 | -2.0 | left foot on the floor |\n"
    "| `ground_feet_on_deck` | -2.0 | -2.0 | right foot on the deck |\n"
    "| `knees_on_deck` | -1.0 | -1.0 | knee resting on the deck |\n"
    "| `tilt` | 1.0 | 1.0 | hold the stance's 0.09 rad roll, no pitch |\n"
    "| `base_height` | 1.0 | 1.0 | hold 0.156 m |\n"
    "| `stance_pose` | 1.0 | 1.0 | joints near the keyframe (ride: left legs only) |\n"
    "| `board_alignment` | 1.0 | 1.0 | board under the left feet, parallel to the robot |\n"
    "| `deck_flat` | -5.0 | -5.0 | deck leaning on its trucks |\n"
    "| `board_still` | -2.0 | n/a | board moving (balance only) |\n"
    "| `track_board_velocity` | n/a | 3.0 | board speed matches the command |\n"
    "| `board_yaw_rate` | n/a | -0.5 | board turning |\n"
    "| `push_air_time` | n/a | 0.5 | right feet take 0.05 to 0.4 s strides |\n"
    "| `action_rate_l2` | -0.01 | -0.01 | jerky actions |\n"
    "| `joint_torques_l2` | -2e-4 | -2e-4 | effort |\n"
    "| `termination` | -100 | -100 | falling, leaving the deck, tipping the board |\n"
  ),
  code('REWARD_WEIGHTS = {\n  # "board_still": -2.0,\n}\n'),
  md(dr_md),
  cells[12],
  md(
    "## 6. Train 🚀\n\n"
    "*Watch it live* starts a viewer when `WATCH_LIVE = True`; the training cell\n"
    "after it takes about 9 min for 1,500 iterations on the G4. The stop button is\n"
    "safe.\n\n"
    "What to watch on W&B (or TensorBoard without a key):\n\n"
    "- `Train/mean_episode_length`: 500 is a full 10 s episode.\n"
    "- `Episode_Reward/feet_on_deck`: 2.0 means both left feet on the deck all"
    " episode.\n"
    "- `Episode_Termination/*`: which termination ends episodes.\n"
    "- Ride only: `Episode_Reward/track_board_velocity` (3.0 is perfect).\n"
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
