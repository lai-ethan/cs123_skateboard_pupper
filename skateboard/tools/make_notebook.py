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
    "source": text.splitlines(True),
  }


def text(i: int) -> str:
  return "".join(cells[i]["source"])


setup = text(2)
setup = setup.replace(
  'REPO_URL = "https://github.com/cs123-stanford/pupper-mjlab.git"',
  'REPO_URL = "https://github.com/lai-ethan/cs123_skateboard_pupper.git"  # team repo',
)
assert "cs123_skateboard_pupper" in setup and 'BRANCH = "main"' in setup

out = [
  md(
    "# Pupper on a skateboard 🛹\n\n"
    "Trains the split-stance skateboard tasks (left feet on the deck, right feet\n"
    "on the floor). Built from `CS123_Pupper_mjlab.ipynb`; the tasks live in\n"
    "`src/mjlab/tasks/pupper_skate/`.\n\n"
    "*Runtime → Change runtime type → **G4 GPU**, then run cells top to bottom.*\n"
  ),
  cells[1],
  code(setup),
  cells[3],
  cells[4],
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
    "ITERATIONS = 3000\n"
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
  cells[11],
  cells[12],
  md(
    "## 6. Train 🚀\n\n"
    "Run *Watch it live* first if you want the viewer, then the training cell\n"
    "(about 18 min for 3,000 iterations on the G4). The stop button is safe.\n\n"
    "What to watch on W&B:\n\n"
    "- `Train/mean_episode_length`: 500 is a full 10 s episode.\n"
    "- `Episode_Reward/feet_on_deck`: 2.0 means both left feet on the deck all"
    " episode.\n"
    "- `Episode_Termination/*`: which termination ends episodes.\n"
    "- Ride only: `Episode_Reward/track_board_velocity` (3.0 is perfect).\n"
  ),
  cells[14],
  cells[15],
  cells[16],
  cells[17],
  cells[18],
  cells[19],
  md(
    "## 8. Deploy\n\n"
    "`policy.json` uploads to the W&B run when training ends. **Do not run a\n"
    "skate policy on the robot without a TA sign-off**, and do the powered-off\n"
    "fit test on the real board first (see `skateboard/PLAN.md`, tasks 10 and 12).\n"
  ),
]
nb["cells"] = out
json.dump(nb, open(dst, "w"), indent=1, ensure_ascii=False)
open(dst, "a").write("\n")
print(f"wrote {dst}: {len(out)} cells")
