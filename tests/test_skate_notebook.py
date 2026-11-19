"""The skate notebook's REWARD_WEIGHTS cell matches the env configs."""

import json
from pathlib import Path

import pytest

from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import (
  pupper_skate_balance_env_cfg,
  pupper_skate_ride_env_cfg,
)
from mjlab.tasks.pupper_skate.pupper_skate_split_env_cfg import (
  pupper_skate_split_balance_env_cfg,
  pupper_skate_split_ride_env_cfg,
  pupper_skate_split_steer_env_cfg,
)

REPO = Path(__file__).parent.parent
NOTEBOOK = REPO / "notebooks" / "CS123_Pupper_Skate.ipynb"
TASKS = (
  "Mjlab-Skate-Balance-Pupper-v3",
  "Mjlab-Skate-Ride-Pupper-v3",
  "Mjlab-SkateSplit-Balance-Pupper-v3",
  "Mjlab-SkateSplit-Ride-Pupper-v3",
)


def _reward_weights_cell() -> str:
  nb = json.loads(NOTEBOOK.read_text())
  cells = [
    "".join(c["source"])
    for c in nb["cells"]
    if c["cell_type"] == "code" and "REWARD_WEIGHTS = {" in "".join(c["source"])
  ]
  assert len(cells) == 1
  return cells[0]


@pytest.mark.parametrize(
  "task, make_cfg",
  [
    ("Mjlab-Skate-Balance-Pupper-v3", pupper_skate_balance_env_cfg),
    ("Mjlab-Skate-Ride-Pupper-v3", pupper_skate_ride_env_cfg),
    ("Mjlab-SkateSplit-Balance-Pupper-v3", pupper_skate_split_balance_env_cfg),
    ("Mjlab-SkateSplit-Ride-Pupper-v3", pupper_skate_split_ride_env_cfg),
    ("Mjlab-SkateSplit-Steer-Pupper-v3", pupper_skate_split_steer_env_cfg),
  ],
)
def test_reward_weights_match_the_env(task, make_cfg):
  scope: dict = {"TASK": task}
  exec(_reward_weights_cell(), scope)
  weights = scope["REWARD_WEIGHTS"]
  rewards = make_cfg().rewards
  # Every key is a term of this task (the train cell skips unknown names
  # silently, so a typo would go unnoticed), and every term is listed.
  assert set(weights) == set(rewards)
  # The notebook ships the env's defaults.
  for name, weight in weights.items():
    assert weight == pytest.approx(rewards[name].weight), name


def test_setup_cell_updates_an_existing_clone():
  nb = json.loads(NOTEBOOK.read_text())
  setup = [
    "".join(c["source"])
    for c in nb["cells"]
    if c["cell_type"] == "code" and "REPO_URL = " in "".join(c["source"])
  ]
  assert len(setup) == 1
  assert "!git -C /content/pupper-mjlab fetch --depth 1 origin {BRANCH}" in setup[0]
  assert "!git -C /content/pupper-mjlab reset --hard FETCH_HEAD" in setup[0]


def _code_cells() -> list[str]:
  nb = json.loads(NOTEBOOK.read_text())
  return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_settings_cell_lists_all_four_tasks():
  settings = [c for c in _code_cells() if "NUM_ENVS = " in c]
  assert len(settings) == 1
  for task in TASKS:
    assert f"#   {task}" in settings[0], task


@pytest.mark.parametrize("task", TASKS)
def test_gait_cell_runs_for_all_four_tasks(task):
  gait = [c for c in _code_cells() if "gait_report.py" in c]
  assert len(gait) == 1
  lines = gait[0].splitlines()
  # One shell command, not indented, so no `if` can skip it.
  commands = [line for line in lines if line.lstrip().startswith("!")]
  assert commands == [
    "!python skateboard/tools/gait_report.py {TASK} {ckpt} {SPEED_FLAG}"
  ]
  scope: dict = {"TASK": task}
  exec(_reward_weights_cell(), scope)
  exec("\n".join(line for line in lines if line.startswith("SPEED_FLAG = ")), scope)
  command = commands[0].format(ckpt="model_1499.pt", **scope)
  assert command.split()[2] == task
  if task == "Mjlab-Skate-Ride-Pupper-v3":
    # The tripod ride's top command.
    assert command.endswith(" --speed 0.3")
  elif "-Ride-" in task:
    assert command.endswith(" --speed 0.4")
  else:
    assert "--speed" not in command


def test_settings_cell_starts_fresh_by_default():
  settings = [c for c in _code_cells() if "NUM_ENVS = " in c]
  assert len(settings) == 1
  scope: dict = {}
  exec(settings[0], scope)
  assert scope["RESUME_FROM"] == ""
  assert "<entity>/mjlab/<run-id>" in settings[0]


def test_train_cell_resumes_only_when_asked():
  train = [c for c in _code_cells() if "launch_training(" in c]
  assert len(train) == 1
  lines = train[0].splitlines()
  # Empty RESUME_FROM: the same call as before warm starts existed.
  fresh = "  launch_training(TASK, TrainConfig(env=env_cfg, agent=agent_cfg))"
  assert lines[-2:] == ["else:", fresh]
  resume = train[0][train[0].index("if RESUME_FROM:") : train[0].index("\nelse:")]
  assert "agent_cfg.resume = True" in resume
  assert "wandb_run_path=RESUME_FROM" in resume
