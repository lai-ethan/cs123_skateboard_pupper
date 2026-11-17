"""The skate notebook's REWARD_WEIGHTS cell matches the env configs."""

import json
from pathlib import Path

import pytest

from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import (
  pupper_skate_balance_env_cfg,
  pupper_skate_ride_env_cfg,
)

REPO = Path(__file__).parent.parent
NOTEBOOK = REPO / "notebooks" / "CS123_Pupper_Skate.ipynb"


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
