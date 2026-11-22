# Gemini Voice Control for the Skateboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Written 2026-11-21. Desk and sim work only. Nothing here moves the robot.

**Goal:** A person says "move forward" or "turn left" to Gemini Pupper and the
robot, already placed on the skateboard, rides or carves for a few seconds,
then stops pushing.

**Architecture:** Gemini Pupper gets a skate mode. The launch file loads the
steer policy as one more inactive controller (`gemini_skate`). Five new tools
(`start_skating`, `skate_forward`, `skate_turn`, `end_skating`, plus the
existing `stop`) send timed speed commands on `/llm_cmd_vel`, the same path
the walking tools use. Every other motion is refused while Pupper is on the
board, in the tool registry and again in the robot API.

**Tech Stack:** Python 3.11, FastAPI tool server, ROS 2 Jazzy `ros2_control`
(`NeuralController`), the repo's fake-robot desk tests, mjlab on MuJoCo Warp
for the sim numbers.

**Two repos:**

- `~/claude/cs123/pupper/gemini_pupper` (called GP below). Tasks 2 to 5. Work
  on a new branch `skate-voice` cut from the current `nicholas/todo4-11`.
- `~/claude/cs123/pupper-mjlab` (called MJ below), branch `skateboard`.
  Tasks 1 and 5.

## What "careful without distance" means here

Pupper has no distance or obstacle sensing, so:

- Commands are timed, never "go 2 metres". One command lasts 1 to 5 s
  (default 3 s). At most 10 s can be queued, so a person has to speak again
  to keep it going.
- One speed: 0.3 m/s, the only speed checked in sim, and the policy file
  clips there anyway.
- Every reply gives distance and turn as estimates from simulation and says
  a person must keep the path clear.
- The board has no brake. Stop means "stop pushing"; the reply says how far
  it rolls on.
- "Turn left" is a slow carve, about 1 to 2 degrees a second in sim. A 5 s
  hard carve turns the board about 10 degrees. Gemini is told this so it
  does not promise a corner.
- Hard carve is 0.5 rad/s, not 0.6: v6 lost the board at +0.6 on one sim
  env (see `PLAN_STEER_V8.md`).
- Dead-man: `cmd_vel_mux` drops Gemini 500 ms after the server goes quiet,
  the joystick outranks Gemini, and the right-stick e-stop still relaxes
  the motors.

## Global Constraints

- **No robot motion.** No `robot/start.sh`, no `lab/chat.py --robot`, no
  `curl` to a POST endpoint, no `ros2` command, no SSH to the robot. The
  plan ends at desk tests and a written checklist.
- **Do not raise limits.** No change to `MoveLimits`, `max_body_angle`,
  gains, joint limits or the policy's `command_clip`. Skate numbers may only
  be at or under: speed 0.3 m/s, carve 0.5 rad/s, 5 s a command, 10 s queued.
- **Do not edit** `~/claude/cs123/pupper/pupper_gait_deploy` (read-only).
- **No secrets** in files, commits or logs: no API keys, robot IP, robot
  password, W&B keys or W&B entity names. Both repos are public.
- **Commits:** `<type>: <description>`, authored by the configured git user
  only. No `Co-Authored-By`, no "Generated with", no AI attribution.
- **No pushes** from a task. The controller pushes at the end (MJ to `team`
  after a secret scan; GP only when Ethan says so).
- **GP tests:** `uv run --no-project --with pyyaml python lab/test.py <group>`
  from the GP repo root (the Mac's `python3` has no `yaml`). The full run
  printed "All 166 checks passed" before this plan; it must still pass.
- **GP style:** 4-space indent, existing long-line style, `from __future__
  import annotations`. Tests are `check(...)` calls in `robot/test_tools.py`.
- **MJ style:** 88 columns, 2-space indent, plain-function tests. Always
  `FORCE_CPU=1 uv run ...`. Run only the touched test file. Do not run
  `make check`. One sim run per shell call.
- **Result messages** are all Gemini learns: a refusal states the reason and
  the limit numbers.
- Keep Peng Xu's credit and the name Pupper untouched.

## File structure

| File | Repo | Responsibility |
| --- | --- | --- |
| `skateboard/tools/timed_ride.py` (new) | MJ | Sim rollout of one timed command: travel, turn, coast |
| `tests/test_skate_timed_ride.py` (new) | MJ | Tests for its two pure functions |
| `robot/skateboard.py` (new) | GP | `SkateConfig`, `load_skate`, controller params, constants. No ROS imports |
| `skate/skate.yaml`, `skate/policy.json`, `skate/README.md` (new) | GP | The numbers, the policy file, how to use it |
| `robot/robot_api.py` (modify) | GP | Skate state, the on-board guard in `switch_to` and `set_following`, `mode()` |
| `robot/launch/gemini_pupper.launch.py` (modify) | GP | Load `gemini_skate` inactive |
| `robot/tool_registry.py` (modify) | GP | `{skate}` in descriptions, the on-board guard in `call`, the new lab module |
| `lab/part8_skate.py` (new) | GP | The skate behaviors and tools |
| `lab/prompt.md` (modify) | GP | A Skateboard section |
| `robot/test_tools.py` (modify) | GP | `Rig(skate=...)`, groups `skate_config`, `skate_robot`, `skate_tools` |
| `skateboard/HARDWARE_TEST.md` (modify) | MJ | Test 6: voice control checklist |

---

### Task 1: sim tool for a timed command (MJ)

**Files:**
- Create: `skateboard/tools/timed_ride.py`
- Test: `tests/test_skate_timed_ride.py`

**Interfaces:**
- Consumes: `load_policy(task, policy_path, speed, *, num_envs, seed,
  yaw_rate)` from `skateboard/tools/policy_loader.py`, which returns
  `(raw_env, wrapped_env, policy)`.
- Produces: a CLI that prints one JSON line with keys `stick`, `resets`,
  `envs`, `ride_m_mean`, `ride_m_max`, `m_per_s`, `turn_deg_mean`,
  `turn_deg_per_s`, `coast_m_mean`, `coast_m_max`, `coast_turn_deg_mean`.
  Task 5 reads `m_per_s`, `turn_deg_per_s`, `coast_m_max` and `resets`.

- [ ] **Step 1: Write the failing tests**

`tests/test_skate_timed_ride.py`:

```python
"""Tests for skateboard/tools/timed_ride.py (the pure functions)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).parent.parent / "skateboard" / "tools"


def _load_timed_ride():
  sys.path.insert(0, str(TOOLS))
  spec = importlib.util.spec_from_file_location("timed_ride", TOOLS / "timed_ride.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


tr = _load_timed_ride()


def _straight(metres_per_step, steps, envs=2):
  """Boards moving along x; env 1 stands still."""
  pos = np.zeros((steps + 1, envs, 2))
  pos[:, 0, 0] = np.cumsum([0.0, *metres_per_step])
  return pos


def test_path_length_adds_up_each_step():
  pos = _straight([1.0, 1.0, 0.5], 3)
  dist = tr.path_length(pos)
  assert dist.shape == (4, 2)
  assert dist[:, 0].tolist() == [0.0, 1.0, 2.0, 2.5]
  assert dist[:, 1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_turn_so_far_unwraps_across_pi():
  yaw = np.array([[3.0], [3.1], [-3.1], [-3.0]])
  turn = tr.turn_so_far(yaw)
  assert turn[-1, 0] == pytest.approx(2 * np.pi - 6.0, abs=1e-9)


def test_summarise_splits_ride_and_coast():
  # 2 ride steps of 0.5 m, then 1 coast step of 0.1 m, at 0.5 s a step.
  pos = _straight([0.5, 0.5, 0.1], 3, envs=1)
  yaw = np.radians([[0.0], [1.0], [2.0], [2.5]])
  row = tr.summarise(pos, yaw, ride_steps=2, step_dt=0.5, keep=np.array([True]))
  assert row["envs"] == 1
  assert row["ride_m_mean"] == pytest.approx(1.0)
  assert row["ride_m_max"] == pytest.approx(1.0)
  assert row["m_per_s"] == pytest.approx(1.0)
  assert row["turn_deg_mean"] == pytest.approx(2.0)
  assert row["turn_deg_per_s"] == pytest.approx(2.0)
  assert row["coast_m_mean"] == pytest.approx(0.1)
  assert row["coast_m_max"] == pytest.approx(0.1)
  assert row["coast_turn_deg_mean"] == pytest.approx(0.5)


def test_summarise_drops_envs_that_reset():
  pos = _straight([0.5, 0.5, 0.1], 3, envs=2)
  yaw = np.zeros((4, 2))
  row = tr.summarise(pos, yaw, 2, 0.5, keep=np.array([True, False]))
  assert row["envs"] == 1
  assert row["ride_m_mean"] == pytest.approx(1.0)


def test_summarise_refuses_when_every_env_reset():
  pos = _straight([0.5, 0.5, 0.1], 3, envs=1)
  with pytest.raises(ValueError, match="every env reset"):
    tr.summarise(pos, np.zeros((4, 1)), 2, 0.5, keep=np.array([False]))
```

- [ ] **Step 2: Run them and see them fail**

Run: `FORCE_CPU=1 uv run pytest tests/test_skate_timed_ride.py -q`
Expected: collection error, `FileNotFoundError` for `timed_ride.py`.

- [ ] **Step 3: Write the tool**

`skateboard/tools/timed_ride.py`:

```python
"""Measure what one timed voice command does in sim: travel, turn and coast.

Gemini Pupper has no distance sensing, so a voice command is a speed and a
yaw (carve) command held for a few seconds, then zero. This tool rolls that
out and prints one JSON line, so the voice tools can quote real numbers.

Usage (repo root; drop FORCE_CPU=1 on a machine with an NVIDIA GPU):
  FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py \\
    skateboard/policies/split_steer_v6_max0.3_policy.json \\
    [--yaw-rate 0.3] [--ride-s 5] [--coast-s 4] [--envs 16] [--seed 0]

The ride phase commands `--speed` and `--yaw-rate` in every env. The coast
phase commands zero (every env is made a standing env), as the robot's
controller sees after the voice command ends. Distances are along the
board's path. An env that resets at any point is dropped and counted in
`resets`. Each env draws its own board and gains from the task's ranges, so
the mean and the max differ.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from policy_loader import load_policy

STEER = "Mjlab-SkateSplit-Steer-Pupper-v3"


def path_length(pos: np.ndarray) -> np.ndarray:
  """Distance along the path so far. `pos` is (T+1, E, 2); returns (T+1, E)."""
  steps = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
  return np.concatenate([np.zeros((1, pos.shape[1])), np.cumsum(steps, axis=0)])


def turn_so_far(yaw: np.ndarray) -> np.ndarray:
  """Heading change since step 0 in rad, unwrapped. `yaw` is (T+1, E)."""
  return np.unwrap(yaw, axis=0) - yaw[0]


def summarise(
  pos: np.ndarray, yaw: np.ndarray, ride_steps: int, step_dt: float, keep: np.ndarray
) -> dict:
  """Travel, turn and coast over the envs in `keep` (a bool mask over E).

  The ride is steps 0 to `ride_steps`; the coast is the rest.
  """
  if not keep.any():
    raise ValueError("every env reset; nothing to measure")
  dist = path_length(pos)[:, keep]
  turn = np.degrees(turn_so_far(yaw)[:, keep])
  ride_s = ride_steps * step_dt
  ride = dist[ride_steps]
  coast = dist[-1] - ride
  return {
    "envs": int(keep.sum()),
    "ride_m_mean": float(ride.mean()),
    "ride_m_max": float(ride.max()),
    "m_per_s": float(ride.mean() / ride_s),
    "turn_deg_mean": float(turn[ride_steps].mean()),
    "turn_deg_per_s": float(turn[ride_steps].mean() / ride_s),
    "coast_m_mean": float(coast.mean()),
    "coast_m_max": float(coast.max()),
    "coast_turn_deg_mean": float((turn[-1] - turn[ride_steps]).mean()),
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
  ap.add_argument("policy", help="model_*.pt checkpoint or exported policy.json")
  ap.add_argument("--task", default=STEER)
  ap.add_argument("--speed", type=float, default=0.3, help="board speed, m/s")
  ap.add_argument("--yaw-rate", type=float, default=0.0, help="carve command, rad/s")
  ap.add_argument("--ride-s", type=float, default=5.0)
  ap.add_argument("--coast-s", type=float, default=4.0)
  ap.add_argument("--envs", type=int, default=16)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  raw, env, policy = load_policy(
    args.task,
    Path(args.policy),
    args.speed,
    num_envs=args.envs,
    seed=args.seed,
    yaw_rate=args.yaw_rate,
  )
  term = raw.command_manager.get_term("twist")
  board = raw.scene["board"]
  ride_steps = round(args.ride_s / raw.step_dt)
  coast_steps = round(args.coast_s / raw.step_dt)

  def snapshot() -> tuple[np.ndarray, np.ndarray]:
    return (
      board.data.root_link_pos_w[:, :2].cpu().numpy().copy(),
      board.data.heading_w.cpu().numpy().copy(),
    )

  obs = env.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  pos, yaw = [], []
  p, y = snapshot()
  pos.append(p)
  yaw.append(y)
  keep = np.ones(args.envs, dtype=bool)
  with torch.no_grad():
    for k in range(ride_steps + coast_steps):
      if k >= ride_steps:
        # Zero command from here on: a standing env is zeroed every step, and
        # no resample may hand the speed back.
        term.is_standing_env[:] = True
        term.time_left[:] = 1e9
      obs, _, done, _ = env.step(policy(obs))
      keep &= ~done.cpu().numpy().astype(bool)
      if k >= ride_steps:
        command = term.vel_command_b.abs().amax(dim=1).cpu().numpy()
        assert (command[keep] == 0.0).all(), "the coast still has a command"
      p, y = snapshot()
      pos.append(p)
      yaw.append(y)

  row = summarise(np.stack(pos), np.stack(yaw), ride_steps, raw.step_dt, keep)
  out = {"stick": args.yaw_rate, "resets": int((~keep).sum())}
  out.update({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
  print(json.dumps(out))


if __name__ == "__main__":
  main()
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `FORCE_CPU=1 uv run pytest tests/test_skate_timed_ride.py -q`
Expected: `5 passed`.

- [ ] **Step 5: One sim run, straight**

Run:
`FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py skateboard/policies/split_steer_v6_max0.3_policy.json`

Expected: one JSON line with `"stick": 0.0`, `"resets": 0`, `m_per_s`
between 0.15 and 0.45, and `coast_m_max` under 1.5. If the assert "the coast
still has a command" fires, or `term.time_left` / `term.is_standing_env`
does not exist, read `src/mjlab/tasks/velocity/mdp/velocity_command.py` and
`src/mjlab/managers/command_manager.py` for the current attribute names, fix
the two lines in the coast branch, and say so in the report. Put the JSON
line in the report.

- [ ] **Step 6: Format and commit**

```bash
uv run ruff format skateboard/tools/timed_ride.py tests/test_skate_timed_ride.py
uv run ruff check --fix skateboard/tools/timed_ride.py tests/test_skate_timed_ride.py
git add skateboard/tools/timed_ride.py tests/test_skate_timed_ride.py
git commit -m "feat: add timed_ride sim tool for voice command estimates"
```

---

### Task 2: skate config and policy file (GP)

**Files:**
- Create: `robot/skateboard.py`, `skate/skate.yaml`, `skate/policy.json`
- Modify: `robot/test_tools.py` (imports, new group `skate_config`, `GROUPS`,
  `PARTS`)

**Interfaces:**
- Produces, all in `robot/skateboard.py`:
  - `SKATE_DIR: Path`, `SKATE_CONTROLLER = "gemini_skate"`,
    `SKATING = "skating"`, `MIN_RIDE_S = 1.0`, `MAX_RIDE_S = 10.0`,
    `SHARPNESS = ("gentle", "medium", "hard")`, `ON_BOARD_TOOLS: set[str]`,
    `ON_BOARD_MESSAGE: str`, `NOT_INSTALLED: str`
  - `SkateConfig` with fields `policy_path: Path`, `speed: float`,
    `carve: dict[str, float]` (rad/s), `travel_m_per_s: float`,
    `turn_deg_per_s: dict[str, float]`, `coast_m: float`,
    `default_seconds: float`, `max_seconds: float`,
    `queue_max_seconds: float`, and `describe() -> str`
  - `load_skate(skate_dir: Path = SKATE_DIR) -> SkateConfig | None`
  - `skate_controller_params(cfg: SkateConfig, base: dict) -> dict`

- [ ] **Step 1: Branch, and check who commits**

```bash
cd ~/claude/cs123/pupper/gemini_pupper
git checkout -b skate-voice
git config user.name; git config user.email
```

Expected: Ethan's name and email. If not, stop and report; do not change
git config.

- [ ] **Step 2: Write the failing tests**

In `robot/test_tools.py`, add `import yaml` to the imports at the top (after
`import types`), and after the line
`from tricks import Trick, WalkConfig  # noqa: E402` add:

```python
import skateboard  # noqa: E402
from skateboard import SKATE_CONTROLLER, SKATING, SkateConfig, load_skate, skate_controller_params  # noqa: E402
```

Before the `# ----- main` banner near the end of the file, add:

```python
# ----------------------------------------------------------------------------- the skateboard
_POLICY_CLIP = {"command_clip": {"vx": [0.0, 0.3], "vy": [0.0, 0.0], "wz": [-0.6, 0.6]},
                "heading_hold": {"kp": 1.0, "clip": 0.1, "yaw_threshold": 0.1, "walk_threshold": 0.05}}


def _write_skate(**changes) -> Path:
    """A skate folder in a fresh temp dir: skate.yaml (with changes) and a policy stub."""
    spec = {"policy": "policy.json", "speed": 0.3, "carve": {"gentle": 0.15, "medium": 0.3, "hard": 0.5},
            "travel_m_per_s": 0.3, "turn_deg_per_s": {"gentle": 0.7, "medium": 1.0, "hard": 1.9},
            "coast_m": 0.3, "default_seconds": 3, "max_seconds": 5, "queue_max_seconds": 10}
    spec.update(changes)
    d = Path(tempfile.mkdtemp(prefix="gemini_pupper_skate_"))
    (d / "skate.yaml").write_text(yaml.safe_dump(spec))
    (d / "policy.json").write_text(json.dumps(_POLICY_CLIP))
    return d


def _skate_error(**changes) -> str:
    try:
        load_skate(_write_skate(**changes))
    except ValueError as e:
        return str(e)
    return ""


async def skate_config() -> None:
    print("Skate: skate.yaml and the policy next to it")
    check("a folder with no skate.yaml means no skate policy is installed",
          load_skate(Path(tempfile.mkdtemp(prefix="gemini_pupper_noskate_"))) is None)
    cfg = load_skate(_write_skate())
    check("a good skate.yaml loads", cfg is not None and cfg.speed == 0.3 and cfg.carve["hard"] == 0.5
          and cfg.max_seconds == 5.0 and cfg.policy_path.name == "policy.json", f"got {cfg}")
    for what, changes, word in [
        ("a speed above the policy's clip", {"speed": 0.4}, "0.3"),
        ("a carve above the policy's clip", {"carve": {"gentle": 0.15, "medium": 0.3, "hard": 0.7}}, "0.6"),
        ("a carve at the heading hold's threshold", {"carve": {"gentle": 0.1, "medium": 0.3, "hard": 0.5}}, "0.14"),
        ("carves that do not get sharper", {"carve": {"gentle": 0.3, "medium": 0.2, "hard": 0.5}}, "sharper"),
        ("a missing sharpness", {"carve": {"gentle": 0.15, "hard": 0.5}}, "medium"),
        ("a command longer than the queue", {"max_seconds": 12}, "queue_max_seconds"),
        ("a queue longer than 10 s", {"queue_max_seconds": 20}, "10"),
        ("a default longer than the longest command", {"default_seconds": 6}, "max_seconds"),
    ]:
        err = _skate_error(**changes)
        check(f"{what} is refused, with the number", bool(err) and word in err, f"error: {err!r}")
    d = _write_skate()
    (d / "policy.json").unlink()
    try:
        load_skate(d)
        err = ""
    except ValueError as e:
        err = str(e)
    check("a missing policy file is refused, and says where it goes", "policy.json" in err, f"error: {err!r}")
    text = cfg.describe()
    check("describe() gives Gemini the numbers", all(w in text for w in ("0.3 m/s", "5 s", "10 s", "1.9")), text)
    check("... and says they are estimates and that Pupper cannot measure distance",
          "estimate" in text and "cannot measure distance" in text, text)
    base = {"model_path": "walk.json", "chord_partner": "x", "game_buttons": True, "max_body_angle": 1.0, "kp_scale": [1.0]}
    params = skate_controller_params(cfg, base)
    check("the skate controller runs the skate policy, with no joystick game features",
          params["model_path"] == str(cfg.policy_path) and params["game_buttons"] is False
          and params["reference_tricks"] is False and "chord_partner" not in params, f"got {params}")
    check("... and keeps the course controller's tilt limit and trims",
          params["max_body_angle"] == 1.0 and params["kp_scale"] == [1.0] and base["game_buttons"] is True)
    real = load_skate()
    check("the repo's own skate/skate.yaml loads", real is not None and real.speed <= 0.3
          and real.carve["hard"] <= 0.5 and real.max_seconds <= 5 and real.queue_max_seconds <= 10, f"got {real}")
    check("the on-board tools are the skate tools, stop and robot_status",
          skateboard.ON_BOARD_TOOLS == {"start_skating", "skate_forward", "skate_turn", "end_skating", "stop", "robot_status"})
```

In `GROUPS`, add `"skate_config": skate_config,` and in `PARTS` add
`"skate": ["skate_config"],`.

- [ ] **Step 3: Run and see it fail**

Run: `uv run --no-project --with pyyaml python lab/test.py skate_config`
Expected: `ModuleNotFoundError: No module named 'skateboard'`.

- [ ] **Step 4: Write `robot/skateboard.py`**

```python
"""The skateboard: skate/skate.yaml and the skate policy next to it.

Pupper rides a mini cruiser in a split stance: left feet on the deck, right
feet pushing on the floor. One policy (pupper-mjlab's
Mjlab-SkateSplit-Steer-Pupper-v3) rides and steers. Its speed command is the
board's speed. Its yaw command is a carve: 0 rides straight (the controller's
heading hold keeps the line), and a value above the hold's threshold asks
Pupper to lean so the board turns, at about a tenth of the command.

Pupper cannot mount the board, measure distance or see obstacles. So skate
mode is started by a person who has placed Pupper, every command is timed and
short, and the numbers Gemini reads are estimates from simulation.

skate.yaml fields:
  policy              the policy JSON, in this folder
  speed               m/s the board is asked to roll at (one speed only)
  carve               gentle / medium / hard -> yaw command in rad/s
  travel_m_per_s      sim estimate: metres per second of riding
  turn_deg_per_s      gentle / medium / hard -> sim estimate of the board's turn
  coast_m             sim estimate: how far the board rolls after the last command
  default_seconds     a command's length when Gemini gives none
  max_seconds         the longest single command
  queue_max_seconds   the most riding that can be queued (at most MAX_RIDE_S)

No skate.yaml means no skate mode: load_skate returns None and the skate
tools say nothing is installed.

This module has no ROS imports, so the launch file can use it too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

SKATE_DIR = Path(__file__).resolve().parent.parent / "skate"
SKATE_CONTROLLER = "gemini_skate"  # the ros2_control controller that runs the skate policy
SKATING = "skating"  # the state: the skate policy holds the legs, Pupper is on the board
MIN_RIDE_S = 1.0  # shorter than this barely gets a push in
MAX_RIDE_S = 10.0  # hardware runs are kept to about 10 s; skate.yaml can only go lower
CARVE_MARGIN = 0.04  # rad/s a carve must clear the heading hold's threshold by
SHARPNESS = ("gentle", "medium", "hard")

# The only tools Gemini may call while Pupper is on the board (tool_registry.call).
ON_BOARD_TOOLS = {"start_skating", "skate_forward", "skate_turn", "end_skating", "stop", "robot_status"}
ON_BOARD_MESSAGE = (
    "Pupper is on the skateboard. There it can only ride forward (skate_forward), carve left or right"
    " (skate_turn) and stop. It cannot walk, go backward, step sideways, turn on the spot, follow or do"
    " tricks. To get off: a person holds the harness, then call end_skating"
)
NOT_INSTALLED = "no skate policy is installed on this robot (skate/skate.yaml is missing), so Pupper can't skate"


@dataclass
class SkateConfig:
    """How Pupper skates: what it is commanded, and what Gemini is told to expect."""

    policy_path: Path
    speed: float  # m/s
    carve: dict[str, float]  # sharpness -> yaw command, rad/s
    travel_m_per_s: float  # sim estimate
    turn_deg_per_s: dict[str, float]  # sharpness -> sim estimate of the board's turn
    coast_m: float  # sim estimate
    default_seconds: float
    max_seconds: float
    queue_max_seconds: float

    def describe(self) -> str:
        """What Gemini reads about skating (the {skate} in a tool description)."""
        g, m, h = (self.turn_deg_per_s[s] for s in SHARPNESS)
        return (f"Pupper rides at {self.speed:g} m/s, about {self.travel_m_per_s:g} m per second of riding, and the"
                f" board rolls on about {self.coast_m:g} m after it stops pushing (no brake). One command lasts"
                f" {MIN_RIDE_S:g} to {self.max_seconds:g} s (default {self.default_seconds:g} s), and at most"
                f" {self.queue_max_seconds:g} s of riding can be queued. Carving is slow: the board turns about"
                f" {g:g}, {m:g} or {h:g} degrees per second for gentle, medium or hard. These are estimates from"
                f" simulation: Pupper cannot measure distance or see obstacles.")


def load_skate(skate_dir: Path = SKATE_DIR) -> SkateConfig | None:
    """The skate config, or None when skate.yaml is not there. Raises ValueError
    for numbers the policy was not trained on or that are over the limits."""
    spec_path = skate_dir / "skate.yaml"
    if not spec_path.exists():
        return None
    spec = yaml.safe_load(spec_path.read_text())
    policy_path = skate_dir / spec["policy"]
    if not policy_path.exists():
        raise ValueError(f"{spec_path}: the policy {policy_path.name} is not in {skate_dir}; copy it there"
                         " (see skate/README.md)")
    policy = json.loads(policy_path.read_text())
    clip = policy["command_clip"]
    hold = policy.get("heading_hold") or {}
    speed = float(spec["speed"])
    top_speed = max(clip["vx"])
    if not 0.0 < speed <= top_speed + 1e-6:
        raise ValueError(f"{spec_path}: speed {speed:g} must be above 0 and at most {top_speed:g} m/s, the policy's clip")
    missing = [s for s in SHARPNESS if s not in (spec.get("carve") or {})]
    if missing:
        raise ValueError(f"{spec_path}: carve needs {list(SHARPNESS)}; missing {missing}")
    carve = {s: float(spec["carve"][s]) for s in SHARPNESS}
    lowest = float(hold.get("yaw_threshold", 0.0)) + CARVE_MARGIN
    top_carve = min(abs(clip["wz"][0]), abs(clip["wz"][1]))
    for s, wz in carve.items():
        if not lowest - 1e-6 <= wz <= top_carve + 1e-6:
            raise ValueError(f"{spec_path}: carve {s} {wz:g} must be from {lowest:g} (clear of the heading hold)"
                             f" to {top_carve:g} rad/s (the policy's clip)")
    if not carve["gentle"] < carve["medium"] < carve["hard"]:
        raise ValueError(f"{spec_path}: carve must get sharper from gentle to medium to hard, not {carve}")
    turn = {s: float(spec["turn_deg_per_s"][s]) for s in SHARPNESS}
    default_s, max_s, queue_s = (float(spec[k]) for k in ("default_seconds", "max_seconds", "queue_max_seconds"))
    if queue_s > MAX_RIDE_S + 1e-6:
        raise ValueError(f"{spec_path}: queue_max_seconds {queue_s:g} is more than {MAX_RIDE_S:g} s")
    if not MIN_RIDE_S <= max_s <= queue_s:
        raise ValueError(f"{spec_path}: max_seconds {max_s:g} must be from {MIN_RIDE_S:g} s to queue_max_seconds {queue_s:g}")
    if not MIN_RIDE_S <= default_s <= max_s:
        raise ValueError(f"{spec_path}: default_seconds {default_s:g} must be from {MIN_RIDE_S:g} s to max_seconds {max_s:g}")
    return SkateConfig(policy_path=policy_path, speed=speed, carve=carve,
                       travel_m_per_s=float(spec["travel_m_per_s"]), turn_deg_per_s=turn,
                       coast_m=float(spec["coast_m"]), default_seconds=default_s, max_seconds=max_s,
                       queue_max_seconds=queue_s)


def skate_controller_params(cfg: SkateConfig, base: dict) -> dict:
    """Parameters for the skate controller: the course default controller's, with the
    skate policy (its JSON carries its stance, gains, command clip and heading hold)
    and no joystick game features. The tilt limit and trims stay the course's."""
    params = dict(base)
    params.pop("chord_partner", None)
    params.update(model_path=str(cfg.policy_path), game_buttons=False, reference_tricks=False)
    return params
```

- [ ] **Step 5: Add the config and the policy file**

`skate/skate.yaml`:

```yaml
# Skate mode for Gemini Pupper. Read by robot/skateboard.py; see its docstring.
# The policy is pupper-mjlab's skateboard/policies/split_steer_v6_max0.3_policy.json.
policy: policy.json
speed: 0.3            # m/s: the only speed checked in sim; the policy clips here
carve:                # yaw command, rad/s (0 rides straight)
  gentle: 0.15
  medium: 0.3
  hard: 0.5           # not 0.6: one sim env lost the board there
# Estimates from simulation (pupper-mjlab skateboard/tools/timed_ride.py).
travel_m_per_s: 0.3
turn_deg_per_s:
  gentle: 0.7
  medium: 1.0
  hard: 1.9
coast_m: 0.3
default_seconds: 3
max_seconds: 5
queue_max_seconds: 10
```

Copy the policy:

```bash
cp ~/claude/cs123/pupper-mjlab/skateboard/policies/split_steer_v6_max0.3_policy.json skate/policy.json
```

- [ ] **Step 6: Run and see it pass, then the whole suite**

Run: `uv run --no-project --with pyyaml python lab/test.py skate_config`
Expected: every line `ok`, ending "All N checks passed." with no FAIL line.

Run: `uv run --no-project --with pyyaml python lab/test.py`
Expected: "All N checks passed." with N above 166 and no FAIL line.

- [ ] **Step 7: Commit**

```bash
git add robot/skateboard.py robot/test_tools.py skate/skate.yaml skate/policy.json
git commit -m "feat: add skate config and the steer v6 policy"
```

---

### Task 3: skate mode in the robot API and the launch file (GP)

**Files:**
- Modify: `robot/robot_api.py`, `robot/launch/gemini_pupper.launch.py`,
  `robot/test_tools.py` (`Rig`, a fixture, group `skate_robot`)

**Interfaces:**
- Consumes: Task 2's `SKATE_CONTROLLER`, `SKATING`, `ON_BOARD_MESSAGE`,
  `SkateConfig`, `load_skate`, `skate_controller_params`.
- Produces:
  - `Robot.skate: SkateConfig | None`
  - `robot.state_now` / `robot.state_after_queue` can be `SKATING`
  - `Robot.switch_to(c)` returns `Result(False, ...)` containing
    "skateboard" when `state_now == SKATING` and `c` is neither `None` nor
    `SKATE_CONTROLLER`
  - `Robot.set_following(True)` refuses while skating
  - `Robot.mode()` message contains "skating" in skate mode
  - In tests: `Rig(tricks=None, skate=None)` and `fixture_skate()`

- [ ] **Step 1: Write the failing tests**

In `robot/test_tools.py`, change `Rig.__init__`. Replace

```python
    def __init__(self, tricks=None):
```

with

```python
    def __init__(self, tricks=None, skate=None):
```

and after the line `robot_api.load_tricks = made` add

```python
        robot_api.load_skate = lambda: skate
```

and after the statement that builds `names` (it ends with
`| {t.walk_controller for t in made().values() if t.walks})`) add

```python
        if skate is not None:
            names.add(SKATE_CONTROLLER)
```

After `skate_config` add:

```python
def fixture_skate() -> SkateConfig:
    return SkateConfig(policy_path=Path("skate.json"), speed=0.3, carve={"gentle": 0.15, "medium": 0.3, "hard": 0.5},
                       travel_m_per_s=0.3, turn_deg_per_s={"gentle": 0.7, "medium": 1.0, "hard": 1.9}, coast_m=0.3,
                       default_seconds=3.0, max_seconds=5.0, queue_max_seconds=10.0)


async def skate_robot() -> None:
    print("Skate: the robot's skate mode")
    rig = Rig()
    try:
        check("with no skate policy installed, the robot has no skate controller",
              rig.ts.skate is None and SKATE_CONTROLLER not in rig.ts.leg_controllers)
    finally:
        rig.close()
    rig = Rig(skate=fixture_skate())
    try:
        check("with one installed, the skate controller is a leg controller",
              rig.ts.skate is not None and SKATE_CONTROLLER in rig.ts.leg_controllers)
        await rig.call("lie_down")
        res = await rig.ts.switch_to(SKATE_CONTROLLER)
        check("switching to the skate controller works from motors idle", res.ok, f"got {res}")
        check("... it alone holds the legs", rig.robot.active() & rig.ts.leg_controllers == {SKATE_CONTROLLER},
              f"active: {rig.robot.active()}")
        check("... and the state is skating", rig.ts.state_now == SKATING, f"state: {rig.ts.state_now}")
        res = await rig.ts.mode()
        check("robot_status says it is skating", "skating" in res.message, res.message)
        res = await rig.ts.switch_to(LOCOMOTION_CONTROLLER)
        check("on the board, the walking policy is refused", not res.ok and "skateboard" in res.message, f"got {res}")
        check("... and nothing switched", rig.robot.active() & rig.ts.leg_controllers == {SKATE_CONTROLLER}
              and rig.ts.state_now == SKATING, f"active: {rig.robot.active()}")
        bow = rig.ts.tricks["bow"]
        res = await rig.ts.switch_to(bow.controller)
        check("... a trick is refused too", not res.ok and "skateboard" in res.message, f"got {res}")
        before = rig.ts.state_after_queue
        res = await rig.ts.set_following(True)
        check("on the board, follow mode is refused", not res.ok and "skateboard" in res.message, f"got {res}")
        check("... the follower stays off and the state is unchanged",
              not rig.robot.follower_on and not rig.ts.following and rig.ts.state_after_queue == before)
        res = await rig.ts.switch_to(None)
        check("relaxing the motors leaves skate mode", res.ok and rig.ts.state_now == robot_api.MOTORS_IDLE
              and not rig.robot.active() & rig.ts.leg_controllers, f"got {res}; state {rig.ts.state_now}")
        res = await rig.ts.switch_to(LOCOMOTION_CONTROLLER)
        check("after that, standing up to walk works again", res.ok and rig.ts.state_now == LOCOMOTION, f"got {res}")
    finally:
        rig.close()
```

In `GROUPS` add `"skate_robot": skate_robot,` and change the `PARTS` entry
to `"skate": ["skate_config", "skate_robot"],`.

- [ ] **Step 2: Run and see it fail**

Run: `uv run --no-project --with pyyaml python lab/test.py skate_robot`
Expected: `AttributeError: 'Robot' object has no attribute 'skate'`.

- [ ] **Step 3: Change `robot/robot_api.py`**

(a) After the line that starts `from tricks import TRICK_FADE_S,` add:

```python
from skateboard import ON_BOARD_MESSAGE, SKATE_CONTROLLER, SKATING, load_skate
```

(b) In the module docstring's list of states near the top of the file, if
it names the states (`LOCOMOTION`, `MOTORS_IDLE`, `FALLEN`), add one line in
the same style: `SKATING (skateboard.py): the skate policy holds the legs,
Pupper is on the board`. After the `FALLEN = "fallen"` line add the comment:

```python
# SKATING (skateboard.py) is a state too: the skate policy holds the legs and Pupper
# is on the skateboard. Only relaxing the motors leaves it (see switch_to).
```

(c) In `Robot.__init__`, replace

```python
        self.tricks: dict[str, Trick] = load_tricks()
        self.leg_controllers = (LEG_CONTROLLERS | {t.controller for t in self.tricks.values() if t.kind == "policy"}
                                | {t.walk_controller for t in self.tricks.values() if t.walks})
```

with

```python
        self.tricks: dict[str, Trick] = load_tricks()
        self.skate = load_skate()  # None: no skate policy installed
        self.leg_controllers = (LEG_CONTROLLERS | {t.controller for t in self.tricks.values() if t.kind == "policy"}
                                | {t.walk_controller for t in self.tricks.values() if t.walks}
                                | ({SKATE_CONTROLLER} if self.skate else set()))
```

(d) In `switch_to`, replace

```python
        activate = [controller] if controller else []
        deactivate = sorted(self.leg_controllers - set(activate))
```

with

```python
        if self.state_now == SKATING and controller not in (None, SKATE_CONTROLLER):
            # Half on a board, any other policy would start with two feet on a rolling
            # deck. The only way out is relaxing the motors, with a person holding Pupper.
            return Result(False, f"can't switch to {controller}: {ON_BOARD_MESSAGE}")
        activate = [controller] if controller else []
        deactivate = sorted(self.leg_controllers - set(activate))
```

Add one sentence to the end of `switch_to`'s docstring: `In skate mode it
refuses every controller but the skate one; relaxing the motors is allowed.`

(e) In `state_of`, replace

```python
        if controller is None:
            return MOTORS_IDLE
```

with

```python
        if controller is None:
            return MOTORS_IDLE
        if controller == SKATE_CONTROLLER:
            return SKATING
```

(f) In `set_following`, replace

```python
        if self.following:
            return Result(True, "already following the nearest person")
        await self.clear_queue()
```

with

```python
        if self.following:
            return Result(True, "already following the nearest person")
        if SKATING in (self.state_now, self.state_after_queue):
            return Result(False, f"can't follow: {ON_BOARD_MESSAGE}")
        await self.clear_queue()
```

(g) In `mode`, replace

```python
        mode = (
            "locomotion, following a person" if self.following
```

with

```python
        mode = (
            "skating: on the skateboard in skate mode (skate_forward and skate_turn ride it; end_skating"
            " relaxes the motors to get off)" if states.get(SKATE_CONTROLLER) == "active"
            else "locomotion, following a person" if self.following
```

- [ ] **Step 4: Load the controller in the launch file**

In `robot/launch/gemini_pupper.launch.py`, after the line
`from tricks import WALK_CONTROLLER, controller_params, ...  # noqa: E402`
add:

```python
from skateboard import SKATE_CONTROLLER, load_skate, skate_controller_params  # noqa: E402
```

In `gemini_controllers()`, after the `for t in tricks:` loop and before the
comment `# The course config writes paths as $(find-pkg-share pkg)/...`, add:

```python
    skate = load_skate()  # skate/skate.yaml: the skateboard policy, if installed
    if skate is not None:
        params[SKATE_CONTROLLER] = {"ros__parameters": {**kind, **skate_controller_params(skate, base)}}
```

Add to the docstring list at the top of the file, after the `gemini_walk`
bullet: `  - gemini_skate, the skateboard policy (skate/skate.yaml), when one
is installed`.

- [ ] **Step 5: Run and see it pass**

Run: `uv run --no-project --with pyyaml python lab/test.py skate_robot`
Expected: every line `ok`, "All N checks passed." with no FAIL line.

Run: `python3 -m py_compile robot/launch/gemini_pupper.launch.py && echo compiled`
Expected: `compiled`. (The launch file cannot run on the Mac; it is checked
on the robot in Task 5's checklist.)

Run: `uv run --no-project --with pyyaml python lab/test.py`
Expected: "All N checks passed." with no FAIL line. Put N in the report.

- [ ] **Step 6: Commit**

```bash
git add robot/robot_api.py robot/launch/gemini_pupper.launch.py robot/test_tools.py
git commit -m "feat: add skate mode to the robot API and launch"
```

---

### Task 4: the skate tools (GP)

**Files:**
- Create: `lab/part8_skate.py`
- Modify: `robot/tool_registry.py`, `robot/test_tools.py` (group
  `skate_tools`)

**Interfaces:**
- Consumes: `robot.skate`, `SKATING`, `SKATE_CONTROLLER`, `ON_BOARD_TOOLS`,
  `ON_BOARD_MESSAGE`, `NOT_INSTALLED`, `MIN_RIDE_S`, `SHARPNESS`;
  `robot.queue`, `robot.switch_to`, `robot.send_speed(vx, vy, wz_deg)`,
  `robot.sleep`, `robot.now`, `robot.busy_until`, `robot.fallen`;
  `prepare_to_move(robot) -> str` from `lab/part3_follow.py`.
- Produces: behaviors `start_skating(robot, placed)`, `ride(robot, seconds,
  direction="straight", sharpness="medium")`, `end_skating(robot, held)`,
  and tools `start_skating`, `skate_forward`, `skate_turn`, `end_skating`.

- [ ] **Step 1: Write the failing tests**

In `robot/test_tools.py`, after `skate_robot` add:

```python
async def _on_board(rig: Rig) -> None:
    """Relax, then start skate mode the way a person and Gemini would."""
    await rig.call("lie_down")
    res = await rig.call("start_skating", placed=True)
    assert res.ok and rig.ts.state_now == SKATING, f"could not start skating: {res}; state {rig.ts.state_now}"


def _ride_twists(rig: Rig, start: int = 0):
    return [(t, m) for t, m in rig.robot.twists()[start:] if abs(m.linear.x) + abs(m.angular.z) > 1e-9]


async def skate_tools() -> None:
    print("Skate: the tools Gemini calls")
    names = ["start_skating", "skate_forward", "skate_turn", "end_skating"]
    rig = Rig()
    try:
        described(rig, names, "skate")
        res = await rig.call("start_skating", placed=True)
        check("with no skate policy installed, start_skating says so", not res.ok and "installed" in res.message, f"got {res}")
        res = await rig.call("skate_forward")
        check("... and so does skate_forward", not res.ok and "installed" in res.message, f"got {res}")
    finally:
        rig.close()

    rig = Rig(skate=fixture_skate())
    try:
        decls = {d["name"]: d for d in tools.declarations(rig.ts)}
        check("skate_forward's description carries the numbers from skate.yaml",
              "0.3 m/s" in decls["skate_forward"]["description"] and "{skate}" not in decls["skate_forward"]["description"],
              decls["skate_forward"]["description"])
        res = await rig.call("skate_forward")
        check("riding before skate mode is refused, and says how to start",
              not res.ok and "start_skating" in res.message and not _ride_twists(rig), f"got {res}")
        res = await rig.call("start_skating", placed=True)
        check("start_skating while standing is refused, and says to lie down first",
              not res.ok and "lie_down" in res.message and not rig.robot.activations(SKATE_CONTROLLER), f"got {res}")
        await rig.call("lie_down")
        res = await rig.call("start_skating", placed=False)
        check("start_skating without a person placing Pupper is refused",
              not res.ok and "place" in res.message and not rig.robot.activations(SKATE_CONTROLLER), f"got {res}")
        res = await rig.call("start_skating", placed=True)
        check("start_skating after placing is accepted", res.ok, f"got {res}")
        check("... the skate policy takes the legs", rig.robot.activations(SKATE_CONTROLLER) and rig.ts.state_now == SKATING,
              f"state {rig.ts.state_now}")
        check("... with no failure in the queue", not rig.failures(), f"{rig.failures()}")
        started = rig.robot.activations(SKATE_CONTROLLER)[0]

        mark = len(rig.robot.twists())
        res = await rig.call("skate_forward", seconds=3)
        ride = _ride_twists(rig, mark)
        check("skate_forward is accepted", res.ok, f"got {res}")
        check("... it waits until the policy has faded in", bool(ride) and ride[0][0] >= started + WALK_READY_S - 1e-6,
              f"first command {ride[0][0] - started:.2f} s after the switch" if ride else "no commands")
        check("... sends 0.3 m/s straight ahead, nothing sideways, no turn",
              bool(ride) and all(abs(m.linear.x - 0.3) < 1e-9 and m.linear.y == 0 and m.angular.z == 0 for _, m in ride))
        check("... for 3 s at 20 Hz", 58 <= len(ride) <= 62, f"{len(ride)} commands")
        last = rig.robot.twists()[-1][1]
        check("... and ends with a zero command", last.linear.x == 0 and last.angular.z == 0)
        check("the reply gives the estimate (about 0.9 m) and says it is one",
              "0.9 m" in res.message and "stimate" in res.message and "obstacle" in res.message, res.message)
        check("... and Pupper is still in skate mode", rig.ts.state_now == SKATING and rig.ts.state_after_queue == SKATING)

        for direction, sharpness, wz in [("left", "hard", 0.5), ("right", "hard", -0.5), ("left", "gentle", 0.15)]:
            mark = len(rig.robot.twists())
            res = await rig.call("skate_turn", direction=direction, sharpness=sharpness, seconds=2)
            ride = _ride_twists(rig, mark)
            check(f"skate_turn {direction} {sharpness} carves at {wz:+g} rad/s while riding at 0.3 m/s",
                  res.ok and bool(ride) and all(abs(m.angular.z - wz) < 1e-3 and abs(m.linear.x - 0.3) < 1e-9 for _, m in ride),
                  f"got {res}; z = {sorted({round(m.angular.z, 3) for _, m in ride})}")
        mark = len(rig.robot.twists())
        res = await rig.call("skate_turn", direction="left")
        ride = _ride_twists(rig, mark)
        check("skate_turn defaults to a medium carve for 3 s",
              res.ok and 58 <= len(ride) <= 62 and all(abs(m.angular.z - 0.3) < 1e-3 for _, m in ride), f"got {res}")
        check("... and its reply estimates the turn in degrees", "degrees" in res.message, res.message)

        mark = len(rig.robot.twists())
        res = await rig.call("skate_forward", seconds=6)
        check("a command over 5 s is refused, with the limit", not res.ok and "5 s" in res.message and not _ride_twists(rig, mark),
              f"got {res}")
        res = await rig.call("skate_forward", seconds=0.5)
        check("a command under 1 s is refused, with the limit", not res.ok and "1 s" in res.message, f"got {res}")

        first = await tools.call(rig.ts, "skate_forward", {"seconds": 5})
        second = await tools.call(rig.ts, "skate_forward", {"seconds": 5})
        third = await tools.call(rig.ts, "skate_forward", {"seconds": 1})
        check("10 s of riding can be queued", first.ok and second.ok, f"got {first}; {second}")
        check("... and more than that is refused, with the limit", not third.ok and "10 s" in third.message, f"got {third}")
        await rig.settle()

        switches = len(rig.robot.switches)
        for name, args in [("stand_up", {}), ("lie_down", {}), ("move", {}), ("turn", {}), ("arc", {}),
                           ("follow", {"enabled": True}), ("play_trick", {"name": "bow"})]:
            res = await rig.call(name, **args)
            check(f"{name} is refused on the board, and says why", not res.ok and "skateboard" in res.message, f"got {res}")
        check("... and none of them touched the controllers or the follower",
              len(rig.robot.switches) == switches and not rig.robot.follower_on and rig.ts.state_now == SKATING)

        mark = len(rig.robot.twists())
        await tools.call(rig.ts, "skate_forward", {"seconds": 5})
        for _ in range(40):
            await _real_sleep(0)
        res = await rig.call("stop")
        ride = _ride_twists(rig, mark)
        last = rig.robot.twists()[-1][1]
        check("stop cuts a ride short", res.ok and 0 < len(ride) < 90, f"got {res}; {len(ride)} commands")
        check("... the last command is zero", last.linear.x == 0 and last.angular.z == 0)
        check("... and Pupper stays in skate mode, balancing", rig.ts.state_now == SKATING
              and SKATE_CONTROLLER in rig.robot.active())

        rig.ts.fallen = SKATE_CONTROLLER
        res = await rig.call("skate_forward")
        check("after a fall, riding is refused and says so", not res.ok and "fell" in res.message, f"got {res}")
        rig.ts.fallen = None

        res = await rig.call("end_skating", held=False)
        check("end_skating without a person holding Pupper is refused",
              not res.ok and "hold" in res.message and rig.ts.state_now == SKATING, f"got {res}")
        res = await rig.call("end_skating", held=True)
        check("end_skating with a person holding relaxes the motors",
              res.ok and rig.ts.state_now == robot_api.MOTORS_IDLE and not rig.robot.active() & rig.ts.leg_controllers,
              f"got {res}; state {rig.ts.state_now}")
        res = await rig.call("stand_up")
        check("after that, stand_up works again", res.ok and rig.ts.state_now == LOCOMOTION, f"got {res}")
    finally:
        rig.close()
```

In `GROUPS` add `"skate_tools": skate_tools,` and change the `PARTS` entry
to `"skate": ["skate_config", "skate_robot", "skate_tools"],`.

- [ ] **Step 2: Run and see it fail**

Run: `uv run --no-project --with pyyaml python lab/test.py skate_tools`
Expected: a FAIL line "lab/ declares start_skating, skate_forward,
skate_turn, end_skating: missing [...]", then more failures.

- [ ] **Step 3: Change `robot/tool_registry.py`**

(a) Replace the import line `from robot_api import CALL_STOPS, Result, Robot`
with:

```python
from robot_api import CALL_STOPS, Result, Robot
from skateboard import ON_BOARD_MESSAGE, ON_BOARD_TOOLS, SKATING
```

(b) Replace the `LAB_MODULES` line with:

```python
LAB_MODULES = ["basics", "part2_walk", "part3_follow", "part4_tricks", "part5_interactive_action", "part8_skate"]
```

(c) In `_describe`, before `return text`, add:

```python
    if "{skate}" in text:
        text = text.replace("{skate}", robot.skate.describe() if robot.skate else "(No skate policy is installed yet.)")
```

and add to the module docstring, after the sentence about `{tricks}` and
`{actions}`: `{skate} is replaced by what skate/skate.yaml says about
skating.`

(d) In `call`, replace

```python
    if t is None:
        return Result(False, f"unknown tool {name}; the tools are {sorted(TOOLS)}")
```

with

```python
    if t is None:
        return Result(False, f"unknown tool {name}; the tools are {sorted(TOOLS)}")
    if SKATING in (robot.state_now, robot.state_after_queue) and name not in ON_BOARD_TOOLS:
        # On the board (or about to be): refuse now, with the reason, instead of queueing
        # a job the robot would refuse later (Robot.switch_to enforces the same rule).
        return Result(False, f"{name} is not possible right now: {ON_BOARD_MESSAGE}")
```

- [ ] **Step 4: Write `lab/part8_skate.py`**

```python
"""Part 8 (final project): skateboard. Finished; read it like basics.py.

Pupper rides a mini cruiser in a split stance: left feet on the deck, right
feet pushing on the floor. One policy rides and steers. Gemini drives it with
the same speed commands the walking tools send (robot.send_speed), but the
rules are different, because Pupper cannot measure distance or see obstacles:

  - A person places Pupper on the board. start_skating only runs after
    Gemini passes placed=true, and only from motors idle.
  - Every command is timed and short (skate.yaml: max_seconds), and only so
    much riding can be queued (queue_max_seconds). To keep going, a person
    has to ask again.
  - One speed. A carve is a slow turn while riding: the yaw command asks
    Pupper to lean, and the board turns at about a tenth of it.
  - No backward, sideways, turning on the spot, following or tricks on the
    board: tool_registry.call refuses those tools, and Robot.switch_to
    refuses the controllers.
  - The board has no brake. Stopping means "stop pushing"; it rolls on.
  - Every reply calls its distance and turn an estimate.

The numbers Gemini reads come from skate/skate.yaml ({skate} in a description).
"""

from __future__ import annotations

from part3_follow import prepare_to_move
from robot_api import FALLEN, FELL_MESSAGE, LLM_CMD_RATE, MOTORS_IDLE, WALK_READY_S, Result, Robot, state_name, with_note
from skateboard import MIN_RIDE_S, NOT_INSTALLED, SHARPNESS, SKATE_CONTROLLER, SKATING, SkateConfig
from tool_registry import Param, tool

PLACEMENT = "left feet on the deck, right feet on the floor, body parallel to the board, a hand on the harness"
NOT_SKATING = ("Pupper is not in skate mode. A person places it on the board first (" + PLACEMENT
               + "); then call start_skating with placed=true")


# ============================================================ what the robot does
def estimate(cfg: SkateConfig, seconds: float, direction: str, sharpness: str) -> str:
    """What a ride should do, in words, for the tool's reply. An estimate, and it says so."""
    text = f" Estimate from simulation, not measured: about {cfg.travel_m_per_s * seconds:.1f} m forward"
    if direction != "straight":
        text += f", curving {direction} by about {cfg.turn_deg_per_s[sharpness] * seconds:.0f} degrees"
    return (text + f", then the board rolls on about {cfg.coast_m:g} m. Pupper cannot see obstacles:"
                   " a person must keep the path clear.")


async def start_skating(robot: Robot, placed: bool) -> Result:
    """Hand the legs to the skate policy, once a person has placed Pupper on the board."""
    cfg = robot.skate
    if cfg is None:  # (1) now
        return Result(False, NOT_INSTALLED)
    state = robot.state_after_queue
    if state == SKATING:
        return Result(True, "already in skate mode, on the board")
    if state not in (MOTORS_IDLE, FALLEN):
        return Result(False, f"can't start skating from {state_name(state)}: call lie_down first so the motors relax,"
                             f" have a person place Pupper on the board ({PLACEMENT}), then call start_skating")
    if not placed:
        return Result(False, f"not started: Pupper cannot get on the board by itself. Ask a person to place it"
                             f" ({PLACEMENT}), and call start_skating with placed=true once they say it is done")
    note = await prepare_to_move(robot)

    async def job() -> Result:  # (2) later
        res = await robot.switch_to(SKATE_CONTROLLER)
        if not res.ok:
            return res
        robot.send_speed(0, 0, 0)
        await robot.sleep(WALK_READY_S)  # the controller eases into the stance, then fades the policy in
        if robot.fallen:
            return Result(False, FELL_MESSAGE)
        return Result(True, "in skate mode: balancing on the board, not rolling yet")

    res = await robot.queue("start skating", job, WALK_READY_S, then=SKATING)  # (3)
    return with_note(note, res)


async def ride(robot: Robot, seconds: float, direction: str = "straight", sharpness: str = "medium") -> Result:
    """Ride forward for ``seconds``, straight or carving left or right, then stop pushing."""
    cfg = robot.skate
    if cfg is None:  # (1) now
        return Result(False, NOT_INSTALLED)
    if robot.fallen:
        return Result(False, FELL_MESSAGE)
    if robot.state_after_queue != SKATING:
        return Result(False, NOT_SKATING)
    if direction not in ("straight", "left", "right") or sharpness not in SHARPNESS:
        return Result(False, f"direction must be straight, left or right and sharpness one of {list(SHARPNESS)}")
    if not MIN_RIDE_S - 1e-6 <= seconds <= cfg.max_seconds + 1e-6:
        return Result(False, f"{seconds:g} s is outside what one skate command may last: {MIN_RIDE_S:g} s to"
                             f" {cfg.max_seconds:g} s. Pupper cannot see where it is going, so rides are short")
    pending = max(0.0, robot.busy_until - robot.now())
    if pending + seconds > cfg.queue_max_seconds + 1e-6:
        return Result(False, f"not queued: {pending:.1f} s is already queued, and with {seconds:g} s more it would pass"
                             f" the {cfg.queue_max_seconds:g} s limit. Wait until Pupper has finished, then ask again")
    wz = 0.0 if direction == "straight" else cfg.carve[sharpness] * (1.0 if direction == "left" else -1.0)
    what = "forward" if direction == "straight" else f"carving {direction} ({sharpness})"

    async def job() -> Result:  # (2) later
        if robot.fallen:
            return Result(False, FELL_MESSAGE)
        if robot.state_now != SKATING:
            return Result(False, f"did not ride: Pupper is in {state_name(robot.state_now)}, not in skate mode")
        end = robot.now() + seconds
        while robot.now() < end - 1e-6:
            if robot.fallen:
                robot.send_speed(0, 0, 0)
                return Result(False, FELL_MESSAGE)
            robot.send_speed(cfg.speed, 0.0, wz * 57.2958)  # send_speed takes deg/s
            await robot.sleep(1.0 / LLM_CMD_RATE)
        robot.send_speed(0, 0, 0)
        return Result(True, f"rode {what} for {seconds:g} s, then stopped pushing")

    res = await robot.queue(f"skate {what} {seconds:.3g}s", job, seconds, then=SKATING)  # (3)
    return Result(True, res.message + estimate(cfg, seconds, direction, sharpness)) if res.ok else res


async def end_skating(robot: Robot, held: bool) -> Result:
    """Leave skate mode: stop pushing and relax the motors. A person must be holding Pupper."""
    if robot.state_after_queue != SKATING:
        return Result(False, "Pupper is not in skate mode, so there is nothing to end")
    if not held:
        return Result(False, "not ended: leaving skate mode relaxes the motors, and Pupper would drop onto the"
                             " board. Ask a person to hold the harness, and call end_skating with held=true once they do")

    async def job() -> Result:
        robot.send_speed(0, 0, 0)
        return await robot.switch_to(None)

    res = await robot.queue("end skating", job, 1.0, then=MOTORS_IDLE)
    return Result(True, res.message + " The motors will relax: lift Pupper off the board. On the floor,"
                                      " stand_up gets it walking again.") if res.ok else res


# ============================================================ what Gemini reads
@tool(
    "start_skating",
    "Start skate mode. Pupper cannot climb onto the skateboard: a person places it first (" + PLACEMENT + ")."
    " Pupper must be relaxed (call lie_down first if it is standing). Ask the person to place it, wait until they"
    " say it is done, and only then call this with placed=true. It takes about 4 s; then Pupper balances on the"
    " board without rolling. {skate}",
    placed=Param("boolean", "true only if a person has just said Pupper is placed on the board. Never guess."),
)
async def start_skating_tool(robot: Robot, placed: bool) -> Result:
    return await start_skating(robot, placed)


@tool(
    "skate_forward",
    "Ride the skateboard straight ahead for a few seconds, then stop pushing. Only in skate mode. Use it for"
    ' "go forward", "ride", "skate". Pupper cannot go a set distance: you pick the seconds. {skate} Before you'
    " call it, the path must be clear; if you are not sure, ask.",
    seconds=Param("number", "How long to ride, in seconds. Leave it out for the default.", required=False),
)
async def skate_forward_tool(robot: Robot, seconds: float | None = None) -> Result:
    cfg = robot.skate
    return await ride(robot, cfg.default_seconds if cfg and seconds is None else (seconds or 0.0))


@tool(
    "skate_turn",
    "Carve left or right while riding forward for a few seconds, then stop pushing. Only in skate mode. Use it"
    ' for "turn left", "go right", "steer". It is a wide, slow curve, not a turn on the spot: say so, and give the'
    " estimate from the reply instead of promising a corner. {skate}",
    direction=Param("string", "Which way to carve.", enum=["left", "right"]),
    sharpness=Param("string", "How hard to lean. Default medium.", enum=list(SHARPNESS), required=False),
    seconds=Param("number", "How long to carve, in seconds. Leave it out for the default.", required=False),
)
async def skate_turn_tool(robot: Robot, direction: str, sharpness: str = "medium", seconds: float | None = None) -> Result:
    cfg = robot.skate
    return await ride(robot, cfg.default_seconds if cfg and seconds is None else (seconds or 0.0), direction, sharpness)


@tool(
    "end_skating",
    "Leave skate mode: Pupper stops pushing and its motors relax, so it drops unless someone holds it. Ask a"
    " person to hold the harness, wait until they say they have it, then call this with held=true. Afterwards the"
    " person lifts Pupper off the board, and stand_up gets it walking again. To only stop riding, call stop instead.",
    held=Param("boolean", "true only if a person has just said they are holding Pupper. Never guess."),
)
async def end_skating_tool(robot: Robot, held: bool) -> Result:
    return await end_skating(robot, held)
```

- [ ] **Step 5: Run and see it pass**

Run: `uv run --no-project --with pyyaml python lab/test.py skate_tools`
Expected: every line `ok`. If "stop cuts a ride short" reports 0 commands,
the fake clock needs more turns before the stop: raise the `range(40)` in
the test to `range(80)` (the check allows up to 90 commands) and say so in
the report.

Run: `uv run --no-project --with pyyaml python lab/test.py`
Expected: "All N checks passed" with no FAIL and no `--` line. Put N in the
report.

- [ ] **Step 6: Commit**

```bash
git add lab/part8_skate.py robot/tool_registry.py robot/test_tools.py
git commit -m "feat: add skate tools for Gemini"
```

---

### Task 5: prompt, docs, measured numbers, hardware checklist (GP and MJ)

**Files:**
- Modify (GP): `lab/prompt.md`, `robot/test_tools.py` (three checks at the
  end of `skate_tools`), `skate/skate.yaml`
- Create (GP): `skate/README.md`
- Modify (MJ): `skateboard/HARDWARE_TEST.md`, `skateboard/README.md`

**Interfaces:**
- Consumes: Task 1's CLI and its JSON keys `m_per_s`, `turn_deg_per_s`,
  `coast_m_max`, `resets`; Task 2's `skate.yaml` keys.

- [ ] **Step 1: Write the failing prompt checks**

At the end of `skate_tools` in GP `robot/test_tools.py`, after the last
`finally:` block, add:

```python
    text = prompt_sections("Skateboard").lower()
    check("prompt.md has a Skateboard section that says a person places Pupper", "place" in text, text[:200])
    check("... that stop comes first", "stop" in text)
    check("... and that distances are estimates and the path must be clear", "estimate" in text and "clear" in text)
```

Run: `uv run --no-project --with pyyaml python lab/test.py skate_tools`
Expected: the three new checks FAIL.

- [ ] **Step 2: Add the section to `lab/prompt.md`**

Between the `# Tricks` section and the `# Safety` section, add:

```markdown
# Skateboard

You can ride a skateboard, but you cannot get on it yourself. A person has to
place you: left feet on the deck, right feet on the floor. Ask them to, wait
until they say it is done, then call start_skating. On the board you can only
ride forward with skate_forward, carve left or right with skate_turn, and
stop. You cannot walk, go backward, turn on the spot, follow or do tricks
there, so say that instead of trying. You cannot measure distance and you
cannot see what is in front of the board. Before every ride, ask if the path
is clear unless the person just told you. Rides are a few seconds long. The
distance and the turn in a tool reply are estimates: say "about", and never
promise an exact distance or a sharp corner. A turn on the board is a wide,
slow curve. If anyone says stop, call stop first; the board rolls on a little
because it has no brake. To get off, ask a person to hold you, then call
end_skating.
```

Run: `uv run --no-project --with pyyaml python lab/test.py`
Expected: "All N checks passed" (N is Task 4's number plus 3).

- [ ] **Step 3: Measure the numbers in sim (MJ, one run per shell call)**

Run each of these from the MJ repo root, one at a time, and keep the JSON
line each prints:

```bash
FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py skateboard/policies/split_steer_v6_max0.3_policy.json --yaw-rate 0
FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py skateboard/policies/split_steer_v6_max0.3_policy.json --yaw-rate 0.15
FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py skateboard/policies/split_steer_v6_max0.3_policy.json --yaw-rate -0.15
FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py skateboard/policies/split_steer_v6_max0.3_policy.json --yaw-rate 0.3
FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py skateboard/policies/split_steer_v6_max0.3_policy.json --yaw-rate -0.3
FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py skateboard/policies/split_steer_v6_max0.3_policy.json --yaw-rate 0.5
FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py skateboard/policies/split_steer_v6_max0.3_policy.json --yaw-rate -0.5
```

Stop and report to Ethan, changing nothing, if any run has `resets` above 0,
or if a left run's `turn_deg_per_s` is not positive, or a right run's is not
negative. A carve level that loses the board or turns the wrong way in sim
is a decision for Ethan, not a number to tune.

- [ ] **Step 4: Put the measured numbers in GP `skate/skate.yaml`**

- `travel_m_per_s`: `m_per_s` of the `--yaw-rate 0` run, 2 decimals.
- `turn_deg_per_s.gentle`: the smaller of `|turn_deg_per_s|` from the +0.15
  and -0.15 runs, 1 decimal. `medium` from ±0.3 and `hard` from ±0.5 the
  same way. The smaller one, so Gemini never promises more turn than the
  weaker side gives.
- `coast_m`: the largest `coast_m_max` of the seven runs, rounded up to
  0.1 m.
- Change nothing else in the file.

Run: `uv run --no-project --with pyyaml python lab/test.py`
Expected: the same "All N checks passed".

- [ ] **Step 5: Write GP `skate/README.md`**

```markdown
# Skate mode

Voice control for Pupper on the skateboard (CS 123 Team 9 final project).
Tools: `lab/part8_skate.py`. Numbers: `skate.yaml`. Config loader:
`robot/skateboard.py`.

- `policy.json` is pupper-mjlab's
  `skateboard/policies/split_steer_v6_max0.3_policy.json`. To change policy,
  copy the new file over it and re-measure the estimates in `skate.yaml` with
  pupper-mjlab's `skateboard/tools/timed_ride.py`.
- What a person says: "lie down", place Pupper, "start skating", "go
  forward", "turn left", "stop", hold Pupper, "get off the board".
- Pupper cannot measure distance or see obstacles. Commands are timed
  (1 to 5 s, at most 10 s queued), the speed is fixed at 0.3 m/s, and
  distances and turns in replies are estimates from simulation.
- Desk test: `python3 lab/test.py skate` (no robot needed).
- A new policy needs a TA's OK before its first run on the robot. The
  hardware checklist is test 6 in pupper-mjlab's
  `skateboard/HARDWARE_TEST.md`.
```

- [ ] **Step 6: Add the checklist to MJ `skateboard/HARDWARE_TEST.md`**

Append at the end of the file (fill the three sim numbers from Step 4 where
the table says so):

```markdown
## 6. Voice control (Gemini Pupper)

Not run yet. Needs Ethan's OK on the day, a TA's sign-off for the policy,
and tests 3 and 5 (split ride, steer by joystick) passed first. The voice
tools assume the ride and the steer work.

Setup, by Ethan on the robot:

- [ ] `git pull` the `skate-voice` branch of gemini_pupper on the robot. The
      robot's copy may be ahead of the Mac snapshot: merge, do not overwrite.
- [ ] `python3 lab/test.py skate` on the robot: all checks pass.
- [ ] `robot/start.sh`, with Pupper lying on the floor, away from the board.
      In `/tmp/gemini-pupper/robot.log` the spawner lists `gemini_skate` as
      loaded.
- [ ] `ros2 param get /gemini_skate model_path` ends in `skate/policy.json`.
- [ ] Joystick connected. Right-stick press is the e-stop and outranks
      Gemini.

Run, hands on the harness the whole time, 4 m of clear floor ahead:

- [ ] Say "lie down". Place Pupper: left feet on the deck, right feet on the
      floor, body parallel to the board. Say "start skating". Pass: Gemini
      asks whether Pupper is placed before it starts; Pupper eases into the
      stance and balances for 10 s without rolling.
- [ ] Say "walk forward". Pass: Gemini refuses and says why. Pupper does not
      move.
- [ ] Say "go forward". Pass: rides about 3 s, then stops pushing.
- [ ] Say "stop" in the middle of a ride. Pass: pushing stops within 1 s.
- [ ] Say "turn left", then "turn right". Pass: the board curves the right
      way; Pupper stays on.
- [ ] Hold the harness, say "get off the board". Pass: Gemini asks if Pupper
      is held, then the motors relax.

Stop the test (e-stop) if a left foot nears the deck edge, a right foot
lands on the deck, the board heads for anything, or a motor faults.

Measure and write down, to replace the sim estimates in `skate/skate.yaml`:

| What | Sim estimate | Measured |
| --- | --- | --- |
| Distance in a 3 s ride (tape on the floor) | 3 × `travel_m_per_s` | |
| Roll after the last push | `coast_m` | |
| Heading change in a 5 s hard carve, each way | 5 × `turn_deg_per_s.hard` | |
```

In MJ `skateboard/README.md`, add one row to the documents table, after the
`HARDWARE_TEST.md` row:

```markdown
| Voice control through Gemini Pupper: timed commands, no distance sensing | [`PLAN_GEMINI_VOICE.md`](PLAN_GEMINI_VOICE.md) |
```

- [ ] **Step 7: Commit both repos**

GP:

```bash
git add lab/prompt.md robot/test_tools.py skate/skate.yaml skate/README.md
git commit -m "feat: add skateboard prompt section and sim-measured estimates"
```

MJ:

```bash
git add skateboard/HARDWARE_TEST.md skateboard/README.md
git commit -m "docs: add voice control hardware checklist"
```

---

## After the tasks (controller)

- Whole-branch review of the GP `skate-voice` diff on the most capable
  model.
- Optional, only if a Gemini key is on the Mac: `python3 lab/chat.py` on the
  fake robot with "start skating", "go forward", "turn left", "walk
  backward", "stop". Never `--robot`.
- Secret scan of both diffs, then push MJ to `team`. Push GP only when Ethan
  says so.
- Append a "Result" section to this file with the measured numbers and what
  the tests printed.

## Self-review

- Spec coverage: "move forward" is `skate_forward` (Task 4); "turn left" is
  `skate_turn` (Task 4); "careful, no distance" is the timed caps, the queue
  cap, the fixed speed, the estimates in every reply and the path-clear rule
  (Tasks 2, 4, 5) plus the sim numbers behind them (Tasks 1, 5); "assume
  ride and steer work" is why nothing here retrains or re-tests the policy,
  and why test 6 depends on tests 3 and 5.
- Placeholders: none. The three "Sim estimate" cells in the checklist name
  the yaml keys on purpose: the numbers are measured in Task 5, Step 3.
- Names checked across tasks: `SKATE_CONTROLLER`, `SKATING`,
  `ON_BOARD_TOOLS`, `ON_BOARD_MESSAGE`, `NOT_INSTALLED`, `MIN_RIDE_S`,
  `SHARPNESS`, `SkateConfig` fields, `load_skate`, `skate_controller_params`,
  `fixture_skate`, `Rig(skate=...)`, tool names, and the JSON keys from
  `timed_ride.py`.

## Accounting

- Skipped: a distance-based command ("go 2 metres"), reverse, obstacle
  stops from the camera, and getting on the board by itself. None is safe
  or possible with this policy and no distance sensing.
- Assumed: steer v6 rides and carves on hardware as in sim; the team's fork
  may change files under `robot/` (the lab calls them provided); the
  `cmd_vel_mux` deadband of 0.05 passes a 0.15 rad/s carve unchanged (its
  source is not on the Mac; test 6 checks the gentle carve).
- Issues: the plan code has not been run. The Mac's gemini_pupper is a
  snapshot from 2026-11-09 and the robot's copy may be ahead, so the branch
  may need a merge on the robot.

## Result (2026-11-21): built and desk-tested, not run on the robot

Nothing here has run on the robot. A TA's OK is needed before the first run.

- **Gemini Pupper repo**, local branch `skate-voice` (8 commits on top of
  `nicholas/todo4-11`, not pushed): skate config and the steer v6 policy, skate
  mode in the robot API and launch, the four tools (`start_skating`,
  `skate_forward`, `skate_turn`, `end_skating`), the prompt section and
  `skate/README.md`. Desk tests: 300 checks pass (166 before this work).
- **This repo:** `skateboard/tools/timed_ride.py` with its tests, and test 6
  in `HARDWARE_TEST.md`.

Sim estimates Gemini quotes (steer v6, 16 envs, 5 s rides, no resets):

| Stick (rad/s) | Speed (m/s) | Board turn (deg/s) | Max coast (m) |
| --- | --- | --- | --- |
| 0 | 0.292 | -0.153 | 0.073 |
| +0.15 / -0.15 | 0.285 / 0.290 | 1.213 / -1.087 | 0.079 / 0.066 |
| +0.3 / -0.3 | 0.288 / 0.292 | 1.658 / -1.369 | 0.084 / 0.073 |
| +0.5 / -0.5 | 0.295 / 0.301 | 2.182 / -2.287 | 0.093 / 0.082 |

`skate.yaml` carries 0.29 m/s, 1.1 / 1.4 / 2.2 deg/s and 0.1 m coast.

### What changed from the plan

- A tilt e-stop while skating keeps walking, tricks and follow refused until a
  person holds Pupper and `end_skating(held=true)` runs (`fell_off_board`).
- "stop" during the switch into skate mode can no longer leave the skate
  controller on unnoticed; no other controller can be switched in while it is
  active.
- A failed job resets the queued state to the real state.
- The web app's debug endpoints get the same on-board refusal as the tools.
- `load_skate` enforces 0.3 m/s, 0.5 rad/s, 5 s a command and 10 s queued as
  ceilings in code.
- The checklist adds: gamepad e-stop before the first ride, a gentle carve,
  zero-command stop, the tip test, and a hands-off-the-gamepad warning.

### Open decisions

- Decided 2026-11-21: the board guard stays on after a gamepad e-stop on the
  board. The e-stop still relaxes every leg at once; then walking, tricks and
  follow stay refused (a spoken "stand up" too) until a person holds Pupper
  and `end_skating(held=true)` runs. Before, the e-stop released the guard, so
  "stand up" would have been accepted with Pupper still on the deck.
- The turn estimates round up slightly past the weaker side (1.087 to 1.1,
  1.369 to 1.4, 2.182 to 2.2 deg/s), as the rounding rule here says. Flooring
  to 1.0 / 1.3 / 2.1 would make "never promises more turn" literally true.
- `robot/robot_api.py`, `robot/tool_registry.py`, `robot/server.py` and the
  launch file are provided code in the Gemini Pupper repo and are changed on
  the branch.

### Known minor gaps (deferred)

- If the switch into skate mode lands more than 0.1 s after "stop", one
  "stand up" can be answered as queued (its job is then refused); in that
  window "lie down" can relax the motors without the held check, and follow
  can leave the skate tools saying "not in skate mode" until "stop". Note the
  real switch time on hardware.
- A `skate.yaml` table given as a bare number raises TypeError, not
  ValueError.
- The 10 s queue cap can be a little strict after a fast job and a little
  loose after a slow one.
- `end_skating` waits behind queued rides ("stop" first).
- `timed_ride.py`: `--ride-s` under one step gives NaN; episode timeouts
  count as resets.
- Sim runs 2 to 7 were launched in one batch; one re-run alone gave an
  identical line.

### Accounting

- Skipped: any run on the robot; `/code-review` (not available in this
  session, subagent reviews were used instead); pushing the Gemini Pupper
  branch (public repo, not yet authorised).
- Assumed: steer v6 works on hardware; the mux deadband passes 0.15 rad/s;
  the Mac copy of the Gemini Pupper repo matches the robot's.
- Issues: the estimates are sim numbers; expect a longer coast on hardware.
