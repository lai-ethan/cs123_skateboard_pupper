b# Pupper Skateboard RL Training Implementation Plan (v3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** train a policy in sim that holds Pupper in a split stance on the Walmart mini cruiser (left feet on the deck, right feet on the floor), then one that pushes the board forward at a commanded speed, and get it onto the real robot by Demo Day (Dec 3).

**Architecture:** the board is a second, unactuated entity in the pupper-mjlab scene. A new task package `tasks/pupper_skate/` builds on the stock `pupper_velocity_fs_env_cfg`, swaps in a split-stance keyframe, and adds contact sensors, rewards, terminations and board randomization. The actor keeps the stock 48-dim observation, so the exported `policy.json` deploys with no robot-side changes. Board state goes to the critic only (the critic is the value network used during training; it never ships).

**Tech stack:** pupper-mjlab (mjlab fork, MuJoCo-Warp), rsl_rl PPO, uv, pytest, Colab G4 GPU, Weights & Biases.

## Global Constraints

- Repo: `github.com/lai-ethan/cs123_skateboard_pupper`, branch `main`. It is the course repo `cs123-stanford/pupper-mjlab` plus our skateboard work. Never push to the course repo.
- All commands run from the repo root unless a step says otherwise.
- Always `uv run ...`. On the Mac add `FORCE_CPU=1`. Training runs on Colab only.
- Keep `PUPPER_MUJOCO_OPTIONS` and 50 Hz control. Changing them hurts sim-to-real.
- Actor observation stays `zero_reference_actor_obs` (48 dims × 20 history). Board state is critic-only.
- HUSKY is CC-BY-NC-4.0: use its structure and ideas, copy no code, credit it.
- No robot motion without the team agreeing first. TA sign-off before the first real run of any new policy.
- `make check` before every commit. Commits `<type>: <description>`, no AI trailers.
- Deck side is left (`DECK_SIDE = "l"`). The keyframe is solved for left only.

## What changed from v2

- The work now lives in the team repo and is committed. Tasks 7, 8 and 13 (commit, publish, README) are done.
- Paths are repo-relative. The stance helper scripts were cleaned up and moved to `skateboard/tools/` (`solve_stance.py`, `check_stance.py`).
- The notebook clones the team repo, branch `main`.
- The board has arrived. Task 10 (measure and refit) is the next thing to do.

## What changed from v1

- Tasks 1 to 6 were built and verified while checking that the plan's code actually runs.
- The stance changed. A level stance is impossible: the deck top is 0.10 m but the legs can only make up about 0.075 m. The stance is now rolled 0.09 rad toward the floor side and spawns already rolled.
- Left feet moved out to y = 0.14 m (v1 guessed 0.12). Board centre sits 0.14 m left of the base (v1: 0.17).
- Riding is now in scope as its own task (`Mjlab-Skate-Ride-Pupper-v3`), not "after the checkpoint".
- Board randomization (mass, friction, bushing stiffness) is in. Deck-height randomization is still out.
- Added: Colab notebook, tuning playbook, real-board refit procedure, deploy steps.

## Design

### Stance (`SKATE_*` in `pupper_constants.py`)

| Constant | Value | Meaning |
| --- | --- | --- |
| `SKATE_POSE` | FR `(0.3, 0.0, -0.594)`, FL `(0.282, -0.479, -0.33)`, BR `(0.3, 0.0, -0.594)`, BL `(0.22, -0.579, -0.229)` | 12 joint angles, order hip, abduction, knee |
| `SKATE_HEIGHT` | 0.156 m | base height once settled |
| `SKATE_ROLL` | 0.09 rad | settled roll, right side lower |
| `SKATE_BOARD_OFFSET` | `(-0.01, 0.14)` m | board centre in the robot's heading frame (x forward, y left) |
| `SKATE_KEYFRAME` | pos z 0.166, rot rolled 0.09, joints above | spawn state |

Left feet: 0.14 m out, 0.070 m below the base. Right feet: default x/y, 0.145 m below the base. With zero actions the stance holds: roll 0.095, height 0.156, board still, no terminations.

### Scene and sensors

- Entities: `robot`, `board` (`get_skateboard_cfg()`, spawned at the offset above, z = wheel radius).
- `nconmax=80`, `njmax=800` (two bodies touching each other and the floor).
- Contact sensors, all per foot: `deck_feet_contact` (left feet on deck), `ground_feet_contact` (right feet on floor), `deck_feet_on_ground`, `ground_feet_on_deck`, `knees_on_deck`. The first two track air time.

### Rewards

Positive terms are `exp(-error²/std²)` or contact counts, so the weight is the most a term can pay per step.

| Term | Balance | Ride | Function | Notes |
| --- | --- | --- | --- | --- |
| `feet_on_deck` | 1.0 | 1.0 | `contact_count` | +1 per left foot on the deck |
| `feet_on_ground` | 0.5 | 0.1 | `contact_count` | +1 per right foot on the floor |
| `deck_feet_on_ground` | -2.0 | -2.0 | `contact_count` | left foot on the floor |
| `ground_feet_on_deck` | -2.0 | -2.0 | `contact_count` | right foot on the deck |
| `knees_on_deck` | -1.0 | -1.0 | `contact_count` | |
| `tilt` | 1.0 | 1.0 | `gravity_tracking` | target gravity xy `(0, -sin 0.09)`, std 0.2 |
| `base_height` | 1.0 | 1.0 | `base_height` | target 0.156, std 0.03 |
| `stance_pose` | 1.0 | 1.0 | `stance_pose` | std 0.3; ride: left-leg joints only |
| `board_alignment` | 1.0 | 1.0 | `board_alignment` | std 0.05 m, 0.3 rad |
| `deck_flat` | -5.0 | -5.0 | `deck_roll_l2` | deck leaning on its trucks |
| `board_still` | -2.0 | none | `board_speed_l2` | |
| `track_board_velocity` | none | 3.0 | `track_board_velocity` | board forward speed vs command, std 0.15 |
| `board_yaw_rate` | none | -0.5 | `board_yaw_rate_l2` | |
| `push_air_time` | none | 0.5 | `feet_air_time` | right-foot strides of 0.05 to 0.4 s, only when command > 0.1 |
| `action_rate_l2` | -0.01 | -0.01 | stock | |
| `joint_torques_l2` | -2e-4 | -2e-4 | stock | |
| `termination` | -100 | -100 | `is_terminated` | |

### Terminations

- `fell_over`: tilt past 0.6 rad (wider than walking, since the stance is tilted).
- `feet_off_deck`: any left foot off the deck for 0.5 s.
- `base_too_low`: base under 0.08 m.
- `board_tipped`: board tilted past 0.3 rad.

### Command, resets, randomization, critic

- Command: balance is always zero. Ride samples forward speed in `RIDE_SPEED_RANGE = (0.0, 0.5)` m/s, 10% of envs standing, no sideways or turn.
- Resets: robot ±1 cm, yaw ±0.05 rad; board x ±3 cm, yaw ±0.05 rad.
- Board randomization at startup: mass and inertia ×0.7 to ×1.3, deck friction 0.6 to 1.2, wheel friction 0.6 to 1.1, bushing stiffness ×0.5 to ×2. Robot randomization (gains, foot friction, encoder bias, centre of mass, inertia, IMU latency) is inherited. There are no random pushes.
- Critic extras (19 dims, critic total 91): board pose in the robot frame, board linear and angular velocity, deck roll angle and rate, per-foot contact flags.

## File Structure

| File | Responsibility | State |
| --- | --- | --- |
| `src/mjlab/asset_zoo/robots/mini_cruiser/skateboard_constants.py` | board model, `SkateboardParams` | done |
| `src/mjlab/asset_zoo/robots/pupper_v3/pupper_constants.py` | `SKATE_*` constants (modified) | done |
| `src/mjlab/tasks/pupper_skate/mdp/{rewards,terminations,observations}.py` | skate MDP terms | done |
| `src/mjlab/tasks/pupper_skate/pupper_skate_env_cfg.py` | balance and ride env configs | done |
| `src/mjlab/tasks/pupper_skate/config/__init__.py` | task registration | done |
| `tests/test_pupper_skateboard.py` | board physics (5 tests) | done |
| `tests/test_pupper_skate.py` | keyframe, terms, env, export, ride (17 tests) | done |
| `notebooks/CS123_Pupper_Skate.ipynb` | Colab training | done, not yet run on Colab |
| `skateboard/tools/solve_stance.py` | solves the stance joint angles for a deck height | done |
| `skateboard/tools/check_stance.py` | drops Pupper on the board in a stance and prints whether it holds | done |
| `skateboard/tools/make_notebook.py` | regenerates the skate notebook from the course notebook | done |
| `skateboard/{README,PLAN,RESEARCH}.md` | start-here guide, this plan, paper notes | done |
| `README.md` | task table rows, HUSKY credit | done |

## Tasks

### Tasks 1 to 6: built and verified

- [x] **Task 1: Split-stance keyframe.** `SKATE_POSE`, `SKATE_HEIGHT`, `SKATE_ROLL`, `SKATE_BOARD_OFFSET`, `SKATE_KEYFRAME`. Tests: angles inside joint limits with 0.2 rad margin; forward kinematics puts left feet at y 0.14, z -0.070 and right feet at z -0.145.
- [x] **Task 2: MDP terms.** Rewards `contact_count`, `gravity_tracking`, `stance_pose`, `board_alignment`, `board_speed_l2`, `board_yaw_rate_l2`, `deck_roll_l2`, `track_board_velocity`; termination `contact_lost`; observations `board_pose_in_robot_frame`, `deck_roll_state`, `contact_flags`. Each has a unit test on a hand-built state.
- [x] **Task 3: Balance task** `Mjlab-Skate-Balance-Pupper-v3`. Tests: 150 zero-action steps hold the stance with no terminations; lifting the robot off the deck fires `feet_off_deck`.
- [x] **Task 4: Registration and export contract.** Tests: both tasks registered; actor is 48 × 20 zero-reference; exported policy has `SKATE_POSE` as its default pose, `command_clip` all zero, no heading hold.
- [x] **Task 5: Ride task** `Mjlab-Skate-Ride-Pupper-v3`. Tests: config (speed range, reward swaps); 100 random-action steps stay finite.
- [x] **Task 6: Colab notebook.** Generated from the course notebook by `skateboard/tools/make_notebook.py`.

Verify any time (on a Mac, keep `FORCE_CPU=1`):

```bash
FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py tests/test_pupper_skateboard.py tests/test_task_configs.py -q
```

Expected: `31 passed`.

### Task 7: Review and commit

- [x] `make check` clean, tests pass, committed.
- [ ] **Look at it** (needs a screen):

```bash
FORCE_CPU=1 uv run play Mjlab-Skate-Balance-Pupper-v3 --agent zero
```

Expected: Pupper stands tilted with left feet on the board and stays there.

### Task 8: Publish so Colab can clone it

- [x] Pushed to `github.com/lai-ethan/cs123_skateboard_pupper`, branch `main`. The notebook's first code cell clones it.

### Task 9: Train balance on Colab (checkpoint, Mon Nov 16)

- [ ] **Step 1:** open `notebooks/CS123_Pupper_Skate.ipynb` in Colab, runtime G4 GPU, run top to bottom. `TASK = "Mjlab-Skate-Balance-Pupper-v3"`, 4096 envs, 3000 iterations (about 18 min). Whoever runs it pastes their own W&B key; never commit a key.
- [ ] **Step 2: Pass criteria** on W&B:
  - `Train/mean_episode_length` ≥ 450 (500 is a full 10 s episode).
  - `Episode_Reward/feet_on_deck` ≥ 1.8 (2.0 is both left feet on the deck all episode).
  - `Episode_Termination/feet_off_deck` and `fell_over` near zero.
- [ ] **Step 3:** run the notebook's "Watch it" cell, screen-record 20 s for the checkpoint.
- [ ] **Step 4: If it fails**, read `Episode_Termination/*` first:
  - `feet_off_deck` dominates: raise `feet_on_deck` to 2.0 and `board_alignment` to 2.0.
  - `fell_over` or `base_too_low` dominates: raise `tilt` and `base_height` to 2.0.
  - Shaky actions in the viewer: `action_rate_l2` to -0.05.
  - Set overrides in the notebook's `REWARD_WEIGHTS` dict, retrain.

Note: zero actions already hold the stance in sim, so balance should train fast. What this run proves is the pipeline and robustness to the robot and board randomization.

### Task 10: Measure the real board and refit (do this first)

**Files:** `skateboard_constants.py` (`SkateboardParams` defaults), `pupper_constants.py` (`SKATE_*`), `tests/test_pupper_skate.py` (the foot-position numbers).

- [ ] **Step 1: Measure.**

| Parameter | How | Formula |
| --- | --- | --- |
| `deck_top_height` | ruler, floor to deck top, board on its wheels | direct |
| `deck_length`, `deck_width`, `wheelbase`, `track`, `wheel_radius` | ruler | direct |
| `mass` | kitchen scale | direct |
| `roll_stiffness` | put mass m (about 1.5 kg) on the deck edge, distance d from the centreline; measure how far the edge drops, Δh | θ = atan(Δh / d); k = m · 9.81 · d / θ |
| `wheel_frictionloss` | roll the board off a ramp of height h, measure the flat coast distance s | F = mass · 9.81 · h / s; frictionloss = F · wheel_radius / 4 |

- [ ] **Step 2:** write the measured values into the `SkateboardParams` defaults. Run `FORCE_CPU=1 uv run pytest tests/test_pupper_skateboard.py -q`. Expected: `5 passed`.
- [ ] **Step 3: If `deck_top_height` moved by more than 5 mm, re-solve the stance.** New left depth = 0.070 + (0.10 − measured height):

```bash
FORCE_CPU=1 uv run python skateboard/tools/solve_stance.py 0.14 <left_depth> 0.145
FORCE_CPU=1 uv run python skateboard/tools/check_stance.py 0.14 0.17 <the 12 angles on the last line>
```

`solve_stance.py` prints the 12 angles and each joint's margin to its limit (keep every margin above 0.2). `check_stance.py` prints roll, base position and board position every second for 5 s. Accept if roll and height are steady over the last 3 prints and the board has not moved.

- [ ] **Step 4:** copy the angles into `SKATE_POSE`, the last roll into `SKATE_ROLL`, the last base z into `SKATE_HEIGHT`. Update the `-0.070` in `test_skate_pose_lifts_the_left_feet_over_the_deck` to the new left depth. Run `make check` and the verify command above. Expected: all pass.
- [ ] **Step 5: Fit test, robot powered off.** Place Pupper by hand in the stance on the real board. Check the left feet reach the deck 0.14 m out and nothing collides. No motors.
- [ ] **Step 6:** commit `fix: refit skateboard model to measured board`, push, retrain balance (Task 9).

### Task 11: Train riding

- [ ] **Step 1:** notebook with `TASK = "Mjlab-Skate-Ride-Pupper-v3"`, 4096 envs, 3000 iterations.
- [ ] **Step 2: Pass criteria:**
  - `Episode_Reward/track_board_velocity` ≥ 2.0 (of 3.0; about 0.1 m/s speed error).
  - `Train/mean_episode_length` ≥ 450.
  - In the viewer the right legs take visible strides and the board rolls straight.
- [ ] **Step 3: Tuning playbook.** Change one thing per run, stop a bad run by iteration 1000.

| Symptom | Change |
| --- | --- |
| Stands still, board does not move | `track_board_velocity` 3.0 → 6.0; `feet_on_ground` → 0.0 |
| Shuffles or drags the right feet | `push_air_time` 0.5 → 1.5 |
| Board moves but left feet slide off | `board_alignment` → 2.0; `feet_on_deck` → 2.0 |
| Board curves away | `board_yaw_rate` -0.5 → -2.0; `deck_flat` -5 → -10 |
| Lunges, then falls | `RIDE_SPEED_RANGE` to `(0.0, 0.3)` in `pupper_skate_env_cfg.py`; `action_rate_l2` → -0.05 |
| Nothing after three runs | see the riding risk under Accounting; fall back to balance for the demo |

- [ ] **Step 4:** record the best run id and the weights that produced it in this file under this task.

### Task 12: First real run

- [ ] **Step 1:** show a TA the sim video and get a sign-off. Balance policy first, ride later.
- [ ] **Step 2:** tape or wedge the board's wheels for the first balance run so it cannot roll.
- [ ] **Step 3:** on the robot (over SSH):

```bash
cd ~/pupper_gait_deploy
./deploy.sh mjlab/<run-id>
```

- [ ] **Step 4:** one person holds the robot by the handle in the stance on the board, a second presses square on the gamepad. Let go only if it holds still for 5 s. Kill switch in hand.
- [ ] **Step 5:** if it shakes: retrain with `action_rate_l2` at -0.05 and the notebook's `KP_MULTIPLIER_RANGE` widened to `(0.5, 1.2)`.
- [ ] **Step 6:** unwedge the wheels, repeat for balance, then the ride policy at the lowest speed.

### Task 13: README and credit

- [x] Root `README.md` has a start-here pointer, both task rows, and the HUSKY credit.

### Schedule

| Date | Milestone |
| --- | --- |
| Fri Nov 13 | code published to the team repo; board in hand |
| Sat Nov 14 | Task 10: measure and refit; first balance run (Task 9) |
| Mon Nov 16 | checkpoint: balance video, pipeline end to end |
| Nov 17 to 24 | Task 11 riding; Task 12 balance on the robot |
| Nov 25 to Dec 1 | ride on the robot, tuning |
| Wed Dec 2 | demo video due |
| Thu Dec 3 | Demo Day |

## Self-review

- Coverage: stance keyframe (Task 1), modeling (Tasks 1, 3, 10), rewards and terminations (Tasks 2, 3, 5 and the Design tables), randomization (Task 3), training (Tasks 9, 11), export and deploy (Tasks 4, 12). No gaps found.
- Names checked against the code: task ids, reward term names, `SKATE_*`, W&B metric names, `REWARD_WEIGHTS`, `RIDE_SPEED_RANGE`.
- Tasks 1 to 6 point at verified files instead of repeating about 900 lines of code here.

## Accounting

- **Skipped:** code listings for Tasks 1 to 6 (the code is in the repo and tested); deck-height randomization; a right-side stance; a push/glide phase clock like HUSKY's.
- **Assumed:** left side on the deck; board is 0.10 m tall and 1.8 kg until measured; W&B project name `mjlab` as in the course notebook.
- **Issues:** riding may not be physically reachable. The right legs have about 1 cm of extension left for pushing, and it has only been tested with random actions. The notebook has not been run on Colab. `uv run play` for the skate tasks has not been run (it opens a viewer). Reward weights are starting values, not tuned.
