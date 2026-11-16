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
- Board randomization (mass, friction, bushing stiffness) is in. Deck-height randomization (±1.5 cm) is in too (Task 14).
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
| `push_stride` | none | 0.3 | `push_stride` | one payment per right-foot stride: planted about 0.4 s, swing 0.2 to 0.5 s, only when command > 0.1 |
| `push_leg_pose` | none | 0.3 | `stance_pose` | right-leg joints near the stance, std 0.6 |
| `joint_vel_l2` | none | -1e-3 | stock | |
| `joint_speed_limit` | none | -0.02 | `joint_speed_over_limit` | speed above 6 rad/s, 75% of the 8 rad/s gait-check limit (a guess at the motor limit) |
| `torque_saturation` | none | -0.5 | `torque_saturation` | torque above 80% of the 3 N·m limit |
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

**Result (Sun Nov 15):** PASS in sim, default weights.
- `bal1`, no pushes: W&B `ethanlai-stanford-university/mjlab/2zxd46yi`.
- `bal2`, pushes + deck-height randomization, 1500 iterations: episode length 500, `feet_on_deck` 2.00. W&B `ethanlai-stanford-university/mjlab/2ace1uqa`. This is the one to try on the robot first.

- [ ] **Step 1:** open `notebooks/CS123_Pupper_Skate.ipynb` in Colab, runtime G4 GPU, run top to bottom. `TASK = "Mjlab-Skate-Balance-Pupper-v3"`, 4096 envs, 1500 iterations (about 50 min on a free T4). Whoever runs it pastes their own W&B key; never commit a key.
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

**Result (Mon Nov 16, 13:01 CST):** `RESULT: PASS`, default weights, full 1500 iterations.
- W&B `ethanlai-stanford-university/mjlab/29p4j81t`, Colab Pro T4, 55 min.
- Last 20 iterations: episode length 487.0 (need 450), `track_board_velocity` 2.380 (need 2.0), `feet_on_deck` 1.873. `board_tipped` 0.03, `base_too_low` 0.01, `feet_off_deck` 0.51, `fell_over` 0.09.
- No reward hacking seen: tracking, feet on deck and episode length rose together while terminations fell.
- `policy.json` on the run was replaced at the end of training (parity error 3.7e-6).
- Earlier run `me6p0pdw` (free T4) was cut off near iteration 1120 by Colab's GPU limit; superseded.
- Do not read `Metrics/twist/error_vel_xy` as tracking error. It sums the robot body's (not the board's) error over the episode, so it grows as episodes get longer.

**Note (Mon Nov 16): the `29p4j81t` gait cannot run on the robot.**
- At 0.4 m/s the front-right hip swings 2.3 rad at 3 Hz and peaks near 39 rad/s.
- Five joints sit at the 3 Nm torque limit.
- "No reward hacking seen" above was wrong. The check looked at the curves, never at the gait.
- `gait_report.py` prints `GAIT: FAIL` for this policy.
- The ride rewards were reshaped in Task 16. Ride needs a fresh 1500-iteration run with them. That run has not happened yet.

- [ ] **Step 1:** notebook with `TASK = "Mjlab-Skate-Ride-Pupper-v3"`, 4096 envs, 1500 iterations.
- [ ] **Step 2: Pass criteria:**
  - `Episode_Reward/track_board_velocity` ≥ 2.0 (of 3.0; about 0.1 m/s speed error).
  - `Train/mean_episode_length` ≥ 450.
  - `Episode_Reward/joint_speed_limit` ≥ -0.1.
  - `Episode_Reward/torque_saturation` ≥ -0.2.
  - `GAIT: PASS` from `skateboard/tools/gait_report.py` at `--speed 0.4` (notebook section 7).
  - In the viewer the right legs take visible strides and the board rolls straight.
- [ ] **Step 3: Tuning playbook.** Change one thing per run, stop a bad run by iteration 1000.

| Symptom | Change |
| --- | --- |
| Stands still, board does not move | `track_board_velocity` 3.0 → 6.0; `feet_on_ground` → 0.0 |
| Shuffles or drags the right feet | `push_stride` 0.3 → 0.6; `push_leg_pose` 0.3 → 0.1 |
| Fast leg flailing (`GAIT: FAIL` on joint speed) | `joint_speed_limit` -0.02 → -0.1; `joint_vel_l2` -1e-3 → -5e-3 |
| Motors pinned at the limit | `torque_saturation` -0.5 → -1.5 |
| Stands still after the reshaping | `push_leg_pose` → 0.1; `joint_vel_l2` → -2e-4; then `track_board_velocity` → 6.0 |
| Board moves but left feet slide off | `board_alignment` → 2.0; `feet_on_deck` → 2.0 |
| Board curves away | `board_yaw_rate` -0.5 → -2.0; `deck_flat` -5 → -10 |
| Lunges, then falls | `RIDE_SPEED_RANGE` to `(0.0, 0.3)` in `pupper_skate_env_cfg.py`; `action_rate_l2` → -0.05 |
| Nothing after three runs | see the riding risk under Accounting; fall back to balance for the demo |

- [ ] **Step 4:** record the best run id and the weights that produced it in this file under this task.

### Task 12: First real run

Do the steps in [`SIM2REAL.md`](SIM2REAL.md) first (measure the board, fit test).

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

### Task 14: Robustness randomization for an unmeasured board

The board is not measured yet, so the policy must tolerate a deck that is not exactly 0.10 m tall and must recover from small shoves. Add three event terms in `_skate_base_env_cfg` style shared code (both tasks get them unless noted), next to the existing board randomization.

**Files:**
- Modify: `src/mjlab/tasks/pupper_skate/pupper_skate_env_cfg.py`
- Test: `tests/test_pupper_skate.py`

**Interfaces:**
- Consumes: `DECK_GEOM_NAME` from `mjlab.asset_zoo.robots.mini_cruiser.skateboard_constants`; `dr.geom_pos` (`src/mjlab/envs/mdp/dr/geom.py`); `mdp.push_by_setting_velocity` (`src/mjlab/envs/mdp/events.py`). Read both functions' signatures before use.
- Produces: event term names `deck_height`, `push_robot`, `nudge_board`, and module constants `DECK_HEIGHT_RANGE = (-0.015, 0.015)`, `PUSH_INTERVAL_S = (2.0, 4.0)`.

Exact values:

| Event name | Mode | Function | Target | Values |
| --- | --- | --- | --- | --- |
| `deck_height` | `startup` | `dr.geom_pos` | board geom `DECK_GEOM_NAME`, z axis only, operation `add` | `DECK_HEIGHT_RANGE = (-0.015, 0.015)` m |
| `push_robot` | `interval`, `interval_range_s=PUSH_INTERVAL_S` | `mdp.push_by_setting_velocity` | robot root | `velocity_range = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}` (linear only) |
| `nudge_board` | `interval`, `interval_range_s=PUSH_INTERVAL_S` | `mdp.push_by_setting_velocity` | board root (`SceneEntityCfg("board")`) | `velocity_range = {"x": (-0.15, 0.15)}` ; **balance task only** |

Rules:
- In `play=True` configs remove `push_robot` and `nudge_board` (the stock play branch already pops `push_robot`; make sure ours is added before that or popped again), and keep `deck_height`.
- If `push_by_setting_velocity` overwrites the whole root velocity (check), that is acceptable for the robot; for the board it must not zero the deck-roll joint. If it cannot target the board's free root cleanly, write a small `mdp/events.py` in the skate package with `nudge_board_velocity(env, env_ids, x_range, asset_cfg)` that adds a sampled x velocity (board heading frame is fine as world x because board yaw stays within ±0.05 rad at reset) to the board root linear velocity, and export it from the skate `mdp/__init__.py`.
- Do not change any reward, termination, observation or the keyframe.

- [ ] **Step 1: Write failing tests** in `tests/test_pupper_skate.py`, following the file's existing style (plain functions; reuse its existing env-building helpers/fixtures):
  - `test_balance_cfg_has_robustness_events`: the balance cfg has `deck_height`, `push_robot`, `nudge_board`; `deck_height` mode is `startup`; both push terms have `interval_range_s == (2.0, 4.0)`.
  - `test_ride_cfg_has_no_board_nudge`: ride cfg has `deck_height` and `push_robot`, not `nudge_board`.
  - `test_play_cfg_drops_pushes`: both tasks with `play=True` have neither `push_robot` nor `nudge_board`.
  - `test_deck_height_is_randomized_per_env`: build the balance env on CPU with 8 envs (same way the file's existing env tests do), read the deck geom z position from the per-env model field, assert the values are not all equal and all lie within ±0.015 m of the nominal value.
  - `test_balance_env_survives_pushes`: 8 envs, step 150 zero-action steps (3 s, covers at least one push interval); assert observations and rewards are finite.
- [ ] **Step 2:** run `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py -q`; the new tests fail.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py tests/test_pupper_skateboard.py tests/test_task_configs.py -q`; all pass. Run `uv run ruff format`, `uv run ruff check`, `uv run ty check` (`make check` has one known upstream pyright failure in `notebooks/watch_live.py:94`; anything else must be clean).
- [ ] **Step 5:** add the three rows to the randomization list in `skateboard/README.md` if it has one, else skip. Commit `feat: add deck-height and push randomization to skate tasks`.

### Task 15: Notebook that runs unattended and by hand

The same notebook must work two ways: a person opens it and runs cells, or "Runtime → Run all" trains and prints a verdict with no prompts. Today the W&B cell blocks on an interactive prompt when no key is pasted, and there is no pass/fail readout.

**Files:**
- Create: `skateboard/tools/run_summary.py`
- Create: `tests/test_skate_run_summary.py`
- Modify: `skateboard/tools/make_notebook.py`, then regenerate `notebooks/CS123_Pupper_Skate.ipynb` with `python3 skateboard/tools/make_notebook.py notebooks/CS123_Pupper_mjlab.ipynb notebooks/CS123_Pupper_Skate.ipynb`
- Modify: `skateboard/README.md` (the "how to train" steps)

**Interfaces:**
- Produces: `run_summary.py` with
  - `PASS_CRITERIA: dict[str, dict[str, float]]` = `{"Mjlab-Skate-Balance-Pupper-v3": {"Train/mean_episode_length": 450.0, "Episode_Reward/feet_on_deck": 1.8}, "Mjlab-Skate-Ride-Pupper-v3": {"Train/mean_episode_length": 450.0, "Episode_Reward/track_board_velocity": 2.0}}` (each value is a minimum).
  - `latest_run_dir(log_root: Path) -> Path` — newest directory matching `log_root/*/*` by mtime; raises `FileNotFoundError` with a clear message if none.
  - `read_scalars(run_dir: Path, last_n: int = 20) -> dict[str, float]` — for every scalar tag in the run's TensorBoard event files, the mean of its last `last_n` points.
  - `verdict(task: str, scalars: dict[str, float]) -> list[tuple[str, float, float, bool]]` — one `(tag, value, minimum, passed)` per criterion; a missing tag counts as failed with value `nan`.
  - CLI: `python skateboard/tools/run_summary.py <task-id> [--log-root logs/rsl_rl] [--run-dir DIR]` prints the iteration count, every `Train/*`, `Episode_Reward/*` and `Episode_Termination/*` scalar sorted by tag, then one `PASS`/`FAIL` line per criterion and a final `RESULT: PASS` or `RESULT: FAIL`. Exit code 0 either way.
- The exact scalar tag names rsl_rl writes are not guaranteed to be the names above. Find the real tags first: `logs/rsl_rl/` in the repo already holds a short local run; if not, make one with `FORCE_CPU=1 CUDA_VISIBLE_DEVICES="" WANDB_MODE=disabled uv run train Mjlab-Skate-Balance-Pupper-v3 --gpu-ids None --env.scene.num-envs 64 --agent.max-iterations 2 --agent.logger tensorboard` (about a minute). If the episode-length tag is spelled differently, use the real spelling in `PASS_CRITERIA` and say so in the report.

Notebook changes (all in `make_notebook.py`; never hand-edit the `.ipynb`):

1. **Settings cell** (the existing task cell) becomes:
```python
TASK = "Mjlab-Skate-Balance-Pupper-v3"
NUM_ENVS = 4096  # halve it if you hit GPU OOM
ITERATIONS = 1500
WATCH_LIVE = False  # True: start the live viewer before training
OPEN_VIEWER = True  # False: skip the interactive viewer at the end
```
2. **W&B cell** replaces the course cell. No interactive prompt, ever:
```python
WANDB_API_KEY = ""  # optional: paste a key, or store it as the Colab secret WANDB_API_KEY

import os

if not WANDB_API_KEY:
  try:
    from google.colab import userdata

    WANDB_API_KEY = userdata.get("WANDB_API_KEY") or ""
  except Exception:
    WANDB_API_KEY = ""

if WANDB_API_KEY:
  import wandb

  os.environ["WANDB_API_KEY"] = WANDB_API_KEY
  wandb.login(key=WANDB_API_KEY, relogin=True)
  LOGGER = "wandb"
else:
  LOGGER = "tensorboard"
  print("No W&B key: logging to TensorBoard only. policy.json stays on this runtime.")
print("logger:", LOGGER)
```
   Rewrite the markdown above it to say the key is optional and what you lose without it (W&B curves and the automatic `policy.json` upload needed for `deploy.sh`).
3. **Watch-live cell:** wrap its body in `if WATCH_LIVE:`.
4. **Train cell:** `agent_cfg.logger = LOGGER` instead of the hard-coded `"wandb"`.
5. **New "Results" cell** right after the train cell (markdown heading `## 7. Did it pass?` plus code): `!python skateboard/tools/run_summary.py {TASK}`.
6. **Viewer cell:** wrap its body in `if OPEN_VIEWER:`. Fix the duplicated section numbers so headings count 1..N in order.
7. The top markdown says: "Run all works unattended; or run cells one at a time."

- [ ] **Step 1: Failing tests** in `tests/test_skate_run_summary.py` (load the script by path with `importlib.util`, since `skateboard/` is not a package). Write event files with `torch.utils.tensorboard.SummaryWriter` into `tmp_path / "exp" / "run1"`:
  - `test_read_scalars_averages_last_points`: tag with 30 points 0..29, `last_n=10` → 24.5.
  - `test_latest_run_dir_picks_newest` and `test_latest_run_dir_raises_when_empty`.
  - `test_verdict_passes_and_fails`: balance task, scalars at 480/1.9 → all passed; at 480/1.0 → second failed; missing tag → failed with nan.
  - `test_cli_prints_result_line`: run `main([...])` with `--run-dir`, capture stdout, assert it ends with `RESULT: PASS` for passing data.
  - `test_notebook_has_no_interactive_login`: load `notebooks/CS123_Pupper_Skate.ipynb`; no code cell contains a bare `wandb.login()`; some cell contains `run_summary.py`; the train cell contains `agent_cfg.logger = LOGGER`.
- [ ] **Step 2:** run them, see them fail.
- [ ] **Step 3:** implement, regenerate the notebook.
- [ ] **Step 4:** `FORCE_CPU=1 uv run pytest tests/test_skate_run_summary.py tests/test_pupper_skate.py -q` passes; `uv run ruff format`, `uv run ruff check`, `uv run ty check` clean for the files you touched (notebook-level ruff errors inherited from the course notebook are known; do not add new ones).
- [ ] **Step 5:** update `skateboard/README.md` training steps to match (key optional, Run all, read the `RESULT:` line). Commit `feat: make skate notebook run unattended with a pass/fail summary`.

### Task 16: Ride gait shaping rewards (hardware-feasible push)

**Why:** run `29p4j81t` passed on metrics but its gait cannot run on the robot. Measured at 0.4 m/s: the front-right hip swings 2.3 rad at 3 Hz and peaks near 39 rad/s, that foot is on the floor 14% of the time, and five joints sit at the 3 Nm limit. Causes: `push_air_time` pays the right feet for time in the air, `stance_pose` covers only the left legs, and nothing penalizes joint speed. This task changes the **ride** rewards only. The balance task and its trained policy must not change.

**Files:**
- Modify: `src/mjlab/tasks/pupper_skate/mdp/rewards.py`, `src/mjlab/tasks/pupper_skate/mdp/__init__.py`, `src/mjlab/tasks/pupper_skate/pupper_skate_env_cfg.py`
- Test: `tests/test_pupper_skate.py`

**Interfaces produced (later tasks use these exact names):**
- `mdp.joint_speed_over_limit(env, limit: float, asset_cfg=_ROBOT) -> Tensor[B]`
- `mdp.torque_saturation(env, limit: float, soft_ratio: float = 0.8, asset_cfg=_ROBOT) -> Tensor[B]`
- `mdp.push_stride(env, sensor_name: str, stance_s: float, air_range: tuple[float, float], command_name: str, command_threshold: float) -> Tensor[B]`
- `JOINT_SPEED_LIMIT: float = 8.0` in `pupper_skate_env_cfg.py`
- Ride reward term names: `push_stride`, `push_leg_pose`, `joint_vel_l2`, `joint_speed_limit`, `torque_saturation`. `push_air_time` is removed.

- [ ] **Step 1: failing tests.** In `tests/test_pupper_skate.py`, next to the other fake-env term tests (use the existing `_fake_env` / `_sensor_env` helpers; give the fake env `step_dt` and `command_manager` attributes where needed, as `test_track_board_velocity_peaks_at_the_commanded_speed` does):
  - `test_joint_speed_over_limit_ignores_speeds_under_the_limit`: `joint_vel = [[7.0, -8.0, 0.0], [10.0, -11.0, 0.0]]`, `limit=8.0` gives `[0.0, 13.0]` (2² + 3²).
  - `test_torque_saturation_ramps_from_the_soft_limit`: `actuator_force = [[2.4, -1.0], [2.7, -3.0], [3.5, 0.0]]`, `limit=3.0`, `soft_ratio=0.8` gives `[0.0, 1.5, 1.0]` (2.4 Nm is the soft limit; 2.7 is halfway; values past the limit clamp to 1 per joint).
  - `test_push_stride_pays_a_planted_stride_once_at_touchdown`: `step_dt = 0.02`, command `[[0.4, 0, 0]] * 5`, two feet per env, `stance_s=0.4`, `air_range=(0.1, 0.5)`, `command_threshold=0.1`. Rows, foot 0 varies and foot 1 is standing (`current_contact_time=1.0`, `last_air_time=0.0`, `last_contact_time=0.0`):
    1. just landed (`current_contact_time=0.02`), `last_air_time=0.2`, `last_contact_time=0.4` → `1 / 0.02 = 50.0`
    2. just landed, `last_air_time=0.2`, `last_contact_time=0.1` (a tap) → `0.25 / 0.02 = 12.5`
    3. just landed, `last_air_time=0.04` (too quick) , `last_contact_time=0.4` → `0.0`
    4. just landed, `last_air_time=0.8` (foot held up), `last_contact_time=0.4` → `0.0`
    5. landed earlier (`current_contact_time=0.06`), `last_air_time=0.2`, `last_contact_time=0.4` → `0.0`
    Then the same call with command `[[0.05, 0, 0]] * 5` returns all zeros.
  - Update `test_ride_task_commands_forward_speed_only`: the nonzero-weight loop covers `track_board_velocity`, `board_yaw_rate`, `push_stride`, `push_leg_pose`, `joint_vel_l2`, `joint_speed_limit`, `torque_saturation`; assert `"push_air_time" not in cfg.rewards`; assert the ride `stance_pose` joint names match only left legs and `push_leg_pose` only right legs; assert none of the five new names is in `pupper_skate_balance_env_cfg().rewards`.
- [ ] **Step 2:** run `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py -q -k "speed_over_limit or torque_saturation or push_stride or ride_task"`; expect failures (missing attributes).
- [ ] **Step 3: implement** in `mdp/rewards.py` (export all three from `mdp/__init__.py`):

```python
def joint_speed_over_limit(
  env: ManagerBasedRlEnv, limit: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Penalize joint speed past what the motors can do: sum((|qd| - limit)+^2)."""
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
  return torch.sum(torch.square(torch.clamp(speed - limit, min=0.0)), dim=1)


def torque_saturation(
  env: ManagerBasedRlEnv,
  limit: float,
  soft_ratio: float = 0.8,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Penalize motors working near their torque limit.

  Per joint: 0 below ``soft_ratio * limit``, rising linearly to 1 at ``limit``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ratio = torch.abs(asset.data.actuator_force[:, asset_cfg.actuator_ids]) / limit
  over = (ratio - soft_ratio) / (1.0 - soft_ratio)
  return torch.sum(torch.clamp(over, 0.0, 1.0), dim=1)


def push_stride(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  stance_s: float,
  air_range: tuple[float, float],
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Reward slow, planted push strides, paid once per touchdown.

  A stride is the stance that ended at lift-off plus the swing that just
  landed. It scores min(stance / stance_s, 1) when the swing lasted within
  ``air_range`` seconds, and 0 otherwise, so quick taps and feet held in the air
  earn little. The score is divided by the step so the weight is the reward for
  one full stride. Paid only above ``command_threshold`` m/s of commanded speed.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  contact_time, air_time = data.current_contact_time, data.last_air_time
  stance_time = data.last_contact_time
  assert contact_time is not None and air_time is not None and stance_time is not None
  landed = (contact_time > 0.0) & (contact_time < env.step_dt + 1.0e-6)
  swing_ok = (air_time >= air_range[0]) & (air_time <= air_range[1])
  score = torch.clamp(stance_time / stance_s, max=1.0)
  reward = torch.sum(score * (landed & swing_ok).float(), dim=1) / env.step_dt
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return reward * (command[:, 0] > command_threshold).float()
```

  Before relying on it, read `_update_air_time_tracking` in `src/mjlab/sensor/contact_sensor.py` and confirm that on the touchdown step `last_air_time` is the swing that just ended and `last_contact_time` is the stance before it. If that is not what the sensor does, stop and report NEEDS_CONTEXT with what it does instead.

  In `pupper_skate_env_cfg.py`: import `PUPPER_EFFORT_LIMIT`; add next to `RIDE_SPEED_RANGE`:

```python
# Joint speed limit [rad/s] the gait check enforces; the ride task's penalty
# starts at 75% of it. A design guess, not a datasheet number: replace it with
# the real motor's limit.
JOINT_SPEED_LIMIT: float = 8.0
```

  and in `pupper_skate_ride_env_cfg` replace the `push_air_time` block and the comment above it, and extend the stance-pose block, so the ride rewards end as:

```python
  # Pushing: pay once per stride for a planted stance of 0.4 s followed by a
  # swing of 0.2 to 0.5 s (only when a speed is commanded). Short stances pay
  # proportionally less, so the speed and torque penalties below are what make a
  # fast tap unattractive. Paying for air time taught a 3 Hz leg windmill.
  r["push_stride"] = RewardTermCfg(
    func=mdp.push_stride,
    weight=0.3,
    params={
      "sensor_name": "ground_feet_contact",
      "stance_s": 0.4,
      "air_range": (0.2, 0.5),
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  r["feet_on_ground"].weight = 0.1
  # Hold the keyframe firmly on the deck-side legs and loosely on the pushing
  # legs, so they stride around the stance instead of flailing.
  r["stance_pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(rf"leg_(front|back)_{DECK_SIDE}_[123]",)
  )
  r["push_leg_pose"] = RewardTermCfg(
    func=mdp.stance_pose,
    weight=0.3,
    params={
      "std": 0.6,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(rf"leg_(front|back)_{GROUND_SIDE}_[123]",)
      ),
    },
  )
  # Keep the motion inside what the motors can do. The sim actuator limits
  # torque but not speed, so speed is limited here. The penalty starts at 75% of
  # the limit the gait check enforces (6 of 8 rad/s), so the best-paid gait
  # stays inside the check.
  r["joint_vel_l2"] = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3)
  r["joint_speed_limit"] = RewardTermCfg(
    func=mdp.joint_speed_over_limit,
    weight=-0.02,
    params={"limit": 0.75 * JOINT_SPEED_LIMIT},
  )
  r["torque_saturation"] = RewardTermCfg(
    func=mdp.torque_saturation, weight=-0.5, params={"limit": PUPPER_EFFORT_LIMIT}
  )
  return cfg
```

- [ ] **Step 4:** `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py tests/test_pupper_skateboard.py tests/test_task_configs.py tests/test_skate_run_summary.py -q` all pass (this includes the slow ride env test that steps the real simulator with every reward term). `uv run ruff format`, `uv run ruff check`, `uv run ty check` clean for touched files.
- [ ] **Step 5:** commit `feat: shape the ride gait for hardware (planted strides, speed and torque limits)`.

### Task 17: Gait report tool and stricter ride pass check

**Why:** `RESULT: PASS` only checked episode length and speed tracking, so the windmill gait passed. Add (a) a rollout tool that measures the gait of any saved policy and prints `GAIT: PASS` or `GAIT: FAIL`, and (b) gait terms in the training-curve pass check.

**Files:**
- Create: `skateboard/tools/policy_loader.py`, `skateboard/tools/gait_report.py`, `tests/test_skate_gait_report.py`
- Modify: `skateboard/tools/render_policy.py` (use the shared loader), `skateboard/tools/run_summary.py`, `tests/test_skate_run_summary.py`

**Interfaces:**
- Consumes from Task 16: `JOINT_SPEED_LIMIT` (8.0) from `mjlab.tasks.pupper_skate.pupper_skate_env_cfg`; reward term names `joint_speed_limit` and `torque_saturation`; `PUPPER_EFFORT_LIMIT` (3.0) from `mjlab.asset_zoo.robots.pupper_v3.pupper_constants`; contact sensor `ground_feet_contact` (two right feet, `data.found` shape `[B, 2]`).
- Produces: `policy_loader.load_policy(task: str, policy_path: Path, speed: float | None) -> tuple[raw_env, wrapped_env, policy]`; `gait_report.gait_metrics(...)`, `gait_report.gait_verdict(...)`, `gait_report.GAIT_LIMITS`.

- [ ] **Step 1: shared loader.** Read `skateboard/tools/render_policy.py`. Move its env construction and policy loading (both the `.pt` runner path and the `policy.json` path) into `skateboard/tools/policy_loader.py` as `load_policy(task, policy_path, speed)`, returning the raw `ManagerBasedRlEnv`, the `RslRlVecEnvWrapper`, and a callable `policy(obs) -> actions`. One env, play mode, CPU. `render_policy.py` keeps its command line and behavior and calls `load_policy`. Tools in this folder are run as scripts, so import the sibling module the way a script in the same folder can (`from policy_loader import load_policy`).
- [ ] **Step 2: failing tests** in `tests/test_skate_gait_report.py` (load the module by path as `tests/test_skate_run_summary.py` does; the test must not need a simulator, so `gait_report.py` must import heavy modules such as the loader only inside `main`, with a comment saying why):
  - `gait_metrics(joint_vel, torque, ground_contact, dt, effort_limit)` takes numpy arrays `[T, 12]`, `[T, 12]`, `[T, 2]` (1.0 = foot on the floor) and returns a dict with exactly these keys:
    - `peak_joint_speed`: the largest per-joint 99th percentile of `|joint_vel|` [rad/s] (per joint, so one fast joint cannot hide among eleven slow ones)
    - `saturated_fraction`: fraction of steps where any joint has `|torque| >= 0.95 * effort_limit`
    - `min_foot_duty`: the smaller of the two feet's fraction of steps on the floor
    - `max_stride_rate`: the larger of the two feet's touchdowns per second (a touchdown is a 0 → 1 change in that foot's contact column; rate = count / (T * dt))
  - Test a calm case: `joint_vel` all 2.0, `torque` all 1.0, both feet in contact for 40 of every 50 steps over 500 steps at `dt=0.02`, expecting `peak_joint_speed == 2.0`, `saturated_fraction == 0.0`, `min_foot_duty == 0.8`, `max_stride_rate == pytest.approx(1.0, abs=0.11)`, and `gait_verdict` all passing.
  - Test a windmill case: one joint at 39.0 rad/s for 5% of steps, one joint at 3.0 Nm for half the steps, foot 0 in contact 1 step of every 16; expect the verdict rows for `peak_joint_speed`, `saturated_fraction`, `min_foot_duty` and `max_stride_rate` to all fail.
  - `GAIT_LIMITS` is exactly `{"peak_joint_speed": ("max", JOINT_SPEED_LIMIT), "saturated_fraction": ("max", 0.05), "min_foot_duty": ("min", 0.4), "max_stride_rate": ("max", 2.0)}`. Because the test cannot import the simulator package cheaply, assert the first limit equals 8.0.
  - `gait_verdict(metrics) -> list[tuple[str, float, str, float, bool]]` rows of `(name, value, "max" | "min", limit, passed)`.
- [ ] **Step 3: implement `gait_report.py`.** Command line: `gait_report.py task policy [--speed 0.4] [--steps 500] [--skip 100]`. It rolls the policy out with `load_policy`, drops the first `--skip` steps (settling), records `robot.data.joint_vel`, `robot.data.actuator_force` and `ground_feet_contact` `found > 0` each step, prints each metric with its limit and PASS or FAIL, then a last line `GAIT: PASS` or `GAIT: FAIL`. Exit code always 0, like `run_summary.py`. Module docstring carries the usage, as the other tools do.
- [ ] **Step 4: run it on the two saved policies** and paste the output into your report:
  - `FORCE_CPU=1 uv run python skateboard/tools/gait_report.py Mjlab-Skate-Ride-Pupper-v3 skateboard/policies/ride_29p4j81t_model_1499.pt --speed 0.4` — expected `GAIT: FAIL` (this is the windmill policy; peak joint speed far above 8 rad/s).
  - `FORCE_CPU=1 uv run python skateboard/tools/gait_report.py Mjlab-Skate-Balance-Pupper-v3 skateboard/policies/balance_2ace1uqa_policy.json` — expected `GAIT: PASS`.
  - Added after the final review: a fifth row, `board_speed_ratio` (mean forward board speed divided by `--speed`, `("min", 0.5)`), and `--speed` turns off standing commands. Deck-height randomization is still plus or minus 1.5 cm; `SIM2REAL.md` says to widen it.
  - If either verdict is not the expected one, report DONE_WITH_CONCERNS with the numbers; do not change `GAIT_LIMITS` to force it.
  - Also confirm the refactored renderer still works: `FORCE_CPU=1 uv run python skateboard/tools/render_policy.py Mjlab-Skate-Balance-Pupper-v3 skateboard/policies/balance_2ace1uqa_policy.json /tmp/sdd_render_check/balance.mp4 --steps 50` writes the mp4 (delete that folder afterwards).
- [ ] **Step 5: training-curve check.** In `run_summary.py`, add to the ride entry of `PASS_CRITERIA`: `"Episode_Reward/joint_speed_limit": -0.1` and `"Episode_Reward/torque_saturation": -0.2` (both are penalties logged as weight × mean cost, so "at least" a small negative number means the penalty is rarely paid). Add a test in `tests/test_skate_run_summary.py` that a ride run with good length and tracking but `Episode_Reward/joint_speed_limit` of -0.5 gets `RESULT: FAIL`, and passes at -0.01 with `torque_saturation` -0.05. Update any existing ride test there that now lacks the two tags.
- [ ] **Step 6:** `FORCE_CPU=1 uv run pytest tests/test_skate_gait_report.py tests/test_skate_run_summary.py tests/test_pupper_skate.py -q` pass; `uv run ruff format`, `uv run ruff check`, `uv run ty check` clean for touched files. Commit `feat: add a gait report tool and gait terms in the ride pass check`.

### Task 18: Notebook and docs for the reshaped ride task

**Files:**
- Modify: `skateboard/tools/make_notebook.py`, `notebooks/CS123_Pupper_Skate.ipynb` (regenerated, never hand-edited), `skateboard/README.md`, `skateboard/CHECKPOINT.md`, `skateboard/PLAN.md` (Task 11 only), `tests/test_skate_run_summary.py` if it checks notebook contents

**Interfaces consumed:** ride reward terms and default weights from Task 16: `push_stride` 0.3, `push_leg_pose` 0.3, `joint_vel_l2` -0.001, `joint_speed_limit` -0.02, `torque_saturation` -0.5; `push_air_time` no longer exists. `skateboard/tools/gait_report.py` from Task 17 prints `GAIT: PASS` or `GAIT: FAIL`.

- [ ] **Step 1: notebook generator.** In `make_notebook.py`:
  - Reward table (section 4): remove the `push_air_time` row; change the `stance_pose` description to `joints near the keyframe (ride: left legs only)` (unchanged) and add rows, Balance column `n/a`: `push_stride` 0.3 "right feet take planted strides: 0.4 s on the floor, then a 0.2 to 0.5 s swing"; `push_leg_pose` 0.3 "right legs stay loosely near the keyframe"; `joint_vel_l2` -1e-3 "joint speed"; `joint_speed_limit` -0.02 "joint speed above 6 rad/s"; `torque_saturation` -0.5 "motors above 80% of their 3 Nm limit".
  - `REWARD_WEIGHTS` cell: remove `"push_air_time"`; add the five terms with those weights under the "Ride only" comment.
  - Section 7 ("Did it pass?"): after the `run_summary.py` cell add a markdown cell and a code cell that run the gait report on the newest checkpoint for the ride task only. Markdown: "### Gait check (ride)\n\nCurves can pass while the gait is one the robot cannot do. This rolls out the newest checkpoint at 0.4 m/s and checks joint speed, torque and how the right feet step. The last line is `GAIT: PASS` or `GAIT: FAIL`. Watch the video too before trusting a pass." Code: find the newest `model_*.pt` under `logs/rsl_rl` (same glob the viewer cell uses for `ckpts`; read that cell) and, `if RIDE:`, run `!python skateboard/tools/gait_report.py {TASK} {ckpt} --speed 0.4`.
  - Regenerate: `python3 skateboard/tools/make_notebook.py notebooks/CS123_Pupper_mjlab.ipynb notebooks/CS123_Pupper_Skate.ipynb`. Confirm the notebook is valid JSON and every code cell parses with `ast.parse` after dropping lines that start with `!` or `%` (if a test already does this, run it).
- [ ] **Step 2: docs.**
  - `skateboard/PLAN.md`, Task 11 only: under the result block add a dated note (Mon Nov 16) that the `29p4j81t` gait is not hardware-feasible (front-right hip 2.3 rad at 3 Hz, peak about 39 rad/s, five joints at the 3 Nm limit), that the line "No reward hacking seen" was wrong because the check never looked at the gait, and that the rewards were reshaped in Task 16 and need a fresh 1500-iteration run. Add pass criteria to Step 2: `Episode_Reward/joint_speed_limit` ≥ -0.1, `Episode_Reward/torque_saturation` ≥ -0.2, and `GAIT: PASS` from `gait_report.py`. In the tuning table replace the `push_air_time` row with `Shuffles or drags the right feet | push_stride 0.3 → 0.6; push_leg_pose 0.3 → 0.1` and add rows `Fast leg flailing (GAIT: FAIL on joint speed) | joint_speed_limit -0.02 → -0.1; joint_vel_l2 -1e-3 → -5e-3`, `Motors pinned at the limit | torque_saturation -0.5 → -1.5`, and `Stands still after the reshaping | push_leg_pose → 0.1; joint_vel_l2 → -2e-4; then track_board_velocity → 6.0`.
  - `skateboard/README.md`: Status line says balance is trained and is the robot candidate, ride must be retrained with the reshaped rewards and the saved ride policy is for reference only; step 3 item 5 also mentions the gait check; step 4 adds the `gait_report.py` command for both saved policies and says the saved ride policy is expected to print `GAIT: FAIL`; the "Where things are" tools row mentions the gait report.
  - `skateboard/CHECKPOINT.md`: read it, then correct any statement that the ride policy is ready for, or next in line for, the robot; say the gait fails the gait check and ride will be retrained first.
  - Plain language, short sentences, no em-dash asides, matching the surrounding docs.
- [ ] **Step 3:** `FORCE_CPU=1 uv run pytest tests/test_skate_run_summary.py tests/test_skate_gait_report.py -q` pass; `uv run ruff format` and `uv run ruff check` clean for `make_notebook.py` (notebook-level ruff errors inherited from the course notebook are known; add no new ones). Commit `docs: notebook and docs for the reshaped ride rewards and gait check`.

### Task 19: Stance audit tool (sim2real check on the pose)

A TA doubted that Pupper can stay balanced in the split stance on the real
board. This tool measures how much margin the stance has in simulation when
the board, the motors and the placement are wrong, so the answer is a table
and not a guess. It does not train anything and changes no task.

**Files:**
- Create: `skateboard/tools/stance_audit.py`
- Modify: `skateboard/tools/policy_loader.py` (one optional keyword)
- Test: `tests/test_skate_stance_audit.py`

**Interfaces:**
- Consumes: `load_policy(task, policy_path, speed, *, render_mode, viewer)`
  from `skateboard/tools/policy_loader.py` (Task 17). A working, unpolished
  prototype of the whole rollout is at
  `.superpowers/sdd/PLAN/audit-prototype.py`; it shows which event names,
  sensors and data fields to use. Port its behaviour, not its style.
- Produces: `support_margin`, `lifted_foot_margins`, `condition_verdict`,
  `CONDITIONS`, `STANCE_LIMITS`, and a command line that ends with
  `STANCE: PASS` or `STANCE: FAIL`.

- [ ] **Step 1: Write the failing tests** in `tests/test_skate_stance_audit.py`
  (load the module by path, the same way `tests/test_skate_gait_report.py`
  does). Pure numpy, no simulator:
  - `support_margin(feet_xy, com_xy)`: `feet_xy` is `(N, 2)`, `com_xy` is
    `(2,)`. Returns the signed distance in metres from the point to the
    nearest edge of the convex hull of the feet, positive inside. Cases: unit
    square `[(0,0),(1,0),(1,1),(0,1)]` with the point `(0.5, 0.5)` gives 0.5;
    `(0.5, 0.1)` gives 0.1; `(0.5, -0.2)` gives -0.2. The result must not
    depend on the order of the feet.
  - `lifted_foot_margins(feet_xy, com_xy, names)`: returns
    `{name: support_margin of the other feet}` for every foot. Case: the unit
    square with the point `(0.4, 0.4)` and names `a, b, c, d` in the order
    above: lifting `a` (the corner at the origin) gives a negative margin
    (about -0.141), lifting `c` gives a positive margin (about +0.141).
  - `condition_verdict(row)` with `STANCE_LIMITS`: a row with `resets` 0,
    `min_deck_contact` 1.0, `max_torque_use` 0.2 and `margin` 0.06 passes;
    changing any one of them to `resets` 1, `min_deck_contact` 0.5,
    `max_torque_use` 0.8 or `margin` 0.01 fails.
  - `STANCE_LIMITS == {"min_deck_contact": 0.9, "max_torque_use": 0.6,
    "margin": 0.03}` and every entry of `CONDITIONS` uses only the override
    keys listed in Step 3.
- [ ] **Step 2: Run them and see them fail.**
  `FORCE_CPU=1 uv run pytest tests/test_skate_stance_audit.py -q`
- [ ] **Step 3: Implement.**
  - `policy_loader.load_policy` gains a keyword `configure: Callable[[Any],
    None] | None = None`, called with the env config after the existing
    edits and before the env is built. Existing callers are unchanged.
  - `stance_audit.py` keeps heavy imports inside `main` (as `gait_report.py`
    does) so the tests stay cheap. Use `scipy.spatial.ConvexHull` (scipy is a
    project dependency) or plain numpy for the hull.
  - Override keys (all optional floats, defaults in brackets): `deck` (0,
    metres added to deck height), `truck` (1, truck stiffness scale), `mu`
    (0.9, deck friction), `kp` (1, position gain scale), `effort` (3.0, N·m),
    `board_x`, `board_y`, `board_yaw`, `robot_yaw` (0; reset pose offsets in
    m and rad), `joint_offset` (0, rad, uniform ± at reset), `push_y` (0, m/s
    added to the robot's sideways velocity at step 200), `board_kick` (0, m/s
    added to the board's forward velocity at step 200), `zero_action` (0; 1
    replaces the policy with zero actions).
  - Every condition pins the other randomized events to fixed values the way
    the prototype does (wheel friction 0.85, board inertia unchanged), so one
    condition changes one thing.
  - `CONDITIONS` (name: overrides), in this order:
    `baseline {}`, `zero_action {zero_action: 1}`, `deck+3cm {deck: 0.03}`,
    `deck-3cm {deck: -0.03}`, `truck_x0.25 {truck: 0.25}`,
    `deck_mu_0.3 {mu: 0.3}`, `kp_x0.6 {kp: 0.6}`, `effort_1.0 {effort: 1.0}`,
    `board_y+4cm {board_y: 0.04}`, `board_y-4cm {board_y: -0.04}`,
    `board_yaw_0.2 {board_yaw: 0.2}`, `joint_offset_0.3 {joint_offset: 0.3}`,
    `push_left_0.6 {push_y: 0.6}`, `push_right_0.6 {push_y: -0.6}`,
    `board_kick_0.8 {board_kick: 0.8}`,
    `worst_combo {deck: 0.02, truck: 0.25, mu: 0.4, kp: 0.7, effort: 2.0}`.
  - Per condition: one env, 500 steps, measure after the first 100. Row
    fields: `resets` (count over the measured steps), `min_deck_contact`
    (lowest per-foot contact share of the two deck feet), `max_torque_use`
    (largest per-joint mean `|actuator_force|` divided by the condition's
    effort limit), `roll_deg` (mean), `board_travel` (m), `margin`
    (`support_margin` of the mean foot positions and mean centre of mass, in
    m), `lift_front_right` and `lift_rear_right` (from
    `lifted_foot_margins`; reported, not part of the verdict).
  - Command line: `stance_audit.py [policy] [--conditions a,b,c]`. `policy`
    defaults to `skateboard/policies/balance_2ace1uqa_policy.json`; the task
    is always `Mjlab-Skate-Balance-Pupper-v3`. An unknown condition name is
    an `ap.error`. Print one line per condition with `PASS`/`FAIL` and the
    row, then `STANCE: PASS` only if every condition run passed.
  - Module docstring: what it is for, the usage line, that a full run takes
    about a minute per condition on a Mac, and that a pass here says nothing
    about the real board's size, grip or the legs' backlash.
- [ ] **Step 4: Run.** Tests pass. Then
  `FORCE_CPU=1 uv run python skateboard/tools/stance_audit.py --conditions baseline,zero_action,effort_1.0,board_kick_0.8`
  and record the output in the report. Expected for `baseline`: no resets,
  margin about 0.066, `lift_front_right` and `lift_rear_right` negative
  (about -0.02 and -0.01). If the numbers disagree with that by a lot, report
  it; do not tune limits.
- [ ] **Step 5:** ruff format, ruff check, ty on the touched files; commit
  `feat: add a stance audit tool for sim2real margin checks`.

## Self-review

- Coverage: stance keyframe (Task 1), modeling (Tasks 1, 3, 10), rewards and terminations (Tasks 2, 3, 5 and the Design tables), randomization (Task 3), training (Tasks 9, 11), export and deploy (Tasks 4, 12). No gaps found.
- Names checked against the code: task ids, reward term names, `SKATE_*`, W&B metric names, `REWARD_WEIGHTS`, `RIDE_SPEED_RANGE`.
- Tasks 1 to 6 point at verified files instead of repeating about 900 lines of code here.

## Accounting

- **Skipped:** code listings for Tasks 1 to 6 (the code is in the repo and tested); a right-side stance; a push/glide phase clock like HUSKY's.
- **Assumed:** left side on the deck; board is 0.10 m tall and 1.8 kg until measured (deck height is randomized ±1.5 cm to cover it); W&B project name `mjlab` as in the course notebook.
- **Issues:** nothing has run on the real robot. The ride run stopped at iteration ~1120 of 1500 (Colab GPU limit), so it has curves but no `RESULT:` verdict. Reward weights are the defaults; they passed, so they were not tuned further. `uv run play` for the skate tasks has not been watched in a viewer.
