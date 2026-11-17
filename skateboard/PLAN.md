b# Pupper Skateboard RL Training Implementation Plan (v8)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** train a policy in sim that holds Pupper centred on the Walmart mini cruiser with all four feet on the deck, then one that reaches a foot down to the floor and pushes the board forward at a commanded speed, and get it onto the real robot by Demo Day (Dec 3).

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
- The stance is centred: all four feet on the deck, body level, board under the middle of the robot. Either side may push. (v1 to v3 used a split stance with the left feet on the deck; Tasks 20 and 21 replace it.)

## What changed from v7

- New Task 26. A study of why the centred ride run (`14m5rs1f`) never pushes
  found a board bug on the way: the deck roll and truck yaw limits were
  compiled as degrees, so the deck could not lean in any run so far. Fix it
  before any more training.
- New Task 27. With the deck free to lean, the centred stance with no policy
  falls in 21 of 64 noisy spawns (5 of 64 before). The cause is the soft end
  of the truck stiffness range, not the keyframe. The hold tests now run at
  nominal stiffness, and training covers 15 to 180 N*m/rad.
- New Task 28. The centred ride run `14m5rs1f` failed: it stands and does not
  push, because standing pays more than a push pose. The centred ride rewards
  now stop charging for one lifted foot and pay for board speed from zero.
- The same study measured that a front-leg push is reachable and holdable from
  the centred stance, and that the ride rewards pay less for the push pose than
  for standing. The ride reward change comes as a later task.

## What changed from v6

- New Task 25. The first centred balance run (`k4kci0oj`) printed `RESULT: PASS`
  but holds four motors at full torque against the joint stops. Both balance
  tasks get a torque-saturation penalty, the balance pass check reads it, and
  the notebook runs the gait check for balance too.

## What changed from v5

- Team decision 2026-11-16: keep two stances and test both on the robot.
  - **Centred** (Tasks 20 to 22): all four feet on the deck. Task ids `Mjlab-Skate-Balance-Pupper-v3` and `Mjlab-Skate-Ride-Pupper-v3`.
  - **Split** (v1 to v3, brought back): left feet on the deck, right feet on the floor. New task ids `Mjlab-SkateSplit-Balance-Pupper-v3` and `Mjlab-SkateSplit-Ride-Pupper-v3`.
- The split stance is refit to the measured board (deck top 10.3 cm, deck 16.1 cm wide) and uses the gait rewards written on 2026-11-16 that were never trained (`push_stride`, `push_leg_pose`, `joint_speed_limit`, `torque_saturation`).
- Task 23 adds the split tasks. Task 24 makes the tools, the notebook and the docs work for both stances.

## What changed from v4

- The final review of v4 (2026-11-16) found the ride task could not learn to push. Measured in sim: the leg reaches 19.3 cm below the base, and the base settles 18.8 cm above the floor on a 6.8 cm deck, 22.4 cm on the stock 10.3 cm deck and 23.4 cm at 11.3 cm. So from the centred stance a foot reaches the floor only on the lowered trucks, and only if the body crouches.
- Nothing rewarded moving a foot toward the floor. Task 22 adds a reward for that, pays for a push only while the board is slower than commanded, and trains ride on the lowered deck only (6.8 to 8.3 cm).
- Balance is unchanged and still covers 6.8 to 11.3 cm.
- The gait check now counts floor strokes, and the notebook always pulls the latest code.

## What changed from v3

- The board is measured (Alexander, 2026-11-16): deck top 103 mm off the floor, deck 161 mm wide and 23 in (584 mm) long, wheel radius 29 mm, track 104 mm, wheelbase 14 3/8 in (365 mm, less accurate). Mass is not measured yet and stays at the 1.8 kg guess.
- The stance is now centred, as the team asked: all four feet on the deck, body level. The split stance is gone. Tasks 20 and 21 do this; where they disagree with the Design tables below, Tasks 20 and 21 win.
- Alexander is modelling lower trucks (deck 2 to 3 cm lower). Deck height is randomized from 6.8 to 11.3 cm to cover both the stock and the lowered board.
- What the prototypes showed: Pupper's default stance is 20 cm wide and the deck is 16.1 cm, so the feet come in to 6.6 cm (front) and 6.0 cm (back) each side, 1.5 to 2 cm inside the deck edge. A first try at 7.2 cm fell off the edge whenever the robot spawned 3 mm off-centre. The stance holds with zero actions on both deck heights. From it, a foot cannot reach the floor on the stock trucks without the body dropping or rolling (the floor is 18.6 cm below the base; the leg reaches about 17 cm). On a 7 cm deck it reaches. So the lower trucks are what make pushing comfortable.

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
- Board randomization (mass, friction, bushing stiffness) is in. Deck-height randomization is in too (Task 14); widened from ±1.5 cm to ±3 cm on 2026-11-16 after the stance audit (`SIM2REAL.md`).
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
- Produces: event term names `deck_height`, `push_robot`, `nudge_board`, and module constants `DECK_HEIGHT_RANGE = (-0.03, 0.03)` (was ±0.015 until 2026-11-16), `PUSH_INTERVAL_S = (2.0, 4.0)`.

Exact values:

| Event name | Mode | Function | Target | Values |
| --- | --- | --- | --- | --- |
| `deck_height` | `startup` | `dr.geom_pos` | board geom `DECK_GEOM_NAME`, z axis only, operation `add` | `DECK_HEIGHT_RANGE = (-0.03, 0.03)` m |
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
  - Added after the final review: a fifth row, `board_speed_ratio` (mean forward board speed divided by `--speed`, `("min", 0.5)`), and `--speed` turns off standing commands. Deck-height randomization was widened to plus or minus 3 cm afterwards.
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

### Task 20: Measured board, centred stance, two-sided pushing

The team measured the board and asked for a centred stance (all four feet on
the deck, either side may push) instead of the split stance. This task changes
the board model, the keyframe, and both skate tasks. The numbers below come
from prototypes that were run in simulation; use them as given.

**Files:**
- Modify: `src/mjlab/asset_zoo/robots/mini_cruiser/skateboard_constants.py`
- Modify: `src/mjlab/asset_zoo/robots/pupper_v3/pupper_constants.py` (`SKATE_*`)
- Modify: `src/mjlab/tasks/pupper_skate/pupper_skate_env_cfg.py`
- Modify: `src/mjlab/tasks/pupper_skate/mdp/rewards.py`
- Modify: `skateboard/tools/solve_stance.py`, `skateboard/tools/check_stance.py`
- Test: `tests/test_pupper_skate.py`, `tests/test_pupper_skateboard.py`

**Interfaces:**
- Produces (Task 21 relies on these names): sensors `deck_feet_contact` and
  `ground_feet_contact`, each now covering all four feet in `FOOT_SITE_NAMES`
  order (FR, FL, BR, BL), both with `track_air_time=True`; sensor
  `knees_on_deck`; reward terms `feet_on_deck`, `feet_on_ground`,
  `knees_on_deck`, `tilt`, `stance_pose`, `board_alignment`, `deck_flat`,
  `board_still` (balance only), `push_stroke` (ride only); constant
  `DECK_HEIGHT_RANGE = (-0.035, 0.01)`.
- Removed: `DECK_SIDE`, `GROUND_SIDE`, sensors and rewards
  `deck_feet_on_ground` and `ground_feet_on_deck`, rewards `base_height`,
  `push_leg_pose`, `push_stride` (function, term and test).

- [ ] **Step 1: board model.** In `SkateboardParams` set the defaults to the
  measured values and say in the comments which are measured:
  `deck_length=0.584`, `deck_width=0.161`, `wheel_radius=0.029`,
  `deck_top_height=0.103`, `wheelbase=0.365` (caliper too short, less
  accurate), `track=0.104`. Leave `mass=1.8` and the truck and wheel values
  alone, marked as guesses. Update `tests/test_pupper_skateboard.py` only where
  it hard-codes an old number; all five board tests must still pass.
- [ ] **Step 2: keyframe.** In `pupper_constants.py`:

  ```python
  # Centred stance: all four feet on the deck and 0.115 m below the base. The
  # front feet sit 0.066 m either side of the centre line and 0.02 m behind
  # their default spot, the back feet 0.060 m. The deck is 0.161 m wide, so
  # the feet have 1.5 to 2 cm to the edge. Any wider and spawn noise drops a
  # foot off the edge.
  SKATE_POSE: tuple[float, ...] = (
    -0.138, -0.299, -0.107,
    0.138, 0.299, 0.107,
    -0.002, -0.264, -0.045,
    0.002, 0.264, 0.045,
  )  # fmt: skip
  # base_link height with the feet just touching a 0.103 m deck. The legs
  # give about 2 cm once loaded.
  SKATE_HEIGHT: float = 0.238
  SKATE_ROLL: float = 0.0  # the centred stance is level
  SKATE_BOARD_OFFSET: tuple[float, float] = (-0.01, 0.0)
  ```

  `SKATE_KEYFRAME` keeps its form. Fix the comments around these constants.
- [ ] **Step 3: stance tools.** `solve_stance.py` takes `FOOT_Y DEPTH` (both
  metres, positive) and solves all four legs to put each foot at its default
  x plus a shift, `±FOOT_Y` sideways and `DEPTH` below the base, printing the
  12 angles in `SKATE_POSE` order and each joint's distance to its limit.
  Front and back legs may differ: options `--back-y` (default `FOOT_Y`) and
  `--front-dx` (default 0, metres forward).
  `solve_stance.py 0.066 0.115 --back-y 0.060 --front-dx -0.02` must print
  the pose of Step 2 to within 0.005 rad. `check_stance.py` keeps its arguments and gains `--deck-height METRES`
  (default: the model's). Both `check_stance.py 0.0 0.248 <pose>` and the same
  with `--deck-height 0.068` must end upright (|roll| and |pitch| under 0.1
  rad after 5 s). Update both docstrings.
- [ ] **Step 4: tests first** in `tests/test_pupper_skate.py`. Replace the
  split-stance tests; keep every test that still applies. Required behaviour:
  - the keyframe puts all four feet over the deck: each foot centre's |y|
    from the board centre is at least 0.01 m inside half the deck width, and
    the pose is inside the joint limits with at least 0.1 rad to spare;
  - with zero actions for 5 s and a clean spawn (no reset noise, nominal
    deck) the balance env does not reset, all four `deck_feet_contact` flags
    are on, no `ground_feet_contact` flag is on, and |roll| and |pitch| stay
    under 0.1 rad;
  - with zero actions for 5 s and the env's normal reset noise, at least 14
    of 16 envs never reset (a prototype got 15 to 16);
  - a foot held off the deck for longer than the grace period ends the
    balance episode (keep the existing test's method, any one foot);
  - `push_stroke` pays `min(last floor contact / stroke_s, 1) / step_dt` on
    the step a foot leaves the floor, nothing on other steps, nothing below
    the command threshold (same fake-sensor style as the old `push_stride`
    test, which is deleted);
  - the ride cfg has `push_stroke`, has no `board_still`, `feet_on_ground`
    or `nudge_board`, and its `feet_off_deck` grace is 1.5 s;
  - `DECK_HEIGHT_RANGE == (-0.035, 0.01)`.
- [ ] **Step 5: env.** In `pupper_skate_env_cfg.py`:
  - Sensors: `deck_feet_contact` = all four foot geoms
    (`^leg_(front|back)_(r|l)_3_collision$`) against the deck, air time
    tracked; `ground_feet_contact` = the same feet against the floor, air time
    tracked; `knees_on_deck` unchanged. Delete the two wrong-surface sensors.
  - Balance rewards: `feet_on_deck` weight 1.0 (so 4.0 when all four are
    down); `feet_on_ground` weight -1.0 (a foot on the floor is wrong in
    balance); `knees_on_deck` -1.0; `tilt` target `(0.0, 0.0)`, std 0.2;
    `stance_pose` unchanged; `board_alignment` unchanged apart from the new
    `SKATE_BOARD_OFFSET`; `deck_flat`, `board_still`, `action_rate_l2`,
    `joint_torques_l2`, `termination` unchanged. Delete `base_height`: the
    deck height is randomized over 4.5 cm, so a fixed world height is wrong,
    and `stance_pose` already fixes the height above the deck.
  - `DECK_HEIGHT_RANGE = (-0.035, 0.01)` with a comment: nominal is the
    measured 0.103 m; the range is 6.8 to 11.3 cm, which covers the stock
    trucks and the planned trucks that sit 2 to 3 cm lower.
  - Terminations unchanged except `fell_over` limit 0.5 (the stance is level
    now) and the comment above them.
  - Ride: `feet_on_deck` weight 0.5; delete `feet_on_ground` and
    `board_still`; `feet_off_deck` grace 1.5 s (a push takes a foot off the
    deck for about a second); `stance_pose` on all joints with weight 0.5 and
    std 0.6 (loose enough for one leg to stride); add

    ```python
    r["push_stroke"] = RewardTermCfg(
      func=mdp.push_stroke,
      weight=0.3,
      params={
        "sensor_name": "ground_feet_contact",
        "stroke_s": 0.25,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    )
    ```

    Keep `track_board_velocity`, `board_yaw_rate`, `joint_vel_l2`,
    `joint_speed_limit`, `torque_saturation` as they are.
  - `mdp.push_stroke(env, sensor_name, stroke_s, command_name,
    command_threshold)` in `rewards.py` replaces `push_stride`: on the step a
    foot's `current_air_time` becomes positive (it is in `(0, step_dt + 1e-6)`),
    score `clamp(last_contact_time / stroke_s, max=1)`; sum over feet, divide
    by `env.step_dt`, zero when the commanded speed is at or below the
    threshold.
  - Rewrite the module docstring and the comments that describe the split
    stance.
- [ ] **Step 6: verify.**
  `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py tests/test_pupper_skateboard.py tests/test_skate_*.py -q`
  must pass. If a `tests/test_skate_*.py` tool test breaks only because of a
  renamed or removed name, make the smallest fix in that tool and say so in
  the report (Task 21 does the real tool work). Also run the balance env with
  zero actions at the two ends of the deck range (deck geom moved by -0.035
  and +0.01) for 5 s each and put the settled roll, pitch, base height and
  contact flags in the report. If either end falls over, stop and report
  BLOCKED with the numbers; do not retune. `test_leaning_right_steers_right`
  misses its tolerance by 0.3 mm with the measured track: fix the test so it
  still proves the steering direction and size with honest margin (for
  example a longer roll), and say what you changed.
- [ ] **Step 7:** ruff format, ruff check and ty on the touched files
  (`make check` also runs pyright, which fails on `notebooks/watch_live.py`,
  a file outside this work); commit
  `feat: centre the skate stance on the measured board`.

### Task 21: Tools, notebook and docs for the centred stance

Task 20 changed the stance and the reward names. This task brings the tools,
the Colab notebook and the docs in line. Nothing here changes training.

**Files:**
- Modify: `skateboard/tools/gait_report.py`, `stance_audit.py`,
  `run_summary.py`, `make_notebook.py`
- Regenerate: `notebooks/CS123_Pupper_Skate.ipynb` (run `make_notebook.py`;
  never edit the notebook by hand)
- Modify: `skateboard/README.md`, `skateboard/SIM2REAL.md`, `README.md`
  (the two skate task rows)
- Test: `tests/test_skate_gait_report.py`, `tests/test_skate_stance_audit.py`,
  `tests/test_skate_run_summary.py`, `tests/test_skate_notebook.py` if present

**Interfaces:**
- Consumes from Task 20: sensors `deck_feet_contact` and
  `ground_feet_contact` (four feet each, order FR, FL, BR, BL); rewards
  `feet_on_deck` (weight 1.0 balance, 0.5 ride), `feet_on_ground` (-1.0,
  balance only), `knees_on_deck`, `tilt`, `stance_pose`, `board_alignment`,
  `deck_flat`, `board_still` (balance only), `push_stroke` (0.3, ride only);
  no `base_height`, `push_stride`, `push_leg_pose`, `deck_feet_on_ground`,
  `ground_feet_on_deck`; `DECK_HEIGHT_RANGE = (-0.035, 0.01)`.

- [ ] **Step 1: `run_summary.py`.** Balance passes at
  `Episode_Reward/feet_on_deck >= 3.6` (four feet, was 1.8 for two). Ride
  criteria unchanged. Update its test.
- [ ] **Step 2: `gait_report.py`.** Read `deck_feet_contact` (four columns)
  instead of `ground_feet_contact`. `min_foot_duty` becomes the smallest
  share of time any foot spends on the deck and `max_stride_rate` the most
  often any foot lands on the deck; limits stay 0.4 and 2.0. Add a printed
  line (not part of the verdict) with each foot's share of time on the floor
  from `ground_feet_contact`, so a reader sees which foot pushes. Update the
  docstrings, the `[T, 2]` shapes and the tests.
- [ ] **Step 3: `stance_audit.py`.** A row passes when all four feet are on
  the deck at least 90% of the time (was the two left feet); the other pass
  limits stay. Report the three-foot margins for all four feet (lift FR, FL,
  BR, BL). Deck-height conditions become offsets that fit the new range:
  -0.035, -0.02, +0.01, +0.02, +0.03. Board sideways offsets become ±0.01
  and ±0.02 m (the feet are 1.5 to 2 cm inside the deck edge). Update the
  tests.
- [ ] **Step 4: notebook.** In `make_notebook.py` make the reward table, the
  `REWARD_WEIGHTS` cell and the "what to watch" text match Task 20 exactly
  (names and weights from the Interfaces block; for anything not listed, read
  `pupper_skate_env_cfg.py`). Describe the stance as centred. Regenerate the
  notebook and check that every key of `REWARD_WEIGHTS` exists in the env's
  reward dict for the matching task (add this as a test if there is not one).
- [ ] **Step 5: docs.**
  - `skateboard/README.md`: board table with the measured values (deck top
    103 mm, width 161 mm, length 584 mm, wheel radius 29 mm, track 104 mm,
    wheelbase 365 mm marked less accurate, mass 1.8 kg marked guess); stance
    described as centred; tool usage lines that changed.
  - `README.md`: the two skate rows say centred stance / push with one foot.
    If `CLAUDE.md` at the repo root is a separate file and not a link to
    `README.md`, make the same change there.
  - `skateboard/SIM2REAL.md`: add a dated section at the top, "Update
    2026-11-16: centred stance", saying: the stance is now centred on the
    measured board; the results table below is for the old split stance and
    has not been rerun; the feet sit 1.5 to 2 cm inside the deck edge
    (front feet 6.6 cm and back feet 6.0 cm from the centre line, deck half
    width 8.05 cm), so sideways placement is the new thing to check; with
    no policy and the usual spawn noise about 9% of spawns tip over, so the
    trained policy has to correct a small sideways error; a deck plate about
    20 cm wide would remove the problem; from this stance a foot
    cannot reach the floor on the stock trucks (floor 18.6 cm below the base,
    leg reach about 17 cm) unless the body drops or rolls, and it can on a
    7 cm deck. Plain language, short sentences, no em dashes.
- [ ] **Step 6:** run
  `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py tests/test_pupper_skateboard.py tests/test_skate_*.py -q`,
  then ruff format, ruff check and ty on the touched files (`make check`
  also runs pyright, which fails on `notebooks/watch_live.py`, a file
  outside this work); commit
  `feat: update skate tools, notebook and docs for the centred stance`.

### Task 22: Make pushing learnable from the centred stance

The final review showed the ride task as built in Task 20 has no path to a
push: the floor is out of reach on the stock deck, and no reward points a foot
toward the floor. This task changes the ride task only. Balance is untouched.

**Files:**
- Modify: `src/mjlab/tasks/pupper_skate/mdp/rewards.py`,
  `src/mjlab/tasks/pupper_skate/mdp/__init__.py` (if it lists names),
  `src/mjlab/tasks/pupper_skate/pupper_skate_env_cfg.py`
- Modify: `skateboard/tools/gait_report.py`, `skateboard/tools/make_notebook.py`
- Regenerate: `notebooks/CS123_Pupper_Skate.ipynb`
  (`python3 skateboard/tools/make_notebook.py notebooks/CS123_Pupper_mjlab.ipynb notebooks/CS123_Pupper_Skate.ipynb`)
- Modify: `skateboard/README.md` (ride reward table or list, deck range for
  ride), `skateboard/SIM2REAL.md` (one bullet in the 2026-11-16 update)
- Test: `tests/test_pupper_skate.py`, `tests/test_skate_gait_report.py`,
  `tests/test_skate_notebook.py`

**Interfaces:**
- Consumes: `ground_feet_contact` and `deck_feet_contact` (four feet, order
  FR, FL, BR, BL), `FOOT_SITE_NAMES`, the `twist` command (column 0 is the
  commanded board speed), `mdp.track_board_velocity` (shows how the board's
  forward speed is read: `board.data.root_link_lin_vel_b[:, 0]`),
  `DECK_HEIGHT_RANGE`, the `deck_height` event.
- Produces: reward functions `foot_reach` and the changed `push_stroke`;
  constant `RIDE_DECK_HEIGHT_RANGE`; ride reward names `foot_reach` and
  `push_stroke`.

- [ ] **Step 1: `foot_reach` reward** in `mdp/rewards.py`:

  ```python
  def foot_reach(
    env: ManagerBasedRlEnv,
    std: float,
    foot_radius: float,
    command_name: str,
    command_threshold: float,
    speed_fraction: float,
    foot_cfg: SceneEntityCfg,
    board_cfg: SceneEntityCfg = SceneEntityCfg("board"),
  ) -> torch.Tensor:
  ```

  It returns `exp(-clearance / std)` where `clearance` is the height of the
  lowest foot's underside above the floor, clamped at 0: the smallest world z
  over the four foot sites, minus `foot_radius`. The floor is the plane z = 0.
  It is zero unless the commanded speed is above `command_threshold` and the
  board's forward speed is below `speed_fraction` times the commanded speed.
  A foot on the deck is at least a deck height up, so this pays almost nothing
  until a foot goes over the edge and down. Resolve the foot sites the way the
  existing code in this package reads foot site positions (look for
  `FOOT_SITE_NAMES` and `site_pos_w`); `foot_cfg` is
  `SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)` with whatever
  `preserve_order` setting that existing code uses.
- [ ] **Step 2: gate `push_stroke` on board speed.** Add parameters
  `speed_fraction: float` and `board_cfg: SceneEntityCfg =
  SceneEntityCfg("board")`. The reward is also zero when the board's forward
  speed is at or above `speed_fraction` times the commanded speed. Update the
  docstring. This stops a policy collecting stroke pay by tapping while the
  board already rolls fast enough.
- [ ] **Step 3: ride config** in `pupper_skate_ride_env_cfg`:
  - new module constant next to `DECK_HEIGHT_RANGE`:

    ```python
    # Ride trains on the lowered trucks only (deck top 6.8 to 8.3 cm). The leg
    # reaches 19.3 cm below the base; on the stock 10.3 cm deck the floor is
    # 22 cm below it, so a push from the centred stance is out of reach there.
    RIDE_DECK_HEIGHT_RANGE: tuple[float, float] = (-0.035, -0.02)
    ```

    and the ride cfg sets the `deck_height` event's `ranges` to it (read how
    the balance cfg passes `DECK_HEIGHT_RANGE` and mirror it; the balance cfg
    keeps `DECK_HEIGHT_RANGE`);
  - `r["foot_reach"]` with weight 0.5 and params `std=0.03`,
    `foot_radius=0.02`, `command_name="twist"`, `command_threshold=0.1`,
    `speed_fraction=0.8`, `foot_cfg` as in Step 1;
  - `push_stroke` gains `speed_fraction=1.0`; its weight stays 0.3;
  - a comment above the two terms saying what each is for, in plain words.
- [ ] **Step 4: tests** in `tests/test_pupper_skate.py` (plain functions,
  follow the style of the existing `push_stroke` test, which builds a small
  fake env):
  - `foot_reach`: lowest foot at z = `foot_radius` pays 1.0; at
    `foot_radius + 0.03` pays `exp(-1)`; a foot below the radius does not pay
    more than 1.0; zero when the command is at the threshold; zero when the
    board speed is at or above 0.8 of the command; only the lowest foot
    counts;
  - `push_stroke`: the existing rows still pass with a slow board; a row with
    the board at the commanded speed pays zero;
  - ride cfg: `foot_reach` weight 0.5 and its params; `push_stroke`
    `speed_fraction` 1.0; the ride `deck_height` range is
    `RIDE_DECK_HEIGHT_RANGE` and the balance one is still
    `DECK_HEIGHT_RANGE`;
  - an env-level check on the ride task with one env, clean spawn, deck offset
    -0.035 and zero actions for 1 s: `foot_reach` is below 0.2 (all feet on
    the deck) and the env does not reset.
- [ ] **Step 5: `gait_report.py`.**
  - `min_foot_duty`: the smallest share of time any foot is in contact with
    the deck or the floor (a pushing foot is on the floor for part of the
    cycle). Limit stays 0.4.
  - `max_stride_rate`: the most often any foot lands on the floor, per second
    (touchdowns from `ground_feet_contact`). Limit stays 2.0.
  - `gait_metrics` therefore takes both contact arrays (`[T, 4]` each). Update
    `main`, the docstrings and the tests; add a test where a foot taps the
    floor 4 times a second and fails `max_stride_rate`, and one where a foot
    alternates deck and floor and passes `min_foot_duty`.
- [ ] **Step 6: notebook.** In `make_notebook.py`:
  - the setup cell must update an existing clone instead of skipping it.
    After the generator's existing `REPO_URL` replacement, replace the two
    clone lines so the cell reads:

    ```python
    if not os.path.exists("/content/pupper-mjlab"):
      !git clone --branch {BRANCH} --depth 1 {REPO_URL} /content/pupper-mjlab
    else:
      !git -C /content/pupper-mjlab fetch --depth 1 origin {BRANCH}
      !git -C /content/pupper-mjlab reset --hard FETCH_HEAD
    ```

    with an assert that the replacement happened;
  - add `foot_reach` (0.5) to the ride reward table and the ride part of the
    `REWARD_WEIGHTS` cell, and say in the "what to watch" text that
    `Episode_Reward/push_stroke` and `Episode_Reward/foot_reach` should rise
    above zero within about 300 iterations, and that a ride run where both
    stay at zero should be stopped;
  - regenerate the notebook; `tests/test_skate_notebook.py` must still pass.
- [ ] **Step 7: docs.** `skateboard/README.md`: ride trains on a 6.8 to 8.3 cm
  deck (lowered trucks), balance on 6.8 to 11.3 cm; list `foot_reach`.
  `skateboard/SIM2REAL.md`: in the "Update 2026-11-16" section add that the
  leg reaches 19.3 cm below the base, the base sits 18.8 cm above the floor on
  a 6.8 cm deck and 22.4 cm on the stock deck, so riding from the centred
  stance needs the lowered trucks; update its accounting section. Plain
  language, short sentences, no em dashes.
- [ ] **Step 8:** run
  `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py tests/test_pupper_skateboard.py tests/test_skate_*.py -q`,
  then ruff format, ruff check and ty on the touched files; commit
  `feat: reward reaching for the floor in the ride task`.

### Task 23: Split-stance tasks on the measured board

Bring the split stance back as a second pair of tasks next to the centred
ones. The split code as last written is at git commit `a110b84` (read old
files with `git show a110b84:<path>`). The centred tasks must not change.

**Files:**
- Create: `src/mjlab/tasks/pupper_skate/pupper_skate_split_env_cfg.py`
- Create: `skateboard/tools/solve_split_stance.py`,
  `skateboard/tools/check_split_stance.py`
- Modify: `src/mjlab/asset_zoo/robots/pupper_v3/pupper_constants.py`,
  `src/mjlab/tasks/pupper_skate/mdp/rewards.py`,
  `src/mjlab/tasks/pupper_skate/mdp/__init__.py`,
  `src/mjlab/tasks/pupper_skate/__init__.py` (task registration)
- Test: create `tests/test_pupper_skate_split.py`

**Interfaces:**
- Consumes: the measured board in `skateboard_constants.py` (unchanged),
  `DECK_HEIGHT_RANGE`, `JOINT_SPEED_LIMIT`, `RIDE_SPEED_RANGE` and
  `PUSH_INTERVAL_S` from `pupper_skate_env_cfg.py` (import them, do not copy).
- Produces:
  - constants `SKATE_SPLIT_POSE` (12 joint angles, FR, FL, BR, BL order),
    `SKATE_SPLIT_HEIGHT`, `SKATE_SPLIT_ROLL`, `SKATE_SPLIT_BOARD_OFFSET` and
    `SKATE_SPLIT_KEYFRAME` in `pupper_constants.py`;
  - `pupper_skate_split_balance_env_cfg(play: bool = False)` and
    `pupper_skate_split_ride_env_cfg(play: bool = False)`;
  - task ids `Mjlab-SkateSplit-Balance-Pupper-v3` and
    `Mjlab-SkateSplit-Ride-Pupper-v3`, registered the same way as the centred
    ids, with the same agent (PPO) config;
  - contact sensors named exactly as at `a110b84` (two deck feet, the left
    ones, and two ground feet, the right ones);
  - reward functions `push_stride` and `push_leg_pose` (and any other reward
    function the `a110b84` split config used that no longer exists), restored
    under their old names next to the centred ones.

- [ ] **Step 1: restore the stance tools.** `solve_split_stance.py` and
  `check_split_stance.py` are the `a110b84` versions of `solve_stance.py` and
  `check_stance.py`, changed only to read the `SKATE_SPLIT_*` constants and
  the current board model. Do not change the centred tools.
- [ ] **Step 2: refit the stance.** The `a110b84` stance was solved for a
  10.0 cm deck guess. Solve it again for the measured board: deck top 0.103 m
  above the floor, deck 0.161 m wide. Requirements for the result:
  - left feet (FL, BL) on the deck, right feet (FR, BR) on the floor;
  - each left foot centre at least 2.5 cm inside the deck's left and right
    edges, and as close to the board's centre line as the legs allow, so the
    robot's weight does not steer the trucks;
  - `check_split_stance.py` holds the pose with zero actions for 10 s: no
    reset, both left feet on the deck and both right feet on the floor at
    least 95% of the time, deck roll under 0.05 rad at the end.
  Write the solved numbers into the `SKATE_SPLIT_*` constants with a comment
  giving the solver command that reproduces them.
- [ ] **Step 3: the env config.** `pupper_skate_split_env_cfg.py` is the
  `a110b84` version of `pupper_skate_env_cfg.py` with these changes only:
  - functions renamed as in Interfaces; it uses `SKATE_SPLIT_*`;
  - shared constants imported from `pupper_skate_env_cfg.py`;
  - the deck height event uses `DECK_HEIGHT_RANGE` (now 6.8 to 11.3 cm);
  - anything that no longer exists in the code base (event or reward
    functions the centred work renamed or changed) is adapted to the current
    function, keeping the `a110b84` behaviour. `push_stroke` now takes
    `speed_fraction`; the split ride task keeps `push_stride` as it was and
    does not use `push_stroke` or `foot_reach`.
  Keep every reward name and weight of the `a110b84` balance and ride
  configs. List them in the report.
- [ ] **Step 4: tests** in `tests/test_pupper_skate_split.py`, plain
  functions. Port the `a110b84` tests of `tests/test_pupper_skate.py` that
  cover the stance, the sensors, the balance and ride reward names and
  weights, `push_stride`, `push_leg_pose` and the terminations. Add:
  - both split ids are registered and build; both centred ids still build;
  - with zero actions and no randomization the split balance env runs 10 s
    without a reset at deck offsets -0.035, 0.0 and +0.01;
  - with the usual spawn noise, 64 envs, 10 s, zero actions: at most 20% of
    envs reset. If this cannot be met, do not loosen it: report the number.
  - the centred configs are unchanged: `pupper_skate_balance_env_cfg()` and
    `pupper_skate_ride_env_cfg()` have the same reward names and weights as
    before this task (compare against a literal dict in the test).
- [ ] **Step 5:** run
  `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py tests/test_pupper_skate_split.py tests/test_pupper_skateboard.py tests/test_skate_*.py -q`,
  then ruff format, ruff check and ty on the touched files; commit
  `feat: bring back the split stance as a second pair of skate tasks`.

### Task 24: Tools, notebook and docs for both stances

**Files:**
- Modify: `skateboard/tools/run_summary.py`, `gait_report.py`,
  `stance_audit.py`, `render_policy.py` (only if it needs it),
  `policy_loader.py` (only if it needs it), `make_notebook.py`
- Regenerate: `notebooks/CS123_Pupper_Skate.ipynb`
- Modify: `skateboard/README.md`, `README.md` (task table: two new rows)
- Test: `tests/test_skate_run_summary.py`, `tests/test_skate_gait_report.py`,
  `tests/test_skate_stance_audit.py`, `tests/test_skate_notebook.py`

**Interfaces:**
- Consumes: the four task ids; the split sensors and reward names from
  Task 23 (read `pupper_skate_split_env_cfg.py`).
- Produces: every tool takes any of the four task ids and picks the stance
  from the id (`"SkateSplit"` in the id means split).

- [ ] **Step 1: `run_summary.py`.** Add pass criteria for the two split ids,
  equal to the criteria the tool had at `a110b84` for balance and ride. The
  centred criteria do not change.
- [ ] **Step 2: `gait_report.py`.** For a split task the metrics are the
  `a110b84` ones (deck feet are the two left feet, pushing feet are the two
  right feet). For a centred task nothing changes. Keep one `GAIT_LIMITS`
  table. Tests for both.
- [ ] **Step 3: `stance_audit.py`.** Add `--task` (default
  `Mjlab-Skate-Balance-Pupper-v3`). For a split task use the `a110b84` pass
  rule (left feet on the deck 90% of the time) and its lift rows (FR, RR);
  deck offsets -0.035, -0.02, +0.01, +0.02, +0.03 for both stances. Tests for
  both.
- [ ] **Step 4: notebook.** The settings cell lists all four task ids in a
  comment, one line each saying what it trains. The reward tables and the
  `REWARD_WEIGHTS` cell cover all four tasks (pick by `TASK`). The gait cell
  runs for both ride tasks. Regenerate; add a test that the `REWARD_WEIGHTS`
  cell's names and weights equal the config's for each of the four tasks.
- [ ] **Step 5: docs.** `skateboard/README.md`: a short table of the two
  stances (what stands where, which task ids, which needs the lowered trucks
  to ride), and the commands with the split ids. Root `README.md`: two rows
  in the task table. Plain language, short sentences, no em dashes.
- [ ] **Step 6:** run the Task 23 Step 5 test command, ruff format, ruff
  check and ty on the touched files; commit
  `feat: skate tools and notebook for both stances`.

### Task 25: Stop balance policies from leaning on the joint stops

**Why:** the first centred balance run (`k4kci0oj`, `RESULT: PASS`) fails the gait
check. All four `_2` joints sit on their joint limit (0.42 rad) and their motors
push at the full 3.0 N·m the whole time (`saturated_fraction = 1.000`). With zero
actions the same stance holds at 17% torque. The policy found that jamming the
legs against the stops makes it stiff. A real motor cannot hold stall torque for
long, so this policy must not go on the robot. The balance tasks have no penalty
that stops it: `joint_torques_l2` at -2e-4 costs 0.007 per step.

**Files:**
- Modify: `src/mjlab/tasks/pupper_skate/pupper_skate_env_cfg.py`
- Modify: `src/mjlab/tasks/pupper_skate/pupper_skate_split_env_cfg.py`
- Modify: `skateboard/tools/run_summary.py`
- Modify: `skateboard/tools/make_notebook.py`, then regenerate
  `notebooks/CS123_Pupper_Skate.ipynb`
- Modify: `skateboard/README.md`
- Test: `tests/test_pupper_skate.py`, `tests/test_pupper_skate_split.py`, the
  run summary and notebook tests that already exist

**Interfaces:**
- Consumes: `mdp.torque_saturation(env, limit, soft_ratio=0.8, asset_cfg)` in
  `src/mjlab/tasks/pupper_skate/mdp/rewards.py` (unchanged), `PUPPER_EFFORT_LIMIT`.
- Produces: a module constant `BALANCE_TORQUE_SOFT_RATIO = 0.5` in
  `pupper_skate_env_cfg.py`, imported by the split config; a reward term named
  `torque_saturation` in both balance tasks.

**Changes:**
- [ ] Both balance reward dicts (centred and split) get
  `"torque_saturation": RewardTermCfg(func=mdp.torque_saturation, weight=-1.0,
  params={"limit": PUPPER_EFFORT_LIMIT, "soft_ratio": BALANCE_TORQUE_SOFT_RATIO})`.
  The penalty starts at 1.5 N·m per motor. Standing with zero actions peaks at
  about 0.5 N·m, so an honest stance pays nothing.
- [ ] Both ride tasks keep their own `torque_saturation` exactly as it is now
  (weight -0.5, default `soft_ratio` 0.8): the ride config already assigns the
  term after copying the balance rewards, so it must still end with those values.
  No other reward name or weight changes in any of the four tasks.
- [ ] `run_summary.py`: a balance run (either stance) also needs
  `Episode_Reward/torque_saturation >= -0.05` (last-20 average) to print
  `RESULT: PASS`. A run whose log has no such key fails that row with a line
  that says the key is missing (old runs predate the term).
- [ ] Notebook: the reward tables and `REWARD_WEIGHTS` for both balance tasks
  list `torque_saturation` (the existing test that the notebook equals the
  config must pass). The gait check cell runs for all four tasks: with
  `--speed 0.4` for the two ride tasks and with no `--speed` for the two
  balance tasks. The W&B cell is not touched.
- [ ] `skateboard/README.md`: one short bullet under the balance task saying
  the balance tasks penalize motors above half their torque limit, and why
  (run `k4kci0oj` leaned on the joint stops).

**Tests (write first, see them fail):**
- [ ] Each balance config has `torque_saturation` with weight -1.0, func
  `mdp.torque_saturation`, `limit == PUPPER_EFFORT_LIMIT`, `soft_ratio == 0.5`.
- [ ] Each ride config has `torque_saturation` with weight -0.5 and no
  `soft_ratio` param (or 0.8).
- [ ] Update the literal centred reward dict pinned in
  `tests/test_pupper_skate_split.py` to include the new balance term; every
  other pinned value stays.
- [ ] Sim test, both balance tasks: 10 s of zero actions, one env, randomization
  as the existing zero-action hold tests set it; the mean of the unweighted
  `torque_saturation` value over the last 5 s is below 0.01.
- [ ] `run_summary` unit tests: a balance log that meets the old criteria with
  `torque_saturation = -0.01` passes; the same with -0.2 fails; the same with
  the key missing fails.
- [ ] Do not loosen any existing threshold. If the zero-action sim test fails
  for a stance, report the measured value; do not raise the 0.01.

**Verify:** `uv run pytest` on the skate test files; `uv run ruff format`,
`uv run ruff check`, `uv run ty check` on the touched files.

**Commit:** `fix: penalize balance policies that lean on the joint stops`

### Task 26: Let the deck lean (board joint limits were compiled as degrees)

**Why:** `skateboard_constants.py` gives the deck roll limit as 0.2 rad and the
truck yaw limit as `steer_gain * roll_limit` rad, but the board spec compiles
angles as degrees (`spec.compiler.degree` is 1). The compiled ranges are
±0.00349 rad for `deck_roll` and ±0.00289 rad for both truck yaw joints. So in
every run so far the deck could not lean or steer, and `deck_flat` was always
zero. A real deck leans. Every policy trained so far is untested against that.

**Files:**
- Modify: `src/mjlab/asset_zoo/robots/mini_cruiser/skateboard_constants.py`
- Modify: `skateboard/README.md` (one bullet in the board section)
- Test: the existing board test file (find it with
  `grep -ln mini_cruiser tests/*.py`), `tests/test_pupper_skate.py`

**Interfaces:**
- Consumes: `get_skateboard_spec()`, `get_skateboard_cfg()`, the params
  dataclass fields `roll_limit` (0.2) and `steer_gain` (0.577).
- Produces: no new names. The compiled joint ranges change.

**Changes:**
- [ ] Make the compiled limits equal the radian values in the dataclass:
  `deck_roll` ±`roll_limit`, each truck yaw ±`steer_gain * roll_limit`. Either
  set the spec's compiler to radians or convert the two ranges; pick the one
  that leaves every other angle-valued attribute of the board spec with the
  value it has today (check for any: `ref`, `springref`, eulers, other ranges)
  and say in the report which you chose and why.
- [ ] The limits must also hold in the model the tasks train on. The board is
  attached to a scene with the robot, and the scene is compiled again: check
  the ranges in a built env's MuJoCo model, not only in the board's own spec.
- [ ] Do not change stiffness, damping, `roll_limit`, `steer_gain`, any reward,
  keyframe or termination.
- [ ] `skateboard/README.md`: one bullet saying the deck can now lean up to
  0.2 rad, and that policies trained before this fix (runs `k4kci0oj`,
  `14m5rs1f`, `2ace1uqa`, `29p4j81t`) saw a deck that could not lean.

**Tests (write first, see them fail):**
- [ ] Board spec compiled alone: `deck_roll` range is ±0.2 rad and each truck
  yaw range is ±(0.577 * 0.2) rad, to 1e-6.
- [ ] Built env for `Mjlab-Skate-Balance-Pupper-v3` and
  `Mjlab-SkateSplit-Balance-Pupper-v3`: the same three ranges in the env's
  compiled model.
- [ ] Sim test on the board alone or in an env: a sideways load on the deck
  edge leans the deck by more than 0.02 rad (it could not exceed 0.0035
  before).
- [ ] Run every existing skate test. The zero-action hold tests of both
  stances are the ones at risk, because both keyframes were tuned on a deck
  that could not lean. Do not loosen a threshold and do not retune a keyframe.
  If one fails, stop, and report for each failing test the measured value, the
  deck roll angle over the hold, and whether the robot falls; status
  DONE_WITH_CONCERNS with the fix committed and the failing tests listed.

**Verify:** `uv run pytest` on the skate test files; `uv run ruff format`,
`uv run ruff check`, `uv run ty check` on the touched files.

**Commit:** `fix: board joint limits were compiled as degrees`

### Task 27: Test the centred stance hold at nominal truck stiffness

**Why:** Task 26 let the deck lean. Two centred zero-action tests then failed
(21 of 64 noisy spawns fall, limit 0.2). The first version of this task blamed
the keyframe. Measurement showed otherwise: with randomization off the centre
of mass is 0.00 mm from the roll axis and the deck does not move. The falls
track the truck stiffness randomization (15 to 60 N*m/rad): at the soft end the
pose cannot hold with no policy. Widening or narrowing the stance did not fix
it, so the keyframe stays.

**Files:**
- Modify: `tests/test_pupper_skate.py`
- Modify: `src/mjlab/tasks/pupper_skate/pupper_skate_env_cfg.py`,
  `pupper_skate_split_env_cfg.py` (truck stiffness range only)
- Modify: `skateboard/README.md` (if it states the range)

**Steps:**
- [ ] The two zero-action hold tests run with the truck stiffness pinned at
  nominal. Thresholds unchanged (0.1 rad tilt, 0.2 fall share). No seed
  picking.
- [ ] Widen the `truck_stiffness` scale from (0.5, 2.0) to (0.5, 6.0), 15 to
  180 N*m/rad, in both stances. The real bushings are unmeasured and built for
  a rider about 20 times heavier than Pupper, so the real deck is likely much
  stiffer than the 30 N*m/rad guess.
- [ ] New test: the centred stance holds with zero actions at 6 times nominal
  stiffness (64 noisy spawns, fall share under 0.2).
- [ ] If a hold test still fails at nominal stiffness, leave it failing and
  report the numbers. Do not change the stance or a threshold.

**Verify:** `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py
tests/test_pupper_skate_split.py tests/test_pupper_skateboard.py`;
`uv run ruff format`, `uv run ruff check`, `uv run ty check` on touched files.

**Commit:** `fix: test the centred stance hold at nominal truck stiffness`

### Task 28: Centred ride rewards that make a push pay

**Why:** the centred ride run `14m5rs1f` learned to stand on the deck and not
push (`track_board_velocity` 1.37 of 3.0, `push_stroke` 0.015). A push study
with the saved policy showed why: standing pays 4.57 reward per step and a
front-leg push pose pays 4.01, so lifting a foot loses reward before the board
moves. A front-leg push is possible (the pose holds for 3 s on a 6.8 cm deck
with a 2 cm margin and 25% peak torque). The split ride task is not touched.

**Files:**
- Modify: `src/mjlab/tasks/pupper_skate/mdp/rewards.py`
- Modify: `src/mjlab/tasks/pupper_skate/mdp/terminations.py`
- Modify: `src/mjlab/tasks/pupper_skate/pupper_skate_env_cfg.py`
  (`pupper_skate_ride_env_cfg` only)
- Modify: `skateboard/README.md` (the centred ride column of the reward table),
  `notebooks/CS123_Pupper_Skate.ipynb` through `skateboard/tools/make_notebook.py`
  if the notebook's reward table lists these weights
- Test: `tests/test_pupper_skate.py`

**Interfaces:**
- `mdp.contact_count(env, sensor_name, max_count: float | None = None)`: with
  `max_count` the count is clamped to it. Default keeps today's behaviour.
- `mdp.contact_lost(env, sensor_name, grace_s, min_lost: int = 1)`: terminate
  when at least `min_lost` of the sensor's geoms have been out of contact for
  longer than `grace_s`. Default keeps today's behaviour.
- New `mdp.board_progress(env, command_name, command_threshold, board_cfg=_BOARD)`:
  `clamp(board_vx / cmd_vx, 0, 1)` where `cmd_vx > command_threshold`, else 0.
  `board_vx` is `board.data.root_link_lin_vel_b[:, 0]`, the same frame as
  `track_board_velocity`.

**Exact values, centred ride only** (`pupper_skate_ride_env_cfg`):

| Term | Now | New |
| --- | --- | --- |
| `feet_on_deck` | weight 0.5, counts 4 feet | weight 0.5, `max_count` 3.0 |
| `feet_off_deck` termination | any foot off for 1.5 s | `min_lost` 2, grace 1.5 s |
| `tilt` std | 0.2 | 0.35 |
| `board_alignment` `std_pos` | 0.05 | 0.08 |
| `foot_reach` weight | 0.5 | 1.5 |
| `push_stroke` weight | 0.3 | 1.0 |
| `board_progress` (new) | none | weight 1.0, `command_threshold` 0.1 |
| `track_board_velocity` | weight 3.0, std 0.15 | unchanged |

The balance task and both split tasks must build with exactly the rewards and
terminations they have now. The split ride config builds on its own function;
check it does not pick these changes up.

**Steps:**
- [ ] Tests first, in `tests/test_pupper_skate.py`, plain functions:
  - `contact_count` with `max_count=3.0` returns 3 when four feet touch and 2
    when two touch; without it returns 4.
  - `contact_lost` with `min_lost=2`: one foot past the grace does not
    terminate, two feet past it do; `min_lost=1` keeps the old result.
  - `board_progress`: 0 with no command, 0 when rolling backwards, 0.5 at half
    the commanded speed, 1.0 at or above it.
  - The built centred ride env has the table's values; the built centred
    balance env still has `feet_on_deck` without `max_count`, `tilt` std 0.2
    and `feet_off_deck` with `min_lost` 1 (or absent) and grace 0.5.
  - The built split ride env has no `board_progress` term.
- [ ] Implement, run the tests.
- [ ] Re-measure the study's comparison and put it in the report: reward per
  step (weight times term, summed) for the standing pose against the
  front-leg push pose on a 6.8 cm deck with the board at rest and command
  0.4. The scripts are in `/Users/ethan/.claude/jobs/bc321137/tmp/pushstudy/`
  (read them, copy what you need into your own scratch directory). Target:
  the push pose pays at least as much as standing. If it does not, report the
  per-term numbers and stop; do not invent further weight changes.
- [ ] Update the README table (and the notebook generator if it lists them).

**Verify:** `FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py
tests/test_pupper_skate_split.py tests/test_pupper_skateboard.py`;
`uv run ruff format`, `uv run ruff check`, `uv run ty check` on touched files.

**Commit:** `feat: centred ride rewards that pay for a push`

## Self-review

- Coverage: stance keyframe (Task 1), modeling (Tasks 1, 3, 10), rewards and terminations (Tasks 2, 3, 5 and the Design tables), randomization (Task 3), training (Tasks 9, 11), export and deploy (Tasks 4, 12). No gaps found.
- Names checked against the code: task ids, reward term names, `SKATE_*`, W&B metric names, `REWARD_WEIGHTS`, `RIDE_SPEED_RANGE`.
- Tasks 1 to 6 point at verified files instead of repeating about 900 lines of code here.

## Accounting

- **Skipped:** code listings for Tasks 1 to 6 (the code is in the repo and tested); a right-side stance; a push/glide phase clock like HUSKY's.
- **Assumed:** centred stance with feet 6.0 to 6.6 cm either side (v4); board mass 1.8 kg until weighed; lowered trucks put the deck 2 to 3 cm lower (balance randomizes deck height 6.8 to 11.3 cm to cover both; ride trains on 6.8 to 8.3 cm only, v5); W&B project name `mjlab` as in the course notebook.
- **Issues:** nothing has run on the real robot. The ride run stopped at iteration ~1120 of 1500 (Colab GPU limit), so it has curves but no `RESULT:` verdict. Reward weights are the defaults; they passed, so they were not tuned further. `uv run play` for the skate tasks has not been watched in a viewer.
