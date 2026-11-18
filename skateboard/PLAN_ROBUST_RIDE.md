# Split Ride That Stays On: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.
> Task numbers continue from `PLAN.md` and its briefs (last used: 40).

**Goal:** A split ride policy (left feet on the deck, right feet pushing on the
floor) that stays on the board for 30 s under the conditions a real robot will
meet, with a tool that proves it.

**Architecture:** First a ride audit tool, so the current policy (`iznjwbka`)
gets a measured baseline. Then the split ride task is changed so that training
sees what the audit tests: longer rides, friction that reaches the foot
contacts, wheel drag, hand-placement error, and rewards that stop the board
drifting out from under the feet. Then one training run on Colab, judged by the
gait report, the ride audit and a video.

**Tech stack:** mjlab (MuJoCo-Warp), rsl_rl PPO, pytest, Colab for training.

## Why (research, 2026-11-18)

Measured on the current split ride policy at 0.3 m/s, 16 envs:

| Condition | Envs that fall off |
|---|---|
| 10 s ride, nominal | 0 to 1 of 16 |
| **30 s ride, nominal** | **5 to 6 of 16**, first at 14.6 s |
| Deck feet grip 0.35 | 3 of 16 |
| Floor feet grip 0.4 | 4 of 16 |
| Board yawed 0.15 rad at placement | 3 of 16 |
| 0.5 m/s sideways shove | 2 of 16 |
| Wheel drag x2 | 3 of 16, speed 0.82 of command |
| Wheel drag x5 | 14 of 16, speed 0.49 of command |

- The board drifts sideways from the robot by 2 to 3 cm per 10 s and nothing
  pulls it back. Training episodes are 10 s, so training never saw the fall.
- The `deck_friction` randomization does nothing to the feet: feet have geom
  priority 1, so every foot contact uses the foot's own friction, and all four
  feet share one draw.
- Wheel drag is a guess and is not randomized.

Turning with the legs (a pivot or skid turn) was measured and is not planned:
the deck feet can pass at most 0.72 N m of yaw to the board and skidding the
wheels takes 2.8 to 4.6 N m. In open-loop probes the robot spun on the deck
(1 to 7 degrees per stroke) and the board did not turn (under 0.13 degrees).
Steering stays gated on HARDWARE_TEST test 0 (truck tilt).

## Global Constraints

- Always `FORCE_CPU=1 uv run ...` on this Mac. Never bare `python`. Do NOT run
  `make check`.
- 88-column lines, 2-space indent, plain-function tests, comment density like
  the surrounding code.
- `uv run ruff format` and `uv run ruff check --fix` on touched files,
  `uv run ty check` on touched source files, pytest one file per command.
- These tasks must behave exactly as before (config, rewards, terminations,
  events, observations, export): `Mjlab-Skate-Balance-Pupper-v3`,
  `Mjlab-Skate-Ride-Pupper-v3`, `Mjlab-SkateSplit-Balance-Pupper-v3`. Shared
  functions get default-preserving keywords only. Do not edit the board or
  robot asset constants.
- The split ride's observation layout must not change (the exporter and the
  robot controller depend on it).
- Stage only your own files by explicit path. Commit on `skateboard` as the
  configured git user. No `Co-Authored-By`, no "Generated with", no AI
  attribution. Do not push. Do not start training. Do not touch anything
  outside this repo.
- The repo is public: no robot address, passwords, keys or W&B entity names in
  any file.

---

### Task 41: Ride audit tool

**Files:**
- Create: `skateboard/tools/ride_audit.py`
- Modify: `skateboard/tools/stance_audit.py` (friction row only)
- Test: the skate tools test file (find with `ls tests | grep -i skate`)

**Background:** `skateboard/tools/stance_audit.py` audits balance policies
under off-nominal conditions. Nothing does that for a ride policy.
`skateboard/tools/gait_report.py` runs a ride policy at a fixed speed but only
at nominal conditions and for 10 s. A working scratch version of the sweep is
at `/Users/ethan/.claude/jobs/bc321137/tmp/research/ride_sweep.py`; read it and
`/Users/ethan/.claude/jobs/bc321137/tmp/research/sim_research.md` section A2
for how each condition was applied. Build the tool from that, in the style of
`stance_audit.py` (argument names, table output, exit code).

**Requirements:**

1. Usage:
   `FORCE_CPU=1 uv run python skateboard/tools/ride_audit.py <task> <policy> --speed 0.3 [--num-envs 16] [--seed 0] [--rows name,name]`.
   `<policy>` is a `.pt` or a policy `.json` (use
   `skateboard/tools/policy_loader.py` as the other tools do).
2. Every row keeps the task's play-mode randomization and overrides one thing.
   Rows, in this order, with these exact names:

   | Row | Override | Steps |
   |---|---|---|
   | `nominal_10s` | none | 500 |
   | `nominal_30s` | none | 1500 |
   | `deck_grip_0.4` | friction of the deck-side (left) foot geoms 0.4 | 500 |
   | `floor_grip_0.4` | friction of the floor-side (right) foot geoms 0.4 | 500 |
   | `board_y_+2cm` | board reset y +0.02 m | 500 |
   | `board_y_-2cm` | board reset y -0.02 m | 500 |
   | `board_yaw_0.15` | board reset yaw +0.15 rad on top of the task's noise | 500 |
   | `shove_side_0.5` | robot root lateral velocity +/-0.5 m/s at step 250, sign alternating by env | 500 |
   | `shove_fwd_0.5` | same, along x | 500 |
   | `wheel_drag_x2` | wheel spin joint damping and frictionloss x2 | 500 |
   | `wheel_drag_x3` | same, x3 | 500 |
   | `truck_x0.5` | truck roll stiffness x0.5 | 500 |
   | `truck_x4` | truck roll stiffness x4 | 500 |
   | `deck_93mm` | deck height 0.093 m | 500 |
   | `deck_113mm` | deck height 0.113 m | 500 |
   | `kp_x0.7` | joint kp x0.7 | 500 |

   Which feet are deck-side and floor-side comes from the task's own contact
   sensor config, not a hard-coded list, so the tool also works on the tripod
   ride (there: three deck feet, one floor foot). If a row cannot be applied
   to a task, print it as `n/a` with the reason; do not crash.
3. Per row report: number of envs that reset at least once (and total
   resets, with the termination names), share of time each deck foot touches
   the deck (mean and worst env), board speed along its own heading divided by
   the command (mean, min env, max env), board lateral offset from the robot
   in the robot's heading frame (min and max across envs over the last 2 s),
   board heading drift (max and mean). Metrics start after step 100.
4. Row verdict PASS when: at most 1 of 16 envs reset (scale with
   `--num-envs`: at most 6.25%, rounded up), mean speed ratio within 0.7 to
   1.3, and worst-env deck-foot contact share at least 0.9. Otherwise FAIL.
   Keep these limits in one module-level dict `RIDE_LIMITS`. Final line
   `RIDE AUDIT: PASS` or `RIDE AUDIT: FAIL (n rows)`; exit code 0 or 1.
5. `--json <path>` writes all rows and metrics as JSON.
6. `stance_audit.py`: its low-friction row sets the deck geom's friction,
   which never reaches the foot contacts (feet have geom priority 1). Make
   that row set the friction of the feet that stand on the deck instead, and
   fix the comment. Change nothing else in that tool.
7. Tests (plain functions): the row table has the 16 names above in order;
   the verdict function passes and fails on hand-made metric dicts at each
   limit; a smoke test runs `nominal_10s` with 2 envs and about 60 steps on
   `Mjlab-SkateSplit-Ride-Pupper-v3` with
   `skateboard/policies/split_ride_iznjwbka_policy.json` and gets finite
   metrics (allow a `--steps` override for this). Mark the smoke test the way
   slow tests in that file are marked, if any are.
8. Run the full audit once on the current policy:
   `FORCE_CPU=1 uv run python skateboard/tools/ride_audit.py Mjlab-SkateSplit-Ride-Pupper-v3 skateboard/policies/split_ride_iznjwbka_policy.json --speed 0.3 --json /Users/ethan/.claude/jobs/bc321137/tmp/ride_audit_iznjwbka.json`
   and paste the table into the report. It is expected to FAIL on several
   rows; that is the baseline, not a bug. If `nominal_10s` fails, stop and
   report.

**Commit:** `feat: ride audit tool, stance audit friction row reaches the feet`

---

### Task 42: Split ride trains for what the audit tests

**Files:**
- Modify: `src/mjlab/tasks/pupper_skate/pupper_skate_split_env_cfg.py`
  (the ride config function; the steer config builds on it and inherits)
- Modify: `src/mjlab/tasks/pupper_skate/mdp/` (new terms only, or
  default-preserving keywords)
- Modify: `notebooks/CS123_Pupper_Skate.ipynb` and
  `skateboard/tools/make_notebook.py` only if they list reward weights or
  terms that this task changes (a test ties the generator to the notebook)
- Test: the skate test files (`ls tests | grep -i skate`)

**Background:** see "Why" above and
`/Users/ethan/.claude/jobs/bc321137/tmp/research/sim_research.md` sections A1
to A3 (file and line references for every event named below). The current
policy rides well for 10 s and loses the deck in a third of envs by 30 s
because the board drifts sideways under the feet.

**Requirements (all on `Mjlab-SkateSplit-Ride-Pupper-v3`; the Steer task
inherits them; no other task changes):**

1. **Episode length** 20 s (train and play).
2. **Foot friction per side.** Replace the single shared foot friction draw
   with two independent draws: deck-side (left) feet 0.5 to 1.4, floor-side
   (right) feet 0.5 to 1.2. Remove the ride's `deck_friction` event if it
   affects no contact that matters, or leave it and say why in a comment.
3. **Wheel drag.** New startup event scaling the wheel spin joints' damping
   and frictionloss by one shared factor per env, 0.5 to 3.0.
4. **Placement.** Board reset y +/-0.02 m (today 0) and yaw +/-0.15 rad
   (today +/-0.05). Check every term that assumes the board starts within
   0.05 rad of world x (there is a comment above `push_robot`); the Steer
   task already has board-heading-frame versions of the affected terms. Use
   them in the ride where a term would be wrong at 0.15 rad, and list each
   term and what you did in the report.
5. **Shoves.** Robot shove range +/-0.35 m/s (today +/-0.2), training only as
   today.
6. **Deck feet must not slide.** Add a penalty on deck-foot slip speed
   relative to the deck (not the world: the deck moves), for the feet in
   contact with the deck. Name `deck_feet_slide`. Weight: the same as the
   existing floor `feet_slide` term. If `mdp.feet_slide` cannot measure
   relative to the board, add a new function.
7. **Board stays under the feet.** Tighten `board_alignment`: `std_pos` 0.03
   (today 0.05) and double its weight. Add a termination `board_drifted`:
   the board's lateral offset from the robot, in the board's heading frame,
   is more than 0.06 m from its value in the split stance keyframe. Compute
   the nominal from the keyframe, not a typed constant. It must be present in
   play as well.
8. **Unchanged:** every other reward weight, the command range, the
   observation layout, the action space, the three tasks named in Global
   Constraints. Do not raise the termination penalty.
9. **Tests:**
   - The ride config has 20 s episodes, two foot friction events with the
     ranges above covering disjoint foot sets that together are all four
     feet, the wheel drag event, the reset and shove ranges, `deck_feet_slide`
     and `board_drifted`.
   - `board_drifted` does not fire in the stance keyframe or at +/-0.02 m
     reset offset, and fires at 0.07 m, at both ends of the deck height range.
   - `deck_feet_slide` is zero when a deck foot moves with the deck, positive
     when it moves relative to the deck, zero for a foot not in contact.
   - Wheel drag event: after reset, each env's four wheels share one factor
     and factors differ across envs.
   - A config-equality or term-list test that the three protected tasks have
     none of the new terms and their existing ranges. Extend the existing
     "other tasks unchanged" tests if there are some.
   - Observation dimension of the ride is what it was.
10. **Smoke runs:** zero-action rollout of the ride (train and play), finite
    rewards. Then
    `FORCE_CPU=1 uv run python skateboard/tools/gait_report.py Mjlab-SkateSplit-Ride-Pupper-v3 skateboard/policies/split_ride_iznjwbka_policy.json --speed 0.3 --deck-height 0.103`
    still runs (the old policy on the new task; its numbers may differ and it
    may reset on `board_drifted`: report them, do not tune to them).
11. **Warm start check:** the old checkpoint
    `skateboard/policies/split_ride_iznjwbka_model_2998.pt` must still load
    into the task's policy network (same observation and action sizes). Say
    in the report how you checked.

**Commit:** `feat: split ride trains on long rides, per-side grip, wheel drag and placement error`

---

### Task 43: Train, judge, export (controller, not a subagent)

1. Push to `team/main` (diff scanned for secrets). Train
   `Mjlab-SkateSplit-Ride-Pupper-v3` on Colab from the notebook, warm start
   from `iznjwbka` `model_2998.pt`, 3000 iterations, run to the end.
2. Judge the final checkpoint at 0.2 and 0.3 m/s, deck 0.103:
   `gait_report.py` (GAIT limits unchanged), `ride_audit.py` (all rows),
   video and contact sheet into `skateboard/media/`.
3. Robot-ready means: `GAIT: PASS` at both speeds, `RIDE AUDIT: PASS`, and a
   video that shows slow planted pushes. If it falls short, decide the fix
   from the failing rows, change one thing, train again.
4. Export with `--clip-vx "(0.0, 0.3)"`, run the robot-side validator
   read-only, update `SIM2REAL.md`, `HARDWARE_TEST.md`, `CHECKPOINT.md`,
   `README.md`. Commit and push.

## Self-review

- Staying on while moving: Tasks 41 (measure), 42 (train for it), 43 (prove).
- Turning with the legs: measured, not feasible; no task. Recorded in "Why".
- Centred ride: parked by request; `ride_audit.py` works on it for later.
- Not done here: wheel sideways friction (needs a board asset change, which
  would alter the balance tasks; straight riding does not skid), floor
  friction as its own knob, constant IMU bias.
