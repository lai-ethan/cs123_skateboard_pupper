# Straight ride plan: measure and reduce the sideways swing

Written 2026-11-26, after the second hardware run.

**Goal:** the split ride goes where the board points: less nose swing on each
kick and less sideways drift, without losing the carpet gains of steer v10.

**Scope:** `Mjlab-SkateSplit-Steer-Pupper-v3` and the sim tools. Balance and
the centred tasks are untouched.

## What the hardware run showed (2026-11-26, Gates carpet, correct placement)

- Split balance holds (IMG_9601).
- Steer v6 and steer v10 both roll the board forward for the whole clip
  (IMG_9606, IMG_9610). The robot stays in place on the deck.
- The nose swings a little on each kick and the board drifts to one side.
- The board looks slower than the 0.3 m/s asked. Not measured: the camera
  moves and no distance was recorded.
- A wired gamepad cable drags from the robot in both clips. Whether the
  trucks were loosened is not known.

## What is not known

- Whether sim shows the same swing. If it does, a reward can train it out.
  If sim rides straight, the swing is a sim-to-real gap (cable, truck lean,
  wheels scrubbing on carpet) and the fix is hardware or randomization, not a
  reward.
- A teammate trained a "v12" off v10 with `board_progress` at weight 1 (their
  numbers: at x10 drag and 0.15 m/s, 0.90 of the command with 3 of 48 boards
  lost, against v10's 0.83 with 7; worse than v10 at 0.3 m/s). It is on their
  branch and is not in this repo yet.

## Global constraints

- No change to the actor observation layout or the action space: any retrain
  resumes from steer v10 (`skateboard/policies/split_steer_v10_model_31996.pt`).
- No change to what the existing fields of `timed_ride.py` print, or to the
  ride audit's rows and verdicts.
- Do not raise motor limits, gains, speed caps or the 0.3 m/s hardware clip.
- Repo style: 88 columns, 2-space indent, plain-function tests. Run only the
  touched test files, with `FORCE_CPU=1 uv run pytest tests/<file>.py`. Do
  not run `make check`. One run per shell call.
- Commits `<type>: <description>`, authored by the configured git user only,
  no AI attribution. Public repo: no W&B entity names, keys, IPs, or local
  home-directory paths.
- Nothing here runs on the robot.

## Task 1: `timed_ride.py` measures straightness and takes a drag level

Files: `skateboard/tools/timed_ride.py`, `tests/test_skate_timed_ride.py`,
`skateboard/README.md` (the tools list, if it describes this tool).

1. New option `--wheel-drag X` (float, default none). When given, the wheel
   drag is pinned at X times nominal in every env, the way the ride audit's
   `wheel_drag_xN` rows pin it (reuse that code path; do not copy the block).
   Without it the tool behaves as today.
2. New fields in the printed JSON, all over the ride phase only (not the
   coast), computed per env and then summarised over the kept envs. "Forward"
   and "left" are the board's heading at the first ride step and the
   direction 90 degrees to its left.
   - `side_m_mean`: mean over envs of the signed leftward displacement of the
     board at the end of the ride, in m.
   - `side_m_abs_max`: the largest size of that displacement over envs.
   - `side_per_m_mean`: mean over envs of (leftward displacement) /
     (forward displacement), for envs whose forward displacement is at least
     0.05 m; `null` if there are none.
   - `yaw_swing_deg_mean`: for each env, take the board yaw (unwrapped) over
     the ride after the first 1.0 s, subtract its least-squares straight
     line, and take the peak-to-peak of what is left, in degrees. Mean over
     envs. This is the nose swing with any steady turn removed.
   - `side_swing_cm_mean`: the same detrended peak-to-peak for the leftward
     position, in cm.
   - `wheel_drag`: X, or `null` when the option is not given.
3. The new maths are small pure functions next to `path_length` and
   `turn_so_far`, so they are tested without a sim.
4. Tests first, seen to fail:
   - a board that moves straight ahead has side 0 and both swings 0;
   - a board that moves along a line 10 degrees to the left of its starting
     heading has `side_per_m_mean` of tan(10 degrees) to 3 decimals and
     swings of 0;
   - a steady turn at a constant yaw rate has `yaw_swing_deg_mean` under
     0.01;
   - a yaw of `2 * sin(2*pi*1.5*t)` degrees on top of a steady turn has
     `yaw_swing_deg_mean` of 4 to within 0.1;
   - a ride shorter than 1.5 s gives `null` for the two swings instead of an
     error;
   - `--wheel-drag 6` pins the wheel joints' damping and frictionloss at 6
     times nominal in every env (check the built env config or model, as the
     audit's own drag test does), and omitting it leaves today's ranges;
   - the existing fields still have the values they had for the existing
     tests.
5. The tool's docstring says what each new field means, in plain sentences.

## Task 2: baseline (controller)

Run `timed_ride.py` on steer v6 and steer v10 at 0.3 m/s and 0.15 m/s, stick
0, at wheel drag x1, x3 and x6, 16 envs, 5 s ride. Record the table in this
file.

## Task 3: decide (controller)

- If v10's `yaw_swing_deg_mean` is at least 2 degrees, or its
  `side_per_m_mean` is at least 0.05 in size, at any of those drag levels:
  the swing is in sim. Add a training task to this plan that charges it, and
  train from v10.
- Otherwise the swing is a sim-to-real gap. Record that, and add to the
  hardware checklist: cable held slack or wireless, trucks tightened for
  straight rides, and a fixed-camera clip with a tape line to measure speed
  and drift.
- Either way: when the teammate's v12 branch is pushed, run it through the
  same audit, steering checks and this baseline before it goes near the
  robot.

## Accounting

- Skipped: the backward gait; any change to the Gemini voice estimates.
- Assumed: the clips are v6 (IMG_9606) and v10 (IMG_9610), from their
  lengths.
- Issues: there is no hardware number for speed or drift yet, only video.
