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
   - `side_swing_cm_mean`: the peak-to-peak of the leftward position over
     the same window after subtracting its least-squares quadratic, in cm. A
     quadratic, not a line, because a steady turn bends the path into a
     parabola in the starting frame; a steady turn with no sway must read
     about 0.
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
   - a steady turn at 3 degrees a second with no sway, at 0.3 m/s for 5 s,
     has `side_swing_cm_mean` under 0.1;
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

## Baseline result (2026-11-26): sim rides straight, so the swing is a sim-to-real gap

`timed_ride.py`, stick 0, 16 boards, 5 s ride, hard-floor ranges with the
wheel drag pinned. Sim only.

| Policy | Asked m/s | Drag | Speed (share of asked) | Resets | Turn (deg/s) | Drift (m sideways per m forward) | Worst board's drift (cm) | Nose swing (deg) | Side sway (cm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v6 | 0.3 | x1 | 1.01 | 0 | -0.29 | -0.015 | 9.5 | 0.89 | 0.20 |
| v6 | 0.3 | x3 | 0.92 | 0 | -0.30 | -0.013 | 8.5 | 0.82 | 0.14 |
| v6 | 0.3 | x6 | 0.70 | 2 | -0.59 | -0.024 | 8.0 | 0.65 | 0.10 |
| v6 | 0.15 | x1 | 1.04 | 0 | -0.43 | -0.018 | 3.9 | 0.33 | 0.05 |
| v6 | 0.15 | x3 | 0.89 | 0 | -0.38 | -0.015 | 3.0 | 0.34 | 0.04 |
| v6 | 0.15 | x6 | 0.59 | 0 | -0.29 | -0.013 | 1.6 | 0.30 | 0.04 |
| v10 | 0.3 | x1 | 1.02 | 0 | -0.38 | -0.024 | 11.9 | 0.85 | 0.20 |
| v10 | 0.3 | x3 | 0.99 | 0 | -0.29 | -0.018 | 10.7 | 0.95 | 0.21 |
| v10 | 0.3 | x6 | 0.92 | 0 | -0.27 | -0.015 | 9.5 | 1.06 | 0.20 |
| v10 | 0.15 | x1 | 1.10 | 0 | -0.45 | -0.020 | 4.1 | 0.28 | 0.05 |
| v10 | 0.15 | x3 | 1.05 | 0 | -0.35 | -0.016 | 3.6 | 0.37 | 0.06 |
| v10 | 0.15 | x6 | 0.97 | 0 | -0.31 | -0.014 | 3.2 | 0.48 | 0.07 |

Negative turn and drift are to the right. Swings read 6 to 15% high for a
pure sine (the detrend), so they are upper estimates.

- v10's nose swing is at most 1.06 degrees and its mean drift at most 2.4 cm
  per metre. Both are under the Task 3 thresholds (2 degrees, 0.05), so the
  swing seen on the robot is not in sim. A reward cannot train out what the
  sim does not show; no training task is added.
- The worst single board drifts 8 to 12 cm over a 5 s ride at 0.3 m/s, so
  some drift is expected on the robot too. The video shows more than that.
- What the sim does not model, and so the likely causes on the robot: the
  gamepad cable dragging from the body, the trucks leaning under each kick if
  they are loose, and the wheels scrubbing sideways on carpet. None is
  measured.
- A side result: at x6 drag v10 holds 0.92 of the asked speed at 0.3 m/s and
  0.97 at 0.15 m/s; v6 holds 0.70 and 0.59. If the robot is slower than
  asked on carpet, v10 should be the faster of the two.

## Task 4: hardware checklist for a straight ride

Files: `skateboard/HARDWARE_TEST.md`, `skateboard/SIM2REAL.md`,
`skateboard/README.md` (Status only).

1. `skateboard/HARDWARE_TEST.md`:
   - In the "2026-11-24 run" area, a "2026-11-26 run" note: with the
     placement guide, split balance held and both steer v6 and steer v10
     rolled the board forward on the Gates carpet for the whole clip (about
     29 s and 28 s); the nose swung on each kick and the board drifted to one
     side; speed was not measured; a wired gamepad cable dragged from the
     robot; truck tightness was not recorded. Say plainly that this is the
     first run that counts as a test, and update the intro sentence that says
     nothing has been tested on the robot.
   - In test 5, a short "Straight ride" block before the carves, for each of
     v6 and v10: gamepad cable held slack above the robot or a wireless
     gamepad; record whether the trucks are stock or loosened, and if the
     board swings, repeat once with the trucks tightened; lay a 2 m tape line
     on the floor, board on the line, camera fixed (not hand-held) looking
     down the line; stick fully forward for 5 s. Record: distance along the
     line, sideways distance from the line at the end, and whether the nose
     swings on each kick.
   - What sim predicts for that ride, from the table above: about 1.4 to 1.5
     m in 5 s on a hard floor; at x6 drag about 1.4 m for v10 and about 1.05
     m for v6; a few cm of drift to one side (the worst sim board 8 to 12
     cm); about 1 degree of nose swing. More than that on the robot is the
     sim-to-real gap this test is measuring.
2. `skateboard/SIM2REAL.md`: a short "Straightness" section with the table
   and the conclusion above, and how to run the tool
   (`timed_ride.py ... --wheel-drag X`), including that swings are only
   comparable at the same `--ride-s`.
3. `skateboard/README.md` Status: the first working hardware run
   (2026-11-26) and a v10 row or line, replacing the "As of 2026-11-20"
   wording.
4. Docs only. Every number comes from the table above or from the note in
   item 1; write "not measured" where there is no number.

## Accounting

- Skipped: the backward gait; any change to the Gemini voice estimates.
- Assumed: the clips are v6 (IMG_9606) and v10 (IMG_9610), from their
  lengths.
- Issues: there is no hardware number for speed or drift yet, only video.
