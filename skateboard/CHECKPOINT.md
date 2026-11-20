# Checkpoint summary: Tuesday Nov 17 (updated Nov 20)

Pupper v3 on a mini cruiser (deck top 103 mm), two stances, each with a balance
and a ride policy. Everything below is simulation only. Nothing has run on the
robot. Demo Day is Dec 3.

## What works

Each policy below prints `GAIT: PASS` on the 103 mm deck. Limits: peak joint
speed at most 12 rad/s, saturated at most 5% of the time, min foot duty at
least 0.4, stride rate at most 2 Hz, board speed 0.5 to 1.5 times the command,
no resets.

| Policy | Run | Policy file | Video |
| --- | --- | --- | --- |
| Centred balance | `zdctdcyf` | `policies/centred_balance_v2_policy.json` | `media/centred_balance_v2.mp4` |
| Split balance | `gt60lfo0` | `policies/split_balance_v3_policy.json` | `media/split_balance_v3.mp4` |
| Split ride v4 | `iznjwbka` | `policies/split_ride_v4_policy.json` | `media/split_ride_v4_0.3ms.mp4` |

- Centred balance: peak joint speed 2.63 rad/s, no saturation, min foot duty
  0.99, no resets.
- Split balance: peak joint speed 0.58 rad/s, right feet on the floor 100% of
  the time, no resets.
- Split ride v4: passed at 0.2, 0.3 and 0.4 m/s on 2026-11-17, on the old box
  deck, before the `board_drifted` and `board_slid` checks existed. It has not
  passed on the current task: on 2026-11-18 its checkpoint reset on
  `board_drifted` in the gait check, and its retrain v7 (`gkk9a6bj`) fails 9
  ride audit rows. Not rerun since `board_slid` and the elliptic cone.
- Heading hold is off in every skate export (kp 0) except split steer v5, v6 and v7.
- File names map to W&B runs in [`policies/README.md`](policies/README.md).

## What does not work

- **Centred ride passes all but one limit.** It is now a tripod ride: three
  feet on the deck, the back right foot pushing on the floor. The sixth run,
  `7fyt12i1`, rides at 0.1, 0.2 and 0.3 m/s with no resets, joint speed under
  7 rad/s, no saturation and at most 1.5 strides per second. It fails
  `min_foot_duty`: the push foot is on the floor 20% of the time (limit 40%),
  because it pushes briefly and then glides.
  Video: `media/centred_ride_v9.mp4`. Provisional policy:
  `policies/centred_ride_v9_max0.3_policy.json`.
- Earlier centred ride runs stood still (`vrn7pp8n`, `2kmu10gl`), lay on the
  deck (`xtbenpi4`), skittered (`9c33unfu`) or took one stride too many
  (`zvx4xc24`). The history is in `SIM2REAL.md`.
- **Steering is a candidate, not a pass.** Split steer v6 (`hfd3qhwr`) is
  now the candidate and v5 is the fallback (see the next two items). Split
  steer v5 (runs `qqm2917n`, then `bxke67fj`) leans to steer on trucks
  loosened to 9 to 18 N·m/rad. On the 103 mm deck at 0.3 m/s it prints
  `GAIT: PASS` at yaw 0, +0.5 and -0.5 with no resets. Carving, the board
  turns at 0.92 (left) and 0.84 (right) of the asked 0.050 rad/s, about 3
  degrees per second. It fails the ride audit
  on 2 of 16 rows: in 30 s rides 2 of 16 boards reset on `board_slid`, and 2
  reset with trucks 4 times stiffer. So about 1 board in 8 slides off in 30 s.
  Policy: `policies/split_steer_v5_max0.3_policy.json`. Videos:
  `media/split_steer_v5_*_0.3ms.mp4`. Earlier steer runs did not turn
  (`jl7i4id0`) or slid off the deck (`SIM2REAL.md`, "Steering").
- **v5 steers on full stick only.** It trained only on full carves (yaw 0.4
  to 0.6 rad/s), so the operator must push the yaw stick all the way.
- **v6 turns on partial stick.** Split steer v6 (`hfd3qhwr`) resumed from v5
  for 3000 iterations on partial turns (yaw 0.1 to 0.6). It passes all 16
  ride audit rows. On the 103 mm deck at 0.3 m/s (seed 0) the board turns
  0.82 to 0.86 of the asked rate at stick ±0.15 and ±0.6, but only 0.58 to
  0.63 at ±0.3. One of 16 boards, the same one each time, resets on
  `board_slid` on left carves of 0.3 to 0.6, so those gait reports fail; on
  two more seeds there was no reset (1 board in 48). At +0.5 it is weaker
  than v5 (0.67 of asked, against 0.92). Policy:
  `policies/split_steer_v6_max0.3_policy.json`. Videos:
  `media/split_steer_v6_*_0.3ms.mp4`.
- **v7 slides less but turns worse.** Split steer v7 (`n5m2zsks`) resumed
  from v6 for 2000 iterations with the deck-foot scuff penalty at -20 (was
  -2) and the slide counted from the episode start. No `board_slid` reset in
  27 gait runs, and under soft leg gains 0 of 16 boards slid out (v6: 4 of
  16) with half the creep (0.21 cm/s, target 0.1). Ride audit 16 of 16. But
  it pulls right: hard right carves (stick -0.5, -0.6) fail on every seed on
  `board_drifted`, +0.6 turns 0.58 of asked (v6: 0.80), and -0.15 turns 2.1
  times asked. It missed the plan's pass bar, so v6 stays the candidate. v7
  is the fallback for straight riding and gentle carves (stick within ±0.3)
  if v6 slides along the deck. The right pull is not diagnosed. Policy:
  `policies/split_steer_v7_max0.3_policy.json` (yaw clipped at ±0.3).
  Videos: `media/split_steer_v7_*_0.3ms.mp4`.
- **No foot stop.** The team will not fit a foot stop or ridge on the deck,
  so staying on the board has to come from the policy. Sliding along the deck
  is the main open risk: in sim v5 lets 2 of 16 boards slide 8 cm within 30 s,
  and v6 lets 1 in 48 slide that far on left carves. v7 lets none, but turns
  worse. Runs must be short, with the robot reset to its marks between runs.
- Centred balance has little sideways room: it fails 7 of 21 stance audit rows
  (board 2 cm off-centre, shoves, soft trucks). Place it within about 1 cm.
- The 12 rad/s joint speed limit is an estimate (motor spec is about 30 rad/s
  with no load), not a measurement.

## What is next

1. Ethan decides on the centred ride: test `7fyt12i1` as it is (last, hand
   held), or train a seventh run that pays the push foot for time on the
   floor.
2. Get Ethan's OK and a TA sign-off for the first hardware session.
3. Run the session from [`HARDWARE_TEST.md`](HARDWARE_TEST.md): board tilt
   (test 0), split balance, centred balance, split ride at 0.2 then 0.3 m/s.
4. Measure truck stiffness and wheel drag (still guesses).
5. Steering test with split steer v6 (`HARDWARE_TEST.md`, test 5): trucks
   loosened to 9 to 18 N·m/rad, start with a small stick, short runs, robot
   reset to its marks between runs. If the robot slides along the deck, try
   v7 with the stick within ±0.3. v5 with full stick is the fallback.
   Needs Ethan's OK and a TA sign-off first.
6. Find the cause of v7's right pull before any further steer retrain.

## What changed from the Nov 17 version

- Centred ride: tripod design, five more runs, `7fyt12i1` exported as a
  provisional candidate.
- Added the steering result and the tilt test.

## What changed on Nov 19

- Split steer v5 exported as the candidate for the first steering test.
- Policies and media renamed to `<stance>_<skill>_v<N>`, with a table of runs.

## What changed on Nov 20

- Two new constraints: the yaw stick can send any value, and there is no
  foot stop on the deck. The steer task now trains partial turns for a v6
  (not trained yet). Foot stop advice is gone; sliding is an open risk.
- Later the same day: split steer v6 (`hfd3qhwr`) trained and evaluated. It
  is the new steering candidate since partial stick works; v5 is the
  fallback. The steering test now uses the v6 file.
- Later still: split steer v7 (`n5m2zsks`) trained to stop the deck slide
  ([`PLAN_DECK_SLIDE.md`](PLAN_DECK_SLIDE.md)). It stops the slide but hurts
  turning, so v6 stays the candidate and v7 is the fallback if v6 slides.

## Accounting

- Skipped: anything on the robot; a stance audit for the ride policies.
- Assumed: the 12 rad/s joint speed limit; truck stiffness 30 N·m/rad.
- Issues: centred ride has no passing policy (`7fyt12i1` fails push foot duty); split
  steer v6 lets 1 board in 48 slide off on left carves and turns about 0.6 of asked
  at mid stick; with no foot stop, sliding along the deck is an open risk; a stick
  right at 0.1 sits between the heading hold and the smallest carve; v7 pulls right and
  fails hard right carves, cause unknown; the robot-side loader has not run a skate
  policy.
