# Checkpoint summary: Tuesday Nov 17 (updated Nov 18)

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
| Centred balance | `zdctdcyf` | `policies/balance_zdctdcyf_policy.json` | `media/balance_zdctdcyf.mp4` |
| Split balance | `gt60lfo0` | `policies/split_balance_gt60lfo0_policy.json` | `media/split_balance_gt60lfo0.mp4` |
| Split ride v4 | `iznjwbka` | `policies/split_ride_iznjwbka_policy.json` | `media/split_ride_iznjwbka_0.3ms.mp4` |

- Centred balance: peak joint speed 2.63 rad/s, no saturation, min foot duty
  0.99, no resets.
- Split balance: peak joint speed 0.58 rad/s, right feet on the floor 100% of
  the time, no resets.
- Split ride v4: passes at 0.2, 0.3 and 0.4 m/s. It replaces `3qmtsv4o`.
- Heading hold is off in all skate exports (kp 0).

## What does not work

- **Centred ride passes all but one limit.** It is now a tripod ride: three
  feet on the deck, the back right foot pushing on the floor. The sixth run,
  `7fyt12i1`, rides at 0.1, 0.2 and 0.3 m/s with no resets, joint speed under
  7 rad/s, no saturation and at most 1.5 strides per second. It fails
  `min_foot_duty`: the push foot is on the floor 20% of the time (limit 40%),
  because it pushes briefly and then glides.
  Video: `media/centred_ride_v9_7fyt12i1.mp4`. Provisional policy:
  `policies/centred_ride_7fyt12i1_max0.3_policy.json`.
- Earlier centred ride runs stood still (`vrn7pp8n`, `2kmu10gl`), lay on the
  deck (`xtbenpi4`), skittered (`9c33unfu`) or took one stride too many
  (`zvx4xc24`). The history is in `SIM2REAL.md`.
- **Steering.** The steer run `jl7i4id0` does not turn the board. In sim the
  robot is too light to tilt the deck far enough (`SIM2REAL.md`, "Steering").
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

## What changed from the Nov 17 version

- Centred ride: tripod design, five more runs, `7fyt12i1` exported as a
  provisional candidate.
- Added the steering result and the tilt test.

## Accounting

- Skipped: anything on the robot; a stance audit for the ride policies.
- Assumed: the 12 rad/s joint speed limit; truck stiffness 30 N·m/rad.
- Issues: centred ride has no passing policy (`7fyt12i1` fails push foot duty); the robot-side loader has not run
  a skate policy.
