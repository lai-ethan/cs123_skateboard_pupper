# Sim2real: will Pupper stay balanced on the board?

The TA's concern after the checkpoint report: "I doubt if Pupper can remain balanced
at this pose on the board." This note is the answer so far. It is version 2,
written 2026-11-17, from simulation only. Nothing here has been tried on the robot.

## Short answer

| Policy | Run | Curves | Gait check | Stance audit | Ready for a first robot test? |
| --- | --- | --- | --- | --- | --- |
| Split balance | `gt60lfo0` | pass | pass | 21 of 21 | **Yes, first candidate** |
| Centred balance | `zdctdcyf` | pass | pass | 14 of 21 | Yes, with care (see below) |
| Split ride | `3qmtsv4o` | pass | **fail, one line** | not run | Not yet (close) |
| Centred ride | `cy8f55bg` | fail | fail | not run | No |

- **Split stance** (left feet on the deck, right feet on the floor): the balance
  policy holds every condition we tried, including soft trucks and several errors
  at once. Test this one first.
- **Centred stance** (all four feet on the deck): the policy stands still and
  passes the real deck height, but the feet are only 1.5 to 2 cm inside the deck
  edge. It falls when the board is 2 cm off-centre, when shoved, or when the
  trucks are four times softer than modelled. On the robot it needs placement
  within about 1 cm, wedged wheels and no shoves.
- **Riding: not yet, in either stance.** The newest split ride policy
  (`3qmtsv4o`) rides at the commanded speed without falling and fails one gait
  line: peak joint speed 9.2 rad/s against a limit of 8, and that limit is a
  guess.
  The centred ride policy falls onto the deck and the board barely moves; it is
  parked until the lowered trucks exist.

## The board, as measured

| Quantity | Value | Source |
| --- | --- | --- |
| Deck top above floor | 103 mm | measured |
| Deck width | 161 mm | measured |
| Wheel radius | 29 mm | measured |
| Wheelbase | 14 3/8 in | measured |
| Track | 104 mm | measured |
| Length | 23 in | measured |
| Mass | 1.523 kg | measured, stock trucks |
| Truck roll stiffness | 30 N·m/rad | **guess** |

Truck stiffness is the top number left to measure. Training randomizes it from
15 to 180 N·m/rad. The split policy passes at a quarter of the modelled value
(7.5); the centred policy does not.

## How it was measured

`skateboard/tools/stance_audit.py` runs a balance policy for 10 s per condition
and changes one thing at a time. All randomization is off, so the only difference
between rows is the condition named.

```bash
# Mac, in ~/claude/cs123/pupper-mjlab (about 9 minutes for all 21 rows)
FORCE_CPU=1 uv run python skateboard/tools/stance_audit.py \
  skateboard/policies/split_balance_gt60lfo0_policy.json \
  --task Mjlab-SkateSplit-Balance-Pupper-v3
```

A row passes when the robot never resets, the deck feet stay on the deck at least
90% of the time, no motor averages above 60% of its torque limit, and the centre
of mass stays at least 3 cm inside the feet.

- **torque** is the busiest motor's mean torque as a share of its limit.
- **margin** is how far the centre of mass is inside the outline of the feet.

The full logs are `policies/split_balance_gt60lfo0_audit.txt` and
`policies/balance_zdctdcyf_audit.txt`.

## Results: split balance `gt60lfo0`

All 21 rows pass with zero resets.

| Condition | Torque | Margin |
| --- | --- | --- |
| Baseline (deck 103 mm) | 0.14 | 6.3 cm |
| No policy (motors hold the pose) | 0.15 | 7.5 cm |
| Deck 3.5 cm or 2 cm lower | 0.22, 0.16 | 6.7, 7.1 cm |
| Deck 1, 2 or 3 cm higher | 0.16, 0.16, 0.44 | 6.9, 7.1, 6.2 cm |
| Trucks 4 times softer | 0.17 | 7.4 cm |
| Deck friction 0.3 | 0.16 | 6.9 cm |
| Motor stiffness 60% | 0.16 | 6.8 cm |
| Torque limit 1.0 N·m (a third) | 0.46 | 6.7 cm |
| Board 1 or 2 cm left or right | 0.16 to 0.17 | 7.0 to 7.4 cm |
| Board turned 0.2 rad | 0.16 | 6.8 cm |
| Joints start 0.3 rad off | 0.17 | 7.4 cm |
| Shove 0.6 m/s left or right | 0.16, 0.17 | 7.2, 7.1 cm |
| Board kicked 0.8 m/s | 0.13 | 5.3 cm (board rolls 8 cm) |
| Several errors at once | 0.25 | 7.2 cm |

- The body leans about 5 degrees toward the deck in every row. That is the
  stance, not a fault.
- Standing on three feet is still not stable: with the front right foot removed
  the centre of mass is 2 cm outside the other three. This matters for riding,
  not for balance.

## Results: centred balance `zdctdcyf`

14 of 21 rows pass.

| Condition | Result | Note |
| --- | --- | --- |
| Baseline (deck 103 mm) | pass | torque 0.16, margin 5.1 cm |
| No policy | pass | |
| Deck 3.5 or 2 cm lower, 1 to 3 cm higher | pass | |
| Deck friction 0.3 | pass | |
| Motor stiffness 60% | pass | |
| Torque limit 1.0 N·m | pass | torque 0.52 |
| Board 1 cm left or right | pass | |
| Board turned 0.2 rad | pass | |
| Joints start 0.3 rad off | pass | |
| **Trucks 4 times softer** | **fail** | 2 resets |
| **Board 2 cm left** | **fail** | 3 resets |
| **Board 2 cm right** | **fail** | 1 reset |
| **Shove 0.6 m/s left or right** | **fail** | 1 reset each |
| **Board kicked 0.8 m/s** | **fail** | no reset, but the board rolls 29 cm and the margin goes to -5 cm |
| **Several errors at once** | **fail** | 1 reset |

The failures share one cause: the feet stand 6.0 to 6.6 cm from the centre line
on a deck that is 8.05 cm wide each side. Any sideways error larger than the
1.5 to 2 cm left over puts a foot off the edge. This is the geometry, not a
training fault, so the policy was not retrained. A deck plate about 20 cm wide
would fix it.

## Checks that carry over to the robot

- The policy file is deploy contract v2, same as the walking labs: 48 numbers per
  step, 20 steps of history (960 inputs), 12 joint targets out. Each local export
  was checked against the trained network (largest difference 2e-08 for the
  centred balance policy).
- Gains and action scale in `policy.json` are the lab's (kp 5.5, kd 0.25, action
  scale 0.75). Balance policies clip the speed command to zero.
- Sensor and motor latency are modelled the same way as the walking tasks.
- In the gait check on the real deck height both balance policies stay far inside
  the motor limits: peak joint speed 0.6 rad/s (split) and 2.7 rad/s (centred)
  against a limit of 8, with no torque saturation.
- Training randomizes deck height from 68 to 113 mm for balance (both stances)
  and for split ride, so both the stock and the lowered trucks are covered.
  Centred ride trains on 68 to 83 mm only: from the centred stance a foot cannot
  reach the floor on the stock trucks.

## What the simulation cannot tell us

- **Truck stiffness and wheel drag.** Still guesses.
- **Foot grip on the real deck.** The sim foot is a sphere on a flat plate. Grip
  tape, the deck's curve and the kicktail are not modelled.
- **Leg backlash and flex.** The sim legs are rigid. Real legs sag under a
  sideways load, and the split stance loads them sideways.
- **Motor speed.** The sim motor has a torque limit and no speed limit. The
  8 rad/s used in the gait check is a first guess.
- **Starting pose.** In sim the robot starts in the stance. On the robot someone
  has to place it there by hand; how the deploy script gets from standing to the
  stance has not been checked.
- **The robot-side code.** Only the policy file was checked here, not the C++
  that loads it.

## Plan before the first real run

Each step needs the one before it. No robot motion without Ethan's OK, and the TA
signs off before the first powered run (PLAN Task 12).

1. **Show a TA** `media/split_balance_gt60lfo0.mp4` and
   `media/balance_zdctdcyf.mp4` and this note.
2. **Fit test, robot powered off.** Place the robot on the board by hand in each
   stance. Check that the feet reach, sit on the deck and not the edge, and that
   nothing collides. For the split stance check that the right feet reach the
   floor.
3. **Split balance, wheels wedged.** One person holds the robot by the handle,
   kill switch in hand. Let go only if it holds still for 5 s. Compare the lean
   (about 5 degrees toward the deck) with the audit.
4. **Split balance, wheels free**, same precautions.
5. **Centred balance, wheels wedged.** Centre the board under the robot to
   within 1 cm. Do not shove it.
6. **Centred balance, wheels free**, only if step 5 holds for 10 s.
7. **Ride only after** a policy prints both `RESULT: PASS` and `GAIT: PASS` and
   the video shows slow planted pushes. `3qmtsv4o` is one line short (joint
   speed); whether to try it at 0.2 m/s, hand held, is the TA's call.

On the robot (over SSH):

```bash
cd ~/pupper_gait_deploy
./deploy.sh mjlab/gt60lfo0   # split balance
./deploy.sh mjlab/zdctdcyf   # centred balance
./deploy.sh mjlab/3qmtsv4o   # split ride, only after step 7
```

## Riding: where it stands

- **Split ride v1 `vzs88cci`:** shuffled (5 steps per second, joint speed 8.6
  rad/s). Reference only.
- **Split ride v2 `ffcxbwup`** (Task 29 rewards): the shuffle is gone (2 steps
  per second) and the board follows the command (0.86 to 1.06 of it). It still
  fails at 0.4 m/s on the real deck: joint speed 10.6 rad/s, motors saturated
  21% of the time, the back right foot never touches the floor, and 5 of 16
  boards reset. At 0.2 m/s the joint speed passes (7.9) but it still saturates
  and falls. Video: `media/split_ride_ffcxbwup_0.4ms.mp4`.
- **Centred ride:** the first run never pushed (`14m5rs1f`). The second, with
  the Task 28 rewards, crashed at iteration 95 when one simulated env diverged.
  The third (`cy8f55bg`, with the Task 30 guard) ran to the end and failed:
  episodes last 363 of 500 steps, motors are saturated the whole time, and in
  the gait check the board moves at 0.15 of the command at best with 51 resets.
  Video: `media/ride_cy8f55bg_0.4ms.mp4` (the robot ends up lying on the deck).
  Parked: it needs trucks that do not exist yet and a deck with no sideways
  room.
- **Split ride v3 `3qmtsv4o`** (Task 30 rewards: more charge for fast joints
  and saturated motors, and for a pushing foot that hovers). Curves pass. On the
  real deck height, 16 boards, no resets at any speed:

  | Command | Joint speed (limit 8) | Saturated | Steps per second | Board speed / command |
  | --- | --- | --- | --- | --- |
  | 0.2 m/s | **9.3** | 0.3% | 2.0 | 0.92 to 1.06 |
  | 0.3 m/s | **9.1** | 2.8% | 2.0 | 0.92 to 1.02 |
  | 0.4 m/s | **9.2** | **5.8%** | 2.0 | 0.93 to 1.01 |

  Both right feet are on the floor a little over half the time. Video:
  `media/split_ride_3qmtsv4o_0.3ms.mp4` (2.7 m in 10 s, no resets). The policy
  file is `policies/split_ride_3qmtsv4o_policy.json`.
- **Next:** find the real motor speed limit (the 8 rad/s is a guess). If it is
  above about 9.5 rad/s this policy passes at 0.2 and 0.3 m/s as it is. If not,
  retrain with a larger `joint_speed_limit` weight.

## What changed from v1

- v1 covered the old split stance on a guessed board, with one policy
  (`2ace1uqa`). v2 covers both stances on the measured board, each with its own
  policy and audit.
- The v1 deck-height failure (motors at their limit on a deck 2.5 cm higher) is
  gone: both policies pass deck heights from 3.5 cm lower to 3 cm higher.
- Board mass is measured (1.523 kg), no longer assumed.
- New finding: the centred stance has too little sideways room on this deck.
- The ride section now reports the failed ride runs in both stances.

## Accounting

- **Skipped:** any hardware test; a stance audit for the ride policies; shoves
  above 0.6 m/s and deck friction below 0.3 (v1 ran them for the old policy).
- **Assumed:** truck stiffness 30 N·m/rad; the 3 N·m torque limit and 8 rad/s
  speed limit; the audit's pass limits (60% torque, 3 cm margin); that lowered
  trucks put the deck at 68 to 83 mm.
- **Issues:** no ride policy passes the gait check (split ride is one line short); the centred stance fails 7 audit rows
  from lack of sideways room; hand-placing the robot in the split stance and the
  robot-side loader are unverified.
