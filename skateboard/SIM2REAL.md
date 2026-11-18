# Sim2real: will Pupper stay balanced on the board?

The TA's concern after the checkpoint report: "I doubt if Pupper can remain balanced
at this pose on the board." This note is the answer so far. It is version 4,
written 2026-11-18, from simulation only. Nothing here has been tried on the robot.

## Short answer

| Policy | Run | Curves | Gait check | Stance audit | Ready for a first robot test? |
| --- | --- | --- | --- | --- | --- |
| Split balance | `gt60lfo0` | pass | pass | 21 of 21 | **Yes, first candidate** |
| Centred balance | `zdctdcyf` | pass | pass | 14 of 21 | Yes, with care (see below) |
| Split ride v4 | `iznjwbka` | pass | pass at 0.2, 0.3, 0.4 m/s | not run | Yes, after both balance tests |
| Centred ride (tripod) | `7fyt12i1` | pass | **fail** on push foot duty only, at 0.1, 0.2, 0.3 m/s | not run | Provisional, last, Ethan's call |

- **Split stance** (left feet on the deck, right feet on the floor): the balance
  policy holds every condition we tried, including soft trucks and several errors
  at once. Test this one first.
- **Centred stance** (all four feet on the deck): the policy stands still and
  passes the real deck height, but the feet are only 1.5 to 2 cm inside the deck
  edge. It falls when the board is 2 cm off-centre, when shoved, or when the
  trucks are four times softer than modelled. On the robot it needs placement
  within about 1 cm, wedged wheels and no shoves.
- **Split ride is ready for a first test.** Split ride v4 (`iznjwbka`) passes
  the gait check on the real deck height at 0.2, 0.3 and 0.4 m/s. It replaces
  `3qmtsv4o`.
- **Centred ride rides in sim but fails one gait limit.** The sixth run
  (`7fyt12i1`) keeps three feet on the deck and pushes with the back right
  foot on the floor. It holds the commanded speed with no falls, slow joints
  and at most 1.5 strides per second. It fails `min_foot_duty`: the push foot
  is on the floor only 20% of the time (see "Riding: where it stands").

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
- Heading hold is off in all skate exports (kp 0).
- In the gait check on the real deck height both balance policies stay far inside
  the motor limits: peak joint speed 0.58 rad/s (split) and 2.63 rad/s (centred)
  against a limit of 12, with no torque saturation.
- Training randomizes deck height from 68 to 113 mm for balance (both stances)
  and for split ride, so both the stock and the lowered trucks are covered.
  Centred ride now trains on 93 to 113 mm, the stock trucks: its push reference
  is solved for the 103 mm deck.

## What the simulation cannot tell us

- **Truck stiffness and wheel drag.** Still guesses.
- **Foot grip on the real deck.** The sim foot is a sphere on a flat plate. Grip
  tape, the deck's curve and the kicktail are not modelled.
- **Leg backlash and flex.** The sim legs are rigid. Real legs sag under a
  sideways load, and the split stance loads them sideways.
- **Motor speed.** The sim motor has a torque limit and no speed limit. The
  12 rad/s used in the gait check is an estimate (the motor spec gives about
  30 rad/s with no load). It is not measured.
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
7. **Split ride `iznjwbka`, hand held,** only after split balance holds
   (steps 3 and 4).
   Start at 0.2 m/s, then 0.3 m/s. It passes the gait check at both.
8. **Centred ride `7fyt12i1`, hand held,** last, and only if Ethan accepts
   that it fails the push foot duty limit (`HARDWARE_TEST.md`, test 4).

The test sheet for this session is [`HARDWARE_TEST.md`](HARDWARE_TEST.md).

On the robot (over SSH):

```bash
cd ~/pupper_gait_deploy
./deploy.sh mjlab/gt60lfo0   # split balance
./deploy.sh mjlab/zdctdcyf   # centred balance
./deploy.sh mjlab/iznjwbka   # split ride v4, only at step 7
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

  | Command | Joint speed (old limit 8) | Saturated | Steps per second | Board speed / command |
  | --- | --- | --- | --- | --- |
  | 0.2 m/s | **9.3** | 0.3% | 2.0 | 0.92 to 1.06 |
  | 0.3 m/s | **9.1** | 2.8% | 2.0 | 0.92 to 1.02 |
  | 0.4 m/s | **9.2** | **5.8%** | 2.0 | 0.93 to 1.01 |

  Both right feet are on the floor a little over half the time. Video:
  `media/split_ride_3qmtsv4o_0.3ms.mp4` (2.7 m in 10 s, no resets). The policy
  file is `policies/split_ride_3qmtsv4o_policy.json`.
  Judged against the old 8 rad/s limit it failed. Against the current 12 rad/s
  estimate its joint speeds would pass, but v4 replaces it anyway.
- **Split ride v4 `iznjwbka`** (PLAN Task 31 rewards: smoother lift-off and
  touchdown, low swings). `GAIT: PASS` on the 103 mm deck at 0.2, 0.3 and
  0.4 m/s. This is the split ride to deploy. Video:
  `media/split_ride_iznjwbka_0.3ms.mp4`. Policy:
  `policies/split_ride_iznjwbka_policy.json`.
- **Centred ride, first design (PLAN Task 32).** The policy followed a clocked
  push-and-glide reference: 0.4 Hz, the back right foot stepped off the deck,
  pushed on the floor and stepped back on. It trained on deck heights 93 to
  113 mm and speeds 0 to 0.4 m/s. Open-loop video of the reference:
  `media/centred_push_reference_openloop.mp4`. Played open loop the push is
  weak: the board reaches only about 0.05 m/s with the push foot down.
- **Centred ride first run `vrn7pp8n`** (1500 iterations): `GAIT: FAIL` at 0.2,
  0.3 and 0.4 m/s. The policy stands on the deck, no foot ever touches the
  floor and the board does not move. Standing earned more reward than pushing.
- **Runs two and three** (`xtbenpi4`, `2kmu10gl`, rewards from commit
  `cd9535a`): both failed. The second lay on the deck and paddled with three
  legs. With that closed off (the body must stay up, only the back right foot
  may leave the deck), the third stood crouched and did not push. Stepping a
  foot off the deck and back on each stroke was too hard to find.
- **Tripod ride (current design).** The robot starts every episode with three
  feet on the deck and the back right foot already on the floor beside the
  board. That foot never has to step on or off: it strokes on the floor and
  the other three carry the body. The reference stroke is 0.8 Hz, the command
  range is 0.1 to 0.3 m/s, and the episode ends if the body drops toward the
  deck or a support foot leaves it for more than 0.3 s.
- **Run four `9c33unfu`** (first tripod run): rides with no resets but
  skitters, about 7 strides per second with motors saturated 19 to 27% of the
  time. `GAIT: FAIL`. Video: `media/centred_ride_v7_9c33unfu.mp4`.
- **Run five `zvx4xc24`** (adds a charge for short strokes, `short_stride`
  -1.5, and `torque_saturation` -1.0): fails only on stride rate, 17 strides
  in 8 s against a limit of 16, at 0.1, 0.2 and 0.3 m/s. Video:
  `media/centred_ride_v8_zvx4xc24.mp4`.
- **Run six `7fyt12i1`** (`short_stride` -3.0, 2026-11-18): the stride rate now
  passes and one limit still fails. On the 103 mm deck:

  | Command | Peak joint speed | Saturated | Strides per s | Push foot on floor | Board speed / command | Resets |
  | --- | --- | --- | --- | --- | --- | --- |
  | 0.1 m/s | 6.3 rad/s | 0% | 1.5 | 20% | 1.07 to 1.39 | 0 |
  | 0.2 m/s | 6.3 rad/s | 0% | 0.75 | 20% | 0.95 to 1.13 | 0 |
  | 0.3 m/s | 6.8 rad/s | 0% | 1.1 | 22% | 0.95 to 1.05 | 0 |

  `GAIT: FAIL` on `min_foot_duty` only (0.20 to 0.22, limit 0.4). The three
  support feet never leave the deck. The push foot gives a short push, then
  hangs in the air while the board glides. The limit was written to catch a
  walking foot held in the air; here it flags a push-and-glide. The limit is
  unchanged and the verdict stands as a fail.
  Video at 0.2 m/s: `media/centred_ride_v9_7fyt12i1.mp4` (2.16 m in 10 s, no
  resets). Policy, capped at 0.3 m/s:
  `policies/centred_ride_7fyt12i1_max0.3_policy.json`.
- **What that means for the robot.** It is a provisional candidate, behind the
  three passing policies. Open risks: the robot stands on three feet for 80%
  of the ride, and at 0.1 m/s the pushes are short taps (about 0.13 s). If
  Ethan wants a pass first, the next change is to pay the push foot for time
  on the floor and train a seventh run.

## Steering: what sim says

Nothing steers yet, and in sim the robot is too light to steer this board.
Every exported ride policy takes a forward speed only, and heading hold is off
in every skate export (the robot's yaw correction would fight the stance).

- **Tried: a split-stance steer task.** `Mjlab-SkateSplit-Steer-Pupper-v3` is
  the split ride plus a commanded board turn rate of up to 0.15 rad/s (a full
  circle in about 40 s). The reward pays only for the board's turn rate, so the
  policy was free to turn with its legs, its feet or a lean. Run `jl7i4id0`
  (1500 iterations) rides well but does not turn: 0.07 of a +0.15 command and
  0.00 of a -0.15 command. It is not exported.
- **Why.** Measured on the board model at 0.3 m/s:
  - The board turns only when the deck tilts: 0.0165 rad/s per degree of tilt.
    0.15 rad/s needs about 9 degrees; 0.1 rad/s needs about 6.
  - Holding 9 degrees takes 3 to 5 N m against the modelled truck bushings
    (30 N m/rad), and 10 to 15 N m at the average stiffness the policy trains
    on (about 98).
  - The robot's whole weight (31 N) on the very edge of the deck gives about
    2.2 N m. That is about 0.07 rad/s on the soft bushing and 0.02 on the
    average one.
  - Pushing the board sideways with the legs turns it only through the tilt it
    causes, or by skidding the wheels. The wheels do not roll sideways.
- **So steering with the legs instead of leaning does not get around it.**
  The legs can only turn this board by tilting the deck or by dragging it
  sideways across the floor.
- **What decides it: the real trucks.** Measure the real board before any more
  steer training (`HARDWARE_TEST.md`, test 0). If the real deck tilts 6 degrees
  or more under the robot's weight at the edge, steering by lean can be
  trained; narrow `truck_stiffness` to the measured value for the steer task
  and tighten the reward (`std` 0.1). Loosening the kingpin nuts makes the
  trucks softer and is worth trying.
- **Cheapest option for Demo Day.** Ride straight and turn the board by hand
  between runs.

## What changed from v1

- v1 covered the old split stance on a guessed board, with one policy
  (`2ace1uqa`). v2 covers both stances on the measured board, each with its own
  policy and audit.
- The v1 deck-height failure (motors at their limit on a deck 2.5 cm higher) is
  gone: both policies pass deck heights from 3.5 cm lower to 3 cm higher.
- Board mass is measured (1.523 kg), no longer assumed.
- New finding: the centred stance has too little sideways room on this deck.
- The ride section now reports the failed ride runs in both stances.

## What changed from v2

- The gait check's joint speed limit is 12 rad/s (an estimate from the motor
  spec), not 8.
- Split ride v4 (`iznjwbka`) passes the gait check at 0.2, 0.3 and 0.4 m/s and
  replaces `3qmtsv4o` in the deploy list.
- Centred ride was redesigned around a clocked push-and-glide reference on the
  stock deck. Its first run failed (stands still); a second run with rebalanced
  rewards is training.
- Balance gait numbers updated from the newest exports, which have heading hold
  off.
- Added the steering section and the hardware test sheet,
  [`HARDWARE_TEST.md`](HARDWARE_TEST.md).

## What changed from v3

- Centred ride became a tripod ride: three feet on the deck, the back right
  foot on the floor from the start. Five more runs are reported.
- The newest run, `7fyt12i1`, rides at 0.1 to 0.3 m/s with no falls and fails
  only the push foot duty limit. It is exported as a provisional candidate.
- The steer run `jl7i4id0` and the tilt measurement are in the steering
  section.

## Accounting

- **Skipped:** any hardware test; a stance audit for the ride policies; shoves
  above 0.6 m/s and deck friction below 0.3 (v1 ran them for the old policy).
- **Assumed:** truck stiffness 30 N·m/rad; the 3 N·m torque limit and the
  12 rad/s speed estimate; the audit's pass limits (60% torque, 3 cm margin);
  that lowered trucks put the deck at 68 to 83 mm.
- **Issues:** centred ride has no passing policy yet (`7fyt12i1` fails the
  push foot duty limit); the centred stance fails 7 audit rows from lack of sideways room;
  hand-placing the robot in the split stance, the robot-side loader and how the
  ride speed is commanded on the robot are unverified.
