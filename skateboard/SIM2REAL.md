# Sim2real: will Pupper stay balanced on the board?

The TA's concern after the checkpoint report: "I doubt if Pupper can remain balanced
at this pose on the board." This note is the answer so far. It is version 6,
written 2026-11-20, from simulation only. Nothing here has been tried on the robot.

## Short answer

| Policy | Run | Curves | Gait check | Stance audit | Ready for a first robot test? |
| --- | --- | --- | --- | --- | --- |
| Split balance | `gt60lfo0` | pass | pass | 21 of 21 | **Yes, first candidate** |
| Centred balance | `zdctdcyf` | pass | pass | 14 of 21 | Yes, with care (see below) |
| Split ride v4 | `iznjwbka` | pass | passed at 0.2, 0.3, 0.4 m/s on 2026-11-17, older task; resets on `board_drifted` on the current one | not run | Not ready; short runs only, Ethan's call |
| Centred ride (tripod) | `7fyt12i1` | pass | **fail** on push foot duty only, at 0.1, 0.2, 0.3 m/s | not run | Provisional, last, Ethan's call |
| Split steer v6 | `hfd3qhwr` | not checked | pass at yaw 0, ±0.15, -0.3, -0.5, -0.6; 1 reset at +0.3, +0.5, +0.6 (0.3 m/s) | ride audit 16 of 16 | Candidate for the first steering test, loosened trucks |
| Split steer v5 | `bxke67fj` | not checked | pass at yaw 0, +0.5, -0.5 (0.3 m/s) | ride audit 14 of 16 | Fallback for the steering test, full stick only |

- **Split stance** (left feet on the deck, right feet on the floor): the balance
  policy holds every condition we tried, including soft trucks and several errors
  at once. Test this one first.
- **Centred stance** (all four feet on the deck): the policy stands still and
  passes the real deck height, but the feet are only 1.5 to 2 cm inside the deck
  edge. It falls when the board is 2 cm off-centre, when shoved, or when the
  trucks are four times softer than modelled. On the robot it needs placement
  within about 1 cm, wedged wheels and no shoves.
- **Split ride is not ready.** Split ride v4 (`iznjwbka`) passed the gait
  check on the real deck height at 0.2, 0.3 and 0.4 m/s on 2026-11-17, on the
  old box deck and before the `board_drifted` and `board_slid` checks existed.
  On 2026-11-18, on the then current task at 103 mm, its checkpoint and the
  next one (`cfcrepfu`) reset on `board_drifted` 27 to 28 times on the box
  deck and 33 to 35 times on the traced deck. Its retrain, v7 (`gkk9a6bj`),
  fails 9 ride audit rows. Not rerun since `board_slid` and the elliptic cone.
- **Centred ride rides in sim but fails one gait limit.** The sixth run
  (`7fyt12i1`) keeps three feet on the deck and pushes with the back right
  foot on the floor. It holds the commanded speed with no falls, slow joints
  and at most 1.5 strides per second. It fails `min_foot_duty`: the push foot
  is on the floor only 20% of the time (see "Riding: where it stands").
- **Split steer v6 turns in proportion to the stick, but is not a full
  pass.** It trained on partial turns and passes all 16 ride audit rows. In
  the gait check one of 16 boards (the same one each time) slides past the
  8 cm limit on left carves of 0.3 to 0.6 on seed 0; seeds 1 and 2 had no
  reset. At mid stick it turns about 0.6 of what it asks (see "Steering: what
  sim says"). v6 is the candidate for the steering test. v5 trained only on
  full carves and stays as the fallback.

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
  skateboard/policies/split_balance_v3_policy.json \
  --task Mjlab-SkateSplit-Balance-Pupper-v3
```

A row passes when the robot never resets, the deck feet stay on the deck at least
90% of the time, no motor averages above 60% of its torque limit, and the centre
of mass stays at least 3 cm inside the feet.

- **torque** is the busiest motor's mean torque as a share of its limit.
- **margin** is how far the centre of mass is inside the outline of the feet.

The full logs are `policies/split_balance_v3_audit.txt` and
`policies/centred_balance_v2_audit.txt`.

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
- Heading hold is off in every skate export (kp 0) except split steer v5,
  which has it on (kp 1.0, clip 0.3 rad/s).
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

1. **Show a TA** `media/split_balance_v3.mp4` and
   `media/centred_balance_v2.mp4` and this note.
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
   (steps 3 and 4), and only if Ethan accepts that it resets on sideways
   drift on the current task. Start at 0.2 m/s, then 0.3 m/s. Short runs.
8. **Centred ride `7fyt12i1`, hand held,** last, and only if Ethan accepts
   that it fails the push foot duty limit (`HARDWARE_TEST.md`, test 4).
9. **Split steer v6, hand held,** only after step 7, on trucks loosened to 9
   to 18 N·m/rad. Partial stick is allowed; start with a small stick. Short
   runs, and the robot reset to its marks between runs (`HARDWARE_TEST.md`,
   test 5). There is no foot stop on the deck. v5 is the fallback, with the
   full yaw stick only.

The test sheet for this session is [`HARDWARE_TEST.md`](HARDWARE_TEST.md).

Install the local files in `policies/`, not the W&B exports: the ride and
steer tests need the capped `*_max0.3_policy.json` files, and `./deploy.sh
mjlab/<run>` fetches a run's W&B export, which is not the capped file. The
files to use:

| Step | File |
| --- | --- |
| 3, 4 | `policies/split_balance_v3_policy.json` |
| 5, 6 | `policies/centred_balance_v2_policy.json` |
| 7 | `policies/split_ride_v4_max0.3_policy.json` |
| 8 | `policies/centred_ride_v9_max0.3_policy.json` |
| 9 | `policies/split_steer_v6_max0.3_policy.json` (fallback: `policies/split_steer_v5_max0.3_policy.json`) |

On the robot (over SSH), after copying the file there, as in
`HARDWARE_TEST.md` test 3. These commands come from the robot repo's usage
and have not yet been run with a skate policy:

```bash
cd ~/pupper_gait_deploy
python3 download_policy.py --file <file> --output policies/test_policy.json
./deploy.sh
```

## Riding: where it stands

- **Split ride v1 `vzs88cci`:** shuffled (5 steps per second, joint speed 8.6
  rad/s). Reference only.
- **Split ride v2 `ffcxbwup`** (Task 29 rewards): the shuffle is gone (2 steps
  per second) and the board follows the command (0.86 to 1.06 of it). It still
  fails at 0.4 m/s on the real deck: joint speed 10.6 rad/s, motors saturated
  21% of the time, the back right foot never touches the floor, and 5 of 16
  boards reset. At 0.2 m/s the joint speed passes (7.9) but it still saturates
  and falls. Video: `media/split_ride_v2_0.4ms.mp4`.
- **Centred ride:** the first run never pushed (`14m5rs1f`). The second, with
  the Task 28 rewards, crashed at iteration 95 when one simulated env diverged.
  The third (`cy8f55bg`, with the Task 30 guard) ran to the end and failed:
  episodes last 363 of 500 steps, motors are saturated the whole time, and in
  the gait check the board moves at 0.15 of the command at best with 51 resets.
  Video: `media/centred_ride_v3_0.4ms.mp4` (the robot ends up lying on the deck).
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
  `media/split_ride_v3_0.3ms.mp4` (2.7 m in 10 s, no resets). The policy
  file is `policies/split_ride_v3_policy.json`.
  Judged against the old 8 rad/s limit it failed. Against the current 12 rad/s
  estimate its joint speeds would pass, but v4 replaces it anyway.
- **Split ride v4 `iznjwbka`** (PLAN Task 31 rewards: smoother lift-off and
  touchdown, low swings). `GAIT: PASS` on the 103 mm deck at 0.2, 0.3 and
  0.4 m/s (2026-11-17, box deck, before `board_drifted` and `board_slid`).
  On the current task it resets on `board_drifted`; see the short answer.
  Video:
  `media/split_ride_v4_0.3ms.mp4`. Policy:
  `policies/split_ride_v4_policy.json`.
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
  time. `GAIT: FAIL`. Video: `media/centred_ride_v7.mp4`.
- **Run five `zvx4xc24`** (adds a charge for short strokes, `short_stride`
  -1.5, and `torque_saturation` -1.0): fails only on stride rate, 17 strides
  in 8 s against a limit of 16, at 0.1, 0.2 and 0.3 m/s. Video:
  `media/centred_ride_v8.mp4`.
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
  Video at 0.2 m/s: `media/centred_ride_v9.mp4` (2.16 m in 10 s, no
  resets). Policy, capped at 0.3 m/s:
  `policies/centred_ride_v9_max0.3_policy.json`.
- **What that means for the robot.** It is a provisional candidate, behind the
  three passing policies. Open risks: the robot stands on three feet for 80%
  of the ride, and at 0.1 m/s the pushes are short taps (about 0.13 s). If
  Ethan wants a pass first, the next change is to pay the push foot for time
  on the floor and train a seventh run.

## Steering: what sim says

Split steer v6 is the candidate for the first hardware steering test
(`HARDWARE_TEST.md`, test 5), not a fully passing policy. It trained on
partial turns and turns in proportion to the stick. Split steer v5 was the
first steer policy to pass all three gait reports (yaw 0, +0.5 and -0.5) with
zero resets, but it needs the full stick; it is the fallback. Steer v2 to v4
also turned the board both ways, but reset in the gait check. The v6 export
has the heading hold on (kp 1.0, clip 0.1 rad/s); v5's has clip 0.3. In sim the hold
holds the board's heading. On the robot it holds the robot's own IMU yaw, so it
cannot see the board: any yaw between robot and board when the hold captures
is kept, not corrected. The other skate exports have the hold off (kp 0).
The runs below show how it got there.

- **First try: a split-stance steer task.** `Mjlab-SkateSplit-Steer-Pupper-v3`
  was the split ride plus a commanded board turn rate of up to 0.15 rad/s.
  Run `jl7i4id0` (1500 iterations) rode well but did not turn: 0.07 of a
  +0.15 command and 0.00 of a -0.15 command. A later run (up to 0.08 rad/s,
  heading hold off; abandoned, no run id recorded, not split steer v2) did
  not learn either. With the hold off, the ride
  policy yawed 0.06 to 0.1 rad off the board and 6 of 8 envs let the board
  walk out sideways within 8 s. Neither is exported.
- **Why the board barely turns.** It turns only when the deck tilts: 0.0165
  rad/s per degree at 0.3 m/s. The tilt comes from the robot's weight off the
  centre line, against the truck bushings. Measured on the board model at 0.3
  m/s, with the deck feet 4 cm off the centre line:

  | Truck stiffness (N m/rad) | Board yaw rate (rad/s) |
  | --- | --- |
  | 12 | 0.07 |
  | 20 | 0.04 |
  | 30 | 0.03 |

  Pushing the board sideways with the legs turns it only through the tilt it
  causes, or by skidding the wheels. The wheels do not roll sideways.
- **The steer task now: a carve signal.** The yaw command has two meanings,
  and the robot's own heading hold tells them apart, so the robot side needs
  no change. The task changed after v5 (see "What changed from v5"). v5 was
  trained on the old version, so the old numbers still apply to v5. v6 was
  trained on the current one.
  - Yaw 0 is the ride. The heading hold is on and its correction holds a
    heading: in sim the board's, on the robot the robot's own IMU yaw. So on
    the robot, place it parallel to the board. The correction is at most 0.1
    rad/s on the current task (0.3 for v5 and for the ride).
  - A carve is a yaw command above 0.1 and up to 0.6 rad/s, either sign. It is
    above the hold's 0.1 rad/s threshold, so the hold lets go and the command
    reaches the policy. The hold's clip is also 0.1, so a command of 0.1 or
    less can only be a correction and a larger one can only be the stick. It
    asks the board to turn at a tenth of the command, 0.01 to 0.06 rad/s, with
    the robot turning along with it. A half stick is a half turn.
  - v5 trained only on full carves, 0.4 to 0.6 rad/s, with the hold's clip at
    0.3. Between 0.1 and 0.35 rad/s its stick looks like a hold correction,
    and it never trained there. So v5 needs the full yaw stick.
  - When the command goes back to 0, the hold captures a new heading and
    holds that: in sim the board's, on the robot its own. A yaw offset to the
    board at that moment stays.
  - The board's turn is paid only during a carve. The board can sit 6 cm off
    the stance's line before the alignment term charges most of its reward
    (the ride allows 3 cm), since a lean needs the deck feet about 4 cm off
    the centre line.
- **The trucks must be loosened.** The steer task trains on 9 to 18 N m/rad.
  The ride trains on 15 to 180, since the real bushings are built for a rider
  about 20 times heavier. Loosen the kingpin nuts on the real board and
  measure the stiffness (`HARDWARE_TEST.md`, test 0) before any steer test on
  the robot. Above about 20 N m/rad the robot cannot lean the board into a
  useful turn without its feet leaving the deck.
- **Sliding along the deck.** The steer policy `188eygof` fell in 4 of 16
  envs over 30 s. Every fall had the same cause. The deck feet crept forward
  along the deck at 0.6 to 2.4 cm/s while touching it, until the front left
  foot ran off the nose (29.2 cm ahead of the board centre) and the body
  pitched nose down. The falls came 17 to 22 cm from the start, a median 21 s
  in, so 7 of 16 came after the 20 s training episode. Training never saw it:
  the drift check read only the sideways offset, and the deck foot slip charge
  was about 1e-4 per step at these speeds. The split ride and steer now end
  the episode once the robot is 8 cm along the deck from where the stance
  puts it, either way (`board_slid`; a reset starts within 6.6 cm of that). A
  new reward, `board_fore_aft` (weight 1.0, 3 cm width), pays for staying put
  long before that. Policies trained before this change have neither.
- **Creeping along the deck.** The steer run `a9x8z66m` trained 3000
  iterations with both. The creep was slower but still there: 14 of 16 envs
  crept forward at 0 to 0.66 cm/s, about 0.3 on average. With `board_slid`
  off for 60 s, none fell in the first 29 s, then 6 of 16 ran the front left
  foot off the nose. No randomised value explains the rate, and more solver
  iterations do not change it, so it comes from the gait. The policy cannot
  see the board, and the 8 cm end arrives 15 to 30 s after the creep starts,
  far past the 2 s that training looks ahead. So a new cost, `board_creep`
  (weight -1.0), charges the creep while it happens, with full cost at 2 cm/s.
  The slide's rate goes through two 1 s averages in a row, so the stride's own
  rocking mostly cancels and only a steady creep is left. 0.3 cm/s costs 0.15
  a step. Policies trained before this change do not have it.
- **Slipping on the deck.** `board_creep` did not cut the creep in 3000
  iterations (run `8tklg4k7`: 0.68 cm/s straight, 1.89 in a left carve).
  More solver iterations did not help, but a different friction cone did.
  MuJoCo's soft contacts let a loaded foot slip slowly under a steady sideways
  load, and real grip tape holds. The tasks already use impratio 10, so the
  change is the cone alone. With `a9x8z66m`, 16 envs, 30 s, 0.3 m/s, yaw 0:

  | MuJoCo contact options | `board_slid` resets | mean creep (cm/s) | mean size |
  | --- | --- | --- | --- |
  | task default (pyramidal cone) | 12 | +0.29 (to the nose) | 0.31 |
  | solver iterations 30 | 12 | +0.27 | 0.31 |
  | elliptic cone | 4 | -0.13 (to the tail) | 0.16 |

  About half of the creep was this slip, and the policy learned a bias against
  it. The split balance, ride and steer now use the elliptic cone with
  impratio 10 (`SPLIT_CONTACT_CONE`, `SPLIT_CONTACT_IMPRATIO`). The centred
  tasks and the walking tasks keep the shared options. Earlier split policies
  were trained with the default contacts.
- **Steer v5 (`qqm2917n`, then `bxke67fj`).** Resumed from v4 `8tklg4k7`, and
  the first steer run trained with the elliptic cone (commit `63ed76a`).
  `qqm2917n` crashed at iteration 23850 when the Colab runtime dropped;
  `bxke67fj` ran the last 150 iterations, to `model_23999.pt`. Gait check on
  the 103 mm deck at 0.3 m/s, 16 boards:

  | Yaw command | Gait check | Resets | Board yaw rate / asked | Worst fore-aft slide | Fastest creep |
  | --- | --- | --- | --- | --- | --- |
  | 0 | pass | 0 | not asked | 6.6 cm | 0.29 cm/s |
  | +0.5 (carve left) | pass | 0 | 0.92 of 0.050 rad/s | 7.0 cm | 0.21 cm/s |
  | -0.5 (carve right) | pass | 0 | 0.84 of 0.050 rad/s | 6.7 cm | 0.35 cm/s |

  The slide limit is 8 cm. v4 for comparison: 3 to 4 resets per gait report,
  and 0.39 of the asked rate carving left. In a 10 s render there were no
  resets, and the board's heading changed +0.50 rad carving left and -0.44 rad
  carving right. Videos: `media/split_steer_v5_straight_0.3ms.mp4`,
  `media/split_steer_v5_carve_left_0.3ms.mp4` and
  `media/split_steer_v5_carve_right_0.3ms.mp4`.
- **Steer v5 ride audit: fail on 2 of 16 rows** (yaw 0, 0.3 m/s, deck 103 mm).
  In `nominal_30s`, 2 of 16 boards reset on `board_slid`. In `truck_x4`, 2
  reset; those trucks are 4 times stiffer, outside the range steering needs
  anyway. The other 14 rows pass, 13 of them with no reset (`shove_fwd_0.5`
  had 1). v3 `a9x8z66m` failed 6 rows. Log:
  `policies/split_steer_v5_audit.txt`.
- **What that meant for the robot.** v5 was the candidate for the first
  hardware steering test before v6, and is now the fallback. There will be
  no foot stop or ridge on the deck, so staying on the board has to come
  from the policy.
  Sliding along the deck is an open risk: in sim 2 of 16 boards slide 8 cm
  within 30 s. Keep runs short and reset the robot to its marks between runs.
  Use the full yaw stick only (v5 trained only on full carves). Loosen the
  trucks to about 9 to 18 N·m/rad first. A hardware run needs Ethan's OK and
  a TA sign-off; neither has been given.
  Policy: `policies/split_steer_v5_max0.3_policy.json` (speed 0 to 0.3 m/s,
  yaw clipped at 0.6 rad/s). Checkpoint:
  `policies/split_steer_v5_model_23999.pt`.
- **Steer v5 on a lower board: it still passes.** Same gait check (0.3 m/s,
  16 boards, 8 s) with the deck top lowered, as smaller wheels or trucks
  would do. No retraining. Every cell is a pass with no reset.

  | Deck top | Straight: worst slide | Carve left: turn, slide | Carve right: turn, slide |
  | --- | --- | --- | --- |
  | 103 mm (now) | 6.6 cm | 0.92 of asked, 7.0 cm | 0.84 of asked, 6.7 cm |
  | 95 mm | 6.8 cm | 0.83 of asked, 6.3 cm | 0.78 of asked, 6.7 cm |
  | 85 mm | 7.4 cm | 0.77 of asked, 6.0 cm | 0.73 of asked, 6.7 cm |
  | 75 mm | 7.6 cm | 0.76 of asked, 6.0 cm | 0.65 of asked, 7.0 cm |

  Two things get worse as the deck drops. The board turns less for the same
  command (0.65 to 0.76 of asked at 75 mm), and the feet slide further when
  riding straight (7.6 cm, against a reset limit of 8 cm). So a lower board
  works with this policy but is not clearly safer in sim. The ride audit was
  not rerun at the lower heights; it passes at 93 and 113 mm.
- **Steer v6 (`hfd3qhwr`).** Resumed from v5 `bxke67fj` for 3000 iterations
  on the current steer task (commit `d3ec53d`), to `model_26998.pt`. It
  trains carves of 0.1 to 0.6 rad/s either sign, asks the board for a tenth
  of the command, and clips the heading hold at 0.1. Gait check on the 103 mm
  deck at 0.3 m/s, 16 boards, seed 0:

  | Yaw command | Gait check | Resets | Board yaw rate / asked |
  | --- | --- | --- | --- |
  | 0 | pass | 0 | not asked |
  | +0.15 (carve left) | pass | 0 | 0.84 |
  | -0.15 (carve right) | pass | 0 | 0.82 |
  | +0.3 | **fail** | 1 `board_slid` (env 12) | 0.58 |
  | -0.3 | pass | 0 | 0.63 |
  | +0.5 | **fail** | 1 `board_slid` (env 12) | 0.67 |
  | -0.5 | pass | 0 | 0.86 |
  | +0.6 | **fail** | 1 `board_slid` (env 12) | 0.80 |
  | -0.6 | pass | 0 | 0.86 |

  The worst fore-aft slide is 6.2 cm straight, up to 8.0 cm on left carves
  (the reset limit is 8 cm) and 5.3 to 5.8 cm on right carves. The three
  failures are the same randomized board, env 12. Left carves at +0.3 and
  +0.5 on seeds 1 and 2 all pass with no reset (0.57, 0.61, 0.94 and 0.76 of
  asked). So 1 board in 48 slid past the limit on left carves.
- **Steer v6 ride audit: pass on 16 of 16 rows** (yaw 0, 0.3 m/s, deck
  103 mm). v5 failed 2 (`nominal_30s` and `truck_x4`). Two rows had 1
  `board_slid` reset each (`nominal_30s`, `shove_fwd_0.5`), inside the
  audit's tolerance. Log: `policies/split_steer_v6_audit.txt`.
- **Steer v6 renders.** 10 s each, 0.3 m/s, no resets. Straight, the board
  travels 2.92 m. The board's heading changed +0.17 rad at stick 0.15, +0.63
  rad at stick 0.6 and -0.20 rad at stick -0.3. Videos:
  `media/split_steer_v6_straight_0.3ms.mp4`,
  `media/split_steer_v6_left_stick0.15_0.3ms.mp4`,
  `media/split_steer_v6_left_stick0.6_0.3ms.mp4` and
  `media/split_steer_v6_right_stick0.3_0.3ms.mp4`.
- **Steer v6 on a lower board.** Same gait check (seed 0) at yaw 0, +0.3 and
  -0.5, deck top 75, 85 and 95 mm, no retraining. Yaw 0 and -0.5 pass at all
  three. +0.3 has the same single `board_slid` reset (env 12) at all three.

  | Deck top | Carve +0.3: turn | Carve -0.5: turn |
  | --- | --- | --- |
  | 95 mm | 0.83 of asked | 0.91 of asked |
  | 85 mm | 0.69 of asked | 0.87 of asked |
  | 75 mm | 0.81 of asked | 0.79 of asked |

- **v6 against v5.** v6 turns in proportion to the stick; v5 needed the full
  stick. v6 passes the full ride audit; v5 failed 2 rows. But v6 turns less
  than asked at mid stick (about 0.6 at 0.3), and at +0.5 it is weaker than
  v5 was (v5: 0.92 and 0.84 of asked at ±0.5, no resets on seed 0).
- **What v6 means for the robot.** v6 is the new candidate for the first
  hardware steering test, because partial stick works in sim. v5 is the
  fallback. Sliding along the deck is still the main open risk: there is no
  foot stop or ridge on the deck, and one sim board in 48 slid past the limit
  on left carves. Keep runs short, start with a small stick, and reset the
  robot to its marks between runs. A stick right at 0.1 sits between the
  heading hold and the smallest carve, so avoid holding it there. Loosen the
  trucks to about 9 to 18 N·m/rad first. A hardware run needs Ethan's OK and
  a TA sign-off; neither has been given.
  Policy: `policies/split_steer_v6_max0.3_policy.json` (speed 0 to 0.3 m/s,
  yaw clipped at ±0.6 rad/s, heading hold kp 1.0, clip 0.1). It passes the
  robot-side validator (`download_policy.py --validate-only`). Checkpoint:
  `policies/split_steer_v6_model_26998.pt`.

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

## What changed from v4

- Split steer v5 (`qqm2917n`, then `bxke67fj`) steers by leaning and is the
  candidate for the first steering test. The steering section has its gait
  check, render and ride audit.
- Policy and media files have descriptive names now; the name to run table is
  [`policies/README.md`](policies/README.md).
- The steering section has a table of steer v5 at deck tops from 75 to
  103 mm, for the choice of wheels and trucks.

## What changed from v5

- Two new team constraints. The operator's yaw stick can send any value, so
  the steer policy must train on partial turns. There will be no foot stop or
  ridge on the deck, so staying on the board has to come from the policy.
- The steer task now trains partial turns: carves of 0.1 to 0.6 rad/s, not
  only 0.4 to 0.6. The board yaw it asks for is a tenth of the command, so
  0.01 to 0.06 rad/s, a half stick for a half turn.
- The steer task clips the heading hold's correction at 0.1 rad/s, not 0.3.
  That equals the hold's yaw threshold, so a command of 0.1 or less is a
  correction and a larger one is the stick. The carve threshold is 0.1 too.
  Steer exports stamp this clip in `heading_hold`; the ride and balance keep
  0.3.
- The board yaw reward's width is 0.02 rad/s, not 0.04, so a partial carve
  and a full one pay differently.
- Split steer v6 (`hfd3qhwr`) trained on these changes: 3000 iterations
  resumed from v5. The steering section has its gait check on three seeds,
  ride audit (16 of 16), renders and a deck height sweep.
- v6 replaces v5 as the candidate for the steering test, since partial stick
  works. v5 is unchanged, still needs the full yaw stick, and is the fallback.
- The plan's step 9 and the file table now name the v6 file.
- Every foot stop recommendation is gone. Sliding along the deck is an open
  risk instead (see "What that means for the robot").
- The tools now accept any yaw inside the steer command clip, and the gait
  report prints the asked board yaw rate for any carve.
- The sim tools replay a `policy.json` with the heading-hold clip in its own
  `heading_hold` block, so v5's file runs with 0.3, as on the robot. For a
  checkpoint, pass `--hold-clip 0.3` to replay v5. With a 0.3 clip the tools
  refuse a yaw of 0.1 to 0.3, where v5 cannot tell the stick from a
  correction.

## Accounting

- **Skipped:** any hardware test; a stance audit for the ride policies; shoves
  above 0.6 m/s and deck friction below 0.3 (v1 ran them for the old policy).
- **Assumed:** truck stiffness 30 N·m/rad; the 3 N·m torque limit and the
  12 rad/s speed estimate; the audit's pass limits (60% torque, 3 cm margin);
  that lowered trucks put the deck at 68 to 83 mm.
- **Issues:** centred ride has no passing policy yet (`7fyt12i1` fails the
  push foot duty limit); the centred stance fails 7 audit rows from lack of sideways room;
  split steer v6 has one `board_slid` reset in 48 boards on left carves and
  turns about 0.6 of asked at mid stick, and with no foot stop on the deck,
  sliding along it is an open risk; a stick right at 0.1 sits between the
  heading hold and the smallest carve;
  hand-placing the robot in the split stance, the robot-side loader and how the
  ride speed is commanded on the robot are unverified.
