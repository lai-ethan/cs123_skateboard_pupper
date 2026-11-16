# Sim2real: will Pupper stay balanced in this pose?

The TA's concern after the checkpoint report: "I doubt if Pupper can remain balanced
at this pose on the board." This note is the answer so far. It is version 1,
written 2026-11-16, from simulation only. Nothing here has been tried on the robot.

## Short answer

- **Standing: yes in sim, with room to spare, except for one thing.** The centre of
  mass sits about 7 cm inside the four feet and the motors use 16% of their torque.
  The stance survives a slippery deck, loose or stiff trucks, weak motors, a board
  placed 7 cm off, and shoves up to 0.6 m/s.
- **The exception is deck height.** If the real deck is 2.5 cm or more higher than
  the model, the trained policy drives the motors to their torque limit. The deck
  height in the model (0.10 m) is still a guess. Measure it before anything else.
- **Pushing: not yet.** Lifting a right foot leaves the centre of mass 1 to 2 cm
  outside the three feet that remain. A push only works if the body shifts toward
  the deck first, or the foot is back down before the robot tips. The saved ride
  policy does neither in a way the motors can follow.

## How it was measured

`skateboard/tools/stance_audit.py` runs the saved balance policy for 10 s per
condition and changes one thing at a time. All randomization is switched off, so
the only difference between rows is the condition named.

```bash
# Mac, in ~/claude/cs123/pupper-mjlab (about 25 s per condition)
FORCE_CPU=1 uv run python skateboard/tools/stance_audit.py
```

A row passes when the robot never resets, the left feet stay on the deck at least
90% of the time, no motor averages above 60% of its torque limit, and the centre of
mass stays at least 3 cm inside the feet.

- **torque** is the busiest motor's mean torque as a share of its limit.
- **margin** is how far the centre of mass is inside the outline of the four feet.
- **lift FR / lift RR** is the same distance with the front-right or rear-right foot
  removed. Negative means the robot would tip if it stood still on three feet.

## Results

Policy: `policies/balance_2ace1uqa_policy.json`. Every row had zero resets unless
noted.

| Condition | Result | Torque | Margin | Note |
| --- | --- | --- | --- | --- |
| Baseline | pass | 0.16 | 7.1 cm | lift FR -2.0 cm, lift RR -1.2 cm |
| No policy (motors hold the pose) | pass | 0.16 | 7.5 cm | the pose holds on its own |
| Deck 3 cm lower | pass | 0.17 | 6.9 cm | |
| Deck 5 cm lower | pass | 0.21 | 7.5 cm | |
| Deck 1 to 2 cm higher | pass | 0.16 | 7.1 to 7.4 cm | |
| **Deck 2.5 cm higher** | **fail** | **0.98** | 9.0 cm | motors at their limit |
| **Deck 3 cm higher** | **fail** | **0.96 to 1.00** | 8.7 to 9.2 cm | three runs, all fail |
| Deck 3 cm higher, no policy | pass | 0.42 | 4.9 cm | the policy is the problem |
| Deck 5 cm higher | pass | 0.58 | 4.9 cm | just under the torque limit |
| Trucks 4 times softer | pass | 0.16 | 6.9 cm | |
| Deck friction 0.3 | pass | 0.16 | 7.0 cm | 0.15 also passes |
| Motor stiffness 60% | pass | 0.15 | 6.9 cm | 40% and 25% also pass |
| Torque limit 1.0 N·m (a third) | pass | 0.48 | 7.1 cm | |
| Torque limit 0.6 N·m | fail | 0.82 | 7.1 cm | still standing |
| Torque limit 0.45 N·m | fail | 1.00 | 6.9 cm | still standing |
| Board 4 cm left or right | pass | 0.16 | 7.0 cm | 7 cm also passes |
| Board turned 0.2 rad | pass | 0.16 | 7.0 cm | |
| Joints start 0.3 rad off | pass | 0.16 | 6.9 cm | |
| Shove 0.6 m/s left or right | pass | 0.16 | 6.9 cm | |
| Shove 1.2 m/s left or right | fail | 0.16 to 0.25 | | one reset each |
| Board kicked 0.8 m/s | pass | 0.14 | 5.9 cm | board rolls 8 cm |
| Board kicked 1.5 m/s | pass | 0.12 | 3.4 cm | board rolls 27 cm |
| Several errors at once | pass | 0.24 | 7.3 cm | deck +2 cm, soft trucks, low grip, weak motors |

The deck-height failure is not steady with height (3 cm fails, 5 cm passes), and
the pose alone holds at 3 cm. So the policy reacts badly to a deck taller than it
trained on (training covered plus or minus 1.5 cm). It does not show that the pose
is impossible there.

## What the simulation cannot tell us

- **Board dimensions.** Deck height, wheelbase, track, mass, truck stiffness and
  wheel drag are guesses (table in `README.md`). Deck height is the one the audit
  shows we are sensitive to.
- **Foot grip on the real deck.** The sim foot is a sphere on a flat plate. Grip
  tape, a curved deck and a kicktail are not modelled.
- **Leg backlash and flex.** The sim legs are rigid. Real legs sag under a sideways
  load, and this pose loads them sideways.
- **Sensor and timing error.** Latency is modelled from the walking tasks. It has
  not been checked for this pose.
- **Motor speed.** The sim motor has a torque limit and no speed limit. The 8 rad/s
  used in the gait check is a first guess.

## Plan before the first real run

Each step needs the one before it. No robot motion without Ethan's OK, and the TA
signs off before the first powered run (PLAN Task 12).

1. **Measure the board** with a ruler and a scale, update
   `skateboard_constants.py`, and rerun the audit. If the deck is more than 2 cm
   off, retrain balance.
2. **Widen deck-height randomization** on the next balance training run, from
   plus or minus 1.5 cm to plus or minus 3 cm, and rerun the audit on the result.
3. **Fit test, robot powered off.** Place the robot on the board by hand in the
   stance. Check that the feet reach, the left feet sit on the deck and not the
   edge, and nothing collides.
4. **Balance with the wheels wedged**, one person holding the robot, kill switch in
   hand. Compare the roll angle and motor load with the audit's baseline row.
5. **Balance with the wheels free**, same precautions.
6. **Ride only after a retrain** prints both `RESULT: PASS` and `GAIT: PASS` and
   the video shows slow planted pushes.

## For the ride retrain

The three-foot numbers say the push needs a body shift toward the deck. The ride
rewards now pay for a planted stance and penalize fast joints and saturated motors,
but nothing rewards the shift directly. If the retrain still tips or windmills, add
a reward for keeping the centre of mass inside the three planted feet while one
right foot is in the air.

## What changed from the checkpoint report

- The checkpoint report claimed the stance holds. This note adds how much margin it
  has, where it breaks, and what sim cannot show.
- New finding: the balance policy fails when the deck is 2.5 cm or more higher than
  modelled.
- New finding: standing on three feet is not stable in this pose.

## Accounting

- **Skipped:** any hardware test; retraining; a three-foot reward term.
- **Assumed:** the board dimensions marked "guess" in `README.md`; the 3 N·m torque
  limit; the pass limits of the audit (60% torque, 3 cm margin).
- **Issues:** deck height must be measured before the first run; pushing needs a
  weight shift the current ride policy does not do.
