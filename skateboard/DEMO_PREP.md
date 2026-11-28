# Test and demo prep (Demo Day 2026-12-03)

One page: what is verified, what is not, and the order to do things in. The
step-by-step sheets are [`HARDWARE_TEST.md`](HARDWARE_TEST.md) (robot tests by
gamepad) and gemini_pupper's `skate/DEMO.md` (the voice demo runbook, branch
`skate-voice`). This page does not replace them.

## What is verified, and how

| Thing | Sim | Desk (fake robot) | Real robot |
| --- | --- | --- | --- |
| Steer v6 ride and carve | 16 of 16 audit rows | n/a | 2026-11-26: rolled the board forward on carpet for a whole clip, by gamepad. Not measured. |
| Steer v10 (carpet drag) | 18 of 21 audit rows; 0.96 of asked speed at 6x wheel drag, 0.58 at 10x | n/a | 2026-11-26: rolled the board forward on carpet for a whole clip, by gamepad. Not measured. |
| Steer v13 (holds its line better) | 18 of 21 audit rows; 20 of 21 steering runs; 25 s drift 16.8 cm mean, 41.4 cm worst (v10: 29.0, 84.7) | n/a | Never run. |
| Voice skate tools (start, forward, turn, stop, end) | n/a | 629 checks pass, including the whole demo scripted for each of the three policies | Never run. |
| Stop, Disconnect, gamepad e-stop, tip-over on the board | n/a | Covered, including tests that fail when each one is deliberately broken | Never run through Gemini Pupper. |

Nothing has been measured on the robot yet: not speed, not drift, not turn
rate, not carpet drag. The distances and angles Pupper says aloud are sim
estimates.

## Known limits to expect

- **Drift.** In sim every policy ends a 25 s ride off to one side (v13 about
  16 cm on average). In the 2026-11-26 clips the board went forward and a bit
  to one side. Keep rides short and re-aim the board between them.
- **Turns are gentle.** Sim estimates are 1 to 2 degrees per second. A 3 s
  "hard" turn is a few degrees, a nudge, not a corner.
- **Carpet.** v6 was trained up to 3x wheel drag, v10 and v13 up to 10x. On a
  thick carpet prefer v10. Hard floor is inside every policy's range.
- **Placement decides everything.** Pupper stands beside the board on the
  board's right, body parallel to it, facing the nose; left feet on the deck's
  middle line (one ahead of the middle, one behind), right feet on the floor.
  The 2026-11-24 failure was placement, not the policy.

## Test session (before Demo Day), in this order

Every step that moves the robot needs Ethan's OK right then, and a TA's OK
before the first run of any policy file new to the robot.

1. **Board check, robot off** (`HARDWARE_TEST.md` test 0): confirm which end
   has the castle picture and that it is the nose; note truck tightness; pull
   the board on the demo floor with a luggage scale if one is to hand.
2. **Gamepad, steer v6** (test 5): placement, a straight ride on the tape
   line with a fixed camera, then gentle carves both ways. Write down nose
   travel in cm and seconds: these are the first real measurements.
3. **Gamepad, steer v10**, same ride, same floor. Compare with v6.
4. **Gamepad, steer v13**, only with a TA's OK for the new file. Look for a
   straighter line than v10. A slow steady curve with the stick centred may be
   IMU yaw drift: note the direction and size.
5. **Voice, steer v6** (test 6 and `skate/DEMO.md`), only after step 2 has
   passed. This is the first time the skate tools run on the robot, so treat
   it as a hardware test: a hand on the harness the whole time, the gamepad in
   a second person's hands.
6. Decide the demo policy from steps 2 to 4. Change it in `skate/skate.yaml`
   only if the chosen file has passed step 2-style riding with a TA's OK.

## Demo Day

- Run `skate/DEMO.md` top to bottom. Its "Before the day" list is the
  checklist; its "Script" is what to say.
- Default policy is `steer_v6`. `steer_v10` and `steer_v13` are options in the
  catalogue; `steer_v13` is marked sim only until step 4 passes.
- Two people: one holds the harness and places Pupper, one holds the gamepad.

## Five things to remember at the robot

1. Only the app's **Disconnect button** stops a ride. If the page closes or
   Wi-Fi drops, queued rides finish (at most 10 s): the **gamepad e-stop** is
   the stop.
2. After the gamepad e-stop the motors relax. Pupper stays "on the board"
   until someone holds it and says "end skating".
3. After changing `skate.yaml`: get Pupper **off the board**, then
   `robot/stop.sh`, then `robot/start.sh`. `start.sh` alone does not load the
   new policy; if it is tried, skate mode switches itself off and says so.
4. Stopping the robot software lets the legs go limp. Hold Pupper first.
5. To get off: "stop", a person holds Pupper, then "end skating".

## Accounting

- Skipped: any robot run; measuring real carpet drag, speed or drift; a
  backward gait.
- Assumed: Demo Day floor is unknown, so both a low-drag (v6) and a
  carpet-trained (v10) policy stay ready; `steer_v6` stays the default until
  the test session says otherwise.
- Issues: the voice tools and steer v13 have never run on the robot; whether
  the castle end is the nose on the real board is not recorded anywhere.
