# First hardware session: test sheet

Date: ________  People: ________________________

Tests 1 and 2 use policies that pass the gait check in sim on the 103 mm deck. Test 3's
policy passed it on an older task and resets on sideways drift on the current one. Test
4 uses one that fails one limit. Test 5 steers: it passes all 16 ride audit rows but
one sim board in 48 slid off on left carves. Its fallback, v7, slides less but fails
hard right carves. None has run on the robot. Sim first, then robot.

## Safety, before any power

- [ ] Ethan's OK for this session.
- [ ] TA sign-off for the first powered run (PLAN Task 12).
- [ ] One person holds the robot's harness or has hands ready the whole run.
- [ ] Balance tests: board on carpet, or wheels chocked.
- [ ] Everyone knows where the e-stop (kill switch) is; one person holds it.
- [ ] First run of each policy at the lowest speed.
- [ ] Deck top measured at 103 mm. If not, stop: the gait checks used 103 mm.
- [ ] Fit test with the robot powered off (SIM2REAL "Plan before the first real
      run", step 2).

Deploy: install the local file named in each test, from `policies/`, as in
test 3. Tests 3 to 5 need the capped `*_max0.3_policy.json` files; tests 1 and
2 use the `_policy.json` files. Do not use `./deploy.sh mjlab/<run>` for these
tests: it fetches the run's W&B export, which is not the capped file. The
file table and commands are in [`SIM2REAL.md`](SIM2REAL.md) ("Plan before the
first real run"). The robot-side commands have not yet been run with a skate
policy. Do not improvise other commands.

Stop any test at once if: a foot slips off the deck edge, a joint buzzes or
shakes, the robot leans past the hands, or anything smells hot.

## 0. Board tilt stiffness (robot off, no power)

Decides whether steering is possible at all (SIM2REAL "Steering: what sim
says"). Takes five minutes.

- Board on a hard floor, wheels chocked. Phone with a level app flat on the
  deck, across the board.
- Hang or press a known weight on one long edge of the deck, midway between
  the trucks: 1 kg, then 2 kg, then 3 kg (the robot is about 3 kg). Read the
  tilt each time.
- Repeat with both kingpin nuts loosened one full turn.
- Steering by lean needs about 7.5 to 15 degrees at 3 kg (9 to 18 N·m/rad,
  worked out below). Under 7.5 degrees the trucks are stiffer than the steer
  policy trained on; over 15 degrees they are softer (under 9 N·m/rad). Either
  way, do not run test 5.
- Tilt at 1 / 2 / 3 kg, stock: ______ / ______ / ______ degrees
- Tilt at 1 / 2 / 3 kg, loosened: ______ / ______ / ______ degrees
- Stiffness k = m x 9.81 x d / tilt, with m in kg, d the weight's distance from
  the centre line in m and the tilt in rad. 3 kg on the edge (d about 0.0805 m,
  half the 161 mm measured deck; the sim's traced outline is 157.1 mm) is
  2.4 N·m, so 9 to 18 N·m/rad, the range test 5
  needs, is about 7.5 to 15 degrees at 3 kg.
- k at 3 kg, loosened: ______ N·m/rad

## 1. Split balance

- Policy: `policies/split_balance_v3_policy.json` (run `gt60lfo0`)
- First because two feet stay on the floor, so a failure is a small one.
- Placement: left feet on the deck, right feet on the floor. Wheels chocked.
- Watch: right feet stay planted; lean about 5 degrees toward the deck, as in
  sim; board does not creep.
- Pass: holds still for 10 s with hands off. Stop: a right foot lifts, the
  left feet slide, or it shakes.
- Notes: ____________________________________________

## 2. Centred balance

- Policy: `policies/centred_balance_v2_policy.json` (run `zdctdcyf`)
- Placement: all four feet on the deck. Centre the board under the robot to
  within about 1 cm. Wheels chocked.
- Watch: feet stay on the deck, not the edge; body still; no shaking.
- Pass: holds still for 10 s with hands off. Stop: any foot near the edge, any
  shake, any sideways drift.
- Notes: ____________________________________________

## 3. Split ride at 0.2 m/s, then 0.3 m/s

- Policy: `policies/split_ride_v4_policy.json` (run `iznjwbka`)
- It passed the gait check on 2026-11-17, on the old box deck. It has not
  passed on the current task: its checkpoint resets on `board_drifted`, the
  board walking sideways out from under the feet. Run it only if Ethan
  accepts that. Keep runs short.
- Only if test 1 passes. Wheels free, clear straight path, hands ready.
- Placement: left feet on the deck, right feet on the floor.
- Speed command: the forward stick. Full stick asks for 1.5 m/s and the policy
  clips it. Use `policies/split_ride_v4_max0.3_policy.json` for this
  session: it clips at 0.3 m/s, so a full push cannot go faster. A light push
  (about one eighth of the stick) is 0.2 m/s. The uncapped file reaches
  0.5 m/s, which has no gait check.
- Install the capped file: copy it to the robot, then in
  `~/pupper_gait_deploy` run
  `python3 download_policy.py --file <file> --output policies/test_policy.json`
  and `./deploy.sh`. Press square to start the policy; right-stick press is
  the e-stop.
- Watch: slow planted pushes with the right feet, about 2 strides per second
  at most; left feet stay on the deck; board rolls straight. Nothing steers.
- Pass at 0.2 m/s: several pushes, board moves, no fall. Then try 0.3 m/s.
  Stop: fast or stamping legs, a left foot off the deck, the board curving
  away, or the board walking sideways from under the left feet.
- Notes at 0.2 m/s: ____________________________________________
- Notes at 0.3 m/s: ____________________________________________

## 4. Centred ride, tripod (provisional: Ethan decides)

- Policy: `policies/centred_ride_v9_max0.3_policy.json` (run `7fyt12i1`).
  It clips at 0.3 m/s. Install it like the capped file in test 3.
- Gait check: `GAIT: FAIL` on one limit. The push foot is on the floor 20% of
  the time (limit 40%). Every other limit passes at 0.1, 0.2 and 0.3 m/s
  (SIM2REAL "Riding: where it stands"). Run this test only if Ethan accepts
  that, and only after tests 1 to 3.
- Placement: front right, front left and back left feet on the deck, body
  centred within about 1 cm. Back right foot on the floor beside the board.
  It starts there; it does not step off the deck.
- Hands on the harness the whole run. The robot stands on three feet for most
  of the ride.
- Speed: a light push of the stick (0.2 m/s) first. Do not test 0.1 m/s: the
  pushes there are short taps.
- Watch: one short push with the back right foot about once a second, then a
  glide with that foot in the air; the other three feet stay on the deck;
  body upright.
- Pass: several push-and-glide cycles at 0.2 m/s with no fall. Stop: a deck
  foot slips toward the edge, the body tips toward the lifted foot, fast
  tapping, or the robot drops onto the deck.
- Notes: ____________________________________________

## 5. Split steer, lean carving (candidate: Ethan decides)

- Policy: `policies/split_steer_v6_max0.3_policy.json` (run `hfd3qhwr`,
  resumed from v5). It clips speed at 0.3 m/s and yaw at 0.6 rad/s, and
  passes the robot-side validator (`download_policy.py --validate-only`).
  Install it like the capped file in test 3.
- This is a candidate for the first steering test, not a fully passing policy.
  Ride audit: `PASS` on 16 of 16 rows. Gait check: `GAIT: PASS` at yaw 0,
  ±0.15, -0.3, -0.5 and -0.6. Left carves of +0.3, +0.5 and +0.6 fail on one
  sim board in 16 (the same one each time), which slid 8 cm along the deck.
  On two more seeds those left carves had no reset: 1 board in 48 in all.
- Fallback: `policies/split_steer_v5_max0.3_policy.json` (`bxke67fj`), full
  yaw stick only. It trained only on full carves and fails 2 of 16 ride
  audit rows. Its heading hold clip is 0.3, not 0.1.
- If the robot slides along the deck with v6: try
  `policies/split_steer_v7_max0.3_policy.json` (run `n5m2zsks`, resumed
  from v6). Keep the stick within ±0.3; the file clips yaw at ±0.3 rad/s
  anyway. It passes the validator and all 16 ride audit rows, and no sim
  board slid out in 27 gait runs. But it pulls right, gentle right carves
  turn up to 2.1 times what they ask, and hard right carves (stick -0.5,
  -0.6) fail in sim on every seed, so it does not get the full stick. The
  cause of the right pull is not known yet. Same heading
  hold as v6 (kp 1.0, clip 0.1). Same placement and trucks.
- Needs Ethan's OK and a TA sign-off for this test, and test 3 passed first.
- Trucks: loosen the kingpin nuts until test 0 reads 9 to 18 N·m/rad. The
  policy trained only on that range. The stock trucks have not been
  measured; they are assumed much stiffer, and then the board will not turn.
- There is no foot stop or ridge on the deck, and none will be fitted.
  Staying on the board comes from the policy alone. Sliding along the deck is
  the main open risk: in sim the left feet slowly slide toward the nose or
  tail, up to 8.0 cm on left carves. Keep each run short, about 10 s, and
  reset the robot to its marks between runs.
- Placement: left feet on the deck, right feet on the floor, as in test 3.
  Place the robot square to the board, body parallel to the deck.
- This file has the heading hold on (kp 1.0, clip 0.1 rad/s). On the robot the
  hold keeps the robot's own IMU yaw, not the board's heading: it cannot see
  the board. Any yaw between robot and board when the hold captures stays
  locked in. After each carve, check the body is still parallel to the board
  before the next run. The other skate files have the hold off.
- Speed: the forward stick, clipped at 0.3 m/s. Sim checked only 0.3 m/s.
- First ride straight at yaw 0. Then carve with a small stick, about 0.15
  rad/s, each way. Only then try more stick, up to full (0.6). Partial stick
  is allowed: v6 trained on carves of 0.1 to 0.6 rad/s, and a half stick asks
  for about half the turn. Do not hold the stick right at 0.1: that sits
  between the heading hold and the smallest carve. Back to yaw 0 holds the
  new heading (the robot's own, see above).
- Expect a slow lean turn. In sim the board turned about 0.8 of what it asks
  at stick 0.15 and 0.6, but only about 0.6 at stick 0.3. In 10 s renders
  the board's heading changed +0.17 rad at stick 0.15, +0.63 rad at stick
  0.6 and -0.20 rad at stick -0.3.
- Watch: the robot leans its weight toward one deck edge; the board turns that
  way; left feet stay put on the deck.
- Pass: the board turns each way for a few seconds with no fall, then rides
  straight again at yaw 0. Stop: a left foot slides toward the nose or tail or
  off the deck, the robot turns but the board does not, or fast stamping legs.
- Notes straight: ____________________________________________
- Notes carve left, small stick: ____________________________________________
- Notes carve right, small stick: ____________________________________________
- Notes carve, more stick: ____________________________________________
- Notes v7, if used (straight, then stick ±0.15, ±0.3): ______________________

## 6. Voice control (Gemini Pupper)

Not run yet. Needs Ethan's OK on the day, a TA's sign-off for the policy,
and tests 3 and 5 (split ride, steer by joystick) passed first. The voice
tools assume the ride and the steer work. Nothing in this test has run on the
robot: every distance and turn Gemini gives is a sim estimate.

Setup, by Ethan on the robot:

- [ ] `git pull` the `skate-voice` branch of gemini_pupper on the robot. The
      robot's copy may be ahead of the Mac snapshot: merge, do not overwrite.
- [ ] `python3 lab/test.py skate` on the robot: all checks pass.
- [ ] `robot/start.sh`, with Pupper lying on the floor, away from the board.
      In `/tmp/gemini-pupper/robot.log` the spawner lists `gemini_skate` as
      loaded.
- [ ] `ros2 control list_controllers` lists `gemini_skate` as inactive. It
      must not be active before "start skating".
- [ ] `ros2 param get /gemini_skate model_path` ends in `skate/policy.json`.
- [ ] Joystick connected. Right-stick press is the e-stop and outranks
      Gemini.

Run, hands on the harness the whole time, 4 m of clear floor ahead:

- [ ] Say "lie down". Place Pupper: left feet on the deck, right feet on the
      floor, body parallel to the board. Say "start skating". Pass: Gemini
      asks whether Pupper is placed before it starts; Pupper eases into the
      stance and balances for 10 s without rolling.
- [ ] Say "walk forward". Pass: Gemini refuses and says why. Pupper does not
      move.
- [ ] Say "go forward". Pass: rides about 3 s, then stops pushing.
- [ ] After that ride, watch the right legs for 5 s. Pass: they stay planted
      and do not push again; the zero command really stops the pushing.
- [ ] Say "stop" in the middle of a ride. Pass: pushing stops within 1 s, and
      the right legs stay still afterwards.
- [ ] Say "turn left", then "turn right". Pass: the board curves the right
      way; Pupper stays on.
- [ ] Tip test, still balancing (no ride queued), both hands on the harness:
      tilt Pupper slowly until the tilt e-stop fires. Say "walk forward", then
      "do a trick". Pass: both are refused and Pupper does not move. Hold
      Pupper, say "get off the board". Pass: the motors relax. Lift Pupper
      onto the floor, say "stand up". Pass: it stands.
- [ ] Hold the harness, say "get off the board". Pass: Gemini asks if Pupper
      is held, then the motors relax. If a ride is still queued, end_skating
      waits for it: say "stop" first to end at once.

Stop the test (e-stop) if a left foot nears the deck edge, a right foot
lands on the deck, the board heads for anything, or a motor faults.

Measure and write down, to replace the sim estimates in `skate/skate.yaml`:

| What | Sim estimate | Measured |
| --- | --- | --- |
| Distance in a 3 s ride (tape on the floor) | 3 × `travel_m_per_s` = 3 × 0.29 = about 0.9 m | |
| Roll after the last push | `coast_m` = 0.1 m | |
| Heading change in a 5 s hard carve, each way | 5 × `turn_deg_per_s.hard` = 5 × 2.2 = about 11° | |

The sim numbers come from `skateboard/tools/timed_ride.py` on steer v6 (16 envs,
5 s ride, no resets at any carve level). The turn is the weaker side's.
