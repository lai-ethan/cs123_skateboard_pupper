# First hardware session: test sheet

Date: ________  People: ________________________

Tests 1 and 2 use policies that pass the gait check in sim on the 103 mm deck. Test 3's
policy passed it on an older task and resets on sideways drift on the current one. Test
4 uses one that fails one limit. Test 5 steers: it passes the gait check but fails 2 of
16 ride audit rows. None has run on the robot. Sim first, then robot.

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

- Policy: `policies/split_steer_v5_max0.3_policy.json` (runs `qqm2917n`, then
  `bxke67fj` for the last 150 iterations). It clips speed at 0.3 m/s and yaw
  at 0.6 rad/s. Install it like the capped file in test 3.
- This is a candidate for the first steering test, not a fully passing policy.
  Gait check: `GAIT: PASS` at yaw 0, +0.5 and -0.5 with no resets. Ride audit:
  `FAIL` on 2 of 16 rows. In a 30 s ride the robot slid 8 cm along the deck
  on 2 of 16 sim boards, about 1 in 8.
- Needs Ethan's OK and a TA sign-off for this test, and test 3 passed first.
- Trucks: loosen the kingpin nuts until test 0 reads 9 to 18 N·m/rad. The
  policy trained only on that range. The stock trucks have not been
  measured; they are assumed much stiffer, and then the board will not turn.
- Fit a foot stop or a small ridge on the deck in front of and behind the left
  feet. In sim the left feet slowly slide along the deck toward the nose or
  tail. Keep each run short, about 10 s.
- Placement: left feet on the deck, right feet on the floor, as in test 3.
  Place the robot square to the board, body parallel to the deck.
- This file has the heading hold on (kp 1.0, clip 0.3 rad/s). On the robot the
  hold keeps the robot's own IMU yaw, not the board's heading: it cannot see
  the board. Any yaw between robot and board when the hold captures stays
  locked in. After each carve, check the body is still parallel to the board
  before the next run. The other skate files have the hold off.
- Speed: the forward stick, clipped at 0.3 m/s. Sim checked only 0.3 m/s.
- First ride straight at yaw 0. Then carve: a yaw command of 0.4 to 0.6 rad/s,
  either way. The file clips yaw at 0.6, so a carve is the full yaw stick.
  Push the yaw stick all the way, quickly; to stop the carve, let it go fully.
  Do not hold it part way or move it slowly: 0.1 to 0.35 rad/s reaches the
  policy and it never trained there. Back to yaw 0 holds the new heading (the
  robot's own, see above).
- Expect a slow lean turn: about 3 degrees per second, a wide arc of about 6.5
  to 7 m radius at 0.3 m/s (sim board yaw rate 0.046 rad/s left, 0.042
  right). In sim, 10 s of carving turned the board 0.50 rad left and 0.44 rad
  right.
- Watch: the robot leans its weight toward one deck edge; the board turns that
  way; left feet stay put on the deck.
- Pass: the board turns each way for a few seconds with no fall, then rides
  straight again at yaw 0. Stop: a left foot slides toward the nose or tail or
  off the deck, the robot turns but the board does not, or fast stamping legs.
- Notes straight: ____________________________________________
- Notes carve left: ____________________________________________
- Notes carve right: ____________________________________________
