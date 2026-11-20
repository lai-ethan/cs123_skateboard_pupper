# First hardware session: test sheet

Date: ________  People: ________________________

Tests 1 to 3 use policies that pass the gait check in sim on the 103 mm deck. Test 4
uses one that fails one limit. None has run on the robot. Sim first, then robot.

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

Deploy: use the commands in [`SIM2REAL.md`](SIM2REAL.md) ("Plan before the
first real run") and PLAN Task 12. Do not improvise other commands.

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
- Steering by lean needs about 6 degrees at 3 kg. Under 3 degrees: do not
  train steering on this board.
- Tilt at 1 / 2 / 3 kg, stock: ______ / ______ / ______ degrees
- Tilt at 1 / 2 / 3 kg, loosened: ______ / ______ / ______ degrees

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
  away.
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
