# Checkpoint summary: Monday Nov 16

Both skateboard policies are trained in simulation and saved to W&B. Nothing has run on the real robot yet.

Balance is the only robot candidate. The ride policy passes the curve check but fails the gait check: its legs move too fast and its motors sit at the torque limit. Ride will be retrained with reshaped rewards before it goes near the robot. That retrain has not run yet.

## What is done

| Policy | W&B run | Sim result |
| --- | --- | --- |
| Balance (with shoves and deck-height randomization) | `ethanlai-stanford-university/mjlab/2ace1uqa` | PASS: episode length 500 of 500, feet on deck 2.00 of 2 |
| Balance (no shoves, backup) | `ethanlai-stanford-university/mjlab/2zxd46yi` | PASS |
| Ride (push the board at 0 to 0.5 m/s) | `ethanlai-stanford-university/mjlab/29p4j81t` | Curves PASS: episode length 487 of 500, speed tracking 2.38 of 3 (need 2.0), feet on deck 1.87 of 2. Gait FAIL: not for the robot |

- Board model, split stance (left feet on deck, right feet on floor), both tasks, tests and the Colab notebook are in this repo.
- Default reward weights passed the curve check for both tasks. For ride, that check missed the gait.
- Each run has a `policy.json` in its W&B Files. Only the balance ones are for the robot.
- Gait check (`skateboard/tools/gait_report.py`): balance prints `GAIT: PASS`; ride at 0.4 m/s prints `GAIT: FAIL` (99th-percentile joint speed 24 to 26 rad/s against a limit of 8, a motor at the torque limit in almost every step, and the right feet stepping more than 2 times per second). The numbers change between runs because the play env is unseeded. Single rows can flip; the overall verdict has not.
- The ride rewards were reshaped for the motors (PLAN Task 16). Ride needs a fresh 1500-iteration run with them; see PLAN Task 11.
- The ride run is the full 1500 iterations (Colab Pro T4, 55 min). An earlier ride run, `me6p0pdw`, stopped at iteration ~1120; use `29p4j81t`.
- Sim videos of both policies are in `skateboard/media/`; the policy files are in `skateboard/policies/`. To replay them, see step 4 of `skateboard/README.md`.

## What to do Monday

0. **Measure the deck height before powering on.** The policy was trained for 8.5 to 11.5 cm (floor to top of deck). Outside that, do not run it; retrain with the real number first.
1. **Show a TA the sim result and get a sign-off.** Balance first. No skate policy runs on the robot without it.
2. **Wedge or tape the wheels** so the board cannot roll on the first try.
3. **Deploy balance** on the robot over SSH:
   ```bash
   cd ~/pupper_gait_deploy
   ./deploy.sh ethanlai-stanford-university/mjlab/2ace1uqa
   ```
4. **First run:** one person holds the robot by the handle in the stance on the board, another starts the policy with the kill switch in hand. Let go only if it holds still for 5 s.
5. **If it shakes:** retrain with `action_rate_l2` at -0.05 and `KP_MULTIPLIER_RANGE` widened to `(0.5, 1.2)` (PLAN Task 12).
6. **If balance holds:** unwedge the wheels and repeat. Do not deploy ride (`29p4j81t`). Ride goes on the robot only after a retrained run prints both `RESULT: PASS` and `GAIT: PASS`, at the lowest speed.
7. **Measure the board** (deck height, wheelbase, width, mass) and send the numbers. The sim uses estimates; see README step 1.

## Known gaps

- The robot-side deploy (`pupper_gait_deploy`) has never been run with a skate policy. Not checked in sim: torso touching the deck, and truck steering direction against the real board.

- The joint speed limit in the ride rewards and the gait check (8 rad/s) is a first guess, not a measured motor limit.
- Board dimensions in sim are estimates. Deck height is randomized ±1.5 cm to cover the error.
- To retrain: open `notebooks/CS123_Pupper_Skate.ipynb` in Colab, paste your own W&B key, Run all. About 55 min for 1500 iterations on a T4, about 9 min on a G4. Never commit a key.
- `Metrics/twist/error_vel_xy` in W&B is not a tracking error; it grows with episode length. Read `Episode_Reward/track_board_velocity` instead.

## What changed from v1

- Ride was retrained for the full 1500 iterations (run `29p4j81t`, Mon Nov 16, 13:01 CST) and has a real `RESULT: PASS`. It replaces the stopgap policy from iteration 1100 of `me6p0pdw`.
- The notebook's section 4 now lists every reward weight, so tuning is a one-number edit.
- Later on Mon Nov 16 the ride gait was checked and fails: it is not robot-ready. The ride rewards were reshaped and the notebook gained a gait check. Ride is no longer next in line for the robot; it will be retrained first.

## Accounting

- Skipped: nothing on the robot; `/code-review` (not available in this setup).
- Assumed: sim board dimensions (estimates until measured).
- Issues: reward weights are the first guess and untested on hardware; expect to tune them after the first robot runs. The reshaped ride rewards have not been trained yet.
