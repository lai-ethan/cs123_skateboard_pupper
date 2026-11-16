# Checkpoint summary: Monday Nov 16

Both skateboard policies are trained in simulation and saved to W&B. Nothing has run on the real robot yet.

## What is done

| Policy | W&B run | Sim result |
| --- | --- | --- |
| Balance (with shoves and deck-height randomization) | `ethanlai-stanford-university/mjlab/2ace1uqa` | PASS: episode length 500 of 500, feet on deck 2.00 of 2 |
| Balance (no shoves, backup) | `ethanlai-stanford-university/mjlab/2zxd46yi` | PASS |
| Ride (push the board at 0 to 0.5 m/s) | `ethanlai-stanford-university/mjlab/me6p0pdw` | Passing by the curves at iteration ~1120 of 1500 |

- Board model, split stance (left feet on deck, right feet on floor), both tasks, tests and the Colab notebook are in this repo.
- Default reward weights worked for both tasks, so they were not tuned.
- Each run has a `policy.json` in its W&B Files, ready for the robot.
- Ride numbers when it stopped: speed tracking about 2.4 (need 2.0, max 3.0), feet on deck about 1.9, episode length about 485 (need 450).

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
6. **If balance holds:** unwedge the wheels and repeat. Ride (`me6p0pdw`) comes after that, at the lowest speed.
7. **Measure the board** (deck height, wheelbase, width, mass) and send the numbers. The sim uses estimates; see README step 1.

## Known gaps

- The robot-side deploy (`pupper_gait_deploy`) has never been run with a skate policy. Not checked in sim: torso touching the deck, and truck steering direction against the real board.

- The ride run was cut off at iteration ~1120 by Colab's free GPU limit, so it has no automatic PASS line. Its `policy.json` was re-exported from the last checkpoint (`model_1100.pt`).
- Board dimensions in sim are estimates. Deck height is randomized ±1.5 cm to cover the error.
- To retrain: open `notebooks/CS123_Pupper_Skate.ipynb` in Colab, paste your own W&B key, Run all. About 50 min for 1500 iterations on a free T4. Never commit a key.
- `Metrics/twist/error_vel_xy` in W&B is not a tracking error; it grows with episode length. Read `Episode_Reward/track_board_velocity` instead.
