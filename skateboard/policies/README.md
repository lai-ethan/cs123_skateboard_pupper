# Policy and media names

Files here and in [`../media/`](../media/) are named
`<stance>_<skill>_v<N>[_<detail>]`. The stance is `centred` or `split`, the skill
is `balance`, `ride` or `steer`, and N counts that stance and skill's runs in the
order they were trained. Suffixes say what the file is: `_policy.json` (export),
`_max0.3_policy.json` (export that clips the speed command at 0.3 m/s),
`_model_<iteration>.pt` (checkpoint), `_audit.txt` (audit log), `_0.3ms` (video
at that speed) and `_125.png` (the still at that frame). The one media file
outside the scheme is `centred_push_reference_openloop.mp4`: it plays the
centred push reference open loop, with no trained policy, so it has no run.

| Name | W&B run | Task id | What it is |
| --- | --- | --- | --- |
| `centred_balance_v2` | `zdctdcyf` | `Mjlab-Skate-Balance-Pupper-v3` | Centred balance. Gait check pass, 14 of 21 stance audit rows. |
| `centred_ride_v3` | `cy8f55bg` | `Mjlab-Skate-Ride-Pupper-v3` | Centred ride that steps off the deck to push. Ends up lying on the deck. |
| `centred_ride_v7` | `9c33unfu` | `Mjlab-Skate-Ride-Pupper-v3` | First tripod ride. Skitters at about 7 strides per second. |
| `centred_ride_v8` | `zvx4xc24` | `Mjlab-Skate-Ride-Pupper-v3` | Tripod ride. Fails the stride rate limit only. |
| `centred_ride_v9` | `7fyt12i1` | `Mjlab-Skate-Ride-Pupper-v3` | Tripod ride, provisional. Fails push foot duty only. |
| `split_balance_v2` | `2ace1uqa` | `Mjlab-Skate-Balance-Pupper-v3` (then the split stance) | First split stance, guessed board. Replays on `Mjlab-SkateSplit-Balance-Pupper-v3`. |
| `split_balance_v3` | `gt60lfo0` | `Mjlab-SkateSplit-Balance-Pupper-v3` | Split balance. 21 of 21 stance audit rows. First robot candidate. |
| `split_ride_v0` | `29p4j81t` | `Mjlab-Skate-Ride-Pupper-v3` (then the split stance) | First split ride, guessed board. Windmills its front right leg; fails the gait check. |
| `split_ride_v1` | `vzs88cci` | `Mjlab-SkateSplit-Ride-Pupper-v3` | Shuffles at 5 steps per second. |
| `split_ride_v2` | `ffcxbwup` | `Mjlab-SkateSplit-Ride-Pupper-v3` | No shuffle, but fails the gait check at 0.4 m/s. |
| `split_ride_v3` | `3qmtsv4o` | `Mjlab-SkateSplit-Ride-Pupper-v3` | No resets; replaced by v4. |
| `split_ride_v4` | `iznjwbka` | `Mjlab-SkateSplit-Ride-Pupper-v3` | Gait check pass at 0.2, 0.3 and 0.4 m/s on the box deck. |
| `split_ride_v7` | `gkk9a6bj` | `Mjlab-SkateSplit-Ride-Pupper-v3` | Traced deck, pays for the heading hold. Fails 9 ride audit rows. |
| `split_steer_v2` | `188eygof` | `Mjlab-SkateSplit-Steer-Pupper-v3` | First carve signal run. Steers both ways; the deck feet creep off the nose. |
| `split_steer_v3` | `a9x8z66m` | `Mjlab-SkateSplit-Steer-Pupper-v3` | Adds `board_slid` and `board_fore_aft`. Creeps slower, still fails 6 audit rows. |
| `split_steer_v5` | `bxke67fj` (resumes `qqm2917n`) | `Mjlab-SkateSplit-Steer-Pupper-v3` | Elliptic friction cone. First steer run to pass all three gait reports with zero resets; fails 2 of 16 audit rows. Fallback for the first steering test. Trained only on full carves (yaw 0.4 to 0.6, hold clip 0.3), so use the full yaw stick. |
| `split_steer_v6` | `hfd3qhwr` (resumes `bxke67fj`) | `Mjlab-SkateSplit-Steer-Pupper-v3` | Partial turns (yaw 0.1 to 0.6, hold clip 0.1). Passes 16 of 16 audit rows; 1 board in 48 slides off on left carves. Candidate for the first steering test. |
| `split_steer_v7` | `n5m2zsks` (resumes `hfd3qhwr`) | `Mjlab-SkateSplit-Steer-Pupper-v3` | Deck-foot scuff penalty -20 (was -2), slide counted from the episode start. No `board_slid` in 27 gait runs; 16 of 16 audit rows. Pulls right; fails hard right carves (-0.5, -0.6). Fallback if v6 slides; export clips yaw at ±0.3. |
| `split_steer_v9` | `d63ab512` (resumes `hfd3qhwr`) | `Mjlab-SkateSplit-Steer-Pupper-v3` | Carpet drag, wheel drag x0.5 to x25. Not exported: passes 16 of 21 audit rows, fails drag x6 to x25 and `shove_fwd_0.5` (v6 passes it), and above drag x10 stands still instead of pushing. |
| `split_steer_v10` | `74jsd4gb` (resumes `d63ab512`) | `Mjlab-SkateSplit-Steer-Pupper-v3` | Carpet drag, wheel drag x0.5 to x10. Passes 18 of 21 audit rows, every row v6 passes; 0.96 of asked speed at drag x6 with no resets, 0.58 at x10 (misses the 0.7 bar). Gait check on three seeds: 22 of 23 runs pass, 1 reset at stick -0.5 (evidence: `split_steer_v10_gait.txt`); gentle left carves turn up to about twice what is asked, right carves 0.6 to 1.0, slight left pull at stick 0. Provisional: run second after v6 to compare on carpet. Export clips speed 0 to 0.3 m/s and yaw at ±0.6, as v6's. |
| `split_steer_v11` | `ioqfatdr` (resumes `74jsd4gb`) | `Mjlab-SkateSplit-Steer-Pupper-v3` | Same code and drag range as v10, 2000 more iterations. Not exported: passes 17 of 21 audit rows (v10: 18), fails `deck_grip_0.4`, a row v6 and v10 pass, and misses the bar at x10 (0.65 with 5 resets; v10 0.58 with 4). Steering as v10: 22 of 23 gait runs pass. |
| `split_steer_v13` | `k43t9fqw` (resumes `74jsd4gb`) | `Mjlab-SkateSplit-Steer-Pupper-v3` | Pays the board for answering the heading hold, tighter yaw in `board_alignment` (`../PLAN_HEADING.md`). Sim, 25 s straight ride, 16 boards, drag x3: sideways drift mean size 16.8 cm, worst board 41.4 cm (v10: 29.0 and 84.7), mean signed -15.5 cm (v10 the same). Spreads less than v10 but misses both lines of its own drift bar (16.8 against 15 cm, 41.4 against 40 cm) and the signed line (within ±8 cm) added for v14. Passes 18 of 21 audit rows, the same three failures as v10 (x10, x15, x25): 0.91 of asked speed at drag x6 with no resets, 0.51 at x10 with no resets (misses the 0.7 bar, as v10). Gait check on three seeds: 20 of 21 runs pass, 1 reset at stick +0.5 (seed 0); evidence `split_steer_v13_audit.txt`, `split_steer_v13_gait.txt`. Provisional: run third after v6 and v10. Export clips speed 0 to 0.3 m/s and yaw at ±0.6, as v10's. Never run on the robot. The steer task at HEAD trains v13's rewards. |
| `split_steer_v14` | `rd2np6mb` (resumes `k43t9fqw`) | `Mjlab-SkateSplit-Steer-Pupper-v3` | v13's rewards plus `board_heading_hold` at weight 1.0 (the task at HEAD keeps the term at 0). Not exported: on the same 25 s ride, drift mean size 17.0 cm, worst board 44.2 cm, mean signed -16.1 cm, no better than v13, so the board-heading reward did not remove the offset. Audit and gait check not run. |

`split_steer_v6` trained on the current steer task: partial turns (yaw 0.1 to
0.6) with the hold clip at 0.1. Its media names give the stick, as in
`split_steer_v6_left_stick0.15_0.3ms.mp4`. There will be no foot stop on the
deck for any steer test, so sliding along the deck is an open risk: in sim
1 of 48 v6 boards slid 8 cm on left carves. Keep runs short and reset the
robot to its marks between runs.

`split_steer_v7` is v6 retrained to stop that slide
([`../PLAN_DECK_SLIDE.md`](../PLAN_DECK_SLIDE.md)). Its `_max0.3_policy.json`
clips both speed (0 to 0.3 m/s) and yaw (±0.3 rad/s); v6's clips yaw at ±0.6.
`split_steer_v7_right_stick0.6_0.3ms.mp4` is a carve the export cannot send.
It shows one board that did not reset; in the gait check 1 or 2 boards in 16
(up to 4 resets) drift off sideways at that stick. v7 pulls right, and the cause
of the pull is not known yet.

`split_balance_v2` and `split_ride_v0` were trained before the split stance had
its own task ids, so they ran under the ids the centred stance uses now.

Runs that count toward N but have no files here:

- `centred_balance_v1`: `k4kci0oj` (held its legs on the joint stops).
- `centred_ride_v1`: `14m5rs1f`. `v2`: crashed at iteration 95, no run id recorded.
  `v4` to `v6`: `vrn7pp8n`, `xtbenpi4`, `2kmu10gl`.
- `split_balance_v1`: `2zxd46yi`.
- `split_ride_v5`: `cfcrepfu` (resumed from v4). `v6`: `wk1xldv1` (resumed from
  v5). v7 resumed from v6. The docs already called `vzs88cci` v1, so the older
  `29p4j81t` is v0. `me6p0pdw`, cut off by Colab before it, has no number.
- `split_steer_v1`: `jl7i4id0`. `v4`: `8tklg4k7`. A lean-steer run between v1
  and v2 was abandoned with no run id recorded and has no number.
