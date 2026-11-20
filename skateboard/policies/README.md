# Policy and media names

Files here and in [`../media/`](../media/) are named
`<stance>_<skill>_v<N>[_<detail>]`. The stance is `centred` or `split`, the skill
is `balance`, `ride` or `steer`, and N counts that stance and skill's runs in the
order they were trained. Suffixes say what the file is: `_policy.json` (export),
`_max0.3_policy.json` (export that clips the speed command at 0.3 m/s),
`_model_<iteration>.pt` (checkpoint), `_audit.txt` (audit log), `_0.3ms` (video
at that speed) and `_125.png` (the still at that frame).

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
| `split_steer_v5` | `bxke67fj` (resumes `qqm2917n`) | `Mjlab-SkateSplit-Steer-Pupper-v3` | Elliptic friction cone. Carves both ways, fails 2 of 16 audit rows. Candidate for the first steering test. |

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
- `split_steer_v1`: `jl7i4id0`. `v4`: `8tklg4k7`.
