# Pupper on a skateboard: start here

CS 123 final project, Team 9. Goal: train a policy in simulation so Pupper
rides the Walmart 23" mini cruiser, then run it on the real robot by Demo Day
(Dec 3).

**Two stances.** Each has a balance task and a ride task. Every tool and the
notebook pick the stance from the task id: `SkateSplit` in the id means split.

| Stance | What stands where | Task ids | Lowered trucks to ride? |
| --- | --- | --- | --- |
| Centred | All four feet on the deck. To ride, Pupper steps the back right foot off the deck, pushes against the floor and steps back on, then glides on four feet. | `Mjlab-Skate-Balance-Pupper-v3`, `Mjlab-Skate-Ride-Pupper-v3` | No. The ride reference is solved for the stock 103 mm deck. |
| Split | Left feet on the deck, right feet on the floor. To ride, the right legs push. | `Mjlab-SkateSplit-Balance-Pupper-v3`, `Mjlab-SkateSplit-Ride-Pupper-v3` | No. The right feet are already on the floor. |

## Where things are

| What | Where |
| --- | --- |
| The plan: design, rewards, tasks, schedule | [`PLAN.md`](PLAN.md) |
| Paper notes (older; the plan wins where they disagree) | [`RESEARCH.md`](RESEARCH.md) |
| Colab training notebook | [`notebooks/CS123_Pupper_Skate.ipynb`](../notebooks/CS123_Pupper_Skate.ipynb) |
| Board model and its dimensions | [`skateboard_constants.py`](../src/mjlab/asset_zoo/robots/mini_cruiser/skateboard_constants.py) |
| Stance (joint angles, height, roll) | `SKATE_*` in [`pupper_constants.py`](../src/mjlab/asset_zoo/robots/pupper_v3/pupper_constants.py) |
| Tasks: scene, rewards, terminations | [`src/mjlab/tasks/pupper_skate/`](../src/mjlab/tasks/pupper_skate/) |
| Tests | `tests/test_pupper_skate.py`, `tests/test_pupper_skate_split.py`, `tests/test_pupper_skateboard.py` |
| Stance helper scripts, video renderer, pass check, gait report | [`tools/`](tools/) |
| Sim2real: stance margins, what sim cannot show, steps before the first real run | [`SIM2REAL.md`](SIM2REAL.md) |
| First hardware session test sheet | [`HARDWARE_TEST.md`](HARDWARE_TEST.md) |
| Trained policies, sim videos, which run each name is | [`policies/`](policies/), [`media/`](media/), [`policies/README.md`](policies/README.md) |

## Status

As of 2026-11-20. Nothing has run on the robot.

| Policy | W&B run | Curves | Gait check | Stance audit | Robot test? |
| --- | --- | --- | --- | --- | --- |
| Split balance | `gt60lfo0` | pass | pass | 21 of 21 | First candidate |
| Centred balance | `zdctdcyf` | pass | pass | 14 of 21 | Second, with careful placement |
| Split ride v4 | `iznjwbka` | pass | passed at 0.2, 0.3, 0.4 m/s on 2026-11-17, older task; resets on `board_drifted` on the current one | not run | Not ready: Ethan's call, short runs only |
| Centred ride (tripod) | `7fyt12i1` | pass | fail on push foot duty only, at 0.1, 0.2, 0.3 m/s | not run | Provisional, last, Ethan's call |
| Split steer v6 | `hfd3qhwr` | not checked | pass at yaw 0, ±0.15, -0.3, -0.5, -0.6; 1 reset at +0.3, +0.5, +0.6 (0.3 m/s) | ride audit 16 of 16 | Steering candidate, loosened trucks, Ethan's call |
| Split steer v7 | `n5m2zsks` | not checked | pass at yaw 0, ±0.15, ±0.3, +0.5, +0.6; fail at -0.5, -0.6 on every seed (`board_drifted`) | ride audit 16 of 16 | Fallback if v6 slides along the deck; stick within ±0.3 |
| Split steer v5 | `bxke67fj` | not checked | pass at yaw 0, +0.5, -0.5 (0.3 m/s) | ride audit 14 of 16 | Steering fallback, full stick only |

- Done: board model on the measured board, both stances, five tasks, tests, notebook, both balance policies.
- Both balance policies and their audit logs are in [`policies/`](policies/); videos are in [`media/`](media/).
- Split ride v4 `iznjwbka` (PLAN Task 31 rewards) passed the gait check on the 103 mm deck at 0.2, 0.3 and 0.4 m/s on 2026-11-17, on the old box deck and before the `board_drifted` and `board_slid` checks existed. It has not passed on the current task: on 2026-11-18 its checkpoint reset on `board_drifted` (sideways drift off the feet) in the gait check, and its retrain, split ride v7 `gkk9a6bj`, fails 9 ride audit rows. Not rerun since `board_slid` and the elliptic cone. Policy: `policies/split_ride_v4_policy.json`.
- Centred ride is a tripod ride now: three feet on the deck, the back right foot pushing on the floor. Run `7fyt12i1` rides at 0.1 to 0.3 m/s with no resets and passes every gait limit but one: the push foot is on the floor 20% of the time (limit 40%). Provisional policy: `policies/centred_ride_v9_max0.3_policy.json`. Video: `media/centred_ride_v9.mp4`. The run history is in [`SIM2REAL.md`](SIM2REAL.md).
- Steering: split steer v6 (`hfd3qhwr`, resumed from v5) is the candidate for the first hardware steering test, not a fully passing policy. It leans to steer on trucks loosened to 9 to 18 N·m/rad and trained on partial turns, so it turns in proportion to the stick. At 0.3 m/s it passes all 16 ride audit rows. In the gait check (seed 0) the board turns 0.82 to 0.86 of the asked rate at stick ±0.15 and ±0.6, but only 0.58 to 0.63 at ±0.3. One of 16 boards, the same one each time, slides past the 8 cm limit on left carves of 0.3 to 0.6; on two more seeds there was no reset, so 1 board in 48. There will be no foot stop or ridge on the deck, so sliding is the main open risk: the test uses short runs, starts with a small stick and resets the robot to its marks between runs. Policy: `policies/split_steer_v6_max0.3_policy.json`. Split steer v5 (`bxke67fj`) is the fallback: it trained only on full carves, so it needs the full yaw stick, and it fails 2 of 16 ride audit rows. Test sheet: [`HARDWARE_TEST.md`](HARDWARE_TEST.md), tests 0 and 5. A hardware run needs Ethan's OK and a TA sign-off.
- Split steer v7 (`n5m2zsks`, resumed from v6) charges deck-foot scuffing ten times harder and counts the slide from the episode start ([`PLAN_DECK_SLIDE.md`](PLAN_DECK_SLIDE.md)). It fixes the slide: no `board_slid` reset in 27 gait runs, 0 of 16 boards slid out under soft leg gains (v6: 4 of 16), and creep halved (still above the 0.1 cm/s target). But it hurts turning: it pulls right, hard right carves (stick -0.5, -0.6) fail on every seed on `board_drifted`, hard left carves turn less (0.58 of asked at +0.6, v6 0.80) and gentle right carves overshoot (2.1 times asked at -0.15). It did not meet the plan's pass bar, so v6 stays the steering candidate. v7 is the fallback for straight riding and gentle carves if v6 slides along the deck on the robot. Policy: `policies/split_steer_v7_max0.3_policy.json`, yaw clipped at ±0.3 rad/s.
- Heading hold is off in every skate export (kp 0) except split steer v6 and v7 (kp 1.0, clip 0.1 rad/s) and v5 (kp 1.0, clip 0.3 rad/s).
- Not done: a centred ride policy that passes the gait check, measuring truck stiffness and wheel drag, anything on the robot. The steps before the first real run are in [`SIM2REAL.md`](SIM2REAL.md); the test sheet is [`HARDWARE_TEST.md`](HARDWARE_TEST.md).

Five tasks are registered:

- `Mjlab-Skate-Balance-Pupper-v3`: hold the centred stance on a board that is free to roll.
  - Both balance tasks penalize any motor above half its torque limit (`torque_saturation`). The first centred balance run, `k4kci0oj`, passed the curve check while it held its legs on the joint stops at full torque, which a real motor cannot keep up.
- `Mjlab-Skate-Ride-Pupper-v3`: the tripod ride. Three feet stay on the deck and the back right foot pushes on the floor so the board rolls at a commanded speed (0.1 to 0.3 m/s), on the stock deck (9.3 to 11.3 cm). Every episode starts in the tripod layout with the push foot already on the floor, so it never steps on or off the deck. The actor's reference slot plays the `skate_push` stroke from `gait_reference.py` at 0.8 Hz. `push_stroke` (3.0) and `foot_reach` (1.0) pay only the back right foot, and only while the other three are on the deck; `board_progress` (4.0) and `track_board_velocity` pay for the commanded speed; `short_stride` (-3.0) charges every push that ends before 0.75 s, so quick taps lose reward; `torque_saturation` (-1.0) charges motors near their limit. The episode ends if the body drops toward the deck (`base_near_deck`) or a support foot is off the deck for more than 0.3 s (`support_feet_off_deck`). Earlier designs had the foot step off the deck each stroke; none learned it (history in [`SIM2REAL.md`](SIM2REAL.md)). Export with `--clip-vx "(0.0, 0.3)"`.
- `Mjlab-SkateSplit-Balance-Pupper-v3`: hold the split stance on a board that is free to roll.
- `Mjlab-SkateSplit-Ride-Pupper-v3`: push with the right legs so the board rolls at a commanded speed (0 to 0.5 m/s). `push_stride` pays for planted strides of the right feet, each stance as long as a 10 cm stroke takes at the command (0.25 to 0.5 s). `action_rate_l2`, `action_acc`, `joint_acc` and `swing_height` charge for snaps at lift-off and touchdown and for swings higher than 3 cm, and the joint speed penalty starts at 5 rad/s (PLAN Task 31).

- `Mjlab-SkateSplit-Steer-Pupper-v3`: the split ride on loosened trucks (9 to 18 N·m/rad) plus a carve signal. Yaw 0 rides straight with the heading hold on; a yaw command above 0.1 and up to 0.6 rad/s, either sign, asks the board to turn at a tenth of it, so a partial stick is a partial turn. The task clips the hold's correction at 0.1 rad/s (the ride keeps 0.3), so a command of 0.1 or less is a correction and a larger one is the stick. Split steer v5 trained on an earlier version with carves of 0.4 to 0.6 only and the 0.3 clip; v6 (`hfd3qhwr`) and v7 (`n5m2zsks`) trained on the current task. `track_board_yaw_rate` and `board_turn_progress` pay for the board's turn. Split steer v5 is the first steer run to pass all three gait reports (yaw 0, +0.5, -0.5) with zero resets; v2 to v4 also turned the board both ways but reset.

The reward tables are in section 4 of the notebook.

## How to start

### 1. The board

The sim uses the numbers below. The sizes are measured on the team's board.
Replace the guesses in `SkateboardParams` when you measure them.

| Parameter | Sim value | Source | How to measure |
| --- | --- | --- | --- |
| Deck top height | 103 mm | measured | ruler, floor to top of deck |
| Deck width | 161 mm | measured | ruler |
| Deck length | 584 mm | measured | ruler |
| Wheel radius | 29 mm | measured | ruler (58 mm wheels) |
| Track | 104 mm | measured | ruler, left to right wheel centre |
| Wheelbase | 365 mm | measured, less accurate | ruler, front axle to rear axle |
| Mass | 1.523 kg | measured 2026-11-16 (stock trucks) | kitchen scale |
| Roll stiffness | 30 N·m/rad | guess | put about 1.5 kg on the deck edge, measure how far the edge drops |
| Wheel drag | 2e-3 N·m per wheel | guess | roll off a ramp of known height, measure the coast distance |

Deck height matters most. The stance is solved for 103 mm. Balance training
randomizes the deck from 6.8 to 11.3 cm, which covers the stock trucks and the
lower trucks we plan to fit. Centred ride trains on 9.3 to 11.3 cm only (the
stock trucks): its push reference is solved for the 103 mm deck, where a
crouch and a 17 degree lean put a foot on the floor. The feet sit only 1.5 to
2 cm inside the deck edge, so sideways placement on the board matters too
(see [`SIM2REAL.md`](SIM2REAL.md)).

With no policy, the centred stance holds on a stiff truck (2 to 7 of 64 noisy
spawns fall at 180 N·m/rad) but tips in about half of them at the guessed 30
N·m/rad, so truck stiffness is the most important board number still to measure.

The deck can now lean up to 0.2 rad and the trucks steer with it. Until
2026-11-16 the board's joint limits were read as degrees, so the deck could
barely lean (0.0035 rad). Policies trained before this fix (runs `k4kci0oj`,
`14m5rs1f`, `2ace1uqa`, `29p4j81t`) saw a deck that could not lean.

### 2. Run the tests (laptop, optional)

Needs [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
git clone https://github.com/lai-ethan/cs123_skateboard_pupper.git
cd cs123_skateboard_pupper
FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py tests/test_pupper_skateboard.py tests/test_task_configs.py -q
```

Expected: `63 passed, 1 xfailed`. `FORCE_CPU=1` is for Macs; drop it on a machine with an
NVIDIA GPU.

To see the stance in a viewer:

```bash
FORCE_CPU=1 uv run play Mjlab-Skate-Balance-Pupper-v3 --agent zero
```

### 3. Train (Colab)

1. Open `notebooks/CS123_Pupper_Skate.ipynb` in Colab (File → Open notebook → GitHub → this repo).
2. Runtime → Change runtime type → G4 GPU.
3. Optional: give it a W&B key, either pasted in the W&B cell or stored once as
   the Colab secret `WANDB_API_KEY`. Without one the run logs to TensorBoard
   only and `policy.json` stays on the runtime (no W&B upload for `deploy.sh`).
   Never commit a key.
4. Set the task in section 3 (a stance's Balance task before its Ride task),
   then Runtime → Run all. Nothing
   prompts. 1,500 iterations take about 9 minutes on a G4, about 55 on a T4. You can also run the cells
   one at a time.
5. Read the last line of section 7, *Did it pass?*: `RESULT: PASS` or
   `RESULT: FAIL`, with the numbers above it. The same check works on any run
   from the repo root: `python skateboard/tools/run_summary.py <task-id>` (on a laptop: `uv run python ...`).
   For either ride task, section 7 also runs the gait check on the newest checkpoint. Its
   last line is `GAIT: PASS` or `GAIT: FAIL`. Curves can pass while the gait is
   one the robot cannot do, so a ride run needs both passes. Watch the video too.

The thresholds live in `PASS_CRITERIA` in `tools/run_summary.py`. What to change
when a run fails is in Tasks 9 and 11 of the plan.

### 4. Replay the trained policies (laptop)

Saved policies are in [`policies/`](policies/), so no W&B login is needed. The
current ones are `split_balance_v3_*`, `centred_balance_v2_*`,
`split_ride_v4_*`, `split_steer_v6_*`, `split_steer_v7_*` (slide fallback) and
`split_steer_v5_*` (full stick fallback). Each name maps to its W&B run in
[`policies/README.md`](policies/README.md).
`split_balance_v2_*` and `split_ride_v0_*` are from the first split stance on a
guessed board; the commands below use them only to show the tools. From the repo
root:

```bash
# Ride, in the interactive viewer
FORCE_CPU=1 uv run play Mjlab-SkateSplit-Ride-Pupper-v3 \
  --checkpoint-file skateboard/policies/split_ride_v0_model_1499.pt --num-envs 1

# Ride or balance, saved as a video plus stills (no viewer needed)
FORCE_CPU=1 uv run python skateboard/tools/render_policy.py Mjlab-SkateSplit-Ride-Pupper-v3 \
  skateboard/policies/split_ride_v0_model_1499.pt out/ride.mp4 --speed 0.4
FORCE_CPU=1 uv run python skateboard/tools/render_policy.py Mjlab-SkateSplit-Balance-Pupper-v3 \
  skateboard/policies/split_balance_v2_policy.json out/balance.mp4
```

To run the gait check on both saved policies:

```bash
FORCE_CPU=1 uv run python skateboard/tools/gait_report.py Mjlab-SkateSplit-Ride-Pupper-v3 \
  skateboard/policies/split_ride_v0_model_1499.pt --speed 0.4
FORCE_CPU=1 uv run python skateboard/tools/gait_report.py Mjlab-SkateSplit-Balance-Pupper-v3 \
  skateboard/policies/split_balance_v2_policy.json
```

The gait check rolls out 16 envs at once (`--envs`), each with its own board
drawn from the task's ranges, built from a fixed seed (`--seed`, default 0), so
two runs give the same answer. Every limit is judged on the worst env, and any
fall in any env fails. Before a policy goes on the robot, add
`--deck-height <m>` to pin every env's deck top to the real board: 0.103 for the
stock trucks, or the measured height of the lowered ones. A ride task needs
`--speed`; the board must then roll at 0.5 to 1.5 times it. Joint speed is
judged on the 99th percentile against 12 rad/s (an estimate from the motor
spec, not a measurement); the true largest joint speed is printed beside it but
is not part of the verdict.

For a centred task the gait check reads the deck and floor contacts of all four
feet. `min_foot_duty` is the smallest share of time any foot touches the deck or
the floor, and `max_stride_rate` the most often any foot lands on the floor, per
second. For a split task it counts only the two right feet, the ones that push:
their time on the floor and how often they land. A line above the verdict shows each
foot's share of time on the floor in the env with the worst foot duty, so you can
see which foot pushes; it is not part of the verdict. The last line is `GAIT: PASS` or `GAIT: FAIL`. On
2026-11-16, on the split tasks, the saved balance policy gave `GAIT: PASS` and
the saved ride policy `GAIT: FAIL` (legs too fast, motors at the torque limit,
the front right foot on the floor only 27% of the time). The board still rolled
fast enough. Those numbers came from one unseeded env, before the check moved
to 16 seeded envs, so they may not repeat.

To play the centred ride's push reference open loop (no policy: the reference
is the action), with each reward term's mean per step:

```bash
FORCE_CPU=1 uv run python skateboard/tools/play_push_reference.py --rewards \
  --video out/push_reference.mp4
```

To check how much balance margin a policy has when the sim is wrong (deck
height, board placement, grip, motors, shoves):

```bash
FORCE_CPU=1 uv run python skateboard/tools/stance_audit.py \
  skateboard/policies/split_balance_v2_policy.json --conditions baseline,board_y+2cm \
  --task Mjlab-SkateSplit-Balance-Pupper-v3
```

`--task` picks the stance; it defaults to `Mjlab-Skate-Balance-Pupper-v3`.

- Centred: a row passes when all four feet stay on the deck at least 90% of the
  time. Each row shows the margin on three feet with each foot lifted (lift FR,
  FL, BR, BL).
- Split: a row passes when both left feet stay on the deck at least 90% of the
  time. Each row shows the margin with a right foot lifted (lift FR, RR).

Both stances use the same conditions. Drop `--conditions` to run all 21; each takes about 10 s on a Mac. On
2026-11-16, on the centred task, the saved balance policy passed `baseline` and
`deck-3.5cm` and failed `board_y+2cm` (28 resets). On the split task it passed
`baseline`. See [`SIM2REAL.md`](SIM2REAL.md).

Each video takes about a minute on a Mac. The balance run kept only its
exported `policy.json`, so it replays through `render_policy.py`, not `play`.
Rendered clips and stills are in [`media/`](media/): ride at 0.4 m/s from both
sides, and balance.

### 5. Robot

No skate policy runs on the robot without a TA sign-off. Steps are in Task 12.

## Credit

The board model follows the structure of HUSKY (Han et al.,
[arXiv:2602.03205](https://arxiv.org/abs/2602.03205)). No HUSKY code is included.
