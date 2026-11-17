# Pupper on a skateboard: start here

CS 123 final project, Team 9. Goal: train a policy in simulation so Pupper
rides the Walmart 23" mini cruiser, then run it on the real robot by Demo Day
(Dec 3).

**Two stances.** Each has a balance task and a ride task. Every tool and the
notebook pick the stance from the task id: `SkateSplit` in the id means split.

| Stance | What stands where | Task ids | Lowered trucks to ride? |
| --- | --- | --- | --- |
| Centred | All four feet on the deck. To ride, Pupper steps one foot off either side, pushes against the floor and steps back on. | `Mjlab-Skate-Balance-Pupper-v3`, `Mjlab-Skate-Ride-Pupper-v3` | Yes. On the stock deck a foot cannot reach the floor. |
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
| Trained policies, sim videos | [`policies/`](policies/), [`media/`](media/) |

## Status

- Done: board model, stance, both training tasks, tests, notebook.
- The stance moved from split (left feet on the deck, right feet on the floor) to centred on 2026-11-16. The split stance came back the same day as a second pair of tasks. The centred tasks must be trained. The saved policies in [`policies/`](policies/) were trained on the split stance.
- Balance (split stance) was trained in sim and passes the gait check (see [`CHECKPOINT.md`](CHECKPOINT.md)).
- Ride must be retrained. Its rewards were reshaped so the gait fits the motors (PLAN Task 16). The saved ride policy passes the curve check but fails the gait check, so it is for reference only. The retrain has not run yet.
- Not done: retraining both tasks for the centred stance, measuring the board's mass, trucks and wheel drag, anything on the robot.

Four tasks are registered:

- `Mjlab-Skate-Balance-Pupper-v3`: hold the centred stance on a board that is free to roll.
  - Both balance tasks penalize any motor above half its torque limit (`torque_saturation`). The first centred balance run, `k4kci0oj`, passed the curve check while it held its legs on the joint stops at full torque, which a real motor cannot keep up.
- `Mjlab-Skate-Ride-Pupper-v3`: push with one foot off either side so the board rolls at a commanded speed (0 to 0.5 m/s). Two rewards teach the push while the board is too slow: `foot_reach` pays more the closer the lowest foot gets to the floor, and `push_stroke` pays once for each planted stroke as the foot lifts off the floor.
- `Mjlab-SkateSplit-Balance-Pupper-v3`: hold the split stance on a board that is free to roll.
- `Mjlab-SkateSplit-Ride-Pupper-v3`: push with the right legs so the board rolls at a commanded speed (0 to 0.5 m/s). `push_stride` pays for planted strides of the right feet.

The reward tables for all four are in section 4 of the notebook.

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
| Mass | 1.8 kg | guess | kitchen scale |
| Roll stiffness | 30 N·m/rad | guess | put about 1.5 kg on the deck edge, measure how far the edge drops |
| Wheel drag | 2e-3 N·m per wheel | guess | roll off a ramp of known height, measure the coast distance |

Deck height matters most. The stance is solved for 103 mm. Balance training
randomizes the deck from 6.8 to 11.3 cm, which covers the stock trucks and the
lower trucks we plan to fit. Ride trains on 6.8 to 8.3 cm only (the lowered
trucks): on the stock deck a foot cannot reach the floor from the centred
stance. The feet sit only 1.5 to 2 cm inside the deck edge, so sideways
placement on the board matters too (see [`SIM2REAL.md`](SIM2REAL.md)).

- The deck can now lean up to 0.2 rad and the trucks steer with it. Until
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

Expected: `46 passed`. `FORCE_CPU=1` is for Macs; drop it on a machine with an
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

The two saved policies are in [`policies/`](policies/), so no W&B login is
needed. Both were trained on the split stance, so replay them with the split
task ids. The split tasks were rebuilt on 2026-11-16, so treat the policies as a
way to try the tools, not as results. From the repo root:

```bash
# Ride, in the interactive viewer
FORCE_CPU=1 uv run play Mjlab-SkateSplit-Ride-Pupper-v3 \
  --checkpoint-file skateboard/policies/ride_29p4j81t_model_1499.pt --num-envs 1

# Ride or balance, saved as a video plus stills (no viewer needed)
FORCE_CPU=1 uv run python skateboard/tools/render_policy.py Mjlab-SkateSplit-Ride-Pupper-v3 \
  skateboard/policies/ride_29p4j81t_model_1499.pt out/ride.mp4 --speed 0.4
FORCE_CPU=1 uv run python skateboard/tools/render_policy.py Mjlab-SkateSplit-Balance-Pupper-v3 \
  skateboard/policies/balance_2ace1uqa_policy.json out/balance.mp4
```

To run the gait check on both saved policies:

```bash
FORCE_CPU=1 uv run python skateboard/tools/gait_report.py Mjlab-SkateSplit-Ride-Pupper-v3 \
  skateboard/policies/ride_29p4j81t_model_1499.pt --speed 0.4
FORCE_CPU=1 uv run python skateboard/tools/gait_report.py Mjlab-SkateSplit-Balance-Pupper-v3 \
  skateboard/policies/balance_2ace1uqa_policy.json
```

For a centred task the gait check reads the deck and floor contacts of all four
feet. `min_foot_duty` is the smallest share of time any foot touches the deck or
the floor, and `max_stride_rate` the most often any foot lands on the floor, per
second. For a split task it counts only the two right feet, the ones that push:
their time on the floor and how often they land. A line above the verdict shows each
foot's share of time on the floor, so you can see which foot pushes; it is not
part of the verdict. The last line is `GAIT: PASS` or `GAIT: FAIL`. On
2026-11-16, on the split tasks, the saved balance policy gave `GAIT: PASS` and
the saved ride policy `GAIT: FAIL` (legs too fast, motors at the torque limit,
the front right foot on the floor only 27% of the time). The board still rolled
fast enough. The numbers change between runs (the play env is unseeded).

To check how much balance margin a policy has when the sim is wrong (deck
height, board placement, grip, motors, shoves):

```bash
FORCE_CPU=1 uv run python skateboard/tools/stance_audit.py \
  skateboard/policies/balance_2ace1uqa_policy.json --conditions baseline,board_y+2cm \
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
