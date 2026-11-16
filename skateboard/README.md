# Pupper on a skateboard: start here

CS 123 final project, Team 9. Goal: train a policy in simulation so Pupper
rides the Walmart 23" mini cruiser, then run it on the real robot by Demo Day
(Dec 3).

**The stance:** left feet on the deck, right feet on the floor. Pupper never
has all four feet on the board. The right legs balance, and later push.

## Where things are

| What | Where |
| --- | --- |
| The plan: design, rewards, tasks, schedule | [`PLAN.md`](PLAN.md) |
| Paper notes (older; the plan wins where they disagree) | [`RESEARCH.md`](RESEARCH.md) |
| Colab training notebook | [`notebooks/CS123_Pupper_Skate.ipynb`](../notebooks/CS123_Pupper_Skate.ipynb) |
| Board model and its dimensions | [`skateboard_constants.py`](../src/mjlab/asset_zoo/robots/mini_cruiser/skateboard_constants.py) |
| Stance (joint angles, height, roll) | `SKATE_*` in [`pupper_constants.py`](../src/mjlab/asset_zoo/robots/pupper_v3/pupper_constants.py) |
| Tasks: scene, rewards, terminations | [`src/mjlab/tasks/pupper_skate/`](../src/mjlab/tasks/pupper_skate/) |
| Tests | `tests/test_pupper_skate.py`, `tests/test_pupper_skateboard.py` |
| Stance helper scripts, video renderer | [`tools/`](tools/) |
| Trained policies, sim videos | [`policies/`](policies/), [`media/`](media/) |

## Status

- Done: board model, stance, both training tasks, tests, notebook. Balance and ride policies are trained in sim (see [`CHECKPOINT.md`](CHECKPOINT.md)).
- Not done: measuring the real board, anything on the robot.

Two tasks are registered:

- `Mjlab-Skate-Balance-Pupper-v3`: hold the stance on a board that is free to roll.
- `Mjlab-Skate-Ride-Pupper-v3`: push so the board rolls at a commanded speed (0 to 0.5 m/s).

## How to start

### 1. Measure the board (do this first)

The sim uses the numbers below. Only the first three come from the listing.
Replace the guesses in `SkateboardParams`, then follow Task 10 in the plan.

| Parameter | Sim value | Source | How to measure |
| --- | --- | --- | --- |
| Deck length | 0.58 m | listing | ruler |
| Deck width | 0.165 m | listing | ruler |
| Wheel radius | 0.03 m | listing | ruler |
| Deck top height | 0.10 m | guess | ruler, floor to top of deck |
| Wheelbase | 0.30 m | guess | ruler, front axle to rear axle |
| Track | 0.12 m | guess | ruler, left to right wheel centre |
| Mass | 1.8 kg | guess | kitchen scale |
| Roll stiffness | 30 N·m/rad | guess | put about 1.5 kg on the deck edge, measure how far the edge drops |
| Wheel drag | 2e-3 N·m per wheel | guess | roll off a ramp of known height, measure the coast distance |

Deck height matters most. The stance is solved for 0.10 m; if the real deck is
more than 5 mm off, re-solve it (Task 10, step 3).

### 2. Run the tests (laptop, optional)

Needs [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
git clone https://github.com/lai-ethan/cs123_skateboard_pupper.git
cd cs123_skateboard_pupper
FORCE_CPU=1 uv run pytest tests/test_pupper_skate.py tests/test_pupper_skateboard.py tests/test_task_configs.py -q
```

Expected: `36 passed`. `FORCE_CPU=1` is for Macs; drop it on a machine with an
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
4. Set the task in section 3 (Balance first), then Runtime → Run all. Nothing
   prompts. 1,500 iterations take about 9 minutes on a G4, about 55 on a T4. You can also run the cells
   one at a time.
5. Read the last line of section 7, *Did it pass?*: `RESULT: PASS` or
   `RESULT: FAIL`, with the numbers above it. The same check works on any run
   from the repo root: `python skateboard/tools/run_summary.py <task-id>` (on a laptop: `uv run python ...`).

The thresholds live in `PASS_CRITERIA` in `tools/run_summary.py`. What to change
when a run fails is in Tasks 9 and 11 of the plan.

### 4. Replay the trained policies (laptop)

The two passing policies are in [`policies/`](policies/), so no W&B login is
needed. From the repo root:

```bash
# Ride, in the interactive viewer
FORCE_CPU=1 uv run play Mjlab-Skate-Ride-Pupper-v3 \
  --checkpoint-file skateboard/policies/ride_29p4j81t_model_1499.pt --num-envs 1

# Ride or balance, saved as a video plus stills (no viewer needed)
FORCE_CPU=1 uv run python skateboard/tools/render_policy.py Mjlab-Skate-Ride-Pupper-v3 \
  skateboard/policies/ride_29p4j81t_model_1499.pt out/ride.mp4 --speed 0.4
FORCE_CPU=1 uv run python skateboard/tools/render_policy.py Mjlab-Skate-Balance-Pupper-v3 \
  skateboard/policies/balance_2ace1uqa_policy.json out/balance.mp4
```

Each video takes about a minute on a Mac. The balance run kept only its
exported `policy.json`, so it replays through `render_policy.py`, not `play`.
Rendered clips and stills are in [`media/`](media/): ride at 0.4 m/s from both
sides, and balance.

### 5. Robot

No skate policy runs on the robot without a TA sign-off. Steps are in Task 12.

## Credit

The board model follows the structure of HUSKY (Han et al.,
[arXiv:2602.03205](https://arxiv.org/abs/2602.03205)). No HUSKY code is included.
