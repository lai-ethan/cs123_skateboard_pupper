# Deck slide plan (steer v7)

Written 2026-11-20. Sim only. Nothing here has run on the robot.

**Goal:** the robot keeps its deck feet in place along the deck while it rides
and steers, with no foot stop or ridge on the deck.

**Scope:** `Mjlab-SkateSplit-Steer-Pupper-v3` only. Steer v6 at stick 0 is the
straight ride, so there is no separate split ride retrain.

## What the diagnosis found (steer v6, run `hfd3qhwr`, 7 rollouts)

- The one sim robot that slides (seed 0, env 12) has soft gains on its deck
  legs: mean kp scale 0.75, the lowest of 16, with both deck abduction joints
  near 0.6. Giving its gains to the other 15 made all of them creep towards
  the nose, and 4 hit `board_slid` in 15 s. Low grip (0.535) and high wheel
  drag alone did not: all 16 rode 15 s clean.
- The creep is a ratchet. When one right foot pushes alone, the deck foot on
  the same end drops to 1 to 2 N (about 10 N otherwise) and scuffs about 5 mm.
  It scuffs back when reloaded. The leftover is about 1 mm per stride.
- The policy cannot see where it stands along the deck. Its observations are
  IMU, command, joint angles and last action. So `board_slid` and
  `board_fore_aft` punish something it cannot sense, and a reset alone can
  start a robot up to 6.2 cm into the 8 cm limit.
- `deck_feet_slide`, the one term that charges the scuff itself, costs 0.0015
  to 0.011 per step. `board_creep` costs 0.07 to 0.33. The scuff term is
  effectively off.
- There is no correction: robots far from centre creep outward faster.

## Global constraints

- No change to the actor observation layout or the action space: v7 resumes
  from v6 checkpoint `model_26998.pt`.
- No new sensors. The real robot has only what the deploy contract lists.
- Changes apply to the split ride and steer tasks through the shared ride cfg.
  Balance tasks and centred tasks keep their behaviour.
- Repo style: 88 columns, 2-space indent, plain-function tests. Run only the
  touched test files, with `FORCE_CPU=1 uv run pytest tests/<file>.py`.
- Commits `<type>: <description>`, authored by the configured git user only,
  no AI attribution. Public repo: no W&B entity names, keys, IPs.

## Task 1: charge the scuff, and measure the slide from the episode start

Files: `src/mjlab/tasks/pupper_skate/pupper_skate_split_env_cfg.py`,
`src/mjlab/tasks/pupper_skate/mdp/terminations.py`,
`src/mjlab/tasks/pupper_skate/mdp/rewards.py`, `mdp/__init__.py`,
`tests/test_pupper_skate_split.py`, and `skateboard/tools/gait_report.py`
only if it needs the new value.

1. `deck_feet_slide` weight in the split ride cfg: `-2.0` to `-20.0`, through
   a new constant `DECK_FEET_SLIDE_WEIGHT = -20.0` with a comment that gives
   the measured costs above. Update the comment at the term.
2. Slide from the start. Today `board_slide` is the fore-aft offset minus the
   keyframe's. Add the slide since the episode's own start:
   - the offset at the first step after each reset is stored per env (follow
     the state pattern `board_creep` already uses, including its `reset`);
   - `board_slid` fires when the slide since the start is more than
     `BOARD_MAX_SLIDE` (0.08 m, unchanged), or when the absolute `board_slide`
     from the keyframe is more than a new `BOARD_MAX_SLIDE_ABS = 0.12` (the
     backstop that keeps the front left foot on the deck);
   - `board_fore_aft` pays on the slide since the start (std unchanged).
   - `board_slide` itself keeps its meaning (tools and tests read it).
   - `board_creep` is unchanged.
   Reset ranges stay as they are, so placement variety remains.
3. Tests (write them first, see them fail): the weight constant reaches the
   term; a robot reset 5 cm off the keyframe that does not move does not fire
   `board_slid` and gets full `board_fore_aft`; one that moves 9 cm from its
   start fires; one that starts 5 cm off and moves 8 cm more the same way
   fires on the absolute backstop or the relative limit, whichever comes
   first, and the test says which; the stored start resets per env on reset.
   Fix existing tests that pin the old behaviour, saying what changed.
4. Update the long comment above `BOARD_MAX_SLIDE` with two or three plain
   sentences on why (the diagnosis above), and the docstrings.

## Task 2: train v7 (controller, Colab)

Resume from v6 (`hfd3qhwr`, `model_26998.pt`) for 2000 iterations.

## Task 3: evaluate and decide (controller)

- Gait report at yaw 0, ±0.15, ±0.3, ±0.5, ±0.6, seeds 0, 1, 2.
- Ride audit, 16 rows.
- The diagnosis rollout with env 12's gains in every env
  (`slide_diag/diag.py`, `kp_transplant` setup).
- Pass bar: no `board_slid` reset on any seed, creep under 0.1 cm/s in the
  `kp_transplant` fleet, ride audit still 16 of 16, board turn no worse than
  v6 by more than 0.1 of asked at any stick.
- If it misses on creep: add a deck-foot minimum-load penalty (3 N floor,
  weight -0.5) and run again. If turning got worse: v6 stays the candidate.
- Videos, export, validator, docs, as for v6.
