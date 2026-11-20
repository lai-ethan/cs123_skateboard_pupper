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

## Results (steer v7, 2026-11-20)

v7 is run `n5m2zsks`: v6 `hfd3qhwr` resumed for 2000 iterations on Task 1's
code (commit `6318e2c`), to `model_28997.pt`. Evaluation as in Task 3: 16
boards, 0.3 m/s, 500 steps, 103 mm deck.

**Verdict: v7 misses the pass bar (2 of 4). It fixes the slide but hurts
turning. v6 stays the steering candidate.**

| Bar | v7 | v6 | Met? |
| --- | --- | --- | --- |
| No `board_slid` reset, any seed | 0 in 27 gait runs | 1 board (env 12) on +0.3, +0.5, +0.6, seed 0 | yes |
| `kp_transplant` creep under 0.1 cm/s | mean 0.21, max 0.31 cm/s; 0 of 16 slid out | mean 0.42, max 0.89 cm/s; 4 of 16 slid out | no |
| Ride audit 16 of 16 | 16 of 16 | 16 of 16 | yes |
| Turn no worse than v6 by > 0.1 of asked | +0.5, +0.6, -0.6 (seed 0) and +0.3 (seed 2) are worse | | no |

What got better:

- No `board_slid` reset anywhere in the gait check (3 seeds, 9 sticks).
- With env 12's soft leg gains in every env, no board slid out and the creep
  halved. It is still twice the 0.1 cm/s target, and every transplanted env
  still creeps towards the nose.
- The ride audit still passes 16 of 16. `nominal_30s` and `deck_113mm` had 1
  `board_slid` reset each, inside the audit's 1-in-16 allowance.

What got worse:

- v7 pulls right. At stick 0 the mean board yaw rate is -0.013 rad/s (v6:
  -0.003).
- Hard right carves fail. Stick -0.5 and -0.6 fail the gait check on every
  seed, on `board_drifted` (the board drifts sideways more than 6 cm), 1 to 4
  boards in 16. v6 passed both on seed 0.
- Left carves are weaker at high stick: +0.5 turns 0.56 of asked (v6: 0.67),
  +0.6 turns 0.58 (v6: 0.80), seed 0.
- Gentle right carves overshoot: 2.12 times the asked rate at -0.15 and 1.32
  at -0.3 (v6: 0.82 and 0.63), seed 0.

Turn as a fraction of the asked board yaw rate, seed 0:

| Stick | v7 | v6 |
| --- | --- | --- |
| +0.15 | 0.79 | 0.84 |
| -0.15 | 2.12 | 0.82 |
| +0.3 | 0.62 | 0.58 |
| -0.3 | 1.32 | 0.63 |
| +0.5 | 0.56 | 0.67 |
| -0.5 | 0.99, **fail** (`board_drifted`) | 0.86 |
| +0.6 | 0.58 | 0.80 |
| -0.6 | 0.72, **fail** (`board_drifted`) | 0.86 |

Decision:

- v6 stays the candidate for the steering test over the full stick range.
- v7 is kept as the low-slide alternative for riding straight and gentle
  carves, stick within ±0.3. Try it on the robot if v6 slides along the deck.
  Its export clips yaw at ±0.3 rad/s, so the robot cannot send it a hard
  carve.
- The plan's fallback for missed creep (a deck-foot minimum-load penalty) was
  not run. The cause of the right pull is not known yet; it should be
  understood before another retrain.
- Neither v6 nor v7 has run on the robot.

Files: `policies/split_steer_v7_model_28997.pt`,
`policies/split_steer_v7_max0.3_policy.json`, `policies/split_steer_v7_audit.txt`,
`media/split_steer_v7_*_0.3ms.mp4`. The full write-up is in `SIM2REAL.md`,
"Steering: what sim says".

- Skipped: the minimum-load penalty retrain; a deck height sweep for v7.
- Assumed: the `kp_transplant` setup (env 12's gains in every env) is a fair
  stress test of the slide.
- Issues: the right pull is not diagnosed; creep is still above target.
