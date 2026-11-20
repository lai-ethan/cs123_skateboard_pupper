# Steer v8 plan: stop the sideways walk-off, charge the straight-line pull

Written 2026-11-20. Sim only. Nothing here has run on the robot.

**Goal:** keep v7's grip along the deck, and get back v6's turning: hard right
carves that do not lose the board sideways, and a board that rides straight
when the stick is at 0.

**Scope:** `Mjlab-SkateSplit-Steer-Pupper-v3` only.

## What changed from v7

- v7 raised the deck-foot scuff charge and measured the fore-aft slide from
  the episode start. Both stay.
- v8 adds two reward terms (below) and resumes from v7, not v6.

## What the diagnosis found (v6 against v7, 5 sticks, seeds 0 and 2)

- Hard right carves under v7 end in `board_drifted` because the deck feet
  slide to the right across the deck while they stay loaded (5 to 10 N, in
  contact 99% of steps). The feet account for 0.8 to 5.1 cm of the 6 cm in
  the last 2 s. They are not stepping and the loads are normal.
- v7 leans further in right carves than v6 (deck roll 3.7 against 2.9 degrees
  at stick -0.6, back left foot 1.4 cm further right) and less in left carves
  (2.1 against 2.8 degrees at +0.6). So it starts a hard right carve about
  1 cm closer to the limit, on a steeper deck.
- No term charges a slow sideways slide before the -100 at 6 cm. The fore-aft
  axis has `board_creep` for exactly this; the sideways axis has nothing.
- The pull at stick 0 (-0.013 rad/s on seeds 0 and 2, -0.003 on seed 1) costs
  the policy 8.5e-5 a step. `board_yaw_rate` is an L2 term at weight -0.5 and
  no other term reads a slowly turning board.
- `deck_feet_slide` already measures slip against the deck's own motion, yaw
  and roll included. No change there.
- Deck-foot loads during the drift are 5 to 10 N, so the 3 N minimum-load
  idea from the v7 plan does not fit this failure. Dropped.
- Not determined: why v7's feet slip faster than v6's under the same sideways
  force, and what picks the envs that drift.

## Global constraints

- No change to the actor observation layout or the action space: v8 resumes
  from v7 checkpoint `model_28997.pt` (run `n5m2zsks`).
- No new sensors.
- The two new terms apply to the steer task. The ride, balance and centred
  tasks keep their behaviour: their reward and termination values must not
  change.
- `DECK_FEET_SLIDE_WEIGHT`, `BOARD_MAX_SLIDE`, `BOARD_MAX_SLIDE_ABS`,
  `board_creep` and `board_drifted` (6 cm) are unchanged.
- The notebook builds its own reward weight table
  (`skateboard/tools/make_notebook.py`, `REWARD_WEIGHTS`). A term missing
  there trains at the wrong weight or not at all. Both new terms must be in
  the table with the weights below, and `notebooks/CS123_Pupper_Skate.ipynb`
  must be regenerated with the tool, not edited by hand.
- Repo style: 88 columns, 2-space indent, plain-function tests. Run only the
  touched test files, with `FORCE_CPU=1 uv run pytest tests/<file>.py`. Do
  not run `make check`.
- Commits `<type>: <description>`, authored by the configured git user only,
  no AI attribution. Public repo: no W&B entity names, keys, IPs.

## Task 1: charge the sideways creep and the straight-line pull

Files: `src/mjlab/tasks/pupper_skate/mdp/rewards.py`, `mdp/__init__.py`,
`src/mjlab/tasks/pupper_skate/pupper_skate_split_env_cfg.py`,
`tests/test_pupper_skate_split.py`, `skateboard/tools/make_notebook.py`,
`notebooks/CS123_Pupper_Skate.ipynb` (regenerated),
`tests/test_skate_notebook.py` if it pins the table.

1. `board_side_creep`: the sideways twin of `board_creep`.
   - Same shape as `board_creep`: the rate of change of the sideways board
     offset (the quantity `board_drifted` reads, in the board's heading
     frame), through two first-order averages in a row of time constant `tau`
     each, absolute value, divided by `rate_cap` and clipped to 1. Per-env
     state reset on episode reset, as `board_creep` does. Share code with
     `board_creep` where that is clean; do not copy the block.
   - The double average is the point: the stride's side-to-side sway (about
     1 cm per 0.5 s) must cancel, and a steady slide must not. A lean is a
     constant offset and costs nothing once it is held.
   - In the steer cfg only: `r["board_side_creep"]`, weight
     `BOARD_SIDE_CREEP_WEIGHT = -1.0`, `tau = BOARD_CREEP_TAU`,
     `rate_cap = BOARD_CREEP_CAP`. A comment at the constant with the three
     numbers from the diagnosis that justify it (feet 0.8 to 5.1 cm in 2 s,
     loads 5 to 10 N, nothing charged before the -100).
2. Straight-line board tracking.
   - Today `track_board_yaw_rate` pays only in carves. Add a parameter
     `straight_min_speed: float | None = None`. When it is set, the term also
     pays outside carves in envs whose forward speed command is at least that
     value, with a target board yaw rate of 0 and the same `std`. Standing
     envs (speed command under it) still get 0. `None` keeps today's
     behaviour exactly.
   - In the steer cfg: `straight_min_speed=STEER_YAW_MIN_SPEED`. Weight stays
     2.0. Update the comment above the term: at std 0.02 a steady 0.013 rad/s
     pull keeps 66% of the term, so it now costs 0.69 a step.
   - `board_yaw_rate` (the L2 term) stays as it is.
3. Tests, written first and seen to fail:
   - `board_side_creep`: a board held at a constant sideways offset costs 0
     after the averages settle; a steady sideways slide at `rate_cap` or more
     costs the full weight; a zero-mean 2 Hz sway of 1 cm amplitude costs
     under 15% of the term; the state of one env resets on its reset and the
     other env's does not.
   - `track_board_yaw_rate` with `straight_min_speed`: a straight env at
     speed with board yaw rate 0 gets 1.0, at 0.013 rad/s gets
     `exp(-(0.013 / 0.02) ** 2)` to 3 decimals, a standing env gets 0, a
     carve env gets what it got before, and `None` reproduces the old values.
   - The steer cfg carries both, with the weights above; the ride cfg has
     neither `board_side_creep` nor `straight_min_speed`.
   - The notebook table has `board_side_creep` at -1.0 and the regenerated
     notebook contains it.
4. Docstrings for both, plain sentences. Update the steer cfg docstring's
   list of what is paid.

## Task 2: train v8 (controller, Colab)

Resume from v7 (`n5m2zsks`, `model_28997.pt`) for 2000 iterations, 4096 envs.

## Task 3: evaluate and decide (controller)

Same runs as v7: gait report at stick 0, ±0.15, ±0.3, ±0.5, ±0.6 on seeds 0,
1, 2; ride audit; the soft-gains rollout.

Pass bar, all of:

- no reset of any kind in the 27 gait runs;
- soft-gains rollout: 0 of 16 slide out, mean creep no worse than v7's 0.21
  cm/s;
- ride audit 16 of 16;
- board turn within 0.1 of asked of v6's value at every stick on seed 0, and
  no stick turning more than 1.3 times what it asks;
- at stick 0 the mean board yaw rate is within ±0.006 rad/s on every seed.

If it passes, v8 replaces v6 as the steering candidate. If it does not, v6
stays, v7 stays the fallback, and the report says which bar it missed.
Videos, export, validator and docs as for v7.

## Accounting

- Skipped: the stiffer deck-foot contact idea (it changes the physics for
  every policy, and there is no evidence yet that the slip is a sim
  artefact); a scuff weight between -2 and -20 (no ablation supports it).
- Assumed: the sideways slide is the cause of the hard right carve resets,
  from 9 events on 2 seeds.
- Issues: the stick-0 pull moved by 0.02 rad/s under a tiny perturbation in
  one diagnosis run, so it is measured on 3 seeds here, not 1.
