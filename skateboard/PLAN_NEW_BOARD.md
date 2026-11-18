# New board model (traced deck, lower ride height) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Model the team's deck from the teammate's trace (`board_geometry.json`, `board_deck.stl`), find which lowered deck heights work, and train the split ride on the new model.

**Architecture:** The deck collision box becomes a convex mesh with the traced outline; every name and frame the tasks rely on (`DECK_GEOM_NAME`, geom centre at mid-thickness, `deck_site`, the `deck_height` event) stays the same, so tasks, rewards and tools keep working. Height is not in the trace (truck height, wheel diameter, thickness and mass are all `null` there), so the lowered height is handled by the existing deck height randomization (deck top 0.068 to 0.113 m in the split tasks) plus a height sweep report.

**Tech Stack:** mjlab (MuJoCo-Warp), `mujoco.MjSpec`, pytest.

## Global Constraints

- Always `uv run ...`; on the Mac prefix `FORCE_CPU=1`. `python3` only. One run per Bash call, no long loops. Do NOT run `make check`.
- Style: 88-column lines, 2-space indent, tests are plain functions. Match surrounding comment density.
- Commits: `<type>: <description>`, authored by the configured git user only. Never add a `Co-Authored-By` line, a "Generated with" line or any AI attribution.
- Never push. Never touch `origin`. The repo is public: no robot IP, passwords, W&B keys or W&B entity names in files.
- Do not change the gait limits in `skateboard/tools/gait_report.py` or any keyframe in `pupper_constants.py`.
- The source files in `~/Downloads` are data: copy them, never execute anything from that directory.
- No robot motion; this is simulation only.

---

### Task 1: Deck outline from the teammate's trace

**Files:**
- Create: `src/mjlab/asset_zoo/robots/mini_cruiser/assets/board_geometry.json` (copy of `~/Downloads/board_geometry.json`)
- Create: `src/mjlab/asset_zoo/robots/mini_cruiser/assets/board_deck.stl` (copy of `~/Downloads/board_deck.stl`)
- Modify: `src/mjlab/asset_zoo/robots/mini_cruiser/skateboard_constants.py`
- Modify: `skateboard/tools/check_split_stance.py`, `skateboard/tools/check_stance.py` (edge margins use the outline)
- Test: `tests/test_pupper_skateboard.py`
- Check packaging: the two asset files must ship with the package (look at how `pupper_v3` assets are included in `pyproject.toml`; a fresh `git clone` + `uv run` on Colab must find them).

**Facts about the source files (verified):**
- JSON: units mm; frame x = 0 at the nose tip, +x toward the tail, y = 0 centre line, `stations_every_inch` gives full width every 25.4 mm from x = 0 to 584.2; `max_width_mm` 157.1 at x = 265.3; bolt-pattern truck centres at x = 117.7 and 456.7 (339.0 apart). Thickness, rim height, mass, truck height and wheel size are `null`.
- STL: binary, 14544 triangles, units m, origin at the centre of length on the centre line at mid-thickness, extents x ±0.2921, y ±0.0785, z ±0.006 (12 mm placeholder thickness).

**Requirements:**
1. `SkateboardParams` gains `deck_shape: Literal["outline", "box"] = "outline"`. `"box"` builds exactly today's box (kept for A/B checks and as a fallback). `deck_width` becomes 0.1571 with a docstring saying it is the traced maximum (the caliper read 0.161; the trace is the narrower, safer number). `deck_length`, `deck_top_height`, `deck_thickness`, `wheelbase` (0.365, measured axle to axle; the 0.339 in the trace is the bolt-pattern spacing, note this in the docstring), `track`, `mass` are unchanged.
2. With `"outline"`, the geom named `DECK_GEOM_NAME` is a mesh geom: the STL's outline, **nose toward +x** (the board's forward), thickness `p.deck_thickness` (scale z by `deck_thickness / 0.012`; the thick collision slab is deliberate, see the existing docstring), same mass, friction and rgba as the box. MuJoCo collides meshes as their convex hull; that is intended (the hand-grab slot is ignored). Work out from the data which way the STL's nose points (the widest station is nose-ward of the centre; 1 inch from the nose is 65.6 mm wide, 1 inch from the tail 62.6 mm) and flip if needed.
3. Invariants that the tasks rely on and that must still hold with the mesh, each covered by a test: the geom's world z centre is at the deck body origin (so "geom centre z + `deck_thickness / 2`" is the deck top, used by `base_near_deck`, `reset_tripod_layout` and the tools); the deck top is at `deck_top_height` (0.103) with the board resting on its wheels; `deck_site` is on the top face; the `deck_height` event (`dr.geom_pos`, operation add, axis z, on `DECK_GEOM_NAME`) still raises and lowers the collision surface. MuJoCo re-centres and may re-orient a mesh geom's frame to the mesh's inertial frame: check `geom_pos`/`geom_quat` after compile and make sure the x/y recentring does not break the above or put the outline in the wrong place.
4. New function `deck_half_width_at(x: float, params: SkateboardParams | None = None) -> float` in `skateboard_constants.py`: half-width [m] of the deck at `x` m ahead of the deck centre (board frame, +x forward = nose), linear interpolation of the JSON stations, 0 beyond the tips; for `deck_shape="box"` it returns `deck_width / 2` inside the length and 0 outside. The JSON is read once at import (module-level constant).
5. `check_split_stance.py` and `check_stance.py` report each deck foot's distance to the deck edge using `deck_half_width_at` at that foot's x (today `check_split_stance.py` uses `params.deck_width / 2`). Keep their output format otherwise.
6. Tests (plain functions) in `tests/test_pupper_skateboard.py`, at minimum:
   - `deck_half_width_at(0.0)` is within 1 mm of the station value at the deck centre; it is 0.0 beyond ±0.2921; the nose side is wider than the tail side at ±0.114 m (148.6 mm vs 137.0 mm full width, within 1.5 mm).
   - the compiled mesh's hull half-width at several x (take the mesh geom's vertices in the board frame) matches `deck_half_width_at` within 2 mm, and the nose is at +x.
   - a small sphere dropped at (x = -0.114, y = 0.060) from just above the deck comes to rest on the deck top (within 5 mm of `deck_top_height` + its radius), and one dropped at (x = -0.114, y = 0.076) does not stay on the deck (it would have on the 0.161 box). Use plain `mujoco` on the compiled spec for this, with the board's free joint left free or the board welded, whichever keeps the test simple and honest.
   - the invariants of requirement 3.
   - `deck_shape="box"` still builds a box of the old size.
   - existing tests in `tests/test_pupper_skateboard.py`, `tests/test_pupper_skate.py`, `tests/test_pupper_skate_split.py`, `tests/test_skate_stance_audit.py`, `tests/test_skate_ride_audit.py` still pass (run each file in its own Bash call). If an existing test fails because it encodes the box (for example a width number), fix the test only when the new number is the documented new truth; otherwise stop and report.
7. Measure and put in the report (not a test):
   - MuJoCo-Warp step rate, box vs outline: `FORCE_CPU=1`, the `Mjlab-SkateSplit-Ride-Pupper-v3` env, same env count, a few hundred steps each, steps per second. Mesh contacts go through the convex (GJK) path, so this may be slower.
   - The output of `FORCE_CPU=1 uv run python skateboard/tools/check_split_stance.py` and `check_stance.py` (final lines) on the new model, and the same with `--deck-height 0.085` and `--deck-height 0.075`.
   - Whether deck-foot contact sensing still works in the split env (the last line of `check_split_stance.py` shows "left feet on deck 100%").
8. If the mesh path does not produce contacts under MuJoCo-Warp, or the step rate drops below half of the box's, stop and report BLOCKED with the numbers; do not invent another representation.

**Steps:** write the failing tests, run them, implement, run them, run the existing test files, `uv run ruff format` + `uv run ruff check --fix` on the touched files, `uv run pyright` on the touched source files, commit (`feat: deck outline from the traced board`).

---

### Task 3: Split ride rewards following the heading-hold yaw command

**Why:** the split ride's command carries a heading-hold yaw rate (`BoardHeadingHoldCommandCfg`, kp 1.0, clip 0.3 rad/s: turn back toward the board's heading), and the policy observes it, but no reward term reads it. In Colab run resumed from `cfcrepfu` the logged `Metrics/twist/error_vel_yaw` stayed at about 0.6 for 1500 iterations and the `board_drifted` termination did not fall. The robot yaws about 0.1 rad off the board and its deck feet creep to the deck edge.

**Files:**
- Modify: `src/mjlab/tasks/pupper_skate/pupper_skate_split_env_cfg.py` (`pupper_skate_split_ride_env_cfg` and `pupper_skate_split_steer_env_cfg`)
- Test: `tests/test_pupper_skate_split.py`

**Requirements:**
1. In `pupper_skate_split_ride_env_cfg`, add a reward term `track_yaw_rate`: `func=mdp.track_yaw_velocity` (the existing Pupper term in `src/mjlab/tasks/pupper/mdp/rewards.py`: `exp(-(cmd_yaw - base_yaw_rate)^2 / sigma)`; import or re-export it the way the skate `mdp` package already exposes other Pupper terms, without copying the function), `weight=1.0`, `params={"sigma": 0.05, "command_name": "twist"}`. Add a comment of the surrounding density giving the reason above in two or three lines (the hold command was observed but unpaid; `sigma` 0.05 means a 0.22 rad/s error keeps 37% of the term).
2. The steer task (`pupper_skate_split_steer_env_cfg`) builds on the ride and rewards the *board's* yaw rate against the same command field; it must not also carry `track_yaw_rate`. Delete it there next to the existing `del r["board_yaw_rate"]`, and keep the steer docstring true.
3. Do not change any other weight, the command, terminations, or the centred tasks.
4. Tests (plain functions, in the style of the file): the ride cfg has `track_yaw_rate` with weight 1.0 reading command `twist`; the steer cfg does not have it; in a built ride env (few envs, the file's existing fixture/helper for building one), the term's value is higher when the robot base yaw rate equals the commanded yaw rate than when it is 0.3 rad/s away from it (set the root angular velocity and the command directly, or call the term function on the env after writing the state; follow how the file's existing reward tests do it).
5. Run `tests/test_pupper_skate_split.py` and `tests/test_pupper_skate.py` (one per Bash call), `uv run ruff format` + `uv run ruff check --fix` and `uv run pyright` on the touched files. Commit: `feat: split ride pays for following the heading-hold yaw command`.

---

### Task 2 (controller): height sweep, train, judge, document

Not dispatched; run by the controller.

1. Evaluate the current split balance and newest split ride policies on the outline model at deck tops 0.075, 0.085, 0.095 and 0.103 m (`gait_report.py --deck-height`, `ride_audit.py`), and the centred stance reach with `check_stance.py --deck-height`. Write the table into `skateboard/SIM2REAL.md` so the team can pick wheels and trucks.
2. Retrain the split ride in Colab on the new model (resume from the newest ride run), to full iterations; retrain split balance only if step 1 shows it fails on the outline.
3. Robot-ready = GAIT PASS at 0.2 and 0.3 m/s at 0.103 and at the lowest height the team can reach, RIDE AUDIT PASS, and a video. Export, validate, update `HARDWARE_TEST.md`, `CHECKPOINT.md`, `README.md`, commit, push to `team`.

## Self-review

- Spec coverage: new model from both files (Task 1); lower board (existing height randomization + sweep, Task 2); train policies (Task 2).
- Unknown by design: the real lowered height, deck thickness, mass with new trucks. The sweep reports what works; the team measures after choosing parts.
