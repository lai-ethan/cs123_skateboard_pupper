# Gemini Pupper Demo Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gemini Pupper's skate mode runs the best skateboard policies we have (by name, switchable with one line), cannot be taken down by a bad config, stops from the debug panel, tells Gemini the truth about placement and drift, and ships with a demo runbook and a scripted desk test of the whole demo.

**Architecture:** `skate/skate.yaml` becomes a small catalogue: shared limits at the top, a `policies:` map with one entry per policy file (its sim estimates and what it has been tested on), and `policy:` naming the one the robot loads. Still one controller (`gemini_skate`) and one loaded policy; switching is an edit plus a restart. A config error turns skate mode off with a spoken reason instead of stopping the server or the launch. The server gains a skate block in `/system/status` and makes the debug panel's STOP and the app's disconnect safe on the board.

**Tech Stack:** Python 3 (FastAPI server, ROS 2 `ros2_control` through `robot/robot_api.py`), PyYAML, the repo's own desk test harness (`lab/test.py` → `robot/test_tools.py`, fake robot `Rig`). Sim measurements come from the pupper-mjlab repo (`skateboard/tools/timed_ride.py`).

Written 2026-11-27. Repos: **GP** = `/Users/ethan/claude/cs123/pupper/gemini_pupper`, branch `skate-voice` (HEAD `752b7c2`, pushed). **MJ** = `/Users/ethan/claude/cs123/pupper-mjlab`, branch `skateboard`.

## What we know (2026-11-27)

- Hardware, 2026-11-26, Gates carpet, gamepad (not through Gemini Pupper):
  split balance held; steer v6 and steer v10 both rolled the board forward
  for a whole clip (about 29 s and 28 s; which clip is which is as reported).
  Nothing was measured. The skate tools in GP have never run on the robot.
- Sim: v10 passes every ride audit row v6 passes and holds 0.92 of the asked
  speed at x6 wheel drag (v6 0.70). v13 and v14 (heading work,
  `PLAN_HEADING.md`) cut sideways drift; one of them is exported at the end of
  that plan.
- GP `skate-voice` today: one policy file (`skate/policy.json`, steer v6), 313
  desk checks pass. Survey findings this plan fixes:
  - the debug panel's STOP calls `/move/velocity`, which is refused on the
    board, so the browser has no working stop there (`skate/README.md` says
    it works);
  - a malformed `skate/skate.yaml` or policy file raises inside
    `Robot.__init__` and inside the launch file, so the server and the whole
    robot stack fail to start;
  - the app's disconnect calls `/walking/deactivate`, which is refused on the
    board, so a ride in progress is not stopped by a disconnect;
  - `stop` and `robot_status` descriptions do not mention skating; the
    placement text does not say which side or which way to face;
  - README, AGENTS.md, `lab/test.py` and module docstrings do not mention
    skate; there is no operator runbook.

## Global Constraints

- GP `AGENTS.md` safety rules override everything. Verbatim:
  - "**Never move the robot without the student's explicit OK, right then.** That includes `robot/start.sh`, `python3 lab/chat.py --robot`, `curl` to any POST endpoint of the robot server, `ros2 topic pub`, `ros2 control switch_controllers`, and anything that activates a controller."
  - "**New trained policies need a TA's OK** before their first run on the robot."
  - "**Respect the limits.** Don't raise `MoveLimits`, `max_body_angle_deg`, joint limits, speed caps or gains to make something "work"."
  - "**No secrets in files.** Never write an API key into the repo, a commit, a log or the web app."
- Nothing in this plan runs on the robot. Desk tests only: from the GP root,
  `uv run --no-project --with pyyaml python lab/test.py [group]`.
- The hard ceilings in `robot/skateboard.py` do not change: `MAX_SPEED = 0.3`,
  `MAX_CARVE = 0.5`, `MAX_COMMAND_S = 5.0`, `MAX_RIDE_S = 10.0`.
- One skate controller, `gemini_skate`, and one loaded policy. No tool lets
  Gemini change the policy.
- Every tool returns a Result whose message Gemini reads word for word: say
  what happened, or what was wrong and the limit or the real choices. Motion
  goes through `robot.queue`; wait with `await robot.sleep`, never
  `time.sleep`. The robot is called Pupper. Keep Peng Xu's credit.
- `robot/` and `web/` are "provided" files in the course repo; this branch
  already changes `robot/`. Keep changes there small and tested, and do not
  change `web/` (it would need a rebuild on the robot).
- Leave the untracked `tricks/sit_stay_shake/` and `tricks/spider_walk/`
  alone.
- Match each file's existing style. Commits `<type>: <description>`, authored
  by the configured git user only, no AI attribution. Both repos are public:
  no keys, IPs, passwords, W&B account names or local home-directory paths.
- Sim numbers are estimates; every doc and message that quotes one says so.

## File structure (GP)

- `skate/skate.yaml`: the catalogue (limits, `policies:`, `policy:`).
- `skate/policies/<name>.json`: one exported policy per entry (moved from
  `skate/policy.json`).
- `robot/skateboard.py`: config loading and validation, `SkateConfig`,
  messages. No ROS imports.
- `robot/robot_api.py`, `robot/launch/gemini_pupper.launch.py`: use the safe
  loader; `Robot.skate_error`.
- `robot/server.py`: `/system/status` skate block; STOP and disconnect on the
  board.
- `lab/part8_skate.py`, `lab/basics.py`, `lab/prompt.md`: what Gemini reads.
- `robot/test_tools.py`: desk tests (new groups `skate_catalogue`,
  `skate_demo`).
- `skate/README.md`, `skate/DEMO.md`, `README.md`, `AGENTS.md`,
  `lab/test.py`, docstrings: documentation.

---

### Task 1: Measure the voice estimates for each policy (controller, MJ)

**Files:**
- Create: `skateboard/policies/VOICE_ESTIMATES.md` (MJ)

- [ ] **Step 1:** For each exported policy that will ship in GP (steer v10
  now; the heading policy when `PLAN_HEADING.md` exports one), run, one at a
  time:

  ```bash
  FORCE_CPU=1 uv run python skateboard/tools/timed_ride.py \
    skateboard/policies/<file>_max0.3_policy.json --yaw-rate=<y>
  ```

  for `<y>` in `0.0 0.15 -0.15 0.3 -0.3 0.5 -0.5` (16 envs, 5 s ride, 4 s
  coast: the tool's defaults).
- [ ] **Step 2:** Stop if any run has `resets > 0` or a turn of the wrong sign.
- [ ] **Step 3:** Record every JSON line in `VOICE_ESTIMATES.md` with the
  rounding rule used for v6 (`PLAN_GEMINI_VOICE.md`): `travel_m_per_s` =
  straight-run `m_per_s` to 2 decimals; each turn estimate = the smaller of
  the left and right `turn_deg_per_s` sizes, to 1 decimal; `coast_m` = the
  largest `coast_m_max` over the seven runs, rounded up to 1 decimal. v6's
  recorded values: 0.29; 1.1 / 1.4 / 2.2; 0.1.
- [ ] **Step 4:** Commit in MJ: `docs: add voice estimates for the shipped skate policies`.

### Task 2: A policy catalogue that cannot take the robot down

**Files (GP):**
- Modify: `robot/skateboard.py`, `skate/skate.yaml`, `robot/robot_api.py`
  (the `load_skate()` call in `Robot.__init__`, about line 228),
  `robot/launch/gemini_pupper.launch.py` (about lines 57-59),
  `lab/part8_skate.py` (the two `NOT_INSTALLED` uses), `robot/test_tools.py`
- Move: `skate/policy.json` → `skate/policies/split_steer_v6_max0.3_policy.json` (`git mv`)
- Create: `skate/policies/split_steer_v10_max0.3_policy.json` (copy of MJ
  `skateboard/policies/split_steer_v10_max0.3_policy.json`)

**Interfaces:**
- Consumes: Task 1's numbers for steer v10 (in the dispatch).
- Produces:
  - `SkateConfig` gains `name: str` (the catalogue key) and `tested: str`.
  - `class SkateConfigError(ValueError)`.
  - `load_skate(skate_dir: Path = SKATE_DIR) -> SkateConfig | None`: None when
    there is no `skate.yaml`; otherwise the selected entry, or
    `SkateConfigError` whose message names the file and the key.
  - `load_skate_safe(skate_dir: Path = SKATE_DIR) -> tuple[SkateConfig | None, str | None]`:
    never raises on a bad config; returns `(cfg, None)`, `(None, None)` when
    not installed, or `(None, "<message>")`.
  - `not_installed_message(error: str | None) -> str`.
  - `Robot.skate_error: str | None`.

- [ ] **Step 1: New yaml schema.** Rewrite `skate/skate.yaml` as:

  ```yaml
  # Skate mode for Gemini Pupper. Read by robot/skateboard.py; see its docstring.
  # To change policy: set `policy` to another name below, then restart the
  # robot stack (robot/stop.sh, robot/start.sh). A new policy needs a TA's OK.
  policy: steer_v6
  speed: 0.3            # m/s: the only speed checked in sim; the policies clip here
  carve:                # yaw command, rad/s (0 rides straight)
    gentle: 0.15
    medium: 0.3
    hard: 0.5           # not 0.6: one sim env lost the board there
  default_seconds: 3
  max_seconds: 5
  queue_max_seconds: 10
  # One entry per policy file. The numbers are estimates from simulation
  # (pupper-mjlab skateboard/tools/timed_ride.py, 16 boards, 5 s rides), not
  # measurements. `tested` says what the policy has done on the real robot.
  policies:
    steer_v6:
      file: policies/split_steer_v6_max0.3_policy.json
      tested: "robot 2026-11-26: rolled forward on carpet by gamepad; not measured"
      travel_m_per_s: 0.29
      turn_deg_per_s: {gentle: 1.1, medium: 1.4, hard: 2.2}
      coast_m: 0.1
    steer_v10:
      file: policies/split_steer_v10_max0.3_policy.json
      tested: "robot 2026-11-26: rolled forward on carpet by gamepad; not measured. Provisional: missed its sim bar at x10 wheel drag"
      travel_m_per_s: <Task 1>
      turn_deg_per_s: {gentle: <Task 1>, medium: <Task 1>, hard: <Task 1>}
      coast_m: <Task 1>
  ```

  The three `<Task 1>` values are given in the dispatch; they are not left in
  the file.

- [ ] **Step 2: Failing tests** in `robot/test_tools.py`, new group
  `skate_catalogue` (add it to `GROUPS` and to `PARTS["skate"]`), using the
  file's existing `_write_skate` / `_load_error` helper style (extend the
  helpers to write the new schema; update `fixture_skate()` to the new
  dataclass fields with `name="steer_v6"`):
  - the repo's own `skate/skate.yaml` loads; `cfg.name == "steer_v6"`;
    `cfg.policy_path` ends with `policies/split_steer_v6_max0.3_policy.json`
    and exists; `cfg.tested` starts with `"robot 2026-11-26"`;
  - every entry under `policies:` in the repo's yaml loads when selected
    (write a temp copy of the yaml with `policy:` set to each name in turn,
    pointing at the real files) and passes every existing validation;
  - `policy:` naming an entry that does not exist is a `SkateConfigError`
    whose message contains `skate.yaml`, the bad name and the valid names;
  - a missing `policies:` map, an entry without `file`, a missing policy
    file, a policy file that is not JSON, a policy JSON that is a list, a
    `heading_hold` that is not a mapping, `carve: 0.3` (not a mapping) and
    `turn_deg_per_s: 2` each raise `SkateConfigError` (not `TypeError`,
    `AttributeError` or `JSONDecodeError`) with the file name and the key in
    the message;
  - `load_skate_safe` returns `(None, None)` for a folder with no yaml,
    `(cfg, None)` for a good one, and `(None, msg)` for each bad case above,
    never raising;
  - `not_installed_message(None)` is the existing `NOT_INSTALLED` text;
    `not_installed_message("skate.yaml: policy 'x' is not one of steer_v6")`
    is `"skate mode is off on this robot because its config has an error
    (skate.yaml: policy 'x' is not one of steer_v6), so Pupper can't skate"`;
  - a `Rig` built with a broken skate config has `robot.skate is None`,
    `robot.skate_error` set, no `gemini_skate` in its leg controllers, and
    `start_skating(placed=True)` answers with that message while `stand_up`
    and walking tools still work.

- [ ] **Step 3: Run** `uv run --no-project --with pyyaml python lab/test.py skate_catalogue`.
  Expected: FAIL (group or names missing).

- [ ] **Step 4: Implement** in `robot/skateboard.py`: the schema above
  (top-level `travel_m_per_s`, `turn_deg_per_s`, `coast_m` and a file-valued
  `policy` are no longer accepted: a yaml in the old shape is a
  `SkateConfigError` saying "skate.yaml uses the old single-policy layout;
  see skate/README.md"); `SkateConfigError`; every existing check kept, now
  raising `SkateConfigError`; type checks before `.get` / `in` so no other
  exception type escapes; `load_skate_safe`; `not_installed_message`. Update
  the module docstring's field list. Then: `Robot.__init__` and the launch
  file call `load_skate_safe()`; on an error they log it at error level
  (launch: print to stderr) and continue with skate off; `Robot` stores
  `skate_error`; `lab/part8_skate.py` uses
  `not_installed_message(robot.skate_error)`; the registry's `{skate}`
  fallback text uses it too. The `Rig` gains a way to inject a load error.

- [ ] **Step 5: Run** the full suite. Expected: `All N checks passed.` with
  N greater than 313 and no FAIL line.

- [ ] **Step 6: Commit** `feat: skate policy catalogue with a safe loader`.

### Task 3: Stop, disconnect and status on the board

**Files (GP):**
- Modify: `robot/server.py` (`/move/velocity`, `/walking/deactivate`,
  `/system/status`), `robot/test_tools.py`

**Interfaces:**
- Consumes: `Robot.skate`, `Robot.skate_error`, `SkateConfig.name`,
  `SkateConfig.tested` (Task 2); `robot.immediate_stop()`,
  `tool_registry.on_board_refusal`, `robot.controller_states()` (existing).
- Produces: `GET /system/status` returns an extra key
  `"skate": {"installed": bool, "policy": str | None, "tested": str | None, "error": str | None, "controller": "active" | "inactive" | "missing"}`.

- [ ] **Step 1: Failing tests** in group `skate_guards` (use the existing
  `_load_server` helper):
  - on the board, `POST /move/velocity` with vx, vy, wz and duration all 0
    (what the debug panel's STOP sends) stops the ride: the queue is empty,
    a zero command was sent, the skate controller is still active, the reply
    is ok and its message contains "stopped"; any non-zero `/move/velocity`
    is still refused with the on-board message;
  - on the board, `POST /walking/deactivate` (what the app sends when it
    disconnects) stops the ride the same way and does NOT relax the motors;
    its message says Pupper is still balancing on the board and that
    end_skating with a person holding it is the way off;
  - off the board both endpoints behave exactly as before (compare with the
    replies on a `Rig` without skate);
  - `GET /system/status` has the `skate` block: with the fixture config,
    `installed` true, `policy` "steer_v6", `error` null, `controller`
    "inactive" before `start_skating` and "active" after; with a broken
    config, `installed` false and `error` set; with no skate config,
    `installed` false, `error` null, `controller` "missing". The existing
    keys of the status reply are unchanged.

- [ ] **Step 2: Run** `... lab/test.py skate_guards`. Expected: FAIL.

- [ ] **Step 3: Implement** in `robot/server.py`. The battery, cpu and
  internet parts of `/system/status` are untouched; the skate block reads
  `tools.robot` state without moving anything (if controller states are
  only available asynchronously, use the same call the existing endpoints
  use). A zero `/move/velocity` and `/walking/deactivate` on the board call
  the same stop the voice `stop` tool uses.

- [ ] **Step 4: Run** the full suite. Expected: all pass.

- [ ] **Step 5: Commit** `fix: stop and disconnect work on the board, status reports skate`.

### Task 4: What Gemini reads

**Files (GP):**
- Modify: `lab/part8_skate.py` (`PLACEMENT`, the `start_skating`
  description), `lab/basics.py` (the `stop` and `robot_status` tool
  descriptions only), `lab/prompt.md` (the Skateboard section),
  `robot/robot_api.py` (`mode()` skating line), `robot/test_tools.py`

**Interfaces:**
- Consumes: `SkateConfig.name` (Task 2), `WALK_READY_S` (existing).

- [ ] **Step 1: Failing tests** (groups `skate_tools` and the prompt checks):
  - `PLACEMENT` is exactly: `"Pupper beside the board on the board's right, body parallel to it and facing the end with the castle picture, left feet on the middle of the deck, right feet on the floor, a hand on the harness"`;
  - the not-placed reply of `start_skating` contains `PLACEMENT`;
  - the `start_skating` description states the wait as
    `f"about {WALK_READY_S:.0f} s"` (no hard-coded 4);
  - the `stop` description says that on the skateboard it stops the pushing
    and the board rolls on a little; the `robot_status` description lists
    skating among the states;
  - `robot.mode()` while skating names the loaded policy:
    `"skating (policy steer_v6): on the skateboard in skate mode (skate_forward and skate_turn ride it; end_skating relaxes the motors to get off)"`;
  - `lab/prompt.md`'s Skateboard section contains "beside the board",
    "castle", "drift", and the queue limit it states equals
    `queue_max_seconds` in the repo's `skate/skate.yaml`.

- [ ] **Step 2: Run** the two groups. Expected: FAIL.

- [ ] **Step 3: Implement.** In `lab/prompt.md` replace the placement
  sentence with: "A person has to place you: beside the board on its right,
  facing the end with the castle picture, your left feet on the middle of
  the deck and your right feet on the floor." and add after the sentence
  about wide, slow curves: "On a long ride you also drift a little to one
  side, so keep rides short and ask the person to straighten the board when
  it has wandered." Change nothing else in the section.

- [ ] **Step 4: Run** the full suite. Expected: all pass.

- [ ] **Step 5: Commit** `feat: tell Gemini where Pupper stands on the board and that it drifts`.

### Task 5: The demo, scripted on the desk, and the runbook

**Files (GP):**
- Modify: `robot/test_tools.py` (new group `skate_demo`, in `GROUPS` and
  `PARTS["skate"]`), `skate/README.md`, `README.md`, `AGENTS.md`,
  `lab/test.py` (docstring), `robot/robot_api.py` (header docstring only),
  `robot/spawn_controllers.py` (docstring only), `docs/architecture.md`
  (state list only)
- Create: `skate/DEMO.md`

- [ ] **Step 1: The demo test** (`skate_demo`), run once for every policy
  name in the repo's `skate/skate.yaml` (load that entry's real file):
  1. from motors idle: `start_skating(placed=False)` is refused with the
     placement text; `start_skating(placed=True)` succeeds; `robot_status`
     says skating and names the policy;
  2. `walk` / `move` is refused with the on-board message;
  3. `skate_forward()` rides `default_seconds`; the reply's distance equals
     `travel_m_per_s * default_seconds` to 1 decimal and says "Estimate";
  4. `skate_turn(direction="left", sharpness="gentle", seconds=3)` rides and
     the reply's turn equals `turn_deg_per_s["gentle"] * 3` to 0 decimals;
  5. `skate_forward(seconds=5)` then `stop` mid-ride: the last command sent
     is zero and the controller is still the skate controller;
  6. gamepad e-stop while riding: motors relaxed, `stand_up` refused,
     `start_skating(placed=True)` works again;
  7. tilt fall while skating: walking refused; `end_skating(held=False)`
     refused; `end_skating(held=True)` relaxes; after the fake robot is set
     upright, `stand_up` succeeds;
  8. the commands sent during every ride never exceed `speed` 0.3 or a yaw
     of 0.5 rad/s, and no single command lasts longer than 5 s.
  The group fails loudly, naming the policy, if any step fails.

- [ ] **Step 2: Run** `... lab/test.py skate_demo`. Expected: PASS for both
  policies if Tasks 2 to 4 are right; a failure here is a real defect in an
  earlier task: stop and report it, do not adjust the test to pass.

- [ ] **Step 3: `skate/DEMO.md`**, an operator runbook with these sections:
  "Before the day" (TA's OK for each policy file; `policy:` chosen in
  `skate.yaml`; `python3 lab/test.py skate` passes; board: loosened trucks,
  which end is the nose); "Start" (the repo's normal start steps, then open
  `/system/status` and check the `skate` block says `installed: true`, the
  expected `policy`, `controller: "inactive"`; if `error` is set, read it,
  fix `skate.yaml`, restart); "Placement" (the `PLACEMENT` text, plus the
  numbers from MJ `skateboard/HARDWARE_TEST.md`'s placement section, quoted
  as sim values); "Script" (the spoken demo: lie down, place, "start
  skating", "go forward", "turn left gently", "stop", "get off the board",
  with what Pupper should say and do at each line, including the sim
  estimate it will quote for the loaded policy); "If something goes wrong"
  (voice "stop"; the debug panel's STOP; the gamepad e-stop and how to
  resume afterwards; a tip on the board; the app disconnecting; a ride that
  drifts; the board not rolling); "Changing policy" (edit `policy:`,
  restart, what each catalogue entry has and has not been tested on);
  "Limits" (0.3 m/s, 1 to 5 s a command, 10 s queued, carve 0.15 / 0.3 /
  0.5 rad/s, no reverse, no steering by distance, hands off the gamepad
  sticks). It says plainly that the skate tools have not run on the robot
  and that every number is a sim estimate.

- [ ] **Step 4: Docs.** `skate/README.md`: describe the catalogue; replace
  "Its stop button works" with what Task 3 made true; point to `DEMO.md`.
  `README.md` "What's where": add `lab/part8_skate.py` and `skate/`; add
  skating to the state list. `docs/architecture.md`: add skating to the
  state list. `AGENTS.md` and `lab/test.py`: list the `skate` group.
  `robot/robot_api.py` header: add `robot.skate`, `robot.skate_error`,
  `robot.fell_off_board` to the API list. `robot/spawn_controllers.py`
  docstring: mention `gemini_skate`.

- [ ] **Step 5: Run** the full suite. Expected: all pass.

- [ ] **Step 6: Commit** `test: script the skate demo on the desk` and
  `docs: add the skate demo runbook`.

### Task 6: Ship the heading policy and choose the default (controller)

- [ ] **Step 1:** When `PLAN_HEADING.md` exports its policy, run Task 1 for
  it, copy its `_max0.3_policy.json` into GP `skate/policies/`, and add its
  catalogue entry with `tested: "sim only: not run on the robot"`.
- [ ] **Step 2:** Run `... lab/test.py skate`. Expected: all pass, with
  `skate_demo` covering the new entry.
- [ ] **Step 3:** Default policy: `steer_v6` stays the `policy:` value unless
  Ethan chooses another; `DEMO.md` "Changing policy" lists the trade-offs.
- [ ] **Step 4:** MJ `skateboard/HARDWARE_TEST.md` test 6 and
  `skateboard/README.md`: point to GP `skate/DEMO.md` and name the catalogue.
- [ ] **Step 5:** Scan both diffs for secrets; push MJ to `team` and GP
  `skate-voice` to `origin`.

## Accounting

- Skipped: a voice tool to switch policy; more than one loaded skate
  controller; any `web/` change; a backward gait; running anything on the
  robot.
- Assumed: Ethan's earlier OK to push `skate-voice` covers this follow-up
  work on the same branch; steer v6 stays the default because it is the
  policy that passed its sim bar and rolled on the robot; the course
  controller's speed caps allow 0.3 m/s (the gamepad runs suggest so; not
  checked in this repo).
- Issues: the skate tools themselves have never run on the robot, so the
  first voice-controlled ride is still a hardware test needing a TA's OK;
  each policy file adds about 6.5 MB to the GP repo.
