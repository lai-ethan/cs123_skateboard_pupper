# Pupper skateboard: research notes (2026-11-09)

Sources: 4 PDFs in `papers/`, the Pupper MJCF (`pupper_v3_complete.mjx.position.xml`, FK run in MuJoCo 4.0) and `pupper_gait_deploy/src`. **Guess** marks numbers not from a source.

## 1. Papers: what Pupper can reuse

**DHAL (2503.01842, Go1, sim-to-real)**
- The left front foot is attached to the deck by a spherical joint in sim (Sec III-C) and on hardware (Sec VI-1; Blind Dexterity Sec V-C). Not a free-board result.
- Board 800×254×110 mm; trucks are PD "springs" with target 0 (App D-B). Isaac wheels lost too much speed, so they added ±U(10,25) N along the motion when |v|>0.3 m/s (App D-A).
- Phase clock T=4 s in training; 2.5 s also worked on hardware (Sec V-A). Low-pass-filtered push/glide indicators plus a "still" flag (Table II).
- Rewards (Table IV): glide feet-on-board 0.3, contact-number 0.3, feet distance 1.8, joint pos 1.2; push lin-vel 1.6, yaw 0.8, orientation −2; wheels-down 0.8, body height exp(−4|Δz−0.15|) 1.0, action rate −0.22, base orientation −25.
- Without the Beta policy: total failure. Single critic: 40–60% success (Table III).
- Randomization (Table VI): friction 0.6–2.0, mass 0–3 kg, push 0.5 m/s every 8 s, delay 0–20 ms. PPO: 4096 envs, 24 steps, λ 0.9, KL 0.01 (Table V); 6–7 h on a 3090, flat ground only (App E).

**PAPL / FiLM (2602.09370, Go1)**
- Steering δ=arctan(γ1·sin(γ2·ψ)) with γ1F=1.0, γ2F=1.12, γ1R=0.7, γ2R=0.9: a 10° roll gives 11.0° front and 6.2° rear axle yaw (Sec III-A, IV-C). Board 0.69×0.27×0.13 m, 0.43 m wheelbase.
- Wheel friction has static, viscous and rolling terms, but **r, μr, τstatic and cω are not given**.
- Phase Tϕ~U(4,12) s: carve ϕ∈[0.2π,0.8π], push [1.2π,1.8π], transition otherwise (no reward; mounting emerges).
- Rewards (Table III): command 5.0·exp(−ε/0.3); body/feet/posture 2.0·exp(−ε/σ) with σ 0.2/0.3/0.4; contact pattern −2.0 (carve: 4 feet on deck; push: right feet on deck, left pushing); clearance −0.4 (0.1 m); termination −6; collision −2.5; foot-on-edge −2.
- Randomization (Table I): payload 0–1.5 kg, CoM ±0.05 m, PD ±10%, deck 3.5–4.5 kg, bushing 30–35 N·m/rad.
- Ablations (Sec V-B): **proprio-only (no exteroception) overturned and drifted often**; no privileged critic stood still; mixture-of-experts collapsed. Hardware results are qualitative only (Sec V-D).

**Reverse-curriculum mounting (2505.06561, A1, sim only)**
- Board 2.1 kg, 575×250 mm; truck β≈α (45° kingpin); bushing k=2 N·m/rad (Sec III-B).
- Spawn on the board first, then widen. A wide initial-state distribution converges to **one foot on the ground** (Sec III-F). Fixed-board policy transferred to a free board.
- PPO (Table II): lr 1e-3 adaptive, 5 epochs, entropy 0.01, λ 0.95.

**Blind Dexterity (2608.29487, G1)**
- Proprio-only mounting: 88–90% success vs 8.2% distilled and 28±39% privileged (Table III). The last action exposes the PD residual, which acts as a contact sensor; dropping it hurt badly (Sec V-A).
- History K=5, asymmetric critic, entropy 0.001, 8000 iterations (Sec IV-A). Reward weights unpublished.

## 2. Recommended sim board model for Pupper (MuJoCo)

**Pupper facts (MJCF FK):**
- Mass **3.17 kg**, not ~2 kg; weigh the robot.
- Hips at x ±0.075 m, y ±0.08. Upper leg 0.085 m, lower leg 0.088 m, foot r=0.02.
- At stand, feet sit at x +0.066/−0.084 and y ±0.10, 0.141 m below the base origin.
- The foot centre reaches at most **0.17 m below the base origin** (0.164 at |y|=0.14).

**Scaling:**
- **Deck top ≤ ~0.06 m.** In a crouch the base sits ~0.11 m above the deck, so a push foot touches the ground only if the deck is ≤ ~0.08 m high. Standard decks are 110–130 mm. Check at the Nov 16 fit tests.
- **Deck ~0.40×0.18 m, board mass 0.5–1.0 kg (guesses).** Feet pull slightly inward on the deck; a side foot reaches the ground at |y|≈0.13.

**Bodies:**
```
board (freejoint)              chassis frame S, ~0.1 kg
 ├─ deck  hinge "deck_roll" axis x, stiffness=k_b, damping=c_b, range ±0.35
 │        box 0.40×0.18×0.012, friction 1.0
 ├─ axle_F (x=+0.13) hinge "yaw_F" axis z, range ±0.4
 │    └─ wheel_FL/FR hinge axis y, damping=c_w, frictionloss=τ_s, sphere r≈0.025
 └─ axle_R (x=−0.13) same, "yaw_R"
```
- **Bushing:** static roll stability needs k_b > m·g·h ≈ 31 N×0.13 m ≈ 4 N·m/rad. A 3 cm CoM shift should give ~10° roll, so k_b ≈ 5.4. Start at **k_b=6, c_b=0.3 (guess)**, then fit to a hanging-weight roll test.
- **Steering:** (a) native `<equality><joint joint1="yaw_F" joint2="deck_roll" polycoef="0 1.12 0 0 0"/>`, rear −0.63 (γ1γ2; linear is fine to ~15°). **Verify MuJoCo-Warp supports joint equalities.** (b) Fallback as in both Go1 papers: position actuators on the axles, with targets set each step by the arctan law.
- **Wheels:** hinge wheels, not explicit forces. Joint `damping` maps to PAPL's cω and `frictionloss` to τstatic plus rolling resistance, which removes DHAL's compensation-force hack. Fit both to a coast-down test (push to 1 m/s, time the stop). Sphere wheels give clean contacts; lateral friction makes the board non-holonomic.
- Turn radius ≈ L/(tanδF+tanδR) ≈ 0.86 m at 10° roll with L=0.26 m.

## 3. Recommended training plan

| Dates | Stage | Exit bar |
|---|---|---|
| Nov 9–11 | Board MJCF; coast-down and lean→turn tests vs the real board | Within 30% (guess) |
| Nov 11–13 | S1 balance on a welded board | 20 s under impulses |
| Nov 13–16 | S2 free board, initial v~U(0,0.6) | **Checkpoint:** 20 s sim rides |
| Nov 16–21 | S3 steer by lean while coasting; first hardware run (TA sign-off) | Yaw error <0.3 rad/s |
| Nov 20–27 | S4 push + recover with phase clock | **Gate Nov 25:** else demo S3 after a hand push |
| Nov 27–Dec 1 | S5 joystick vx/yaw, hardware loop | Video Dec 2 |

**Actor obs (no C++ changes):** `base_ang_vel, projected_gravity, command, joint_pos_rel, joint_vel_rel, last_action, gait_reference`. Use a 5–20-frame history and an MLP no bigger than the shipped 960→256→128→128→128 (19 ms budget). Board speed is unobservable while gliding (leg kinematics reveal it only during push stance), so keep the vx reward loose.

**Critic obs (privileged):** board pose and velocity in the body frame, ψ, ψ̇, δF/δR, per-foot deck/ground contact, foot positions in the board frame, randomization parameters.

**Phase clock via `gait_reference`** (`gait_reference.hpp`; wall-clock driven, so 52 Hz doesn't skew it):
- Table rows = default_pos + [sinϕ, cosϕ, carve, push, 0…].
- Set `dir_threshold=0`, `gallop_speed` above max vx, `gallop_table` = a copy, and clip vx≥0, or the phase runs backwards.
- A zero command blends the signal to zero, which works as DHAL's "still" flag. Training must mirror mjlab's `reference_offset`.
- Add `heading_hold:{kp:0}` (gait_reference enables heading hold by default). Fixed T per export, e.g. 3 s (guess).

**Needs C++ changes:** FiLM layers, the Beta head (mean α/(α+β) isn't a dense layer), DHAL's encoders, board state. Use a Gaussian policy with initial std 0.3 and action_scale 0.5 (guesses).

**Rewards** (single critic; PAPL Table III weights, distance σ halved for Pupper's size, guess):
- All phases: alignment 2.0; body height 1.0·exp(−4|Δz−0.11|) (0.11 guess); wheels down 0.8; termination −6; collision −2.5; foot-on-edge −2; orientation −2, raised later toward DHAL's −25; smoothness at pupper-mjlab defaults.
- Glide (S1–S3): yaw 5.0·exp(−|cω−ωz|/0.3); body over deck 2.0 (σ 0.1); feet 2.0 (σ 0.15); posture 2.0 (σ 0.4); contact −2.0·(4−n_deck); ground contact −1 per foot (DHAL P_ground, blocks the kickstand).
- Push (S4): vx 5.0·exp(−ε/0.3); stance contact −2.0·(2−n); clearance −0.4 below 0.05 m (guess).

**Terminations:** tilt >0.8 rad, non-foot contact, board roll >0.5 rad, base >0.25 m from the board (guess; PAPL 0.5 m). 20 s episodes.

**Randomization:** payload 0–0.3 kg and CoM ±0.02 m (guesses, PAPL ratio); friction 0.6–1.5; kp 5.0–6.0, kd 0.2–0.3; board mass ±15%; k_b ±40% and wheel loss ±50% until measured (guesses); delay 0–20 ms (DHAL); impulses every 3 s. Spawn on the board first.

**PPO:** 4096 envs, 24 steps, 5 epochs, 4 minibatches, lr 1e-3 adaptive, KL 0.01, γ 0.99, λ 0.95, clip 0.2, entropy 0.005. 5–8k iterations per stage, warm-starting each from the last.

## 4. Risks, ranked

1. **Reach.** ~~Pushing off a normal deck is impossible.~~ Resolved by the v2 split stance (§5): standard decks work with ~10–15° body roll; a ≤8 cm deck keeps the body level.
2. **Degenerate optima:** standing still (PAPL), one foot down (mounting paper), glide-only, avoiding the board (DHAL Sec IV-B). Counter with the phase clock and contact terms.
3. **Proprio-only drift** (PAPL ablation). Blind Dexterity suggests history plus last action can suffice. Demo short and on flat ground.
4. **Board physics mismatch.** Measure, fit, randomize.
5. **Torque:** 3 N·m with soft kp 5.5. Log saturation in sim.
6. **Deploy format:** dense MLP only.
7. **52 vs 50 Hz:** phase is fine (wall clock); the 4% step mismatch is covered by delay randomization.
8. **E-stop:** default max_body_angle is 0.52 rad (1.5 in the student slot); set it explicitly. Tether the board.

## 5. v2: split stance (team decision, 2026-11-09)

The team's plan: one side's two feet ride the deck and the other side's two feet stay on the floor to balance or push, like a scooter. Pupper never puts all four feet on the board. Left/right split assumed; "deck side" = left below (pick one, like regular/goofy). **This supersedes the 4-feet-on-deck parts of §2–3.**

**Geometry (MJCF FK, feet within ±4 cm of the hip in x):**
- Ground-side foot depth below the base: 0.10–0.17 m. Deck-side foot depth: 0.045–0.17 m with the foot at |y| 0.06–0.10.
- With a **level body**, deck height h is feasible while (ground depth) − h ≥ 0.045. With ground legs at a working depth of ~0.13–0.14 (room for a push stroke, guess), **h ≤ ~0.08 m** keeps the body level.
- **A standard 11–13 cm deck also works** with the body rolled ~10–15° toward the ground side (≈0.2 m lateral foot spread × sin θ absorbs 3.5–5 cm). That is under the 0.52 rad default e-stop. So **reach is no longer a blocker**; a low deck still gives more push stroke.

**Stability: always keep ≥3 contacts.** Two feet on the same side form a line, not a support polygon, so lifting both ground feet means balancing a 3 kg robot on a knife-edge. Recommended gait:
- Deck feet (FL, BL) stay planted on the deck centerline, which keeps their load from leaning the deck and steering by accident.
- Ground feet (FR, BR) **alternate**: one pushes (stance, sweeping backward relative to the body) while the other swings forward. That is a two-leg walk on the ground side and a static stance on the deck side. "Coast" = both ground feet down and still, or one lightly hovering (guess: drag braking is small at these speeds).
- This maps onto `gait_reference` with no C++ changes: phase the table so only FR/BR move (deck-side rows stay at the riding pose). Warm-start from the shipped walking policy is plausible since the ground side is walking.

**Board model changes:**
- Deck load is ~half the weight (~15 N), so scale the bushing: k_b ≈ 3 N·m/rad (guess, half of §2). Deck can be narrower (≥0.12 m; the feet only need the centerline).
- Spawn with the board under the deck-side feet at y ≈ +0.10 and the board parallel to the body.
- Steering now has two levers: deck lean (deck feet shifting toe/heel) and ground feet pushing sideways or angled. The second is easier and likely dominates; reward yaw tracking and let the policy pick.

**Reward/termination changes from §3:**
- Contact: −2.0 per deck-side foot off the deck; −2.0 per ground-side foot touching the deck; −1.0 when fewer than 3 feet are in contact (guess).
- Delete DHAL's ground-contact penalty and the "4 feet on deck" term; the "one foot down" optimum (mounting paper) is now the intended behavior.
- New failure modes to guard: **splits** (board drifts away laterally or ahead) → terminate when the deck centerline is >0.06 m (guess) laterally from the deck-side hips or yaws >0.5 rad from the body; **shuffling** (deck feet walk instead of ride) → penalize deck-foot slip and air time on the deck side.
- Body roll target: 0 on a low deck; on a standard deck use the measured static roll as the posture target rather than penalizing it.

**Simplified plan:** S1 split-stance balance on a welded board → S2 free board, hand-pushed start, coast and hold line → S3 ground-leg push/recover (gate Nov 25) → S4 yaw commands. Nov 16 checkpoint = S1–S2 in sim plus the physical fit test (stand Pupper in split stance on the actual board and measure body roll).

**Candidate board (Walmart Mammygol 23" mini cruiser, $33.99, arrives Nov 12):** deck 23×6.5 in (0.58×0.165 m) PP plastic with a 30° kicktail; 60×45 mm 78A light-up wheels; 3 in trucks. Deck height and mass are not listed: estimate ~0.10 m and ~1.5–2 kg (guesses), so measure both. Bushings are sized for 30–100 kg riders, so expect almost no lean-steer under ~15 N unless loosened or swapped for soft ones.

## What changed from v1
- Stance is split (2 deck + 2 ground), not 4 on the deck. Reach risk downgraded from blocker to "lower deck is better"; standard decks work with ~10–15° roll.
- New ≥3-contact alternating ground-leg gait; new "splits" and "shuffling" failure modes; ground-contact penalty removed.
- Bushing stiffness halved for half the deck load.

## Accounting
- Skipped: pupper-mjlab source (not in the snapshot), so mjlab APIs and reward defaults are unverified.
- Assumed: geometry from the MJCF; "guess" numbers are physics-scaled estimates.
- Noticed: PAPL omits its wheel constants; DHAL kept a foot attached.
