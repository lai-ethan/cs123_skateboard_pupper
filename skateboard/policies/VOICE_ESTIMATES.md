# Voice estimates for the shipped skate policies

Sim only. These are the numbers Gemini Pupper quotes for each policy in its
`skate/skate.yaml` catalogue. They are estimates, not measurements on the robot.

Method: `skateboard/tools/timed_ride.py <policy json> --yaw-rate=<y>` at 0.3 m/s,
16 boards, 5 s ride, 4 s coast, hard-floor ranges, for y in 0, ±0.15, ±0.3, ±0.5.
Rounding: travel = the straight run's `m_per_s` to 2 decimals; each turn = the
smaller of the left and right `turn_deg_per_s` sizes, to 1 decimal; coast = the
largest `coast_m_max` over the seven runs, rounded up to 1 decimal.

| Policy | travel_m_per_s | turn gentle / medium / hard (deg/s) | coast_m |
| --- | --- | --- | --- |
| `split_steer_v6_max0.3_policy.json` | 0.29 | 1.1 / 1.4 / 2.2 | 0.1 |
| `split_steer_v10_max0.3_policy.json` | 0.30 | 0.8 / 0.9 / 1.1 | 0.2 |

v6's runs are recorded in `PLAN_GEMINI_VOICE.md`. v10 turns less to the left than
to the right at every carve level, so its turn estimates are the left-carve values.

## steer v10 runs (2026-11-27, no resets, every turn the right way)

```
{"stick": 0.0, "resets": 0, "envs": 16, "ride_m_mean": 1.504, "m_per_s": 0.301, "turn_deg_mean": -0.982, "turn_deg_per_s": -0.196, "coast_m_mean": 0.068, "coast_m_max": 0.091}
{"stick": 0.15, "resets": 0, "envs": 16, "ride_m_mean": 1.485, "m_per_s": 0.297, "turn_deg_mean": 4.055, "turn_deg_per_s": 0.811, "coast_m_mean": 0.065, "coast_m_max": 0.091}
{"stick": -0.15, "resets": 0, "envs": 16, "ride_m_mean": 1.492, "m_per_s": 0.298, "turn_deg_mean": -6.104, "turn_deg_per_s": -1.221, "coast_m_mean": 0.069, "coast_m_max": 0.118}
{"stick": 0.3, "resets": 0, "envs": 16, "ride_m_mean": 1.508, "m_per_s": 0.302, "turn_deg_mean": 4.515, "turn_deg_per_s": 0.903, "coast_m_mean": 0.068, "coast_m_max": 0.1}
{"stick": -0.3, "resets": 0, "envs": 16, "ride_m_mean": 1.493, "m_per_s": 0.299, "turn_deg_mean": -7.647, "turn_deg_per_s": -1.529, "coast_m_mean": 0.067, "coast_m_max": 0.117}
{"stick": 0.5, "resets": 0, "envs": 16, "ride_m_mean": 1.539, "m_per_s": 0.308, "turn_deg_mean": 5.634, "turn_deg_per_s": 1.127, "coast_m_mean": 0.074, "coast_m_max": 0.104}
{"stick": -0.5, "resets": 0, "envs": 16, "ride_m_mean": 1.544, "m_per_s": 0.309, "turn_deg_mean": -11.929, "turn_deg_per_s": -2.386, "coast_m_mean": 0.073, "coast_m_max": 0.121}
```
