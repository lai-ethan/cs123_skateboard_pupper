"""Print a skate training run's key scalars and a PASS/FAIL verdict.

Usage (repo root, after training):
  python skateboard/tools/run_summary.py Mjlab-Skate-Balance-Pupper-v3
  python skateboard/tools/run_summary.py <task-id> --run-dir logs/rsl_rl/<exp>/<run>

Reads the TensorBoard event files rsl_rl writes into the run directory (the W&B
logger writes them too). The last line is `RESULT: PASS`, `RESULT: FAIL` or,
when there is nothing to read, `RESULT: UNKNOWN`. The exit code is always 0.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_RIDE_CRITERIA = {
  "Train/mean_episode_length": 450.0,
  "Episode_Reward/track_board_velocity": 2.0,
  # Penalties, logged as weight x mean cost: near zero means rarely paid.
  "Episode_Reward/joint_speed_limit": -0.1,
  "Episode_Reward/torque_saturation": -0.2,
}
# Minimum value of each tag, averaged over the run's last points.
PASS_CRITERIA: dict[str, dict[str, float]] = {
  "Mjlab-Skate-Balance-Pupper-v3": {
    "Train/mean_episode_length": 450.0,
    # +1 per foot on the deck: 3.6 is four feet down 90% of the time.
    "Episode_Reward/feet_on_deck": 3.6,
  },
  "Mjlab-Skate-Ride-Pupper-v3": dict(_RIDE_CRITERIA),
  "Mjlab-SkateSplit-Balance-Pupper-v3": {
    "Train/mean_episode_length": 450.0,
    # Two feet on the deck: 1.8 is both down 90% of the time.
    "Episode_Reward/feet_on_deck": 1.8,
  },
  "Mjlab-SkateSplit-Ride-Pupper-v3": dict(_RIDE_CRITERIA),
}
SHOWN_PREFIXES = ("Train/", "Episode_Reward/", "Episode_Termination/")


def latest_run_dir(log_root: Path) -> Path:
  """Newest run directory (`log_root/<experiment>/<run>`) by mtime."""
  runs = [p for p in Path(log_root).glob("*/*") if p.is_dir()]
  if not runs:
    raise FileNotFoundError(f"No runs under {log_root}: train first.")
  return max(runs, key=lambda p: p.stat().st_mtime)


def _load_series(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
  """Every scalar tag in the event files directly inside run_dir, by step."""
  by_tag: dict[str, dict[int, float]] = {}
  for path in sorted(Path(run_dir).glob("events.out.tfevents.*")):
    if not path.is_file():
      continue
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    for tag in acc.Tags()["scalars"]:
      points = by_tag.setdefault(tag, {})
      for event in acc.Scalars(tag):
        points[event.step] = event.value
  return {tag: sorted(points.items()) for tag, points in by_tag.items()}


def read_scalars(run_dir: Path, last_n: int = 20) -> dict[str, float]:
  """Mean of the last `last_n` points of every scalar tag in the run."""
  return {
    tag: sum(v for _, v in series[-last_n:]) / len(series[-last_n:])
    for tag, series in _load_series(run_dir).items()
  }


def verdict(
  task: str, scalars: dict[str, float]
) -> list[tuple[str, float, float, bool]]:
  """(tag, value, minimum, passed) per criterion; a missing tag fails as nan."""
  rows = []
  for tag, minimum in PASS_CRITERIA[task].items():
    value = scalars.get(tag, math.nan)
    rows.append((tag, value, minimum, value >= minimum))
  return rows


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Summarize a skate training run.")
  parser.add_argument("task", choices=sorted(PASS_CRITERIA))
  parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
  parser.add_argument("--run-dir", type=Path, default=None)
  args = parser.parse_args(argv)

  try:
    run_dir = args.run_dir or latest_run_dir(args.log_root)
  except FileNotFoundError as err:
    print(err)
    print("RESULT: UNKNOWN")
    return 0
  print("run:", run_dir)
  series = _load_series(run_dir)
  if not series:
    print(
      "No TensorBoard event files in this run; the summary needs the"
      " TensorBoard logger."
    )
    print("RESULT: UNKNOWN")
    return 0

  # Tags ending in /time are logged against elapsed seconds, not iterations.
  steps = [s[-1][0] for tag, s in series.items() if not tag.endswith("/time")]
  print("iterations:", max(steps) + 1)
  scalars = read_scalars(run_dir)
  for tag in sorted(scalars):
    if tag.startswith(SHOWN_PREFIXES):
      print(f"  {tag:<45} {scalars[tag]:10.3f}")
  rows = verdict(args.task, scalars)
  for tag, value, minimum, passed in rows:
    print(f"{'PASS' if passed else 'FAIL'}  {tag} = {value:.3f} (need >= {minimum})")
  print("RESULT:", "PASS" if all(row[3] for row in rows) else "FAIL")
  return 0


if __name__ == "__main__":
  sys.exit(main())
