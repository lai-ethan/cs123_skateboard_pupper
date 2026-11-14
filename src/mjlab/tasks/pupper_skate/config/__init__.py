from mjlab.tasks.pupper.rl import PupperOnPolicyRunner
from mjlab.tasks.pupper_gait.rl_cfg import pupper_gait_ppo_runner_cfg
from mjlab.tasks.pupper_skate.pupper_skate_env_cfg import (
  pupper_skate_balance_env_cfg,
  pupper_skate_ride_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task


def _skate_runner_cfg():
  """Separate experiment dir so skate runs do not land among the gait runs."""
  cfg = pupper_gait_ppo_runner_cfg()
  cfg.experiment_name = "pupper_skate"
  return cfg


register_mjlab_task(
  task_id="Mjlab-Skate-Balance-Pupper-v3",
  env_cfg=pupper_skate_balance_env_cfg(),
  play_env_cfg=pupper_skate_balance_env_cfg(play=True),
  rl_cfg=_skate_runner_cfg(),
  runner_cls=PupperOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Skate-Ride-Pupper-v3",
  env_cfg=pupper_skate_ride_env_cfg(),
  play_env_cfg=pupper_skate_ride_env_cfg(play=True),
  rl_cfg=_skate_runner_cfg(),
  runner_cls=PupperOnPolicyRunner,
)
