'''
Environment wrapper for habitat-RL 
Code taken from habitat-api/baselines/train_ppo.py
'''

import random

import numpy as np

import habitat
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.sims.habitat_simulator import SIM_NAME_TO_ACTION, SimulatorActions

from .registry import register_env
from .utils import cfg as cfg_baseline
'''
Snippet for experimenting with Environment

from rl.envs.gibson import gibson_env
from rl.hparams.gibson_ppo import ppo
hparams = ppo()
env = gibson_env(hparams)

while True:
  action = env.action_space.sample()
  obs, rewards, done, _ = env.step(action)
  plt.imshow(obs['rgb'])
  plt.show()
  _ = input('Press [enter] to con')
  plt.close()

'''


class NavRLEnv(habitat.RLEnv):

  def __init__(self, config_env, config_baseline, dataset):
    self._config_env = config_env.TASK
    self._config_baseline = config_baseline
    self._previous_target_distance = None
    self._previous_action = None
    self._episode_distance_covered = None
    super().__init__(config_env, dataset)

  def reset(self):
    self._previous_action = None

    observations = super().reset()

    self._previous_target_distance = self.habitat_env.current_episode.info[
        "geodesic_distance"]
    return observations

  def step(self, action):
    self._previous_action = action
    base_return = super().step(action)

    return base_return

  def get_reward_range(self):
    return (
        self._config_baseline.BASELINE.RL.SLACK_REWARD - 1.0,
        self._config_baseline.BASELINE.RL.SUCCESS_REWARD + 1.0,
    )

  def get_reward(self, observations):
    reward = self._config_baseline.BASELINE.RL.SLACK_REWARD

    current_target_distance = self._distance_target()
    reward += self._previous_target_distance - current_target_distance
    self._previous_target_distance = current_target_distance

    if self._episode_success():
      reward += self._config_baseline.BASELINE.RL.SUCCESS_REWARD

    return reward

  def _distance_target(self):
    current_position = self._env.sim.get_agent_state().position.tolist()
    target_position = self._env.current_episode.goals[0].position
    distance = self._env.sim.geodesic_distance(current_position,
                                               target_position)
    return distance

  def _episode_success(self):
    if (self._previous_action == SIM_NAME_TO_ACTION[SimulatorActions.STOP.value]
        and self._distance_target() < self._config_env.SUCCESS_DISTANCE):
      return True
    return False

  def get_done(self, observations):
    done = False
    if self._env.episode_over or self._episode_success():
      done = True
    return done

  def get_info(self, observations):
    info = {}

    if self.get_done(observations):
      info["spl"] = self.habitat_env.get_metrics()["spl"]

    return info


@register_env
def gibson_env(hparams):
  basic_config = cfg_env(
      config_file=hparams.task_config,
      config_dir='/home/paperspace/Habitat-TF/rl/envs/configs')
  scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
  config_env = cfg_env(
      config_file=hparams.task_config,
      config_dir='/home/paperspace/Habitat-TF/rl/envs/configs')
  config_env.defrost()

  if len(scenes) > 0:
    random.shuffle(scenes)
    config_env.DATASET.POINTNAVV1.CONTENT_SCENES = scenes
  for sensor in hparams.sensors:
    assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
  config_env.SIMULATOR.AGENT_0.SENSORS = hparams.sensors
  config_env.freeze()
  config_baseline = cfg_baseline()

  dataset = PointNavDatasetV1(config_env.DATASET)

  config_env.defrost()
  config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
  config_env.freeze()

  env = NavRLEnv(
      config_env=config_env, config_baseline=config_baseline, dataset=dataset)

  return env
