import numpy as np

from ..utils.logger import log_scalar
from .algos.action_function.registry import get_action_function
from .algos.compute_gradient.registry import get_gradient_function
from ..models.registry import get_models


class Agent():

  def __init__(self, sess, model, env, memory, hparams):
    self._sess = sess
    self._model = model
    self._env = env
    self._memory = memory
    self._hparams = hparams
    self._all_rewards = []
    self._episode_rewards = []
    self._action_function = get_action_function(self._hparams.action_function)
    self._state_processor_vars = None
    self.masks = None
    if hparams.pixel_input:
      self._state_processor = get_models(hparams, names=hparams.state_processor)
    self._grad_function = get_gradient_function(self._hparams.grad_function)
    self.build()

  def _log_rewards(self, rewards):
    self._episode_rewards.extend(rewards)
    self._all_rewards.extend(rewards)

  def _log_results(self, reset_episode=False):
    total_episode_rewards = np.sum(self._episode_rewards)
    total_rewards = np.sum(self._all_rewards)
    rewards_mean = np.divide(total_rewards, self._hparams.episode + 1)

    log_scalar("epsiode_rewards", total_episode_rewards)
    log_scalar("total_rewards", total_rewards)
    log_scalar("rewards_mean", rewards_mean)

    episode_msg = "episode %d\trewards: %f" % (self._hparams.episode + 1,
                                               total_episode_rewards)
    if 'epsilon' in self._hparams.action_function:
      episode_msg += "\tepsilon: %.4f" % (self._hparams.epsilon)

    print(episode_msg)

    if reset_episode:
      self._episode_rewards = []

  def process_states(self, states):
    """ return processed raw pixel input otherwise return raw states"""
    if self._hparams.pixel_input:
      return self._state_processor(states)
    return states

  def build(self):
    """Construct TF graph."""
    raise NotImplementedError

  def observe(self, last_states, actions, rewards, done, states):
    """Allow agent to update internal state, etc.

    Args:
      last_state: list of Tensor of previous states.
      action: list of Tensor of actions taken to reach `state`.
      reward: list of Tensor of rewards received from environment.
      state: list of Tensor of new states.
      done: list of boolean indicating completion of episode.
    """
    raise NotImplementedError

  def act(self, state):
    """Select an action to take.

    Args:
      state: a Tensor of states.
    Returns:
      action: a Tensor of the selected action.
    """
    raise NotImplementedError

  def update(self):
    """Called at the end of an episode. Compute updates to models, etc."""
    raise NotImplementedError

  def get_all_rewards(self):
    return self._all_rewards

  def set_all_rewards(self, all_rewards):
    self._all_rewards = all_rewards
