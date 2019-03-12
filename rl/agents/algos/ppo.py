import tensorflow as tf

from .utils import compute_discounted_rewards, one_hot
from ..agent import Agent
from ..registry import register
from ...utils.logger import log_scalar
from ...models.registry import get_models


@register
class PPO(Agent):
  """ Proximal Policy Optimization """

  def __init__(self, sess, models, env, memory, hparams):
    self._actor = models['PPOActor']
    self._critic = models['PPOCritic']
    self._old_policy = get_models(hparams, names="PPOActor")
    super().__init__(sess, models, env, memory, hparams)

  def act(self, state):
    action_distribution = self._sess.run(
        self.probs, feed_dict={self.states: state[None, :]})
    return self._action_function(self._hparams, action_distribution)

  def observe(self, last_states, actions, rewards, done, states):
    discounts = [self._hparams.gamma] * len(last_states)
    actions = one_hot(actions, self._hparams.num_actions)

    self._memory.add_samples(last_states, actions, rewards, discounts, done,
                             states)

    self._log_rewards(rewards)

    if done[-1]:
      self._log_results(reset_episode=True)
      self.update()

  def build(self):
    self.states = tf.placeholder(
        tf.float32, [None] + self._hparams.state_shape, name="states")
    self.rewards = tf.placeholder(tf.float32, [None], name="rewards")
    self.actions = tf.placeholder(
        tf.int32, [None, self._hparams.num_actions], name="actions")

    states = self.process_states(self.states)

    # compute discounted reward
    with tf.variable_scope("reward_discount"):
      discounted_reward = compute_discounted_rewards(self._hparams.gamma,
                                                     self.rewards)

      if self._hparams.normalize_reward:
        mean, var = tf.nn.moments(discounted_reward, 0)
        # avoid division by zero
        std = tf.sqrt(var) + 1e-10
        discounted_reward = (discounted_reward - mean) / std

    self.logits = self._actor(states)
    self.probs = tf.nn.softmax(self.logits, -1)

    self.oldpi_logits = self._old_policy(states, scope="old_policy")

    self.value = self._critic(states)

    self.advantage = discounted_reward - self.value

    self.losses, self.train_ops = self._grad_function(
        {
            "oldpi_logits": self.oldpi_logits,
            "curr_logits": self.logits
        },
        self.actions,
        self.advantage,
        self._hparams,
        var_list={
            "actor_vars":
            tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="PPOActor"),
            "critic_vars":
            tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="PPOCritic")
        })

  def update(self):
    if self._hparams.training:
      pi_vars, oldpi_vars = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES,
          scope="PPOActor"), tf.get_collection(
              tf.GraphKeys.TRAINABLE_VARIABLES, scope="old_policy")

      pi_vars = sorted(pi_vars, key=lambda v: v.name)
      oldpi_vars = sorted(oldpi_vars, key=lambda v: v.name)

      replace_op = [
          tf.assign(oldpi, pi) for oldpi, pi in zip(oldpi_vars, pi_vars)
      ]

      self._sess.run(replace_op)

      _, _, states, actions, rewards, _, _ = self._memory.sample()

      for _ in range(self._hparams.num_actor_steps):
        a_loss, _ = self._sess.run(
            [self.losses['a_loss'], self.train_ops['a_train_op']],
            feed_dict={
                self.states: states,
                self.actions: actions,
                self.rewards: rewards,
            })
        log_scalar("actor_loss", a_loss)

      for _ in range(self._hparams.num_critic_steps):
        c_loss, _ = self._sess.run(
            [self.losses['c_loss'], self.train_ops['c_train_op']],
            feed_dict={
                self.states: states,
                self.actions: actions,
                self.rewards: rewards,
            })
        log_scalar("critic_loss", c_loss)

    self._memory.clear()
