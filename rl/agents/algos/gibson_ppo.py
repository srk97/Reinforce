import tensorflow as tf
import numpy as np

from .utils import compute_discounted_rewards, one_hot
from ..agent import Agent
from ..registry import register
from ...utils.logger import log_scalar
from ...models.registry import get_models


@register
class Gibson_PPO(Agent):
  """ Proximal Policy Optimization """

  def __init__(self, sess, models, env, memory, hparams):
    self._actor = models['actor']
    self._critic = models['critic']
    self._old_policy = get_models(hparams, names=hparams.models['actor'])
    super().__init__(sess, models, env, memory, hparams)

  def act(self, state, recurrent_state=None):
    self.masks = None
    if type(self._env).__name__ == 'NavRLEnv':
      state_pixel = np.concatenate((state['rgb'], state['depth']),
                                   axis=2)[None, :]
      point_goal = state['pointgoal'][None, :]
      hidden_states, action_distribution = self._sess.run(
          [self.computed_recurrent_states, self.probs],
          feed_dict={
              self.states: state_pixel,
              self.recurrent_states: recurrent_state,
              self.point_goal: point_goal
          })
      hidden_states = np.squeeze(hidden_states, axis=0)
      return self._action_function(self._hparams,
                                   action_distribution), hidden_states
    else:
      action_distribution = self._sess.run(
          self.probs, feed_dict={self.states: state[None, :]})
      return self._action_function(self._hparams, action_distribution)

  def observe(self,
              last_states,
              actions,
              rewards,
              done,
              states,
              last_recurrent_states=None,
              recurrent_states=None):
    discounts = [self._hparams.gamma] * len(last_states)
    actions = one_hot(actions, self._hparams.num_actions)

    if type(self._env).__name__ == 'NavRLEnv':
      last_rgbd = []
      rgbd = []
      last_point_goal = []
      point_goal = []
      #print(type(last_states))
      for i in range(len(last_states)):
        last_rgbd.append(
            np.concatenate((last_states[i]['rgb'], last_states[i]['depth']),
                           axis=2))
        rgbd.append(
            np.concatenate((states[i]['rgb'], states[i]['depth']), axis=2))
        last_point_goal.append(last_states[i]['pointgoal'])
        point_goal.append(states[i]['pointgoal'])

      self._memory.add_samples(last_rgbd, actions, rewards, discounts, done,
                               rgbd, last_recurrent_states, recurrent_states,
                               last_point_goal, point_goal)
    else:
      self._memory.add_samples(last_states, actions, rewards, discounts, done,
                               states)
    self._log_rewards(rewards)

    if done[-1]:
      self._log_results(reset_episode=True)
      self.update()

  def build(self):

    if type(self._env).__name__ == 'NavRLEnv':
      self.point_goal = tf.placeholder(tf.float32, [None, 2], name='pointgoals')
      self.recurrent_states = tf.placeholder(
          tf.float32, [None, self._hparams.hidden_size],
          name='recurrent_states')
    self.states = tf.placeholder(
        tf.float32, [None] + self._hparams.state_shape, name="states")
    self.rewards = tf.placeholder(tf.float32, [None], name="rewards")
    self.actions = tf.placeholder(
        tf.int32, [None, self._hparams.num_actions], name="actions")

    processed_states = self.process_states(self.states)

    if self._hparams.pixel_input:
      self.cnn_vars = self._state_processor.trainable_weights
    else:
      self.cnn_vars = None

    states_critic = processed_states

    # compute discounted reward
    with tf.variable_scope("reward_discount"):
      discounted_reward = compute_discounted_rewards(self._hparams.gamma,
                                                     self.rewards)

      if self._hparams.normalize_reward:
        mean, var = tf.nn.moments(discounted_reward, 0)
        # avoid division by zero
        std = tf.sqrt(var) + 1e-10
        discounted_reward = (discounted_reward - mean) / std

    if type(self._env).__name__ == 'NavRLEnv':
      print(tf.shape(processed_states), tf.shape(self.point_goal))
      actor_states = tf.concat([processed_states, self.point_goal], axis=1)
      print(self.masks)
      _, self.computed_recurrent_states, self.logits = self._actor(
          actor_states, self.recurrent_states, self.masks)
      _, _, self.oldpi_logits = self._old_policy(
          actor_states, self.recurrent_states, self.masks, scope="old_policy")
    else:
      actor_states = processed_states
      self.logits = self._actor(processed_states)
      self.oldpi_logits = self._old_policy(actor_states, scope="old_policy")

    self.probs = tf.nn.softmax(self.logits, -1)

    self.value = self._critic(states_critic)

    self.advantage = discounted_reward - self.value

    #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    self.losses, self.train_ops = self._grad_function(
        {
            "oldpi_logits": self.oldpi_logits,
            "curr_logits": self.logits
        },
        self.actions,
        self.advantage,
        self._hparams,
        # Figure out a way to make state processor vars optional
        var_list={
            "actor_vars": self._actor.trainable_weights,
            "critic_vars": self._critic.trainable_weights,
            "state_processor_vars": self.cnn_vars
        })

  def update(self):

    if self._hparams.training:
      pi_vars, oldpi_vars = self._actor.trainable_weights, self._old_policy.trainable_weights
      pi_vars = sorted(pi_vars, key=lambda v: v.name)
      oldpi_vars = sorted(oldpi_vars, key=lambda v: v.name)

      replace_op = [
          tf.assign(oldpi, pi) for oldpi, pi in zip(oldpi_vars, pi_vars)
      ]

      self._sess.run(replace_op)

      if type(self._env).__name__ == 'NavRLEnv':
        _, _, states, recurrent_states, actions, rewards, self.masks, _, _, point_goals, _ = self._memory.sample(
        )
        print("Update shapes:")
        print("\tstates: ", states.shape)
        print("\trecurrent states: ", recurrent_states.shape)
        a_feed_dict = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.recurrent_states: recurrent_states,
            self.point_goal: point_goals
        }
      else:
        _, _, states, actions, rewards, done, _ = self._memory.sample()
        a_feed_dict = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards
        }

      if self._hparams.pixel_input:
        _ = self._sess.run([self.train_ops['cnn_train_op']],
                           feed_dict=a_feed_dict)

      for _ in range(self._hparams.num_update_steps):
        a_loss, _ = self._sess.run(
            [self.losses['a_loss'], self.train_ops['a_train_op']],
            feed_dict=a_feed_dict)

        c_loss, _ = self._sess.run(
            [self.losses['c_loss'], self.train_ops['c_train_op']],
            feed_dict={
                self.states: states,
                self.actions: actions,
                self.rewards: rewards,
            })

        log_scalar("actor_loss", a_loss)
        log_scalar("critic_loss", c_loss)

      self._memory.clear()
