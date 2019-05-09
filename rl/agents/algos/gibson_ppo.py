import tensorflow as tf
import numpy as np

from .utils import compute_discounted_rewards_numpy, one_hot
from ..agent import Agent
from ..registry import register
from .utils import copy_variables_op
from ...utils.logger import log_scalar
from ...models.registry import get_model


@register
class Gibson_PPO(Agent):
  """ Proximal Policy Optimization """

  def __init__(self, sess, hparams):
    super().__init__(sess, hparams)
    self._actor = get_model(hparams, register="GibsonPPOActor", name="actor")
    self._critic = get_model(hparams, register="GibsonPPOCritic", name="critic")
    self.target_actor = get_model(hparams,
                                  register="GibsonPPOActor",
                                  name="target_actor")
    self.build()

  def act(self, state, worker_id, recurrent_state=None):
    self.masks = None
    if self._hparams.env == 'gibson_env':
      state_pixel = np.concatenate((state['rgb'], state['depth']),
                                   axis=2)[None, :]
      point_goal = state['pointgoal'][None, :]
      hidden_states, action_distribution = self._sess.run(
          [self.computed_recurrent_states, self.probs],
          feed_dict={
              self.last_states: state_pixel,
              self.recurrent_states: recurrent_state,
              self.point_goal: point_goal
          })
      hidden_states = np.squeeze(hidden_states, axis=0)
      return self._action_function(self._hparams,
                                   action_distribution), hidden_states
    else:
      action_distribution = self._sess.run(
          self.probs, feed_dict={self.last_states: state[None, :]})
      return self._action_function(self._hparams, action_distribution)

  def observe(self,
              last_state,
              action,
              reward,
              done,
              state,
              worker_id=0,
              last_recurrent_state=None,
              recurrent_state=None):

    action = one_hot(action, self._hparams.num_actions)
    memory = self._memory[worker_id]
    last_state_rgbd = np.concatenate((last_state['rgb'], last_state['depth']),
                                     axis=2)
    state_rgbd = np.concatenate((state['rgb'], state['depth']), axis=2)
    last_pointgoal, pointgoal = last_state['pointgoal'], state['pointgoal']

    memory.add_sample(last_state=last_state_rgbd,
                      action=action,
                      reward=reward,
                      discount=self._hparams.gamma,
                      done=done,
                      state=state_rgbd,
                      last_recurrent_state=last_recurrent_state,
                      recurrent_state=recurrent_state,
                      last_pointgoal=last_pointgoal,
                      pointgoal=pointgoal)

    if memory.size() == self._hparams.num_steps:
      self.update(worker_id)

  def reset(self, worker_id=0):
    self._memory[worker_id].clear()

  def clone_weights(self):
    self.target_actor.set_weights(self._actor.get_weights())

  def _build_target_update_op(self):
    with tf.variable_scope("update_target_networks"):
      self.target_update_op = copy_variables_op(source=self._actor,
                                                target=self.target_actor)

  def update_targets(self):
    self._sess.run(self.target_update_op)

  def build(self):

    if self._hparams.env == 'gibson_env':
      self.point_goal = tf.placeholder(tf.float32, [None, 2], name='pointgoals')
      self.recurrent_states = tf.placeholder(tf.float32,
                                             [None, self._hparams.hidden_size],
                                             name='recurrent_states')
    self.last_states = tf.placeholder(tf.float32,
                                      [None] + self._hparams.state_shape,
                                      name="states")
    self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
    self.discounted_rewards = tf.placeholder(tf.float32, [None],
                                             name="discounted_rewards")
    self.actions = tf.placeholder(tf.int32, [None, self._hparams.num_actions],
                                  name="actions")

    processed_states = self.process_states(self.last_states)

    if self._hparams.pixel_input:
      self.cnn_vars = self._state_processor.trainable_weights
    else:
      self.cnn_vars = None

    states_critic = processed_states
    print(self._hparams.env)
    if self._hparams.env == 'gibson_env':
      actor_states = tf.concat([processed_states, self.point_goal], axis=1)
      _, self.computed_recurrent_states, self.logits = self._actor(
          actor_states, self.recurrent_states, self.masks)
      _, _, self.target_logits = self.target_actor(actor_states,
                                                   self.recurrent_states,
                                                   self.masks)
    else:
      actor_states = processed_states
      self.logits = self._actor(processed_states)
      self.target_logits = self.target_actor(actor_states, scope="old_policy")

    self.probs = tf.nn.softmax(self.logits, -1)

    self.values = self._critic(states_critic)[:, 0]

    losses, train_ops = self._grad_function(
        logits={
            "target_logits": self.target_logits,
            "logits": self.logits
        },
        actions=self.actions,
        advantages=self.advantages,
        values=self.values,
        discounted_rewards=self.discounted_rewards,
        hparams=self._hparams,
        var_list={
            "actor_vars": self._actor.trainable_weights,
            "critic_vars": self._critic.trainable_weights,
            "cnn_vars": self.cnn_vars
        })

    self.actor_loss = losses['actor_loss']
    self.critic_loss = losses['critic_loss']
    self.actor_train_op = train_ops['actor_train_op']
    self.critic_train_op = train_ops['critic_train_op']
    self.state_processor_train_op = train_ops['state_processor_train_op']

    self._build_target_update_op()

  def update(self, worker_id=0):
    if not self._hparams.training:
      return

    memory = self._memory[worker_id]
    states = np.concatenate((
        memory.get_sequence('last_state'),
        memory.get_sequence('state', indices=[-1]),
    ))

    rewards = memory.get_sequence('reward')
    dones = memory.get_sequence('done')
    self.masks = dones
    values = self._sess.run(self.values, feed_dict={self.last_states: states})
    discounted_rewards = compute_discounted_rewards_numpy(
        self._hparams, rewards, dones, values[-1])
    memory.set_sequence('discounted_reward', discounted_rewards)
    advantages = discounted_rewards - values[:-1]
    if self._hparams.normalize_reward:
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    memory.set_sequence('advantage', advantages)

    for _ in range(self._hparams.num_epochs):
      for batch in memory.shuffled_batches(self._hparams.batch_size,
                                           randomize=False):
        feed_dict = {
            self.last_states: batch.last_state,
            self.actions: batch.action,
            self.advantages: batch.advantage,
            self.discounted_rewards: batch.discounted_reward,
            self.recurrent_states: batch.recurrent_state,
            self.point_goal: batch.pointgoal
        }

      self._sess.run(self.state_processor_train_op, feed_dict=feed_dict)

      actor_loss, _ = self._sess.run([self.actor_loss, self.actor_train_op],
                                     feed_dict=feed_dict)
      log_scalar("loss/actor/worker_%d" % worker_id, actor_loss)

      critic_loss, _ = self._sess.run([self.critic_loss, self.critic_train_op],
                                      feed_dict=feed_dict)
      log_scalar("loss/critic/worker_%d" % worker_id, critic_loss)

      memory.clear()

      self.update_targets()
