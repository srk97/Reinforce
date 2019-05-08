import numpy as np
import tensorflow as tf


def compute_discounted_rewards(hparams, rewards):
  with tf.variable_scope("reward_discount"):
    size = tf.to_float(tf.shape(rewards)[0])
    mask = tf.sequence_mask(tf.range(size, 0, -1), size, dtype=tf.float32)

    def fn(i):
      return tf.pow(hparams.gamma,
                    tf.range(size - i - 1, -i - 1, -1, dtype=tf.float32))

    gammas = mask * tf.map_fn(fn, tf.range(size, dtype=tf.float32))
    discounted_reward = tf.reduce_sum(gammas * rewards[None, ::-1], -1)

    if hparams.normalize_reward:
      mean, var = tf.nn.moments(discounted_reward, 0)
      # avoid division by zero
      std = tf.sqrt(var) + 1e-10
      discounted_reward = (discounted_reward - mean) / std

    return discounted_reward


def discounted_rewards_episode(rewards, discount):
  future_cumulative_reward = 0
  discounted_rewards = np.empty_like(rewards, dtype=np.float32)
  for i in range(len(rewards) - 1, -1, -1):
    discounted_rewards[i] = rewards[i] + discount * future_cumulative_reward
    future_cumulative_reward = discounted_rewards[i]

  return discounted_rewards


def compute_discounted_rewards_numpy(hparams, rewards, done, last_value=None):
  if last_value is not None:
    rewards = np.concatenate((rewards, [last_value]))
  # Add 1 so indices mark episode starts.
  indices = np.where(done == True)[0] + 1
  indices = list(np.concatenate(([0], indices, [len(rewards)])))
  discounted_rewards = []
  for idx in range(len(indices) - 1):
    start_idx = indices[idx]
    end_idx = indices[idx + 1]
    rewards_slice = rewards[start_idx:end_idx]
    discounted_rewards_slice = discounted_rewards_episode(
        rewards_slice, hparams.gamma)
    discounted_rewards = np.concatenate(
        (discounted_rewards, discounted_rewards_slice))
  if last_value is not None:
    discounted_rewards = discounted_rewards[:-1]
  return discounted_rewards


def one_hot(indices, depth):
  return list(np.eye(depth)[indices])


def copy_variables_op(source, target):

  source_vars = sorted(source.trainable_weights, key=lambda v: v.name)
  target_vars = sorted(target.trainable_weights, key=lambda v: v.name)

  return [
      tf.assign(target_var, source_var)
      for target_var, source_var in zip(target_vars, source_vars)
  ]
