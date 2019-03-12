import numpy as np
import tensorflow as tf


def compute_discounted_rewards(gamma, rewards):

  size = tf.to_float(tf.shape(rewards)[0])
  mask = tf.sequence_mask(tf.range(size, 0, -1), size, dtype=tf.float32)

  def fn(i):
    return tf.pow(gamma, tf.range(size - i - 1, -i - 1, -1, dtype=tf.float32))

  gammas = mask * tf.map_fn(fn, tf.range(size, dtype=tf.float32))
  discounted_reward = tf.reduce_sum(gammas * rewards[None, ::-1], -1)

  return discounted_reward


def one_hot(indices, depth):
  return list(np.eye(depth)[indices])
