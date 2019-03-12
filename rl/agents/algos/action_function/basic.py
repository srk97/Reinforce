import random
import numpy as np
import tensorflow as tf

from .registry import register


@register
def random_action(hparams, distribution):
  return random.randint(0, hparams.num_actions - 1)


@register
def epsilon_action(hparams, distribution):
  if random.random() < hparams.epsilon:
    return random_action(hparams, distribution)
  else:
    return max_action(hparams, distribution)


@register
def max_action(hparams, distribution):
  return np.argmax(distribution)


@register
def non_uniform_random_action(hparams, distribution):
  return np.random.choice(range(hparams.num_actions), p=distribution.ravel())


@register
def uniform_random_action(hparams, distribution):
  h = np.random.uniform(size=distribution.shape)
  return np.argmax(distribution - np.log(-np.log(h)))


@register
def normal_noise_action(hparams, action):
  return np.clip(
      np.random.normal(action, hparams.variance), hparams.action_low,
      hparams.action_high)
