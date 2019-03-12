import tensorflow as tf

from ..model import Model
from ..registry import register


@register
class basic(Model):

  def __init__(self, hparams):
    super().__init__(hparams)
    self.layer1 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)
    self.layer2 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)
    self.layer3 = tf.layers.Dense(units=hparams.num_actions)

  def call(self, states, scope="model"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      layer = self.layer1(states)
      layer = self.layer2(layer)
      return self.layer3(layer)


@register
class PPOActor(Model):

  def __init__(self, hparams):
    super().__init__(hparams)
    self.layer1 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)
    self.layer2 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)
    self.layer3 = tf.layers.Dense(units=hparams.num_actions)

  def call(self, states, scope="PPOActor"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      layer = self.layer1(states)
      layer = self.layer2(layer)
      return self.layer3(layer)


@register
class PPOCritic(Model):

  def __init__(self, hparams):
    super().__init__(hparams)
    self.layer1 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)
    self.layer2 = tf.layers.Dense(units=1)

  def call(self, states, scope="PPOCritic"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      layer = self.layer1(states)
      return self.layer2(layer)
