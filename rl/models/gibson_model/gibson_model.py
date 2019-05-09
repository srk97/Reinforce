import tensorflow as tf
import numpy as np
from tensorflow import AUTO_REUSE as reuse

from ..model import Model
from ..registry import register


@register
class GibsonPixelProcessor(Model):

  def __init__(self, hparams, name="GibsonPixelProcessor"):
    super().__init__(hparams, name)
    self.layer1 = tf.layers.Conv2D(
        filters=32,
        kernel_size=([hparams.kernel_sizes[0]] * 2),
        strides=([hparams.strides[0]] * 2),
        activation=tf.nn.relu)
    self.layer2 = tf.layers.Conv2D(
        filters=64,
        kernel_size=([hparams.kernel_sizes[1]] * 2),
        strides=([hparams.strides[1]] * 2),
        activation=tf.nn.relu)
    self.layer3 = tf.layers.Conv2D(
        filters=32,
        kernel_size=([hparams.kernel_sizes[2]] * 2),
        strides=([hparams.strides[2]] * 2),
        activation=tf.nn.relu)

    self.layer4 = tf.layers.Flatten()
    self.layer5 = tf.layers.Dense(hparams.hidden_size, activation=tf.nn.relu)

  def call(self, states):
    with tf.variable_scope(self.name, reuse=reuse):
      layer = self.layer1(states)
      layer = self.layer2(layer)
      layer = self.layer3(layer)
      layer = self.layer4(layer)
      cnn_output = self.layer5(layer)

      return cnn_output


@register
class GibsonPPOActor(Model):

  def __init__(self, hparams, name="GibsonPPOActor"):
    super().__init__(hparams, name)
    self.layer1 = tf.keras.layers.GRU(
        units=hparams.hidden_size, return_state=True)
    self.layer2 = tf.keras.layers.Dense(hparams.num_actions)

  def call(self, states, hidden_states, masks=None):
    with tf.variable_scope(self.name, reuse=reuse):
      if masks is None:
        states = tf.expand_dims(states, axis=0)
        #hidden_states = tf.expand_dims(hidden_states, axis=0)
        rnn_out, hidden_states = self.layer1(
            states, initial_state=hidden_states)
      else:
        done_ = np.where(masks == True)[0]
        indices = np.concatenate(([0], done_, [128]))
        out = []
        for i in range(len(indices) - 1):
          start_idx = indices[i]
          end_idx = indices[i + 1]
          rnn_input = tf.expand_dims(states[start_idx:end_idx], axis=0)
          rnn_out, hidden_states = self.layer1(
              rnn_input, initial_state=hidden_states * masks[start_idx])
          out.append(rnn_out)
        rnn_out = tf.concat(out, axis=1)

      out = self.layer2(rnn_out)

      return rnn_out, hidden_states, out


@register
class GibsonPPOCritic(Model):

  def __init__(self, hparams, name="GibsonPPOCritic"):
    super().__init__(hparams, name)
    self.layer1 = tf.layers.Dense(1)

  def call(self, states):
    with tf.variable_scope(self.name, reuse=reuse):
      return self.layer1(states)
