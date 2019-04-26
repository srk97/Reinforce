import tensorflow as tf

from ..model import Model
from ..registry import register


@register
class GibsonPixelProcessor(Model):

  def __init__(self, hparams):
    super().__init__(hparams)
    self.layer1 = tf.layers.Conv2D(filters=32,
                                   kernel_size=([hparams.kernel_sizes[0]] * 2),
                                   strides=([hparams.strides[0]] * 2),
                                   activation=tf.nn.relu)
    self.layer2 = tf.layers.Conv2D(filters=64,
                                   kernel_size=([hparams.kernel_sizes[1]] * 2),
                                   strides=([hparams.strides[1]] * 2),
                                   activation=tf.nn.relu)
    self.layer3 = tf.layers.Conv2D(filters=32,
                                   kernel_size=([hparams.kernel_sizes[2]] * 2),
                                   strides=([hparams.strides[2]] * 2),
                                   activation=tf.nn.relu)

    self.layer4 = tf.layers.Flatten()
    self.layer5 = tf.layers.Dense(hparams.hidden_size, activation=tf.nn.relu)

  def call(self, states, scope="GibsonPixelProcessor"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      layer = self.layer1(states)
      layer = self.layer2(layer)
      layer = self.layer3(layer)
      layer = self.layer4(layer)
      cnn_output = self.layer5(layer)

      return cnn_output


@register
class GibsonPPOActor(Model):

  def __init__(self, hparams):
    super().__init__(hparams)
    self.layer1 = tf.keras.layers.GRU(units=hparams.hidden_size,
                                      return_state=True)
    self.layer2 = tf.keras.layers.Dense(hparams.num_actions)

  def call(self, states, hidden_states, masks, scope="GibsonPPOActor"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

      def fn_act():
        return self.layer1(states, initial_state=hidden_states)

      def fn_update():
        T = self._hparams.n_steps

        states = tf.reshape(states, [1, T, states.shape[1:]])

        indices = tf.where(tf.equal(masks, True))
        indices_final = tf.dynamic_partition(indices,
                                             tf.range(indices.shape[0]),
                                             indices.shape[0])

        indices_final.insert(tf.constant([[0]], dtype=tf.int64), 0)
                                             
        for i in range(len(indices_final)):


      cond_op = tf.cond(tf.equal(tf.shape(0), 1), fn_act, fn_update)
      states = tf.expand_dims(states, axis=0)
      rnn_out, new_hidden = self.layer1(states, initial_state=hidden_states)
      out = self.layer2(rnn_out)

      return rnn_out, new_hidden, out


@register
class GibsonPPOCritic(Model):

  def __init__(self, hparams):
    super().__init__(hparams)
    self.layer1 = tf.layers.Dense(1)

  def call(self, states, scope="GibsonPPOCritic"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      return self.layer1(states)
