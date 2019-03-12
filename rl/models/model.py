import os
import random
import tensorflow as tf

from ..utils.logger import log_graph


class Model(tf.keras.Model):

  def __init__(self, hparams):
    super().__init__()
    self._hparams = hparams

  def save_graph(self):
    """ write model graph to TensorBoard """
    log_graph()
