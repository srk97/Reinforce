import tensorflow as tf

from .registry import register
from .utils import HParams


@register
def default():
  return HParams(
      model=None,
      sys=None,
      env=None,
      agent=None,
      reward_augmentation=None,
      output_dir=None,
      episode=0,
      steps=0,
      memory_update_priorities=False,
      memory_priority_control=0,
      memory_priority_compensation=1,
      n_steps=1,
      seed=1234,
      state_latent_size=256,
      state_processor="CNN")
