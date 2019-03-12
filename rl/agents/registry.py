import tensorflow as tf

from .memory import ProportionalMemory
_AGENTS = dict()


def register(fn):
  global _AGENTS
  _AGENTS[fn.__name__] = fn
  return fn


def get_agent(sess, model, env, memory, hparams):
  return _AGENTS[hparams.agent](sess, model, env, memory, hparams)


def get_memory(hparams):
  return ProportionalMemory(
    hparams.memory_size,
    hparams.memory_priority_control,
    hparams.memory_priority_compensation)
