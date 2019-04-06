import tensorflow as tf

_MODELS = dict()


def register(fn):
  global _MODELS
  _MODELS[fn.__name__] = fn
  return fn


def get_models(hparams, names=None):
  '''
  names: a string or a list of models to be fetched
  scope_name: Scope within which the model is defined
  '''

  if isinstance(names, str):
    return _MODELS[names](hparams)
  elif isinstance(names, list):
    models = {name: _MODELS[name](hparams) for name in names}
    return models
  elif len(hparams.models) == 1:
    return _MODELS[hparams.models[0]](hparams)
  else:
    print(_MODELS)
    return {name: _MODELS[hparams.models[name]](hparams) for name in hparams.models.keys()}
