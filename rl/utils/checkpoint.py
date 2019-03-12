import os
import pickle
import tensorflow as tf


class Checkpoint():
  """ Save and restore using tf.train.Saver()

  Save hparams to pickle file with format
  {
    checkpoints: list of checkpoint_path chronological  order
    checkpoint_path: (hparams, all_rewards)
  }
  """

  def __init__(self, sess, hparams, agent):
    self._sess = sess
    self._hparams = hparams
    self._agent = agent
    # list of checkpoints
    self._checkpoints = []
    self._run_dir = hparams.run_output_dir
    self._pickle = os.path.join(self._run_dir, 'checkpoint.pickle')
    self._saver = tf.train.Saver()

  def save(self):
    if not self._hparams.training:
      return

    save_path = os.path.join(self._run_dir,
                             'model.ckpt-%d' % self._hparams.episode)
    path_prefix = self._saver.save(self._sess, save_path)

    self._checkpoints.append(path_prefix)

    with open(self._pickle, "wb") as file:
      pickle.dump({
          'checkpoints': self._checkpoints,
          path_prefix: (self._hparams, self._agent.get_all_rewards())
      }, file)

    print("saved checkpoint at %s" % path_prefix)

  def restore(self):
    """ Restore from latest checkpoint
    Returns:
      restored: boolean, True if restored from a checkpoint, False otherwise.
    """
    latest_checkpoint = tf.train.latest_checkpoint(self._run_dir)

    if latest_checkpoint is None:
      if not self._hparams.training:
        raise FileNotFoundError("no checkpoint found in %s" % self._run_dir)
      return False

    self._saver.restore(self._sess, latest_checkpoint)

    with open(self._pickle, "rb") as file:
      checkpoints = pickle.load(file)

      self._checkpoints = checkpoints['checkpoints']
      (checkpoint, all_rewards) = checkpoints[latest_checkpoint]

      # restore hparams
      self._hparams.steps = checkpoint.steps
      if self._hparams.training:
        self._hparams.episode = checkpoint.episode

      # restore rewards
      self._agent.set_all_rewards(all_rewards)

    return True
