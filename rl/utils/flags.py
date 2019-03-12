import os
import time
import getpass
import subprocess
import tensorflow as tf

from .sys import get_sys


def log_hparams(hparams):
  if not tf.gfile.Exists(hparams.output_dir):
    tf.gfile.MakeDirs(hparams.output_dir)

  with tf.gfile.GFile(os.path.join(hparams.output_dir, 'hparams.txt'),
                      'a') as file:
    file.write("{}\n".format(time.ctime()))
    file.write("{}\n".format(str(vars(hparams))))


def validate_flags(FLAGS):
  messages = []
  if not FLAGS.sys:
    messages.append("Missing required flag --sys")
  if not FLAGS.hparams:
    messages.append("Missing required flag --hparams")

  if len(messages) > 0:
    raise Exception("\n".join(messages))

  return FLAGS


def update_hparams(FLAGS, hparams):
  hparams.train_episodes = FLAGS.train_episodes
  hparams.eval_episodes = FLAGS.eval_episodes
  hparams.copies = FLAGS.copies
  hparams.render = FLAGS.render
  hparams.record_video = FLAGS.record_video
  hparams.save_every = FLAGS.save_every
  hparams.sys = FLAGS.sys
  hparams.env = FLAGS.env or hparams.env
  hparams.training = FLAGS.training
  if hparams.env is None:
    print("please specify training environment")
    exit()

  sys = get_sys(FLAGS.sys)
  hparams.output_dir = FLAGS.output_dir or os.path.join(
      sys.output_dir, getpass.getuser(), FLAGS.hparams)

  log_hparams(hparams)

  return hparams
