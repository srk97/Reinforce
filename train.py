import os
import random
import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor

from rl.utils import flags
from rl.envs.registry import get_env
from rl.utils.checkpoint import Checkpoint
from rl.utils.logger import init_logger
from rl.models.registry import get_models
from rl.hparams.registry import get_hparams
from rl.agents.registry import get_agent, get_memory


def init_flags():
  tf.flags.DEFINE_string("hparams", None, "Which hparams to use.")
  tf.flags.DEFINE_string("sys", None, "Which system environment to use.")
  tf.flags.DEFINE_string("env", None, "Which RL environment to use.")
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_integer("train_episodes", 400,
                          "Number of episodes to train the agent")
  tf.flags.DEFINE_integer("eval_episodes", 10,
                          "Number of episodes to evaluate the agent")
  tf.flags.DEFINE_integer("save_every", 100,
                          "Number of episodes to save one checkpoint")
  tf.flags.DEFINE_boolean("training", True, "training or testing")
  tf.flags.DEFINE_integer("copies", 1,
                          "Number of independent training/testing runs to do.")
  tf.flags.DEFINE_boolean("render", False, "Render game play")
  tf.flags.DEFINE_boolean("record_video", False, "Record game play")


def init_random_seeds(hparams):
  tf.set_random_seed(hparams.seed)
  random.seed(hparams.seed)
  np.random.seed(hparams.seed)


def init_hparams(FLAGS):
  flags.validate_flags(FLAGS)

  tf.reset_default_graph()

  hparams = get_hparams(FLAGS.hparams)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams)

  return hparams


def log_start_of_run(FLAGS, hparams, run):
  tf.logging.warn(
      "\n-----------------------------------------\n"
      "BEGINNING RUN #%s:\n"
      "\t hparams: %s\n"
      "\t env: %s\n"
      "\t agent: %s\n"
      "\t output_dir: %s\n"
      "-----------------------------------------\n" %
      (run, FLAGS.hparams, hparams.env, hparams.agent, hparams.output_dir))


def _run(FLAGS):
  hparams = init_hparams(FLAGS)

  init_random_seeds(hparams)

  for run in range(hparams.copies):
    hparams.run_output_dir = os.path.join(hparams.output_dir, 'run_%d' % run)
    log_start_of_run(FLAGS, hparams, run)

    with tf.Session() as sess:
      init_logger(hparams)
      env = get_env(hparams)
      models = get_models(hparams)
      memory = get_memory(hparams)
      agent = get_agent(sess, models, env, memory, hparams)
      checkpoint = Checkpoint(sess, hparams, agent)

      if hparams.training:
        if len(hparams.models) > 1:
          for name, model in models.items():
            model.save_graph()
        else:
          models.save_graph()

      restored = checkpoint.restore()
      if not restored:
        sess.run(tf.global_variables_initializer())

      max_episodes = hparams.train_episodes if hparams.training else hparams.eval_episodes
      while hparams.episode < max_episodes:

        state = env.reset()
        done_ = False

        # run until game is finished
        while not done_:

          last_states = []
          actions = []
          rewards = []
          done = []
          states = []

          # run n steps
          for _ in range(hparams.n_steps):

            if hparams.render:
              env.render()

            last_state = state

            action = agent.act(state)
            state, reward, done_, info = env.step(action)

            last_states.append(last_state)
            actions.append(action)
            rewards.append(reward)
            done.append(done_)
            states.append(state)

            hparams.steps += 1

            if done_:
              break

          agent.observe(last_states, actions, rewards, done, states)

        hparams.episode += 1

        if hparams.episode % hparams.save_every == 0 or hparams.episode == max_episodes:
          checkpoint.save()

      env.close()

    hparams = init_hparams(FLAGS)


def main(_):
  FLAGS = tf.app.flags.FLAGS
  _run(FLAGS)


if __name__ == "__main__":
  init_flags()
  tf.app.run()
