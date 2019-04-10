import tensorflow as tf

from .registry import register
from .utils import clip_grad_norm


@register
def mean_squared_error(preds, targets, hparams, weights=1.0):
  loss = tf.losses.mean_squared_error(
      predictions=preds, labels=targets, weights=weights)

  train_op = tf.train.AdamOptimizer(
      learning_rate=hparams.learning_rate).minimize(loss)
  return loss, train_op


@register
def ppo(logits, actions, advantage, hparams, var_list):
  '''
  logits: A dict containing logits corresponding to old and current policies
  var_list: A dict containing trainable variables of the actor and critic
  '''

  oldpi_logits, new_logits = logits['oldpi_logits'], logits['curr_logits']
  critic_loss = tf.reduce_mean(tf.square(advantage))

  def log_probs(prob_dist):
    return -tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prob_dist, labels=actions)

  ratio = tf.exp(log_probs(new_logits) - log_probs(oldpi_logits))

  clipped_ratio = tf.clip_by_value(ratio, 1 - hparams.clipping_coef,
                                   1 + hparams.clipping_coef)
  surrogate_objective = tf.minimum(clipped_ratio * advantage, ratio * advantage)

  actor_loss = -tf.reduce_mean(surrogate_objective)
  actor_train_op = tf.train.AdamOptimizer(hparams.actor_lr)
  critic_train_op = tf.train.AdamOptimizer(hparams.critic_lr)
  cnn_train_op = tf.no_op()

  if hparams.pixel_input:
    cnn_train_op = tf.train.AdamOptimizer(hparams.critic_lr)
    cnn_loss = (actor_loss + critic_loss) / 2
    cnn_train_op = cnn_train_op.minimize(
        cnn_loss, var_list=var_list['state_processor_vars'])

  if hparams.clip_grad_norm:
    actor_train_op = clip_grad_norm(actor_train_op, actor_loss,
                                    hparams.max_grad_norm,
                                    var_list['actor_vars'])
    critic_train_op = clip_grad_norm(critic_train_op, critic_loss,
                                     hparams.max_grad_norm,
                                     var_list['critic_vars'])
  else:
    actor_train_op = actor_train_op.minimize(
        actor_loss, var_list=var_list['actor_vars'])
    critic_train_op = critic_train_op.minimize(
        critic_loss, var_list=var_list['critic_vars'])

  return {
      "a_loss": actor_loss,
      "c_loss": critic_loss
  }, {
      "a_train_op": actor_train_op,
      "c_train_op": critic_train_op,
      "cnn_train_op": cnn_train_op
  }
