import tensorflow as tf


def clip_grad_norm(train_opt,
                   loss,
                   clip_value,
                   var_list=tf.get_collection(
                       tf.GraphKeys.TRAINABLE_VARIABLES)):
  grads = train_opt.compute_gradients(loss, var_list=var_list)
  clipped_grads = [
      (tf.clip_by_norm(grad, clip_value), var) for grad, var in grads
  ]
  train_op = train_opt.apply_gradients(clipped_grads)

  return train_op
