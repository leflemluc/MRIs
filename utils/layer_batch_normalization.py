import tensorflow as tf

def layer_batch_normalization(x, n_out, phase_train):
    """
    Defines the network layers
    input:
        - x: input vector of the layer
        - n_out: integer, depth of input maps - number of sample in the batch
        - phase_train: boolean tf.Varialbe, true indicates training phase
    output:
        - batch-normalized maps
    """

    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)

    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    # tf.nn.moment: https://www.tensorflow.org/api_docs/python/tf/nn/moments
    # calculate mean and variance of x
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')

    # tf.train.ExponentialMovingAverage:
    # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    # Maintains moving averages of variables by employing an exponential decay.
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    # tf.cond: https://www.tensorflow.org/api_docs/python/tf/cond
    # Return true_fn() if the predicate pred is true else false_fn()
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var, beta, gamma, 1e-3, True)
    # normed = tf.nn.batch_normalization(reshaped_x, mean, var, beta, gamma, 1e-3, True)

    return tf.reshape(normed, [-1, n_out])
